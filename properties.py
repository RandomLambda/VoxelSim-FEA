# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene-level settings and the lists of bearings / loads.

Holds the boundary-condition model (bearings/loads, compute-mode picker,
verbose log) for a single linear-elastic solve: material is a real isotropic
material (E in GPa, yield in MPa -- deliberately different units, matching
how datasheets are normally quoted), and there is no "keep-out" concept --
the part IS the analysis domain, voxelized as-is.
"""

import traceback

import bpy
from bpy.props import (
    PointerProperty, CollectionProperty, IntProperty, FloatProperty,
    FloatVectorProperty, BoolProperty, StringProperty, EnumProperty,
)
from bpy.types import PropertyGroup

from .core import backend


def _compute_mode_items(self, context):
    """Dynamic items for compute_mode. GPU/MULTI_GPU are only ever shown in
    the GPU edition (core/backend.py's GPU_BUILD flag -- see build_extension.py).

    MULTI_GPU / MULTI_CPU are offered but stay opt-in: AUTO never selects
    them below compute_plan.MULTI_GPU_DOF/MULTI_CPU_DOF, since a single
    solve only pays the multi-device dispatch/sync overhead once, and
    whether that one-time cost is worth it depends on the problem size and
    hardware.
    """
    items = [
        ('AUTO', "Auto (recommended)",
         "Pick CPU or GPU automatically from problem size: small grids stay "
         "on the CPU (a GPU would be slower once you count transfer "
         "overhead), larger grids move to the GPU (GPU edition only)"),
        ('CPU', "CPU (single process)",
         "Always numpy on one process, using the BLAS thread pool "
         "(see CPU Threads below)"),
    ]
    if backend.is_gpu_build():
        items.append((
            'GPU', "GPU (single device)",
            "Always the graphics card via Cu-Py; falls back to CPU "
            "automatically if Cu-Py/CUDA is not usable"))
        items.append((
            'MULTI_GPU', "Multi-GPU (all devices)",
            "Split the solve's matrix-vector product across every visible "
            "CUDA device. Only helps on a large grid AND only if your "
            "hardware/driver keeps the per-device dispatch overhead low -- "
            "this is a single solve, not hundreds of iterations, so that "
            "one-time overhead is paid once, but it is still not "
            "guaranteed to win. Falls back to single GPU if only one "
            "device is visible"))
    items.append((
        'MULTI_CPU', "Multi-process CPU",
        "Split the solve's matrix-vector product across worker processes "
        "instead of relying on BLAS threading alone. Mainly worth trying "
        "on a very large grid with no usable GPU"))
    return items


def ensure_material_preview_shading(context):
    """Flip every open 3D viewport to Material Preview shading. A vertex-
    color result overlay is baked as an Attribute->Emission material, which
    Blender's default Solid shading never evaluates -- without this the
    result object just looks like a plain grey/theme-colored solid and the
    result is invisible. Called whenever the visible result changes AND
    every time a solve finishes (so it also self-heals if a viewport was
    manually switched back to Solid/Wireframe between runs)."""
    try:
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
    except Exception:
        pass   # never let a UI convenience break the caller


def _on_result_view_update(self, context):
    ensure_material_preview_shading(context)
    # Deferred import: operators.py imports this module at load time, so
    # importing it back at properties.py's OWN module scope would be a real
    # cycle. Importing it here, inside a callback that only runs long after
    # both modules finished loading, is safe -- see operators.py's
    # apply_result_visibility for what this actually does (show the object
    # matching the new selection, hide the other three).
    from . import operators
    operators.apply_result_visibility(context, self)


def _on_inputs_visible_update(self, context):
    # Deferred import: same cycle-avoidance as _on_result_view_update above.
    from . import operators
    operators.apply_inputs_visibility(context, self)


def _on_color_mode_update(self, context):
    # Deferred import: same cycle-avoidance as _on_result_view_update above.
    # Purely a display setting -- no re-solve, just re-bake the last solve's
    # cached fields with the new shading (see operators.refresh_result_colors).
    from . import operators
    operators.refresh_result_colors(context)


def _on_wireframe_helpers_update(self, context):
    # Only the OFF transition acts: turning it ON just changes what happens
    # to helpers added FROM NOW ON (see _ListAdd.execute) -- existing solid
    # helpers are left alone, matching "wenn nicht, weiterhin als was es
    # war" (no forced re-wireframing on the ON edge). Turning it OFF forces
    # every already-wireframed helper back to solid immediately, so the
    # checkbox always reflects what's on screen rather than only affecting
    # objects added after the toggle.
    if self.wireframe_helpers:
        return
    from . import operators
    operators.force_helpers_solid(self)


def _is_mesh(self, obj):
    return obj.type == 'MESH'


class FEA_Bearing(PropertyGroup):
    """A support: voxels inside this object get their DOFs fixed."""
    obj: PointerProperty(
        type=bpy.types.Object, poll=_is_mesh,
        name="Support mesh",
        description="Where the part is held / anchored. Grid nodes inside "
                    "this mesh are clamped. Place a small closed mesh "
                    "overlapping the part where it is actually supported",
    )
    fix_x: BoolProperty(
        name="X", default=True,
        description="Prevent movement along X at this support. Turn off to "
                    "allow the part to slide along X here (e.g. a roller)")
    fix_y: BoolProperty(
        name="Y", default=True,
        description="Prevent movement along Y at this support")
    fix_z: BoolProperty(
        name="Z", default=True,
        description="Prevent movement along Z at this support")


class FEA_Load(PropertyGroup):
    """A load: force vector applied to voxels inside this object."""
    obj: PointerProperty(
        type=bpy.types.Object, poll=_is_mesh,
        name="Load mesh",
        description="Where an external force is applied. The force is "
                    "spread over the grid nodes inside this mesh. Place a "
                    "small closed mesh overlapping the part where the load "
                    "acts",
    )
    force: FloatVectorProperty(
        name="Force", subtype='XYZ', size=3, default=(0.0, 0.0, -100.0),
        description="Force direction and magnitude in NEWTONS (N), X/Y/Z. "
                    "Unlike topology optimization, this is a REAL load -- "
                    "it directly scales the reported stress and safety "
                    "factor. Example: (0, 0, -100) is 100 N straight down",
    )


class FEA_Settings(PropertyGroup):
    # --- Geometry inputs ---
    part: PointerProperty(
        name="Part", type=bpy.types.Object, poll=_is_mesh,
        description="The actual part to analyze -- this IS the analysis "
                    "domain, voxelized and solved exactly as modeled (no "
                    "material added or removed). Must be a closed "
                    "(watertight) mesh",
    )
    bearings: CollectionProperty(type=FEA_Bearing)
    bearings_index: IntProperty(default=0)
    loads: CollectionProperty(type=FEA_Load)
    loads_index: IntProperty(default=0)

    # --- Discretization ---
    resolution: IntProperty(
        name="Resolution", default=48, min=4, soft_max=200,
        description="Voxels along the part's longest edge. Higher = finer "
                    "stress detail but much slower and more memory (cost "
                    "grows roughly with the cube). 40-64 is a good start; "
                    "raise for thin features the coarse grid might miss",
    )

    # --- Compute backend ---
    show_compute_advanced: BoolProperty(
        name="", default=False,
        description="Show compute backend detail (thread count, device "
                    "status). Auto-opens when you pick a non-Auto Compute "
                    "mode",
    )
    compute_mode: EnumProperty(
        name="Compute", default=0,  # dynamic items require an index, not 'AUTO'
        update=lambda self, context: setattr(
            self, "show_compute_advanced", self.compute_mode != 'AUTO'),
        items=_compute_mode_items,
        description="Which compute backend the solver uses for its one "
                    "linear solve. Auto is right for almost everyone",
    )
    cpu_threads: IntProperty(
        name="CPU Threads", default=0, min=0, soft_max=64,
        description="BLAS/worker-process thread count. 0 = automatic (all "
                    "logical cores minus one)",
    )
    verbose_log: BoolProperty(
        name="Verbose solver log", default=True,
        description="Print the chosen compute backend, device/worker "
                    "counts, BLAS thread count and CG iteration/convergence "
                    "info to the console (Window > Toggle System Console)",
    )

    # --- Material. Units are deliberately DIFFERENT for the two fields,
    # matching how datasheets are normally quoted: stiffness in GPa (fewer
    # trailing zeros -- 210 instead of 210000000000), strength in MPa (300
    # instead of 300000000). Converted to Pa once, at the solve boundary --
    # see operators.py's _build_job / _run_inprocess_solve. ---
    material_preset: EnumProperty(
        name="Material", default='STEEL',
        items=[
            ('STEEL', "Steel", "Generic structural steel"),
            ('ALUMINUM', "Aluminum", "Generic structural aluminum"),
            ('ABS', "Plastic (ABS)", "Generic ABS engineering plastic"),
            ('PP', "Plastic (PP)", "Generic polypropylene"),
            ('CUSTOM', "Custom", "Enter your own E / Poisson / yield values"),
        ],
        description="Isotropic material preset. Applying a preset (button "
                    "below) copies reference values into the editable fields "
                    "-- pick Custom to enter your own from a datasheet",
    )
    youngs_modulus: FloatProperty(
        name="E", default=210.0, min=0.0001, soft_max=1000.0, precision=3,
        description="Young's modulus in GIGAPASCALS (GPa). Unlike topology "
                    "optimization, this is the REAL material stiffness -- it "
                    "directly sets the absolute displacement and stress "
                    "scale, not just a relative shape driver. "
                    "1 GPa = 1000 MPa = 1e9 Pa",
    )
    poisson: FloatProperty(
        name="Poisson", default=0.30, min=0.0, max=0.49,
        description="Poisson's ratio (how much the material bulges sideways "
                    "when compressed)",
    )
    yield_strength: FloatProperty(
        name="Yield Strength", default=300.0, min=0.0, soft_max=5000.0,
        description="Yield strength in MEGAPASCALS (MPa), used only to "
                    "compute the safety-factor result (yield / von Mises "
                    "stress). Set to 0 to disable the safety-factor result",
    )

    # --- Visualization: which single result object is currently shown.
    # All four are (re)built on every solve regardless of this setting (it's
    # cheap -- just coloring, no re-solve), so switching here is instant and
    # never re-runs the FEA. ---
    result_view: EnumProperty(
        name="Show Result", default='STRESS',
        update=_on_result_view_update,
        items=[
            ('NONE', "None", "Hide every result object"),
            ('STRESS', "Von Mises Stress",
             "Surface mesh colored by von Mises stress (blue = low, red = "
             "high). The color scale ignores the top ~1% of elements (the "
             "singular stress spikes that always sit right at point loads/"
             "supports on a voxel grid); the true unclamped max is still "
             "reported in the status line and the legend"),
            ('DISPLACEMENT', "Displacement",
             "Surface mesh colored by displacement magnitude in mm (blue = "
             "least movement, red = most)"),
            ('SAFETY', "Safety Factor",
             "Surface mesh colored by safety factor = yield / von Mises "
             "stress (blue/green = safe, red = at or below yield). "
             "Requires Yield Strength > 0"),
            ('CROSS_SECTION', "Cross-Section (3D)",
             "VoxelSimFEA_StressCloud: one real, colored cube per active "
             "voxel (von Mises stress), near-zero elements dropped, with a "
             "Geometry Nodes clip plane (VoxelSimFEA_ClipPlane Empty) so you "
             "can slice into the part"),
        ],
        description="Which single result object is visible right now -- "
                    "every result from the last solve stays available in "
                    "the 'VoxelSim FEA Results' collection, this just switches "
                    "which one is shown (and drives the on-screen legend)",
    )
    color_mode: EnumProperty(
        name="Shading", default='BANDED',
        update=_on_color_mode_update,
        items=[
            ('CONTINUOUS', "Pixel",
             "Full-resolution color gradient -- every voxel cell (surface "
             "results) or cube (cross-section) gets its own exact color "
             "along a smooth blue-to-red scale"),
            ('BANDED', "Contour",
             "Discrete color bands, like elevation contour lines: every "
             "value within a band gets the SAME flat color, with a crisp "
             "edge wherever the band changes -- the classic FEA 'banded "
             "contour' plot. Band count set below"),
        ],
        description="How the stress/displacement/safety color scale is "
                    "rendered. Purely a display setting -- re-colors the "
                    "last solve's results instantly, no re-run needed",
    )
    contour_bands: IntProperty(
        name="Bands", default=12, min=2, soft_max=40,
        update=_on_color_mode_update,
        description="Number of discrete color bands when Shading is set to "
                    "Contour",
    )

    # --- Viewport overlay toggles ---
    show_legend: BoolProperty(
        name="Show Legend", default=True,
        description="Show the on-screen color-scale legend in the 3D "
                    "viewport while a result is visible",
    )
    show_convergence: BoolProperty(
        name="Show Convergence Plot", default=True,
        description="Show the CG residual-vs-iteration overlay while "
                    "solving, and leave the finished curve on screen "
                    "afterwards",
    )
    inputs_visible: BoolProperty(
        name="Show Setup (Part / Supports / Loads)", default=True,
        update=_on_inputs_visible_update,
        description="Show the part and the support/load helper objects in "
                    "the viewport. Automatically turned off right after a "
                    "solve finishes, since they only ever cover the result "
                    "overlay -- toggle back on to edit the setup and "
                    "re-run",
    )
    wireframe_helpers: BoolProperty(
        name="Wireframe Helper-Objects", default=True,
        update=_on_wireframe_helpers_update,
        description="Show support/load helper objects as wireframe as soon "
                    "as they're assigned (the previous, always-on behaviour). "
                    "Turn off to leave newly-assigned helpers at whatever "
                    "display they already had. Turning this off also forces "
                    "every already-wireframed helper back to solid "
                    "immediately",
    )

    # --- Runtime state (not user-facing) ---
    running: BoolProperty(default=False)
    status: StringProperty(default="")


_CLASSES = (FEA_Bearing, FEA_Load, FEA_Settings)


def register():
    # NOTE: hasattr(cls, "bl_rna") is NOT a reliable "is this class
    # currently registered" check -- Blender never removes the bl_rna
    # attribute after unregistering a class, so hasattr keeps returning True
    # even though bpy.utils.unregister_class() would then raise. try/except
    # RuntimeError is the only reliable guard.
    registered = []
    try:
        for cls in _CLASSES:
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
            bpy.utils.register_class(cls)
            registered.append(cls)
        bpy.types.Scene.blendfea = PointerProperty(type=FEA_Settings)
    except Exception:
        print("blendfea.properties.register() failed -- see traceback below "
              "for the REAL cause (best-effort rollback follows, its own "
              "errors are swallowed so they don't hide this one):")
        traceback.print_exc()
        for cls in reversed(registered):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
        raise


def unregister():
    if hasattr(bpy.types.Scene, "blendfea"):
        del bpy.types.Scene.blendfea
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
