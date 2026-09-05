# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar UI for VoxelSim FEA (View3D > N panel) + an on-screen color-scale
legend drawn directly into the 3D viewport.

The legend is a small gradient bar + two value labels, drawn with the
``gpu``/``blf`` APIs via a ``SpaceView3D`` draw handler (the supported,
non-deprecated Blender drawing APIs). It shows whatever operators.py's
``set_legend`` was last called with (title/unit/bottom-of-bar/top-of-bar),
and only while a result object is selected to be visible
(``result_view != 'NONE'``).

Anchored bottom-right of each 3D viewport (computed from the region's own
width every frame, so it tracks viewport resizing) rather than bottom-left,
since Blender's own tool shelf/toolbar occupies the left edge of every 3D
viewport.
"""

import os
import traceback

import bpy
import blf
import gpu
from bpy.types import Panel
from gpu_extras.batch import batch_for_shader

from .core.backend import is_gpu_build, gpu_status, gpu_usable, gpu_device_count
from .core import materials
from .core.extract import _jet_rgb


class FEA_PT_base:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VoxelSim FEA"


class FEA_PT_main(FEA_PT_base, Panel):
    bl_idname = "FEA_PT_main"
    bl_label = "VoxelSim FEA"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendfea

        col = layout.column(align=True)
        col.prop(s, "part", text="Part")
        col.operator("blendfea.set_part", icon='MESH_CUBE')


class FEA_PT_bc(FEA_PT_base, Panel):
    bl_idname = "FEA_PT_bc"
    bl_parent_id = "FEA_PT_main"
    bl_label = "Supports & Loads"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendfea

        layout.prop(s, "wireframe_helpers")

        box = layout.box()
        box.label(text="Supports (bearings)", icon='CON_PIVOT')
        box.operator("blendfea.add_bearing", icon='ADD')
        for i, b in enumerate(s.bearings):
            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(b, "obj", text="")
            op = row.operator("blendfea.remove_item", text="", icon='X')
            op.collection_name = "bearings"
            op.index = i
            r2 = col.row(align=True)
            r2.prop(b, "fix_x")
            r2.prop(b, "fix_y")
            r2.prop(b, "fix_z")

        box = layout.box()
        box.label(text="Loads", icon='FORCE_FORCE')
        box.operator("blendfea.add_load", icon='ADD')
        for i, ld in enumerate(s.loads):
            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(ld, "obj", text="")
            op = row.operator("blendfea.remove_item", text="", icon='X')
            op.collection_name = "loads"
            op.index = i
            # ld.force's own property name has no unit suffix (a
            # FloatVectorProperty's overall label is thrown away by the X/Y/Z
            # sub-widget layout anyway) -- state the unit explicitly right
            # above the field instead of burying it in the tooltip only.
            col.label(text="Force (Newton, N):")
            col.prop(ld, "force", text="")


class FEA_PT_material(FEA_PT_base, Panel):
    bl_idname = "FEA_PT_material"
    bl_parent_id = "FEA_PT_main"
    bl_label = "Material"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendfea

        row = layout.row(align=True)
        row.prop(s, "material_preset", text="")
        # One confirm button for whichever material the dropdown above is
        # currently set to (reads s.material_preset itself, see
        # FEA_OT_apply_material_preset.execute) -- used to be one CHECKMARK
        # button per preset (4 buttons in a row), which just meant hunting
        # for the right one instead of picking from the dropdown.
        sub = row.row(align=True)
        sub.enabled = s.material_preset != 'CUSTOM'
        sub.operator("blendfea.apply_material_preset", text="",
                     icon='CHECKMARK')
        if s.material_preset != 'CUSTOM':
            preset = materials.get(s.material_preset)
            if preset:
                layout.label(text=preset["note"], icon='INFO')

        col = layout.column(align=True)
        split = col.split(factor=0.85, align=True)
        split.prop(s, "youngs_modulus", text="E")
        split.label(text="GPa")
        split = col.split(factor=0.85, align=True)
        split.prop(s, "poisson")
        split.label(text="-")
        split = col.split(factor=0.85, align=True)
        split.prop(s, "yield_strength", text="Yield Strength")
        split.label(text="MPa")


class FEA_PT_settings(FEA_PT_base, Panel):
    bl_idname = "FEA_PT_settings"
    bl_parent_id = "FEA_PT_main"
    bl_label = "Settings"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendfea

        col = layout.column(align=True)
        col.prop(s, "resolution")

        box = layout.box()
        header = box.row(align=True)
        header.prop(s, "show_compute_advanced",
                    icon='TRIA_DOWN' if s.show_compute_advanced else 'TRIA_RIGHT',
                    icon_only=True, emboss=False)
        header.prop(s, "compute_mode")
        if s.show_compute_advanced:
            col = box.column(align=True)
            threads_text = ("CPU Threads: 0 (auto)" if s.cpu_threads == 0
                            else "CPU Threads")
            col.prop(s, "cpu_threads", text=threads_text)
            cores = os.cpu_count() or 1
            col.label(text=f"CPU: {cores} logical cores detected", icon='INFO')
            if is_gpu_build():
                n_gpu = gpu_device_count() if gpu_usable() else 0
                col.label(text=gpu_status(),
                          icon='CHECKMARK' if gpu_usable() else 'INFO')
                if gpu_usable():
                    col.label(text=f"GPU devices visible: {n_gpu}")
            col.prop(s, "verbose_log")

        layout.prop(s, "result_view")
        row = layout.row(align=True)
        row.prop(s, "color_mode", expand=True)
        if s.color_mode == 'BANDED':
            layout.prop(s, "contour_bands")
        row = layout.row(align=True)
        row.prop(s, "show_legend", icon='COLOR', text="Legend")
        row.prop(s, "show_convergence", icon='SEQ_HISTOGRAM', text="Convergence")


class FEA_PT_run(FEA_PT_base, Panel):
    bl_idname = "FEA_PT_run"
    bl_parent_id = "FEA_PT_main"
    bl_label = "Run"

    def draw(self, context):
        layout = self.layout
        s = context.scene.blendfea

        row = layout.row()
        row.scale_y = 1.5
        row.enabled = not s.running
        row.operator("blendfea.run", icon='PLAY', text="Run FEA")

        if s.running:
            layout.label(text="Running - press ESC to cancel", icon='REC')
        if s.status:
            layout.label(text=s.status)

        layout.prop(s, "inputs_visible",
                   icon='HIDE_OFF' if s.inputs_visible else 'HIDE_ON')


_CLASSES = (FEA_PT_main, FEA_PT_bc, FEA_PT_material, FEA_PT_settings, FEA_PT_run)


# ---------------------------------------------------------------------------
# On-screen legend (viewport overlay, not a Panel -- always visible in every
# 3D viewport while a result object is selected, not just when the N-panel
# is open)
# ---------------------------------------------------------------------------

_LEGEND = {"visible": False, "title": "", "unit": "", "bottom": "", "top": "",
          "left_unit": "", "left_bottom": "", "left_top": ""}
_draw_handle = None
_conv_draw_handle = None

_BAR_W = 22
_BAR_H = 180
_MARGIN_RIGHT = 90     # clears the N-panel's own resize handle/scrollbar
_MARGIN_BOTTOM = 50


def set_legend(title, unit, bottom, top,
               left_unit="", left_bottom="", left_top=""):
    """Called by operators.py after every finished solve (and whenever the
    result_view dropdown changes) with what the color scale currently means.
    ``bottom``/``top`` are pre-formatted value strings, not raw numbers, so
    the caller controls exact rendering (e.g. safety factor's
    ">= 10 (safe)" vs. a plain stress number).

    ``left_*`` are optional: a second set of labels drawn on the left of the
    same gradient bar (e.g. displacement in mm while the bar is colored by
    stress). Left blank (default) means no second label column is drawn.
    """
    _LEGEND.update(title=title, unit=unit, bottom=bottom, top=top,
                   left_unit=left_unit, left_bottom=left_bottom,
                   left_top=left_top, visible=True)


def clear_legend():
    _LEGEND["visible"] = False


def _draw_legend_callback():
    if not _LEGEND.get("visible"):
        return
    context = bpy.context
    s = getattr(context.scene, "blendfea", None)
    if s is None or s.result_view == 'NONE' or not s.show_legend:
        return
    region = context.region
    if region is None or region.width <= 0:
        return

    bar_x = max(10, region.width - _MARGIN_RIGHT - _BAR_W)
    bar_y = _MARGIN_BOTTOM

    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        for i in range(32):
            t0 = i / 32
            t1 = (i + 1) / 32
            y_a = bar_y + t0 * _BAR_H
            y_b = bar_y + t1 * _BAR_H
            r, g, b = [float(c) for c in _jet_rgb_scalar((t0 + t1) * 0.5)]
            verts = [(bar_x, y_a), (bar_x + _BAR_W, y_a),
                    (bar_x + _BAR_W, y_b), (bar_x, y_b)]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
            shader.bind()
            shader.uniform_float("color", (r, g, b, 0.95))
            batch.draw(shader)
        # Thin border so the bar reads clearly against any background.
        border = [(bar_x, bar_y), (bar_x + _BAR_W, bar_y),
                 (bar_x + _BAR_W, bar_y + _BAR_H), (bar_x, bar_y + _BAR_H),
                 (bar_x, bar_y)]
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": border})
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.6))
        batch.draw(shader)
        gpu.state.blend_set('NONE')

        font_id = 0
        blf.size(font_id, 13)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        title_w, _ = blf.dimensions(font_id, _LEGEND["title"])
        blf.position(font_id, bar_x + _BAR_W - title_w, bar_y + _BAR_H + 8, 0)
        blf.draw(font_id, _LEGEND["title"])

        blf.size(font_id, 12)
        unit = _LEGEND["unit"]
        top_txt = f'{_LEGEND["top"]} {unit}'.strip()
        bottom_txt = f'{_LEGEND["bottom"]} {unit}'.strip()
        blf.position(font_id, bar_x + _BAR_W + 8, bar_y + _BAR_H - 12, 0)
        blf.draw(font_id, top_txt)
        blf.position(font_id, bar_x + _BAR_W + 8, bar_y, 0)
        blf.draw(font_id, bottom_txt)

        # Second, left-hand label column on the SAME bar (e.g. displacement
        # mm alongside a stress-colored bar) -- right-aligned so it reads
        # cleanly against the bar's left edge regardless of text length.
        left_unit = _LEGEND.get("left_unit", "")
        if left_unit or _LEGEND.get("left_top") or _LEGEND.get("left_bottom"):
            left_top_txt = f'{_LEGEND["left_top"]} {left_unit}'.strip()
            left_bottom_txt = f'{_LEGEND["left_bottom"]} {left_unit}'.strip()
            w_top, _ = blf.dimensions(font_id, left_top_txt)
            w_bot, _ = blf.dimensions(font_id, left_bottom_txt)
            blf.position(font_id, bar_x - 8 - w_top, bar_y + _BAR_H - 12, 0)
            blf.draw(font_id, left_top_txt)
            blf.position(font_id, bar_x - 8 - w_bot, bar_y, 0)
            blf.draw(font_id, left_bottom_txt)
    except Exception:
        # A drawing hiccup (e.g. an unusual GPU backend) must never crash the
        # viewport's normal draw loop -- just skip the legend for this frame.
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Convergence overlay: a small residual-vs-iteration plot, updated live while
# the CG solve runs and left showing the completed curve once the solve
# finishes, so it's visible at a glance whether the residual was creeping
# down and got cut off, or genuinely stalled/diverged.
# ---------------------------------------------------------------------------

_CONV = {"points": [], "tol": 1e-6}
_CONV_W = 220
_CONV_H = 110
# Stacked above the legend bar (sharing its right edge) rather than anchored
# to the viewport's top-right corner, which Blender's own navigation gizmo
# occupies.
_CONV_GAP_ABOVE_LEGEND = 40
_CONV_MAX_POINTS = 4000   # thin older points rather than grow unbounded


def reset_convergence():
    _CONV["points"] = []
    _CONV["tol"] = 1e-6


def push_convergence_point(it, resid, tol):
    """Called live, once per throttled status update, while a solve runs."""
    _CONV["tol"] = float(tol)
    pts = _CONV["points"]
    pts.append((int(it), float(resid)))
    if len(pts) > _CONV_MAX_POINTS:
        _CONV["points"] = pts[::2]   # halve density, keep it O(1) amortized


def set_convergence_history(history, tol):
    """Called once, after a solve finishes, with the FULL per-iteration
    residual list core.fea.VoxelFEA.solve() collected -- replaces whatever
    partial live trace was streamed, so the final plot is exact rather than
    only as fine as the 0.15s status-write throttle allowed."""
    if not history:
        return
    _CONV["tol"] = float(tol)
    _CONV["points"] = [(i + 1, float(r)) for i, r in enumerate(history)]


def _draw_convergence_callback():
    context = bpy.context
    s = getattr(context.scene, "blendfea", None)
    if s is None or not getattr(s, "show_convergence", True):
        return
    pts = _CONV["points"]
    if not pts:
        return
    region = context.region
    if region is None or region.width <= 0:
        return

    import numpy as np   # local import, same rationale as _jet_rgb_scalar's

    # Right edge lines up with the legend bar's own right edge (bar_x +
    # _BAR_W, see _draw_legend_callback), and it sits directly above the
    # legend's title text -- both always fully inside the region regardless
    # of viewport size, since both offsets are clamped the same way the
    # legend already clamps its own bar_x.
    legend_right = max(10, region.width - _MARGIN_RIGHT - _BAR_W) + _BAR_W
    x0 = max(10, legend_right - _CONV_W)
    y0 = _MARGIN_BOTTOM + _BAR_H + _CONV_GAP_ABOVE_LEGEND
    y1 = y0 + _CONV_H
    # If the viewport is too short for both to fit, keep the convergence
    # plot fully on-screen (clamped below the top edge) rather than letting
    # it run off the top -- the legend, anchored to the bottom, is always
    # visible either way.
    if y1 > region.height - 10:
        y1 = max(region.height - 10, _CONV_H + 10)
        y0 = y1 - _CONV_H

    tol = max(_CONV["tol"], 1e-12)
    # log10 residual axis, clamped to [tol, 1] -- CG's resid RATIO starts at
    # ~1 (r0/||f||) and should shrink toward tol; values outside that band
    # are clipped for drawing only, the raw numbers are unaffected.
    lo, hi = np.log10(tol), 0.0
    span = max(hi - lo, 1e-9)
    it_max = max(p[0] for p in pts)
    it_max = max(it_max, 1)

    def to_xy(it, resid):
        tx = it / it_max
        ly = np.clip(np.log10(max(resid, tol * 1e-3)), lo, hi)
        ty = np.clip((ly - lo) / span, 0.0, 1.0)
        return (x0 + tx * _CONV_W, y0 + ty * _CONV_H)

    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')

        # Background panel so the plot reads against any part of the scene.
        bg = [(x0, y0), (x0 + _CONV_W, y0), (x0 + _CONV_W, y1), (x0, y1)]
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": bg})
        shader.bind()
        shader.uniform_float("color", (0.05, 0.05, 0.05, 0.55))
        batch.draw(shader)

        # Dashed target-tolerance line at the top (ly = hi corresponds to
        # resid=1, ly = lo to resid=tol -- tol itself is the bottom edge).
        border = [(x0, y0), (x0 + _CONV_W, y0),
                 (x0 + _CONV_W, y1), (x0, y1), (x0, y0)]
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": border})
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
        batch.draw(shader)

        # The residual curve itself.
        if len(pts) >= 2:
            line = [to_xy(it, r) for it, r in pts]
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": line})
            shader.bind()
            converged = pts[-1][1] <= _CONV["tol"]
            color = (0.3, 1.0, 0.4, 0.9) if converged else (1.0, 0.75, 0.2, 0.9)
            shader.uniform_float("color", color)
            batch.draw(shader)
        gpu.state.blend_set('NONE')

        font_id = 0
        blf.size(font_id, 12)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, x0 + 4, y1 - 14, 0)
        blf.draw(font_id, "CG convergence (log residual)")
        blf.size(font_id, 11)
        last_it, last_r = pts[-1]
        blf.position(font_id, x0 + 4, y0 + 4, 0)
        blf.draw(font_id, f"it {last_it}  resid {last_r:.2e}  "
                          f"target {_CONV['tol']:.1e}")
    except Exception:
        traceback.print_exc()


def _jet_rgb_scalar(t):
    """Single-value convenience wrapper around core.extract._jet_rgb (which
    is array-shaped) -- avoids pulling numpy into a per-frame draw callback
    just to color one bar segment."""
    import numpy as np
    return _jet_rgb(np.array([t]))[0]


def register():
    global _draw_handle, _conv_draw_handle
    # NOTE: hasattr(cls, "bl_rna") is NOT a reliable "is this class
    # currently registered" check -- see properties.py for the full
    # rationale. try/except RuntimeError is the only reliable guard.
    registered = []
    try:
        for cls in _CLASSES:
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
            bpy.utils.register_class(cls)
            registered.append(cls)
        if _draw_handle is None:
            _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                _draw_legend_callback, (), 'WINDOW', 'POST_PIXEL')
        if _conv_draw_handle is None:
            _conv_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                _draw_convergence_callback, (), 'WINDOW', 'POST_PIXEL')
    except Exception:
        print("blendfea.ui.register() failed -- see traceback below for "
              "the REAL cause (best-effort rollback follows, its own "
              "errors are swallowed so they don't hide this one):")
        traceback.print_exc()
        for cls in reversed(registered):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
        raise


def unregister():
    global _draw_handle, _conv_draw_handle
    if _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        except Exception:
            pass
        _draw_handle = None
    if _conv_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_conv_draw_handle, 'WINDOW')
        except Exception:
            pass
        _conv_draw_handle = None
    clear_legend()
    reset_convergence()
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
