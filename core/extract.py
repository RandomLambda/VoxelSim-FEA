# SPDX-License-Identifier: GPL-3.0-or-later
"""
Result visualization: scalar field -> colors, on DEDICATED result meshes.

A linear FEA here runs once on an already-designed part, so each result
(Stress / Displacement / Safety) is baked onto its own dedicated, independent
COPY of the part's mesh (see ``duplicate_result_object``), with all inherited
material slots cleared first so our Attribute->Emission material in slot 0
is the only one that can possibly be visible. Coloring the user's own part
object in place was tried first and found fragile: a part with its own
multi-material setup could leave some faces showing the user's material
since only slot 0 was replaced, and the object's own display settings could
fight ours.

``apply_stress_colors`` samples the voxel-grid scalar field (von Mises
stress, a displacement-magnitude field, or a derived safety-factor field --
see operators.py) directly on that mesh: a smooth per-VERTEX trilinear blend
(``field_vertex_colors``) for the continuous, node-centered displacement
field, or a flat per-FACE, no-interpolation nearest-cell color
(``field_face_colors``) for the element-centered stress/safety fields, which
genuinely jump voxel-to-voxel and would otherwise show Gouraud-interpolation
artifacts (see field_face_colors' own docstring).

The voxel-cloud cross-section (every active/solid element as its own real,
colored cube, clipped by a movable Empty via Geometry Nodes) shows which
internal voxels are most stressed.
"""

import numpy as np

try:
    import bpy
    import bmesh
    _HAS_BPY = True
except Exception:
    _HAS_BPY = False


# ---------------------------------------------------------------------------
# Colormap + trilinear sampling
# ---------------------------------------------------------------------------

def _band_t(t, bands):
    """Quantize ``t`` (in [0, 1]) into ``bands`` discrete levels, each
    mapped to its own band midpoint -- what turns the colormap into a
    "Contour"/banded plot (properties.color_mode == 'BANDED'): a flat color
    per band with a hard step at every band boundary, the same look most
    FEA packages call a "banded" contour plot (as opposed to "smooth"/
    continuous). ``bands`` <= 1 or falsy returns ``t`` unchanged (the
    original fully-continuous "Pixel" behaviour) -- callers don't need to
    special-case that themselves.
    """
    if not bands or bands <= 1:
        return t
    idx = np.minimum((t * bands).astype(np.int64), int(bands) - 1)
    return (idx + 0.5) / float(bands)


def _jet_rgb(t, bands=None):
    """Classic FEA blue -> cyan -> green -> yellow -> red colormap.

    t : array-like in [0, 1] (values outside are clamped). Returns (N, 3)
    RGB floats in [0, 1]. Pure numpy, headless-testable.

    Used both for von Mises stress (high t = high stress = red) and for a
    safety-factor field, where the caller passes an already-inverted/negated
    field so that t=1 still means "worst" (lowest safety factor) -- see
    operators.py's ``_safety_field``.

    bands : forwarded to _band_t -- pass an int to get discrete "Contour"
    color bands instead of the continuous default.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    t = _band_t(t, bands)
    r = np.clip(np.minimum(4 * t - 1.5, -4 * t + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4 * t - 0.5, -4 * t + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4 * t + 0.5, -4 * t + 2.5), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def extrapolate_field_into_margin(field3d, active3d, iterations=3):
    """Extend field3d a few voxels past the active region by nearest-active
    averaging, so a trilinear sample near the true surface always blends
    between real material values instead of the artificial 0.0 that
    field3d holds outside the active region.

    Without this, a surface vertex near a sloped or staggered boundary
    blends toward the artificial zero purely due to where the voxel grid
    falls relative to the real geometry, showing up as "blue rings" on
    sloped faces unrelated to the actual stress state. Iterative 6-neighbour
    averaging, ``iterations`` layers deep, fills a thin margin with the
    nearest real value; 2-3 layers is enough since a real surface is never
    more than about one voxel outside the last active center. Node-based
    fields (displacement) are already correct at the boundary and don't need
    this.
    """
    field3d = np.asarray(field3d, dtype=np.float64).copy()
    known = np.asarray(active3d, dtype=bool).copy()
    if not known.any() or known.all():
        return field3d

    def _accumulate(total, count, dst, src):
        total[dst] += np.where(known[src], field3d[src], 0.0)
        count[dst] += known[src]

    for _ in range(max(1, int(iterations))):
        if known.all():
            break
        total = np.zeros_like(field3d)
        count = np.zeros(field3d.shape, dtype=np.int32)
        sl = slice(None)
        _accumulate(total, count, (slice(1, None), sl, sl), (slice(None, -1), sl, sl))
        _accumulate(total, count, (slice(None, -1), sl, sl), (slice(1, None), sl, sl))
        _accumulate(total, count, (sl, slice(1, None), sl), (sl, slice(None, -1), sl))
        _accumulate(total, count, (sl, slice(None, -1), sl), (sl, slice(1, None), sl))
        _accumulate(total, count, (sl, sl, slice(1, None)), (sl, sl, slice(None, -1)))
        _accumulate(total, count, (sl, sl, slice(None, -1)), (sl, sl, slice(1, None)))

        newly = (~known) & (count > 0)
        if not newly.any():
            break
        field3d[newly] = total[newly] / count[newly]
        known = known | newly
    return field3d


def _sample_field_trilinear(field3d, origin, vsize, points):
    """Trilinear sample of a voxel-centered scalar field at world points.

    Points outside the grid are clamped to the boundary cell. For an
    element-centered field that is 0.0 outside the active region, run it
    through ``extrapolate_field_into_margin`` first (see
    ``field_vertex_colors``); this function itself has no masking.
    """
    import itertools
    nx, ny, nz = field3d.shape
    origin = np.asarray(origin, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0)
    fi = (points - origin) / vsize - 0.5
    f0 = np.floor(fi).astype(int)
    fr = fi - f0
    out = np.zeros(len(points))
    for corner in itertools.product((0, 1), repeat=3):
        w = np.ones(len(points))
        for d in range(3):
            w *= fr[:, d] if corner[d] else (1.0 - fr[:, d])
        ix = np.clip(f0[:, 0] + corner[0], 0, nx - 1)
        iy = np.clip(f0[:, 1] + corner[1], 0, ny - 1)
        iz = np.clip(f0[:, 2] + corner[2], 0, nz - 1)
        out += w * field3d[ix, iy, iz]
    return out


def _sample_field_nearest(field3d, origin, vsize, points):
    """Nearest-cell sample of a voxel-centered scalar field at world points --
    the exact value the element itself holds, no blending with neighbours.

    Matches the lookup ``voxel_cloud_to_object`` uses, so sampling with this
    function reproduces the voxel-cloud's own per-cell contrast exactly.
    Used by ``field_face_colors`` for flat per-face coloring only -- never
    for per-vertex coloring, since a discontinuous nearest-cell value on
    individual vertices that then get Gouraud-interpolated across a triangle
    produces visible diagonal banding (see field_face_colors' docstring).
    """
    nx, ny, nz = field3d.shape
    origin = np.asarray(origin, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0)
    idx = np.floor((points - origin) / vsize).astype(int)
    ix = np.clip(idx[:, 0], 0, nx - 1)
    iy = np.clip(idx[:, 1], 0, ny - 1)
    iz = np.clip(idx[:, 2], 0, nz - 1)
    return field3d[ix, iy, iz]


def _local_to_world(local_verts, matrix_world):
    """Transform (N,3) object-local vertex coords to world space.

    field3d/origin are always defined in world space (see voxelize.Grid),
    while Mesh.vertices.foreach_get('co') returns object-local coordinates.
    Sampling field3d directly with local coords would only be correct for an
    object at the world origin with an identity transform; for any
    translated/rotated part it would silently sample the wrong voxels.
    ``matrix_world`` should be the result object's matrix (duplicate_result_
    object sets it equal to the source part's matrix_world, so either works).
    """
    local_verts = np.asarray(local_verts, dtype=np.float64)
    if len(local_verts) == 0:
        return local_verts
    mw = np.array(matrix_world, dtype=np.float64)  # 4x4
    homog = np.empty((len(local_verts), 4))
    homog[:, :3] = local_verts
    homog[:, 3] = 1.0
    return (homog @ mw.T)[:, :3]


def field_vertex_colors(field3d, origin, vsize, verts, vmax, vmin=0.0,
                        active3d=None):
    """Map a per-voxel scalar field onto per-vertex RGBA colors, smoothly
    (plain trilinear -- see _sample_field_trilinear). Returns (N, 4) float
    RGBA in [0, 1]. This is "Pixel" shading's continuous path -- for
    "Contour" (banded) shading see _apply_contour_colors /
    _ensure_contour_material instead, which band in the shader after
    interpolation rather than baking pre-quantized colors here.

    Only used for fields that are themselves already continuous/smooth
    (currently: displacement, a node-centered field with no natural jumps
    between neighbouring samples) -- for element-centered fields (stress,
    safety) that do have real cell-to-cell jumps, use field_face_colors
    instead (see its docstring).

    active3d : pass the solve's active-element mask for element-centered
    fields -- the field is extrapolated a few voxels past the true material
    boundary first (see extrapolate_field_into_margin) so sampling near the
    real surface never blends toward the artificial zero outside the part.
    Leave None for fields already correct at the boundary (displacement).
    """
    verts = np.asarray(verts, dtype=float)
    if len(verts) == 0:
        return np.zeros((0, 4))
    if active3d is not None:
        field3d = extrapolate_field_into_margin(field3d, active3d)
    vals = _sample_field_trilinear(field3d, origin, vsize, verts)
    span = max(float(vmax) - float(vmin), 1e-12)
    t = (vals - float(vmin)) / span
    rgb = _jet_rgb(t)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=-1)


def field_face_colors(field3d, origin, vsize, face_points, vmax, vmin=0.0,
                      active3d=None):
    """Map a per-voxel scalar field onto per-FACE (flat, no interpolation)
    RGBA colors -- one color per face, its own nearest-cell value, mirroring
    how voxel_cloud_to_object colors each independent cube.

    Any per-vertex scheme (trilinear, nearest, or a blend) still gets
    Gouraud-interpolated by the GPU across each triangle, which is fine for
    a smooth field but produces visible diagonal streaking for
    element-centered fields (stress/safety) that genuinely jump from one
    voxel to the next, following the mesh's own triangulation rather than
    the underlying field. Giving every face one flat color removes that
    failure mode: the surface reads as many small flat facets, same as the
    voxel cloud. ``_ensure_surface_resolution`` subdivides the mesh down to
    roughly half a voxel per edge so those facets stay small enough to show
    local detail.

    face_points : (N, 3) sample point per face (its center) -- see
    apply_stress_colors, which computes these from the mesh's own polygons.
    This is "Pixel" shading's exact-per-cell path -- for "Contour" (banded)
    shading see _apply_contour_colors / _ensure_contour_material instead.
    Returns (N, 4) float RGBA in [0, 1].
    """
    face_points = np.asarray(face_points, dtype=float)
    if len(face_points) == 0:
        return np.zeros((0, 4))
    if active3d is not None:
        field3d = extrapolate_field_into_margin(field3d, active3d)
    vals = _sample_field_nearest(field3d, origin, vsize, face_points)
    span = max(float(vmax) - float(vmin), 1e-12)
    t = (vals - float(vmin)) / span
    rgb = _jet_rgb(t)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=-1)


# ---------------------------------------------------------------------------
# Baking colors onto an existing Blender mesh (the part itself)
# ---------------------------------------------------------------------------

_RESULT_ATTR = "BlenderFEAResult"
_RESULT_MAT = "BlenderFEA_ResultPreview"


def _ensure_result_material():
    """Attribute -> Emission material: reads the color attribute straight
    through, so the heatmap looks the same regardless of scene lighting or
    viewport shading mode (it's data, not a lit surface)."""
    mat = bpy.data.materials.get(_RESULT_MAT)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(_RESULT_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    emit = nt.nodes.new('ShaderNodeEmission')
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = _RESULT_ATTR
    out.location = (300, 0)
    emit.location = (100, 0)
    attr.location = (-150, 0)
    nt.links.new(attr.outputs['Color'], emit.inputs['Color'])
    nt.links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat


_CONTOUR_ATTR = "BlenderFEAResultT"
_CONTOUR_MAT = "BlenderFEA_ResultPreview_Contour"


def _ensure_contour_material(bands):
    """Build/update the procedural "Contour" shading material.

    Reads the raw continuous t value (0..1, already vmin/vmax-normalized)
    from the _CONTOUR_ATTR point attribute, which the GPU Gouraud-
    interpolates smoothly across every face, quantizes it into ``bands``
    discrete steps in the shader (after interpolation, per rendered pixel),
    then computes the same jet colormap _jet_rgb uses in numpy as a shader
    node graph, emitted flat (Attribute -> Emission).

    Banding is done after interpolation rather than before (pre-quantizing
    per vertex and letting Gouraud blend between flat colors) because the
    latter produces a soft linear blend zone at each band edge that follows
    the mesh's vertex layout, not the field. Quantizing the interpolated
    value per-pixel instead makes each band edge trace the field's true
    iso-value line, limited only by mesh resolution (see
    _ensure_surface_resolution).

    One shared material for every Contour-shaded result object; rebuilt from
    scratch each call rather than diffed/reused, since that's cheap enough
    to redo on every recolor.
    """
    mat = bpy.data.materials.get(_CONTOUR_MAT)
    if mat is None:
        mat = bpy.data.materials.new(_CONTOUR_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    n_attr = nt.nodes.new('ShaderNodeAttribute')
    n_attr.attribute_type = 'GEOMETRY'
    n_attr.attribute_name = _CONTOUR_ATTR
    n_attr.location = (-1000, 0)

    n_bands = nt.nodes.new('ShaderNodeValue')
    n_bands.outputs[0].default_value = float(max(1, int(bands)))
    n_bands.location = (-1000, -250)

    n_bands_m1 = nt.nodes.new('ShaderNodeMath')
    n_bands_m1.operation = 'SUBTRACT'
    n_bands_m1.inputs[1].default_value = 1.0
    n_bands_m1.location = (-1000, -400)
    nt.links.new(n_bands.outputs[0], n_bands_m1.inputs[0])

    # Clamp the raw attribute to [0, 1] before banding: a real stress/safety
    # value can sit above vmax (the color scale clips the top ~1% of
    # singular spikes) or slightly below 0 from float noise, and an
    # un-clamped value would produce an out-of-range band index below.
    n_tclamp = nt.nodes.new('ShaderNodeClamp')
    n_tclamp.inputs['Min'].default_value = 0.0
    n_tclamp.inputs['Max'].default_value = 1.0
    n_tclamp.location = (-800, 100)
    nt.links.new(n_attr.outputs['Fac'], n_tclamp.inputs['Value'])

    # Banding: idx = floor(t * bands); t_banded = (idx + 0.5) / bands, same
    # formula as extract._band_t but as shader nodes so it runs per-pixel on
    # the interpolated value. idx is capped at bands-1 so t == 1.0 lands in
    # the last band instead of producing an out-of-range t_banded.
    n_mulb = nt.nodes.new('ShaderNodeMath')
    n_mulb.operation = 'MULTIPLY'
    n_mulb.location = (-600, -100)
    nt.links.new(n_tclamp.outputs['Result'], n_mulb.inputs[0])
    nt.links.new(n_bands.outputs[0], n_mulb.inputs[1])

    n_floor = nt.nodes.new('ShaderNodeMath')
    n_floor.operation = 'FLOOR'
    n_floor.location = (-450, -100)
    nt.links.new(n_mulb.outputs[0], n_floor.inputs[0])

    n_idxclamp = nt.nodes.new('ShaderNodeMath')
    n_idxclamp.operation = 'MINIMUM'
    n_idxclamp.location = (-300, -100)
    nt.links.new(n_floor.outputs[0], n_idxclamp.inputs[0])
    nt.links.new(n_bands_m1.outputs[0], n_idxclamp.inputs[1])

    n_addhalf = nt.nodes.new('ShaderNodeMath')
    n_addhalf.operation = 'ADD'
    n_addhalf.inputs[1].default_value = 0.5
    n_addhalf.location = (-150, -100)
    nt.links.new(n_idxclamp.outputs[0], n_addhalf.inputs[0])

    n_divb = nt.nodes.new('ShaderNodeMath')
    n_divb.operation = 'DIVIDE'
    n_divb.location = (0, -100)
    nt.links.new(n_addhalf.outputs[0], n_divb.inputs[0])
    nt.links.new(n_bands.outputs[0], n_divb.inputs[1])

    t_banded = n_divb.outputs[0]

    # 4t and -4t, shared by all three jet channels below.
    n_4t = nt.nodes.new('ShaderNodeMath')
    n_4t.operation = 'MULTIPLY'
    n_4t.inputs[1].default_value = 4.0
    n_4t.location = (200, 250)
    nt.links.new(t_banded, n_4t.inputs[0])

    n_neg4t = nt.nodes.new('ShaderNodeMath')
    n_neg4t.operation = 'MULTIPLY'
    n_neg4t.inputs[1].default_value = -4.0
    n_neg4t.location = (200, -450)
    nt.links.new(t_banded, n_neg4t.inputs[0])

    def _clamped_min(a_offset, b_offset, y):
        """r = clamp(min(4t + a_offset, -4t + b_offset), 0, 1) -- one jet
        channel, matching _jet_rgb's r/g/b formulas exactly. Closes over
        n_4t/n_neg4t (4t and -4t, shared by all three channels)."""
        n_a = nt.nodes.new('ShaderNodeMath')
        n_a.operation = 'ADD'
        n_a.inputs[1].default_value = a_offset
        n_a.location = (200, y + 60)
        nt.links.new(n_4t.outputs[0], n_a.inputs[0])

        n_b = nt.nodes.new('ShaderNodeMath')
        n_b.operation = 'ADD'
        n_b.inputs[1].default_value = b_offset
        n_b.location = (200, y - 60)
        nt.links.new(n_neg4t.outputs[0], n_b.inputs[0])

        n_min = nt.nodes.new('ShaderNodeMath')
        n_min.operation = 'MINIMUM'
        n_min.location = (400, y)
        nt.links.new(n_a.outputs[0], n_min.inputs[0])
        nt.links.new(n_b.outputs[0], n_min.inputs[1])

        n_clamp = nt.nodes.new('ShaderNodeClamp')
        n_clamp.inputs['Min'].default_value = 0.0
        n_clamp.inputs['Max'].default_value = 1.0
        n_clamp.location = (600, y)
        nt.links.new(n_min.outputs[0], n_clamp.inputs['Value'])
        return n_clamp.outputs['Result']

    r = _clamped_min(-1.5, 4.5, 250)
    g = _clamped_min(-0.5, 3.5, 0)
    b = _clamped_min(0.5, 2.5, -250)

    n_combine = nt.nodes.new('ShaderNodeCombineColor')
    n_combine.mode = 'RGB'
    n_combine.location = (800, 0)
    nt.links.new(r, n_combine.inputs[0])
    nt.links.new(g, n_combine.inputs[1])
    nt.links.new(b, n_combine.inputs[2])

    n_emit = nt.nodes.new('ShaderNodeEmission')
    n_emit.location = (1000, 0)
    nt.links.new(n_combine.outputs['Color'], n_emit.inputs['Color'])

    n_out = nt.nodes.new('ShaderNodeOutputMaterial')
    n_out.location = (1200, 0)
    nt.links.new(n_emit.outputs['Emission'], n_out.inputs['Surface'])

    return mat


def _bake_scalar_attribute(obj, name, values):
    """Write a (N,) float array (N = vertex count) onto obj as a POINT-
    domain FLOAT attribute (a plain scalar attribute, not a color one).

    Used for Contour shading: the raw continuous t value has to reach the
    GPU as-is so _ensure_contour_material's shader can quantize it into
    bands after Gouraud interpolation (see that function's docstring).

    An existing attribute of the same name but the wrong type/domain is
    dropped and recreated, same rationale as _bake_colors.
    """
    me = obj.data
    if len(me.vertices) == 0:
        return
    attr = me.attributes.get(name)
    if attr is not None and (attr.domain != 'POINT' or attr.data_type != 'FLOAT'):
        me.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = me.attributes.new(name=name, type='FLOAT', domain='POINT')
    attr.data.foreach_set('value', np.asarray(values, dtype=float).ravel())
    me.update()


def _bake_colors(obj, rgba, domain='POINT'):
    """Write an RGBA array onto obj as a FLOAT_COLOR color attribute, plus
    the flat Attribute->Emission preview material in slot 0.

    domain='POINT' : one color per vertex, Gouraud-interpolated by the GPU --
    right for a smooth/continuous field (displacement) or the voxel cloud.
    domain='CORNER' : one color per face-corner, every corner of a face
    given the same value (see _bake_face_colors), for a flat per-face color
    -- right for an element-centered field (stress, safety) that jumps from
    one voxel to the next (see field_face_colors' docstring).

    A color-attribute's domain can't be changed in place once created, so an
    existing attribute in the wrong domain is dropped and recreated.

    Material slots below slot 0 are left untouched.
    """
    me = obj.data
    if len(me.vertices) == 0:
        return
    attr = me.color_attributes.get(_RESULT_ATTR)
    if attr is not None and attr.domain != domain:
        me.color_attributes.remove(attr)
        attr = None
    if attr is None:
        attr = me.color_attributes.new(
            name=_RESULT_ATTR, type='FLOAT_COLOR', domain=domain)
    attr.data.foreach_set('color', np.asarray(rgba, dtype=float).ravel())
    try:
        me.color_attributes.active_color_name = _RESULT_ATTR
    except Exception:
        pass
    me.update()

    mat = _ensure_result_material()
    if len(me.materials) == 0:
        me.materials.append(mat)
    else:
        me.materials[0] = mat


def _bake_vertex_colors(obj, rgba):
    """POINT-domain convenience wrapper -- see _bake_colors."""
    _bake_colors(obj, rgba, domain='POINT')


def _bake_face_colors(obj, face_rgba):
    """Expand one RGBA per FACE to one per face-CORNER (every corner of a
    face repeats that face's own color, so nothing gets Gouraud-interpolated
    within or across faces) and bake as a CORNER-domain attribute -- see
    field_face_colors / _bake_colors for why this, not POINT-domain, is used
    for element-centered fields (stress, safety).
    """
    me = obj.data
    if len(me.polygons) == 0:
        return
    face_rgba = np.asarray(face_rgba, dtype=float)
    loop_totals = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get('loop_total', loop_totals)
    loop_face_index = np.repeat(np.arange(len(me.polygons)), loop_totals)
    loop_rgba = face_rgba[loop_face_index]
    _bake_colors(obj, loop_rgba, domain='CORNER')


_SUBDIV_MAX_VERTS = 1_000_000  # hard cap: never let a fine grid + a big/
                               # simple part explode into an unusable count.
_SUBDIV_MAX_PASSES = 4          # corrective passes for whatever grid_fill
                                # doesn't perfectly equalize in one shot --
                                # NOT the mechanism that reaches target
                                # resolution (that's the direct cuts-count
                                # computation below, done in ~1 pass
                                # regardless of how large the part is).
_SURFACE_SUBDIV_FACTOR = 0.5    # subdivide down to HALF the voxel size, not
                                # a full voxel -- one vertex per voxel is too
                                # coarse for field_vertex_colors to trace a
                                # smooth field's real gradient across a face;
                                # ~2x2 vertices per voxel gives the Gouraud
                                # interpolation enough samples to follow it.
                                # Used for displacement (always) and for
                                # stress/safety's Contour-mode PRE-check --
                                # see _CONTOUR_SUBDIV_FACTOR below for why
                                # Contour needs much finer than this.
_CONTOUR_SUBDIV_FACTOR = 0.15   # Contour shading's band edges are only as
                                # smooth as the mesh resolves the underlying
                                # field; the default 0.5 factor is too coarse
                                # and renders a visible zigzag following
                                # triangle edges rather than the field's
                                # curve, so Contour subdivides several times
                                # finer (still capped by _SUBDIV_MAX_VERTS).


def _triangulate_ngons(mesh):
    """In-place: convert every polygon with more than 4 sides into clean
    triangles (bmesh.ops.triangulate, BEAUTY method). Tris and quads are
    left untouched.

    Runs on every duplicated result object before any subdivision, because
    bmesh.ops.subdivide_edges' use_grid_fill=True (see
    _ensure_surface_resolution) cleanly fills a quad with a regular grid but
    can fall back to a fan fill on an ngon -- a fan fill radiates thin, long
    triangles from a single hub vertex, and interpolating a real field
    across one produces a long unphysical color spike/streak across the
    surface. Triangulating ngons first with BEAUTY (well-shaped triangles)
    avoids that fallback entirely.

    Non-fatal: any bmesh hiccup just leaves the mesh's ngons as they were.
    """
    if not _HAS_BPY:
        return
    try:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        ngons = [f for f in bm.faces if len(f.verts) > 4]
        if ngons:
            bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY',
                                  ngon_method='BEAUTY')
            bm.to_mesh(mesh)
            mesh.update()
        bm.free()
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] ngon triangulation skipped: {exc}")


def _ensure_surface_resolution(mesh, target_edge_len):
    """In-place, shape-preserving subdivision of ``mesh`` so no edge is much
    longer than ``target_edge_len`` (normally the solve's voxel size).

    Vertex colors are linearly (Gouraud) interpolated across each face by
    the GPU, so a plain low-poly part would only ever show one smooth blend
    per face regardless of how exact the per-vertex sample is. Subdividing
    the surface down to roughly voxel resolution gives ``field_face_colors``
    (see apply_stress_colors) small enough facets to resolve per-element
    detail, and gives ``field_vertex_colors`` enough vertices to trace a
    smooth field's real gradient.

    The cut count for each pass is computed directly from the current
    longest offending edge (ceil(longest / target) - 1) so the target
    resolution is reached in a single pass regardless of part size; the
    remaining passes only mop up whatever one subdivide_edges call didn't
    fully equalize (e.g. ngons from grid_fill on non-quad topology).

    Uses plain "SIMPLE" edge subdivision (no smoothing/shrinking, so the
    part's actual shape never changes). Non-fatal: any bmesh hiccup just
    leaves the mesh at whatever resolution it reached. If the vertex cap is
    hit before reaching target_edge_len everywhere, that is reported (see
    below) instead of silently shipping a coarser-than-intended preview.
    """
    if not target_edge_len or target_edge_len <= 0:
        return
    try:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        capped = False
        for _ in range(_SUBDIV_MAX_PASSES):
            long_edges = [e for e in bm.edges
                         if e.calc_length() > target_edge_len * 1.1]
            if not long_edges:
                break
            longest = max(e.calc_length() for e in long_edges)
            cuts = max(1, int(np.ceil(longest / target_edge_len)) - 1)
            # Cap `cuts` itself, not just skip the pass: with
            # use_grid_fill=True, subdividing every edge of a quad by `cuts`
            # fills its interior with a (cuts+1) x (cuts+1) grid -- roughly
            # cuts**2 new vertices per face, not cuts -- so the budget is
            # computed as sqrt(available / face count) to match that
            # quadratic scale rather than risk a huge allocation.
            budget = max(1, _SUBDIV_MAX_VERTS - len(bm.verts))
            max_cuts_affordable = max(1, int(np.sqrt(budget / max(1, len(bm.faces)))))
            if cuts > max_cuts_affordable:
                cuts = max_cuts_affordable
                capped = True
            bmesh.ops.subdivide_edges(
                bm, edges=long_edges, cuts=cuts,
                use_grid_fill=True, use_single_edge=True)
            if len(bm.verts) >= _SUBDIV_MAX_VERTS:
                capped = True
                break
        if capped:
            print(f"[BlenderFEA] result-preview subdivision hit its "
                 f"{_SUBDIV_MAX_VERTS}-vertex cap before reaching the full "
                 f"voxel resolution (target edge length "
                 f"{target_edge_len:.6g}) -- surface preview is coarser "
                 f"than the voxel grid for this part/resolution combo.")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] surface subdivision for result preview skipped: {exc}")


def duplicate_result_object(part_obj, name, collection=None, vsize=None,
                            subdiv_factor=None):
    """Create (or refresh) a standalone copy of part_obj's mesh to use as a
    dedicated result object.

    A dedicated copy -- never the user's own part -- is what makes the
    overlay material reliable: if part_obj already has 2+ material slots and
    some faces use slot 1+ (not slot 0), only ever replacing slot 0 (the old
    v1.x approach, coloring the part in place) left those faces showing the
    user's own material instead of ours -- a real, plausible cause of "the
    result overlay didn't show up". Here every inherited material slot is
    cleared and every face's material_index reset to 0, so slot 0 (which
    _bake_vertex_colors will fill with our Attribute->Emission material) is
    the only slot that can possibly be visible.

    vsize : pass the solve's voxel size to also subdivide the copy down to
    roughly ``vsize * subdiv_factor`` (see ``_ensure_surface_resolution``).
    Leave None to skip subdivision (e.g. Pixel-mode stress/safety objects,
    which get their resolution from voxel-grid bisecting inside
    apply_stress_colors instead).
    subdiv_factor : overrides _SURFACE_SUBDIV_FACTOR -- pass
    _CONTOUR_SUBDIV_FACTOR for a Contour-shaded object (see that constant's
    docstring). None uses the default (displacement's usual resolution).
    """
    if not _HAS_BPY:
        raise RuntimeError("duplicate_result_object requires Blender")
    mesh_copy = part_obj.data.copy()
    mesh_copy.materials.clear()
    if len(mesh_copy.polygons):
        idx = np.zeros(len(mesh_copy.polygons), dtype=np.int32)
        mesh_copy.polygons.foreach_set('material_index', idx)
    _triangulate_ngons(mesh_copy)
    if vsize:
        factor = subdiv_factor if subdiv_factor is not None else _SURFACE_SUBDIV_FACTOR
        _ensure_surface_resolution(mesh_copy, float(vsize) * factor)

    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, mesh_copy)
        target = collection if collection is not None else bpy.context.scene.collection
        target.objects.link(obj)
    else:
        old = obj.data
        obj.data = mesh_copy
        if old.users == 0:
            bpy.data.meshes.remove(old)
    obj.matrix_world = part_obj.matrix_world.copy()
    obj.display_type = 'TEXTURED'   # never Wireframe/Bounds -- see module docstring
    return obj


def _apply_contour_colors(obj, field3d, origin, vsize, vmax, vmin,
                          active3d, bands):
    """Contour shading: sample the field smoothly (plain trilinear, same as
    field_vertex_colors) at every real vertex, bake the raw continuous t
    value (not a color) onto the mesh, and let _ensure_contour_material's
    shader do the banding/coloring per-pixel after Gouraud interpolation.

    Used for all three fields (stress, safety, displacement) whenever
    Contour shading is selected, regardless of whether the field is
    element- or node-centered -- Pixel mode keeps the exact-per-voxel-cell
    distinction between them; Contour mode smooths both the same way.
    """
    me = obj.data
    verts = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', verts)
    verts_local = verts.reshape(-1, 3)
    verts_world = _local_to_world(verts_local, obj.matrix_world)
    verts_sample = _pick_sample_space(obj, verts_local, verts_world,
                                      field3d, origin, vsize)

    field = field3d
    if active3d is not None:
        field = extrapolate_field_into_margin(field, active3d)
    vals = _sample_field_trilinear(field, origin, vsize, verts_sample)
    span = max(float(vmax) - float(vmin), 1e-12)
    t = np.clip((vals - float(vmin)) / span, 0.0, 1.0)

    _bake_scalar_attribute(obj, _CONTOUR_ATTR, t)
    mat = _ensure_contour_material(bands)
    if len(me.materials) == 0:
        me.materials.append(mat)
    else:
        me.materials[0] = mat


def apply_stress_colors(obj, field3d, origin, vsize, vmax, vmin=0.0,
                        active3d=None, bands=None):
    """Bake a scalar field (von Mises stress, displacement magnitude, or a
    safety-factor field) onto obj -- nothing is re-meshed or approximated.
    ``obj`` is normally a dedicated result object from
    ``duplicate_result_object``, not the user's own part.

    bands : an int selects "Contour" shading (properties.color_mode ==
    'BANDED') -- see _apply_contour_colors. None/<=1 (the default) is
    "Pixel" shading, selected by active3d instead:

    - active3d given (stress, safety): the mesh is first cut exactly along
      every voxel-grid plane it crosses (_bisect_to_voxel_grid) so every
      face lies within one voxel cell, then given flat per-face coloring,
      one nearest-cell sample per face (field_face_colors). Also
      extrapolates the field a few voxels past the true material boundary
      first (see extrapolate_field_into_margin).
    - active3d=None (displacement): smooth per-vertex trilinear coloring
      (field_vertex_colors) -- a node-centered field with no real jumps to
      preserve.
    """
    if not _HAS_BPY:
        return
    me = obj.data
    if len(me.vertices) == 0:
        return

    if bands is not None and bands > 1:
        _apply_contour_colors(obj, field3d, origin, vsize, vmax, vmin,
                              active3d, bands)
        return

    if active3d is not None:
        if len(me.polygons) == 0:
            return
        bisect_matrix = _pick_bisect_matrix(obj, field3d, origin, vsize)
        _bisect_to_voxel_grid(obj, origin, vsize, field3d.shape, bisect_matrix)
        me = obj.data  # bisected in place, but reread after bm.to_mesh/update
        if len(me.polygons) == 0:
            return
        centers_local = np.empty(len(me.polygons) * 3)
        me.polygons.foreach_get('center', centers_local)
        centers_local = centers_local.reshape(-1, 3)
        centers_world = _local_to_world(centers_local, obj.matrix_world)
        centers_sample = _pick_sample_space(obj, centers_local, centers_world,
                                            field3d, origin, vsize)
        rgba = field_face_colors(field3d, origin, vsize,
                                 centers_sample, vmax, vmin,
                                 active3d=active3d)
        _bake_face_colors(obj, rgba)
        return

    verts = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', verts)
    verts_local = verts.reshape(-1, 3)
    verts_world = _local_to_world(verts_local, obj.matrix_world)
    verts_sample = _pick_sample_space(obj, verts_local, verts_world,
                                      field3d, origin, vsize)

    rgba = field_vertex_colors(field3d, origin, vsize,
                               verts_sample, vmax, vmin,
                               active3d=active3d)
    _bake_vertex_colors(obj, rgba)


def _bbox_overlap_fraction(verts, field_min, field_max):
    """Fraction (0..1) of ``verts``' own AABB that overlaps [field_min,
    field_max] on every axis simultaneously -- 1.0 means the points' bbox
    sits entirely inside the field's box, 0.0 means no overlap on at least
    one axis. Used to pick which vertex coordinates actually correspond to
    the field's own domain (see ``_pick_sample_space``)."""
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    lo = np.maximum(vmin, field_min)
    hi = np.minimum(vmax, field_max)
    overlap = np.clip(hi - lo, 0.0, None)
    span = np.clip(vmax - vmin, 1e-9, None)
    return float(np.min(overlap / span))


def _pick_sample_space(obj, verts_local, verts_world, field3d, origin, vsize):
    """Decide whether ``verts_local`` (raw mesh.vertices 'co', object-local)
    or ``verts_world`` (the same points transformed by obj.matrix_world) are
    the ones that actually line up with field3d's own world-space domain,
    and return whichever one does.

    field3d/origin are always world-space (see voxelize.Grid), so
    verts_world is the correct choice whenever obj.matrix_world is a
    faithful copy of the solved part's transform (see
    duplicate_result_object). Checking which candidate's bounding box
    actually overlaps the field's own box guards against a stale/mismatched
    matrix_world or a part-side modifier that changes the evaluated shape
    used for voxelization but not the raw data copied here -- self-
    corrects to whichever coordinates are meaningful, and warns loudly if
    neither overlaps well rather than silently baking wrong colors.
    """
    dims = np.asarray(field3d.shape, dtype=float)
    field_min = np.asarray(origin, dtype=float)
    field_max = field_min + dims * float(vsize)

    world_overlap = _bbox_overlap_fraction(verts_world, field_min, field_max)
    local_overlap = _bbox_overlap_fraction(verts_local, field_min, field_max)

    # Prefer world-space (the mathematically correct choice whenever
    # matrix_world is right) as long as it overlaps at all better than local
    # does; only fall back to local if local is the one that actually
    # matches the field's domain.
    if local_overlap > world_overlap:
        chosen, chosen_name, frac = verts_local, "local", local_overlap
    else:
        chosen, chosen_name, frac = verts_world, "world", world_overlap

    if frac < 0.5:
        print(f"[BlenderFEA] WARNING: '{obj.name}' result-mesh vertices "
             f"don't line up well with the solved field's own bounding box "
             f"even after checking both local and world coordinates "
             f"(best overlap: {chosen_name}-space, {frac:.0%}) -- the "
             f"baked colors are likely wrong. world bbox of verts: "
             f"{np.round(verts_world.min(axis=0), 4).tolist()}.."
             f"{np.round(verts_world.max(axis=0), 4).tolist()}; field bbox: "
             f"{np.round(field_min, 4).tolist()}.."
             f"{np.round(field_max, 4).tolist()}; obj.matrix_world="
             f"{[list(np.round(row, 4)) for row in np.array(obj.matrix_world)]}")
    return chosen


def _pick_bisect_matrix(obj, field3d, origin, vsize):
    """Return obj.matrix_world if world-transformed vertices are the ones
    that actually line up with field3d's own domain, or None if the mesh's
    own local coordinates already line up (i.e. no transform needed). Same
    self-correcting overlap check as _pick_sample_space, reused here so
    _bisect_to_voxel_grid slices along the grid planes in the right space.
    """
    me = obj.data
    if len(me.vertices) == 0:
        return obj.matrix_world
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    verts_local = co.reshape(-1, 3)
    verts_world = _local_to_world(verts_local, obj.matrix_world)
    dims = np.asarray(field3d.shape, dtype=float)
    field_min = np.asarray(origin, dtype=float)
    field_max = field_min + dims * float(vsize)
    local_overlap = _bbox_overlap_fraction(verts_local, field_min, field_max)
    world_overlap = _bbox_overlap_fraction(verts_world, field_min, field_max)
    if local_overlap > world_overlap:
        return None
    return obj.matrix_world


def _bisect_to_voxel_grid(obj, origin, vsize, dims, matrix,
                          max_faces=_SUBDIV_MAX_VERTS):
    """Cut obj's mesh (in place) along every voxel-grid plane (axis-aligned,
    in the space ``matrix`` maps TO -- see _pick_bisect_matrix) that crosses
    its own bounding box, on all three axes, so that afterwards every
    resulting face lies ENTIRELY within one voxel cell.

    This is what makes field_face_colors / _bake_face_colors's flat
    per-face coloring exact and independent of the source mesh's own
    triangulation: a face confined to a single cell can only ever need one
    correct color (that cell's own value), so visible facet boundaries are
    dictated purely by the voxel grid, not by generic edge-length
    subdivision (which could leave a single face straddling a cell boundary
    or a sliver triangle sitting on an unrelated diagonal seam).

    matrix : obj.matrix_world to bisect in world space, or None to bisect
    directly in the mesh's own local coordinates (see _pick_bisect_matrix);
    origin/vsize/dims must already be expressed in that same space.

    Non-fatal and face-count-capped like _ensure_surface_resolution: any
    bmesh hiccup, or hitting max_faces before every plane is cut, just
    leaves the mesh partially sliced (reported, never silent).
    """
    if not _HAS_BPY:
        return
    me = obj.data
    if len(me.polygons) == 0:
        return
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        if matrix is not None:
            bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)

        origin = np.asarray(origin, dtype=float)
        nx, ny, nz = dims
        capped = False
        for axis, n in enumerate((nx, ny, nz)):
            if capped or n < 2:
                continue
            coords = [v.co[axis] for v in bm.verts]
            if not coords:
                continue
            lo, hi = min(coords), max(coords)
            plane_no = [0.0, 0.0, 0.0]
            plane_no[axis] = 1.0
            for i in range(1, n):
                plane_val = origin[axis] + i * vsize
                if plane_val <= lo or plane_val >= hi:
                    continue  # doesn't cross this mesh's own bbox -- no-op
                if len(bm.faces) >= max_faces:
                    capped = True
                    break
                plane_co = [0.0, 0.0, 0.0]
                plane_co[axis] = plane_val
                geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
                bmesh.ops.bisect_plane(
                    bm, geom=geom, plane_co=plane_co, plane_no=plane_no,
                    clear_inner=False, clear_outer=False)

        if matrix is not None:
            bmesh.ops.transform(bm, matrix=matrix.inverted(), verts=bm.verts)

        if capped:
            print(f"[BlenderFEA] voxel-grid surface slicing for '{obj.name}' "
                 f"hit its {max_faces}-face cap before every grid plane was "
                 f"cut -- some faces may still straddle a voxel-cell "
                 f"boundary (coarser-than-intended color detail for this "
                 f"part/resolution combo).")
        bm.to_mesh(me)
        bm.free()
        me.update()
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] voxel-grid surface slicing skipped for "
             f"'{obj.name}': {exc}")


def clear_result_colors(obj):
    """Remove the baked color attribute + preview material slot, if present
    (used when the user turns the surface overlay off)."""
    if not _HAS_BPY or obj is None:
        return
    me = obj.data
    attr = me.color_attributes.get(_RESULT_ATTR)
    if attr is not None:
        try:
            me.color_attributes.remove(attr)
        except Exception:
            pass
    mat = bpy.data.materials.get(_RESULT_MAT)
    if mat is not None and len(me.materials) and me.materials[0] == mat:
        me.materials[0] = None


# ---------------------------------------------------------------------------
# Voxel-cloud bake: every active element as its own real, independently-
# colored cube, for a cross-section view. Deliberately real Python-built cube
# geometry (verts + faces, colored exactly like the surface) rather than
# Geometry-Nodes instancing, since instancing + color attributes through
# Realize Instances is a known-fragile combination. Geometry Nodes' only job
# here is the clip itself, driven by a movable Empty.
# ---------------------------------------------------------------------------

_CLIP_GROUP = "BlenderFEA_VoxelClip"
_CLIP_EMPTY = "BlenderFEA_ClipPlane"
_CLIP_MODIFIER = "BlenderFEA_VoxelClip"

_CUBE_LOCAL_VERTS = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
], dtype=float)
_CUBE_LOCAL_FACES = np.array([
    [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
    [3, 2, 6, 7], [0, 3, 7, 4], [1, 5, 6, 2],
], dtype=np.int64)


def _active_voxel_centers(active3d, origin, vsize):
    """World-space centers of every True cell in active3d, in the same flat
    (C/'ij') order as active3d.ravel() - so a field3d[active3d] slice lines
    up 1:1 with these points for coloring."""
    nx, ny, nz = active3d.shape
    origin = np.asarray(origin, dtype=float)
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing='ij')
    centers = np.stack([ix, iy, iz], axis=-1).astype(float)
    centers = origin + (centers + 0.5) * vsize
    return centers[np.asarray(active3d, dtype=bool)]


def _voxel_cubes(active3d, origin, vsize):
    """One independent cube per active element: (N*8, 3) verts, (N*6, 4)
    faces. Vertices for cube i occupy [8*i : 8*i+8], in the same order as
    _active_voxel_centers - so per-voxel data (e.g. stress) can be repeated
    8x to color every cube's own vertices without any lookup."""
    centers = _active_voxel_centers(active3d, origin, vsize)
    n = len(centers)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    verts = (centers[:, None, :]
            + _CUBE_LOCAL_VERTS[None, :, :] * vsize).reshape(-1, 3)
    faces = (_CUBE_LOCAL_FACES[None, :, :]
            + (np.arange(n, dtype=np.int64) * 8)[:, None, None]).reshape(-1, 4)
    return verts, faces


def _cubes_mesh_to_object(verts, faces, name, collection=None):
    """Lean mesh builder for the voxel-cube cloud: no per-polygon smooth-
    shading loop / validate() pass -- each voxel should read as a flat-shaded
    cube, and a Python per-polygon loop doesn't scale to the millions of
    faces a fine, dense grid produces."""
    if not _HAS_BPY:
        raise RuntimeError("_cubes_mesh_to_object requires Blender")
    verts = np.asarray(verts, dtype=float)
    vlist = verts.tolist() if len(verts) else []
    flist = faces.tolist() if isinstance(faces, np.ndarray) else [list(f) for f in faces]

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vlist, [], flist)
    mesh.update()

    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, mesh)
        target = collection if collection is not None else bpy.context.scene.collection
        target.objects.link(obj)
    else:
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
    return obj


# Cull elements below this fraction of the vmax..vmin color range from the
# cloud entirely, not just color them dark -- otherwise the exterior is a
# solid low-value shell you'd have to clip through before seeing anything.
_CLOUD_MIN_FRAC = 0.05


def voxel_cloud_to_object(field3d, active3d, origin, vsize, vmax, vmin,
                          name, collection=None, min_frac=_CLOUD_MIN_FRAC,
                          bands=None):
    """Build (or reuse) the cross-section object: one real, fully-colored
    cube for every active element that clears min_frac of the color range.
    Colors are exact (no interpolation) - each cube's 8 vertices just take
    that element's own field value.

    bands : forwarded to _jet_rgb -- an int for discrete "Contour" shading
    (properties.color_mode == 'BANDED'), None/<=1 for the default
    continuous-per-cell gradient, so the cross-section cloud stays in sync
    with whatever shading mode the surface results are using.
    """
    active_mask = np.asarray(active3d, dtype=bool)
    field3d = np.asarray(field3d)
    threshold = float(vmin) + min_frac * max(float(vmax) - float(vmin), 1e-12)
    keep = active_mask & (field3d >= threshold)

    verts, faces = _voxel_cubes(keep, origin, vsize)
    obj = _cubes_mesh_to_object(verts, faces, name, collection=collection)
    if len(verts):
        vals = field3d[keep]
        span = max(float(vmax) - float(vmin), 1e-12)
        t = (vals - float(vmin)) / span
        rgb = _jet_rgb(t, bands)
        rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=-1)
        _bake_vertex_colors(obj, np.repeat(rgba, 8, axis=0))
    return obj


def _build_voxel_clip_group(grp):
    """Geometry it receives in, geometry clipped by a plane out - nothing
    else. The plane is defined by an Empty's world position + local Z axis:
    a face survives if dot(face_position - empty.location, empty.z_axis) <=
    0. 'Enable Clip' gates the whole thing so the cloud stays intact until
    you deliberately turn slicing on.
    """
    iface = grp.interface
    iface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    iface.new_socket("Clip Empty", in_out='INPUT',
                     socket_type='NodeSocketObject')
    enable_in = iface.new_socket("Enable Clip", in_out='INPUT',
                                 socket_type='NodeSocketBool')
    enable_in.default_value = False

    nodes, links = grp.nodes, grp.links
    nodes.clear()

    n_in = nodes.new('NodeGroupInput')
    n_in.location = (-700, 0)
    n_out = nodes.new('NodeGroupOutput')
    n_out.location = (500, 0)

    n_objinfo = nodes.new('GeometryNodeObjectInfo')
    n_objinfo.transform_space = 'ORIGINAL'
    n_objinfo.location = (-450, -250)

    n_rot = nodes.new('ShaderNodeVectorRotate')
    n_rot.rotation_type = 'EULER_XYZ'
    n_rot.inputs['Vector'].default_value = (0.0, 0.0, 1.0)
    n_rot.location = (-250, -250)

    n_pos = nodes.new('GeometryNodeInputPosition')
    n_pos.location = (-250, -450)

    n_sub = nodes.new('ShaderNodeVectorMath')
    n_sub.operation = 'SUBTRACT'
    n_sub.location = (-50, -350)

    n_dot = nodes.new('ShaderNodeVectorMath')
    n_dot.operation = 'DOT_PRODUCT'
    n_dot.location = (150, -350)

    n_cmp = nodes.new('FunctionNodeCompare')
    n_cmp.data_type = 'FLOAT'
    n_cmp.operation = 'GREATER_THAN'
    n_cmp.inputs['B'].default_value = 0.0
    n_cmp.location = (350, -350)

    n_del = nodes.new('GeometryNodeDeleteGeometry')
    n_del.domain = 'FACE'
    n_del.mode = 'ALL'
    n_del.location = (150, 100)

    n_switch = nodes.new('GeometryNodeSwitch')
    n_switch.input_type = 'GEOMETRY'
    n_switch.location = (300, 0)

    links.new(n_in.outputs['Clip Empty'], n_objinfo.inputs['Object'])
    links.new(n_objinfo.outputs['Rotation'], n_rot.inputs['Rotation'])
    links.new(n_pos.outputs['Position'], n_sub.inputs[0])
    links.new(n_objinfo.outputs['Location'], n_sub.inputs[1])
    links.new(n_sub.outputs['Vector'], n_dot.inputs[0])
    links.new(n_rot.outputs['Vector'], n_dot.inputs[1])
    links.new(n_dot.outputs['Value'], n_cmp.inputs['A'])
    links.new(n_cmp.outputs['Result'], n_del.inputs['Selection'])
    links.new(n_in.outputs['Geometry'], n_del.inputs['Geometry'])

    links.new(n_in.outputs['Enable Clip'], n_switch.inputs['Switch'])
    links.new(n_in.outputs['Geometry'], n_switch.inputs['False'])
    links.new(n_del.outputs['Geometry'], n_switch.inputs['True'])
    links.new(n_switch.outputs['Output'], n_out.inputs['Geometry'])


_CLIP_GROUP_REQUIRED_SOCKETS = {"Clip Empty", "Enable Clip"}


def _ensure_voxel_clip_group():
    """Get-or-create the voxel-clip Geometry Nodes group.

    Also self-heals a stale/incompatible cached group: if a node group named
    _CLIP_GROUP already exists in the .blend but is missing one of the input
    sockets this add-on expects (e.g. left over from an interrupted rebuild
    or a manual edit), it is removed and rebuilt from scratch rather than
    silently reused. This matters because ensure_stress_cloud_geonodes only
    assigns the Clip Empty / Enable Clip modifier inputs when those sockets
    exist, so a stale group missing a socket would otherwise fail silently
    (the modifier does nothing until the user manually re-picks the Empty).
    """
    grp = bpy.data.node_groups.get(_CLIP_GROUP)
    if grp is not None:
        names = {item.name for item in grp.interface.items_tree
                 if getattr(item, "item_type", None) == 'SOCKET'
                 and item.in_out == 'INPUT'}
        if _CLIP_GROUP_REQUIRED_SOCKETS <= names:
            return grp
        try:
            bpy.data.node_groups.remove(grp)
        except Exception:  # noqa: BLE001
            pass
        grp = None
    grp = bpy.data.node_groups.new(_CLIP_GROUP, 'GeometryNodeTree')
    try:
        _build_voxel_clip_group(grp)
    except Exception:
        bpy.data.node_groups.remove(grp)
        raise
    return grp


def _ensure_clip_empty(collection=None):
    obj = bpy.data.objects.get(_CLIP_EMPTY)
    if obj is not None:
        return obj
    obj = bpy.data.objects.new(_CLIP_EMPTY, None)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = 1.0
    target = collection if collection is not None else bpy.context.scene.collection
    target.objects.link(obj)
    return obj


def ensure_stress_cloud_geonodes(obj, collection=None, enable_clip=None):
    """Attach (or refresh) the voxel-clip modifier: once 'Enable Clip' is on,
    it deletes every cube face on one side of the BlenderFEA_ClipPlane Empty.

    enable_clip : None leaves the modifier's current toggle alone; True/False
    forces it - used to flip the cross-section on automatically once a solve
    finishes, so you see the cut immediately instead of a solid shell.

    Never raises - a Geometry Nodes API mismatch on some Blender version
    should just mean "no cross-section today", not a failed solve.
    """
    if not _HAS_BPY:
        return
    try:
        grp = _ensure_voxel_clip_group()
        empty = _ensure_clip_empty(collection)
        mod = obj.modifiers.get(_CLIP_MODIFIER)
        is_new = mod is None
        if mod is None or mod.type != 'NODES':
            mod = obj.modifiers.new(_CLIP_MODIFIER, 'NODES')
        mod.node_group = grp
        ids = {item.name: item.identifier for item in grp.interface.items_tree
              if getattr(item, "item_type", None) == 'SOCKET'
              and item.in_out == 'INPUT'}
        if "Clip Empty" in ids:
            mod[ids["Clip Empty"]] = empty
        if "Enable Clip" in ids:
            if enable_clip is not None:
                mod[ids["Enable Clip"]] = bool(enable_clip)
            elif is_new:
                mod[ids["Enable Clip"]] = False   # off by default - opt in
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] voxel clip setup skipped: {exc}")
