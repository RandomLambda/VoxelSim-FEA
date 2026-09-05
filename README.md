# VoxelSim FEA
VoxelSim FEA runs a real linear-elastic finite element analysis on any closed
mesh, straight inside Blender. Pick the part, mark its supports and loads,
choose a material, and solve — no external FEA package, no export/import
round-trip.

**Real material values.** Four built-in isotropic presets (structural
steel, structural aluminum, ABS, PP) plus a Custom slot for your own
datasheet numbers. Stiffness is entered in GPa, yield strength in MPa —
matching how material datasheets are normally quoted — and the solver
reports actual von Mises stress in MPa and displacement in mm, not a
unitless placeholder.

**Four ways to look at the result**, all built from the same solve and kept
side by side in a "VoxelSim FEA Results" collection so you can flip between
them instantly without re-solving:
- Von Mises stress, mapped onto the part's own surface
- Displacement magnitude, same surface mapping, in millimeters
- Safety factor (yield ÷ von Mises stress), so you can see at a glance
  where the design is closest to yielding
- A 3D voxel cross-section with a Geometry-Nodes clip plane you can drag
  through the part to inspect internal stress

An on-screen color-scale legend is drawn directly in the viewport so the
color mapping is never a guess.

**Boundary conditions** are just mesh objects: drop a small mesh where the
part is held (a support, with per-axis X/Y/Z locking for rollers), another
where a load acts (a real force vector in Newtons), and VoxelSim FEA voxelizes
the part and assembles the model automatically.

**Compute backend**: Auto (recommended) or CPU, plus multi-process CPU
. The matrix-free conjugate-gradient solver core is shared with Blendtopo (our topology-optimization add-on) — 
proven on real hardware there, adapted here for a single, absolute-value
linear solve instead of an iterative relative-shape one.
