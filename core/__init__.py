# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-numpy linear FEA core (no bpy imports at module scope where avoidable
-- voxelize.py is the one exception, since it necessarily reads scene
geometry; everything solver-side stays headless-testable)."""
