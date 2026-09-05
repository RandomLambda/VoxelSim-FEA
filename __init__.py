# SPDX-License-Identifier: GPL-3.0-or-later
"""
VoxelSim FEA - Linear FEA (real material values) for solid mesh parts.

Voxelize a part, solve one linear-elastic load case with a matrix-free
conjugate-gradient FEA solver, and color the part's own surface -- plus an
optional 3D cross-section cloud -- by real von Mises stress or safety factor
(yield / stress), using one of four built-in isotropic material presets
(steel, aluminum, ABS, PP) or custom values (E / yield entered in MPa).
"""

from . import properties
from . import operators
from . import ui

_MODULES = (properties, operators, ui)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    for mod in reversed(_MODULES):
        mod.unregister()


if __name__ == "__main__":
    register()
