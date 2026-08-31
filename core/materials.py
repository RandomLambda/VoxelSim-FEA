# SPDX-License-Identifier: GPL-3.0-or-later
"""
Isotropic material presets.

Four built-in presets (structural steel, aluminum, ABS, PP) plus "Custom"
for anything else. Units match how the UI shows and edits them directly
(properties.py's youngs_modulus / yield_strength fields) and how datasheets
are normally quoted: stiffness (E) in gigapascals (GPa), strength (yield) in
megapascals (MPa) -- deliberately different units, not a typo. The solver
works in SI Pascals internally; both conversions happen once, at the
boundary where a job is handed to core.solve_worker (see operators.py).

Values below are typical/generic reference figures for each material
family, not a datasheet for any specific alloy, grade or supplier -- a
reasonable default for a first-pass check, not a substitute for certified
material data on a safety-critical part.
"""

# Each preset: label, E [GPa], nu [-], yield_strength [MPa], density [kg/m^3],
# short note shown in the tooltip.
PRESETS = {
    "STEEL": {
        "label": "Steel (structural, generic)",
        "youngs_modulus": 210.0,
        "poisson": 0.30,
        "yield_strength": 300.0,
        "density": 7850.0,
        "note": "Generic mild/structural steel. Real alloys range roughly "
                "200-500 MPa yield -- check your grade.",
    },
    "ALUMINUM": {
        "label": "Aluminum (structural, generic)",
        "youngs_modulus": 66.0,
        "poisson": 0.33,
        "yield_strength": 250.0,
        "density": 2700.0,
        "note": "Generic wrought structural aluminum. Cast/other alloys and "
                "tempers differ substantially -- check your alloy and temper.",
    },
    "ABS": {
        "label": "Plastic (ABS, generic)",
        "youngs_modulus": 2.3,
        "poisson": 0.35,
        "yield_strength": 45.0,
        "density": 1050.0,
        "note": "Generic ABS. Polymers are often rate-/temperature-dependent "
                "and can creep under sustained load -- this is a "
                "linear-elastic snapshot only.",
    },
    "PP": {
        "label": "Plastic (PP, generic)",
        "youngs_modulus": 1.5,
        "poisson": 0.42,
        "yield_strength": 30.0,
        "density": 905.0,
        "note": "Generic polypropylene. Softer and more ductile than ABS -- "
                "real grades vary widely (homopolymer vs. copolymer). "
                "Linear-elastic snapshot only, same caveat as ABS.",
    },
}

# Stable order for the UI dropdown/preset buttons.
PRESET_ORDER = ("STEEL", "ALUMINUM", "ABS", "PP")


def get(key):
    """Preset dict for ``key`` (one of PRESET_ORDER), or None if unknown."""
    return PRESETS.get(key)
