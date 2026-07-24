__version__ = "0.0.1"

# Optional: `G2C_APPLY_SOLUTIONS` binds worked implementations from
# `g2c/solutions/` onto their scaffold targets at import time, so notebooks
# and pytest runs can exercise the filled-in package without editing
# `g2c/<topic>/`. Unset (default) leaves scaffolds untouched.
#
#   G2C_APPLY_SOLUTIONS=1        every module (also: all/true/yes/on)
#   G2C_APPLY_SOLUTIONS=01-07    Modules 01 through 07 only
#   G2C_APPLY_SOLUTIONS=07,09b   specific modules
#   G2C_APPLY_SOLUTIONS=0        nothing (also: false/no/off)
#
# Selecting a subset lets a student hand back only the modules they are not
# debugging, instead of discarding every implementation they got right.
import os as _os

_g2c_solutions = _os.environ.get("G2C_APPLY_SOLUTIONS")
if _g2c_solutions is not None:
    from g2c.solutions import apply as _apply
    from g2c.solutions._selection import parse_selectors as _parse

    _selectors = _parse(_g2c_solutions)
    # None means "everything"; an empty list means an explicit off switch.
    if _selectors is None or _selectors:
        _apply(_selectors)
    del _apply, _parse, _selectors

del _os, _g2c_solutions
