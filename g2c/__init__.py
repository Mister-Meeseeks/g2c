__version__ = "0.0.1"

# Optional: if `G2C_APPLY_SOLUTIONS=1`, bind every worked implementation
# from `g2c/solutions/` onto its scaffold target at import time. Lets
# notebooks and pytest runs exercise the filled-in package without
# editing `g2c/<topic>/`. Unset (default) leaves scaffolds untouched.
import os as _os

if _os.environ.get("G2C_APPLY_SOLUTIONS"):
    from g2c.solutions import apply as _apply

    _apply()
    del _apply

del _os
