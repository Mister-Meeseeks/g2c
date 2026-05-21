"""Canonical implementations for g2c pedagogical scaffolds.

`g2c/` itself stays pristine — every pedagogical function is left as
`# TODO` + `raise NotImplementedError` for the student to fill in. This
package mirrors that tree and, when `apply()` is called, binds the
implementations onto the matching scaffold targets at runtime.

Typical use in a worked notebook:

    import g2c.solutions
    g2c.solutions.apply()

    # Now g2c.autodiff.value.Value.__add__ etc. are live.
    from g2c.autodiff.value import Value
    a = Value(2.0) + Value(3.0)

The scaffold invariant is enforced by `tests/test_scaffold_invariant.py`:
if a pedagogical implementation ever leaks into `g2c/` (the body of a
scaffolded function gets replaced with real code), that test fails by
name so the regression is obvious.

Mirror file convention (recipe B from the design):

* Each pedagogical class in `g2c/<topic>/<file>.py` gets a
  `_<ClassName>Impl` holder class in `g2c/solutions/<topic>/<file>.py`.
  The holder's methods are bound onto the real class by `apply()`.
* Each pedagogical module-level function gets a same-named function
  in the mirror file. Bound onto the target module by `apply()`.
* Methods reference the target class by its imported name (e.g.
  `Value(...)` inside `_ValueImpl.__add__` constructs a real `Value`,
  because `Value` is imported from `g2c.autodiff.value` at the top of
  the mirror file).
"""
from __future__ import annotations

from ._patcher import apply

__all__ = ["apply"]
