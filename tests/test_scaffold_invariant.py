"""Pedagogical functions in g2c/ must stay scaffolded.

If a worked implementation ever leaks back into a `g2c/<topic>/<file>.py`
function whose canonical implementation lives in `g2c/solutions/`, this
test fails — by qualified name — so the regression is immediately
visible in CI.

How we identify "this function should be a scaffold":

  * For every Python module under `g2c/solutions/` (skipping `__init__`
    files and the private patcher), we AST-walk and collect:
      - methods of every `_<ClassName>Impl` holder class, paired with
        `g2c.<...>.<ClassName>.<method>`
      - module-level functions, paired with `g2c.<...>.<func>`
  * For each pair, we open the matching file in `g2c/` and confirm
    that the named function is a direct scaffold — its body, after
    any docstring, contains the `# TODO` comment and a top-level
    `raise NotImplementedError`.

Keeping this guarantee as a pytest invariant means it survives
`pytest -q` without a separate CI step — if a worked implementation
ever leaks into `g2c/`, one of the parametrized targets fails by
qualified function name.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
G2C = REPO_ROOT / "g2c"
SOLUTIONS = G2C / "solutions"


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_not_implemented_raise(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def _function_is_scaffold(node, segment: str) -> bool:
    body = node.body[1:] if node.body and _is_docstring(node.body[0]) else node.body
    return "# TODO" in segment and any(
        _is_not_implemented_raise(s) for s in body
    )


def _collect_mirror_targets() -> list[tuple[Path, str | None, str]]:
    """Return [(target_file, class_name_or_None, function_name), ...]."""
    targets: list[tuple[Path, str | None, str]] = []
    for path in sorted(SOLUTIONS.rglob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        rel = path.relative_to(SOLUTIONS)
        target_file = G2C / rel
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                # Only `_<Name>Impl` holders refer to scaffold classes.
                name = stmt.name
                if not (name.startswith("_") and name.endswith("Impl")):
                    continue
                target_class = name[1:-4]
                for inner in stmt.body:
                    if isinstance(
                        inner, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        targets.append((target_file, target_class, inner.name))
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name.startswith("_"):
                    continue
                targets.append((target_file, None, stmt.name))
    return targets


def _find_function(
    source: str, class_name: str | None, func_name: str
) -> tuple[ast.AST, str] | None:
    tree = ast.parse(source)
    if class_name is None:
        for stmt in tree.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name == func_name
            ):
                return stmt, ast.get_source_segment(source, stmt) or ""
        return None
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == class_name:
            for inner in stmt.body:
                if (
                    isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and inner.name == func_name
                ):
                    return inner, ast.get_source_segment(source, inner) or ""
            return None
    return None


def _format_qualname(
    target_file: Path, class_name: str | None, func_name: str
) -> str:
    rel = target_file.relative_to(REPO_ROOT).with_suffix("")
    dotted = ".".join(rel.parts)
    if class_name is None:
        return f"{dotted}.{func_name}"
    return f"{dotted}.{class_name}.{func_name}"


_TARGETS = _collect_mirror_targets()


@pytest.mark.parametrize(
    "target_file,class_name,func_name",
    _TARGETS,
    ids=[_format_qualname(*t) for t in _TARGETS],
)
def test_target_is_scaffolded(
    target_file: Path, class_name: str | None, func_name: str
) -> None:
    """The target of every g2c/solutions mirror entry must be a scaffold."""
    assert target_file.exists(), (
        f"mirror targets nonexistent file: {target_file.relative_to(REPO_ROOT)}"
    )
    source = target_file.read_text()
    found = _find_function(source, class_name, func_name)
    assert found is not None, (
        f"target function not found in {target_file.relative_to(REPO_ROOT)}: "
        f"{class_name or '<module>'}.{func_name}"
    )
    node, segment = found
    assert _function_is_scaffold(node, segment), (
        f"pedagogical impl leaked into "
        f"{_format_qualname(target_file, class_name, func_name)} — "
        "function is no longer a `# TODO` + `raise NotImplementedError` "
        "scaffold. Either revert the leak or remove the corresponding "
        "entry from g2c/solutions/."
    )


def test_collection_nonempty() -> None:
    """Sanity: g2c/solutions/ should contain mirror files."""
    assert len(_TARGETS) > 0, (
        "no mirror targets discovered under g2c/solutions/ — "
        "did the mirror tree get wiped?"
    )
