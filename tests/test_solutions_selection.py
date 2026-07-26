"""Per-module solutions selection.

Covers the selector grammar, the module->mirror map, and the behavior a stuck
student actually depends on: handing back some modules while keeping their own
work in the rest.

Behavioral cases run in a subprocess on purpose. `apply()` mutates the imported
`g2c` package in place, so patching in-process would leak into every later test
in the session and make the suite order-dependent. A subprocess also matches
how a student invokes it — the env var is read at interpreter start.

Suggested order to implement & turn green: none. This file tests course
infrastructure, not a student deliverable, and should be green at all times.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from g2c.solutions._patcher import _iter_mirror_modules
from g2c.solutions._selection import (
    MIRRORS,
    MODULE_ORDER,
    MODULE_TARGETS,
    SelectionError,
    normalize_module,
    parse_selectors,
    resolve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, env_value: str | None) -> subprocess.CompletedProcess:
    """Run `script` in a clean interpreter, optionally with the env var set."""
    import os

    env = dict(os.environ)
    env.pop("G2C_APPLY_SOLUTIONS", None)
    if env_value is not None:
        env["G2C_APPLY_SOLUTIONS"] = env_value
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# The map must track the mirror tree exactly.
# --------------------------------------------------------------------------


def test_map_covers_every_mirror_file():
    """A mirror file missing from the map can never be selected by module."""
    on_disk = {n.replace("g2c.solutions.", "") for n in _iter_mirror_modules()}
    assert on_disk - set(MIRRORS) == set()


def test_map_has_no_stale_entries():
    """A map entry with no mirror file would silently bind nothing."""
    on_disk = {n.replace("g2c.solutions.", "") for n in _iter_mirror_modules()}
    assert set(MIRRORS) - on_disk == set()


def test_every_module_id_is_reachable():
    for module_id in MODULE_ORDER:
        assert normalize_module(module_id) == module_id


# --------------------------------------------------------------------------
# Selector grammar.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("7", "07"),
        ("07", "07"),
        ("3b", "03b"),
        ("03B", "03b"),
        ("9b", "09b"),
        ("20", "20"),
        (" 07 ", "07"),
    ],
)
def test_normalize_module_accepts_launcher_shapes(token, expected):
    assert normalize_module(token) == expected


@pytest.mark.parametrize("token", ["", "99", "attention", "nonsense"])
def test_normalize_module_rejects_non_modules(token):
    assert normalize_module(token) is None


def test_range_spans_course_order_including_lettered_modules():
    """01-04 must pick up 03b, which sorts between 03 and 04 by course order."""
    resolved = resolve(["01-04"])
    topics = {name.split(".")[2] for name in resolved}
    assert topics == {"autodiff", "tensors", "nn", "training", "tokenizer"}


def test_reversed_range_is_accepted():
    assert resolve(["07-01"]) == resolve(["01-07"])


def test_topic_selector_expands_to_all_its_files():
    resolved = resolve(["sampling"])
    assert len(resolved) == 6
    assert all(".sampling." in name for name in resolved)


def test_single_mirror_file_selector():
    resolved = resolve(["attention.multi_head"])
    assert set(resolved) == {"g2c.solutions.attention.multi_head"}


def test_unknown_selector_raises_with_guidance():
    with pytest.raises(SelectionError) as excinfo:
        resolve(["modle7"])
    message = str(excinfo.value)
    assert "modle7" in message
    # The error must tell a stuck student what they *can* type.
    assert "modules:" in message and "topics:" in message


@pytest.mark.parametrize("value", ["1", "all", "true", "YES", "on"])
def test_parse_selectors_all_tokens(value):
    assert parse_selectors(value) is None


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  "])
def test_parse_selectors_off_tokens(value):
    """`G2C_APPLY_SOLUTIONS=0` must mean off, not 'non-empty so apply all'."""
    assert parse_selectors(value) == []


@pytest.mark.parametrize(
    "value,expected",
    [
        ("01-07", ["01-07"]),
        ("7,9b", ["7", "9b"]),
        ("07 09b", ["07", "09b"]),
        ("attention, 09b", ["attention", "09b"]),
    ],
)
def test_parse_selectors_splits_lists(value, expected):
    assert parse_selectors(value) == expected


# --------------------------------------------------------------------------
# Modules 07/08 and 09/16 share files. Selecting one must not leak the other.
# --------------------------------------------------------------------------


def test_modules_07_and_08_do_not_leak_into_each_other():
    """Both are `g2c/attention/`; a topic-level filter would conflate them."""
    assert set(resolve(["07"])) == {"g2c.solutions.attention.self_attention"}
    assert set(resolve(["08"])) == {"g2c.solutions.attention.multi_head"}


def _transformer_lm_allows(selectors: list[str]) -> dict[str, bool]:
    targets = resolve(selectors)["g2c.solutions.transformer.transformer_lm"]
    return {
        member: any(t.allows(member) for t in targets)
        for member in ("forward", "forward_cached")
    }


def test_module_09_gets_forward_but_not_the_module_16_cache_path():
    allows = _transformer_lm_allows(["09"])
    assert allows == {"forward": True, "forward_cached": False}


def test_module_16_gets_forward_cached_but_not_module_09s_forward():
    """Selecting 16 must not hand over the transformer forward pass."""
    allows = _transformer_lm_allows(["16"])
    assert allows == {"forward": False, "forward_cached": True}


def test_selecting_both_09_and_16_unions_the_members():
    allows = _transformer_lm_allows(["09", "16"])
    assert allows == {"forward": True, "forward_cached": True}


def test_module_12_maps_to_no_package():
    """Module 12 is experiments over Module 10's trainer; it ships no deliverable."""
    assert MODULE_TARGETS["12"] == ()
    assert resolve(["12"]) == {}


# --------------------------------------------------------------------------
# Behavior: partial application really is partial.
# --------------------------------------------------------------------------


# A scaffold's code object references NotImplementedError; a bound impl's does
# not. Checking `co_names` beats calling the function (scaffolds raise) and
# beats reading source (which depends on file layout), and it works uniformly
# for methods and free functions.
PROBE = """
from g2c.autodiff.value import Value
from g2c.attention.self_attention import SelfAttention

def live(fn):
    return "NotImplementedError" not in fn.__code__.co_names

print(f"autodiff={live(Value.__mul__)} attention={live(SelfAttention.forward)}")
"""


def test_selecting_module_01_leaves_module_07_scaffolded():
    result = _run(PROBE, "01")
    assert result.returncode == 0, result.stderr
    assert "autodiff=True attention=False" in result.stdout


def test_selecting_all_applies_everything():
    result = _run(PROBE, "1")
    assert result.returncode == 0, result.stderr
    assert "autodiff=True attention=True" in result.stdout


def test_unset_applies_nothing():
    result = _run(PROBE, None)
    assert result.returncode == 0, result.stderr
    assert "autodiff=False attention=False" in result.stdout


def test_zero_applies_nothing():
    result = _run(PROBE, "0")
    assert result.returncode == 0, result.stderr
    assert "autodiff=False attention=False" in result.stdout


def test_range_applies_the_whole_span():
    result = _run(PROBE, "01-07")
    assert result.returncode == 0, result.stderr
    assert "autodiff=True attention=True" in result.stdout


def test_bad_selector_fails_loudly_at_import():
    """A typo must not silently leave the student on scaffolds."""
    result = _run("import g2c", "modul7")
    assert result.returncode != 0
    assert "unknown solutions selector" in result.stderr


def test_apply_with_no_argument_still_binds_everything():
    """Backward compatibility: existing callers pass nothing."""
    script = (
        "import g2c.solutions as s;"
        "swapped = s.apply();"
        "print(len(swapped))"
    )
    result = _run(script, None)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 100
