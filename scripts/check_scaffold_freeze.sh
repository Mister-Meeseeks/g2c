#!/usr/bin/env bash
# Check that the frozen scaffold surface is unchanged since the release ref.
#
# Post-release, files under g2c/ — excluding g2c/solutions/ and
# g2c/notebook_extras/ — are the student edit surface: students fill in the
# scaffold bodies in place, so any upstream change to those files lands as a
# merge conflict inside somebody's homework. This check diffs that surface
# against the freeze ref recorded in .github/scaffold-freeze and fails on any
# modification or deletion. Additions are fine — a brand-new module's files
# merge cleanly into every working copy.
#
# .github/scaffold-freeze format ('#' starts a comment, blank lines ignored):
#   first entry:  the git ref (normally a release tag) the surface is frozen at
#   later entries: paths intentionally changed since the freeze — the
#                  breaking-change ledger. Each should carry a trailing comment
#                  telling a student who already edited that file how to patch
#                  their copy.
# No ref → the freeze is not active yet (pre-release); the check passes.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=.github/scaffold-freeze

entries=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [[ -n "$line" ]] && entries+=("$line")
done < "$CONFIG"

if [[ ${#entries[@]} -eq 0 ]]; then
    echo "scaffold-freeze: no freeze ref in $CONFIG — pre-release, nothing to check."
    exit 0
fi

ref="${entries[0]}"

if ! git rev-parse --verify --quiet "${ref}^{commit}" > /dev/null; then
    echo "scaffold-freeze: freeze ref '$ref' not found — shallow clone or missing tag?" >&2
    exit 1
fi

pathspec=("g2c/" ":(exclude)g2c/solutions/" ":(exclude)g2c/notebook_extras/")
if [[ ${#entries[@]} -gt 1 ]]; then
    for exception in "${entries[@]:1}"; do
        pathspec+=(":(exclude)$exception")
    done
fi

changed="$(git diff --name-only --diff-filter=MD "$ref" HEAD -- "${pathspec[@]}")"

if [[ -n "$changed" ]]; then
    {
        echo "scaffold-freeze: student edit surface changed since '$ref':"
        echo "$changed" | sed 's/^/  /'
        echo
        echo "Students edit these files in place, so upstream changes to them merge-"
        echo "conflict with student work. Route the fix to g2c/solutions/, tests/,"
        echo "docs/, or rubrics instead. If the scaffold change is genuinely"
        echo "unavoidable, add the path to .github/scaffold-freeze with a comment"
        echo "telling students how to patch an already-edited copy."
    } >&2
    exit 1
fi

echo "scaffold-freeze: student edit surface unchanged since '$ref'."
