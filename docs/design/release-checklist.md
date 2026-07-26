# Release Checklist

Maintainer-facing: the ordered gates for cutting a public course release. The
sequencing constraints are real — the history rewrite (if any) must precede
the tag, the tag must precede the freeze ref, and the checkpoint release
should be uploaded before the URL canary's Monday run goes red.

## One-time decisions (resolve before v1.0, then prune this section)

- [ ] **Git history rewrite.** ~170MB of `.git`, mostly superseded image
  blobs. A rewrite drops a fresh clone to a fraction of that but invalidates
  every SHA — including the `git_commit` provenance stamped into any
  already-published checkpoint manifests. Decide **before** uploading
  `checkpoints-v1`. If yes: compress images first, rewrite, force-push,
  retag, and re-stamp or retrain the reference checkpoints.
- [x] **Course name.** Keeping *From Gradients to ChatGPT* (nominative use;
  decided 2026-07-26). Revisit only if actually challenged.
- [ ] **Image weight.** Convert `docs/` PNGs to WebP (~145MB → ~15MB at q82,
  measured). Must precede the history rewrite to pay off inside `.git`.

## Checkpoint release (independent of course version tags)

- [ ] All four reference models retrained at a public commit:
  `StoryLM-1M/5M/30M-base`, `TinyLLM-30M-base`.
- [ ] Ladder is monotonic: val loss 1M > 5M > 30M. (The May dev artifacts
  were not — the 30M was stopped at 12k steps and lost to the 5M.)
- [ ] `.venv/bin/python scripts/package_checkpoints.py --update-script`
- [ ] Commit the `checkpoints.sh` checksum diff and push.
- [ ] `gh release create checkpoints-v1 data/cache/checkpoint-dist/*.tar.gz
  --title "Reference checkpoints v1" --notes "Fetched by ./checkpoints.sh"`
- [ ] `python3 scripts/check_dataset_urls.py` — all green.
- [ ] Fetch smoke test: move one local artifact aside, run
  `./checkpoints.sh`, confirm download + `load_model_artifact_with_tokenizer`.
- Re-publishing after a retrain: `gh release upload checkpoints-v1
  data/cache/checkpoint-dist/*.tar.gz --clobber` (URLs stay stable; bump the
  tag only for breaking changes like a tokenizer swap).

## Course release (vX.Y)

- [ ] Pre-release dry runs done: notebooks 09B+ executed on real hardware (CI
  covers only 00–08 — the 09B–20 range is where stale-code bugs hide), and a
  Leg 0 cold start on a clean Mac following the README verbatim.
- [ ] `G2C_APPLY_SOLUTIONS=1 pytest` green;
  `pytest tests/test_scaffold_invariant.py` green.
- [ ] All CI green on the release commit (tests, bootstrap, docs,
  dataset-urls, scaffold-freeze).
- [ ] Tag: `git tag vX.Y && git push --tags`.
- [ ] Write the tag into the first non-comment line of
  `.github/scaffold-freeze`; commit and push.
- [ ] **Verify the freeze actually went live**: scaffold-freeze workflow green
  on `main`, and a deliberate scratch-branch edit to a file under `g2c/`
  makes it fail. This rehearsal has never been run — do not skip it on v1.0.
- [ ] GitHub Release notes: what changed, and the standing promise that
  updates never modify the student edit surface.
- [ ] Docs site deployed and spot-checked (parts tables identical across
  README / docs index / syllabus).

## Post-release standing rules

- Scaffold fixes route to the solutions mirror, tests, docs, or rubrics. A
  truly unavoidable scaffold change gets a `.github/scaffold-freeze` ledger
  entry with a note telling students how to patch an edited copy.
- Support cadence: this is a side project — responses within days to a week.
  Stuck-report issues triage first; they are the course's only telemetry.

## Announce

- [ ] Canary and CI green, and quiet, for a full week beforehand.
- [ ] Front-door editorial pass: README, `docs/index.md`, `docs/syllabus.md`.
- [ ] The announcement advertises Part I as a complete standalone course
  ("finish line, not a checkpoint") — completion design is the whole game for
  self-study, and the ~11-module version will finish more students than the
  20-module framing.
- [ ] Beta cohort feedback incorporated (Leg A from Module 00; Leg B entering
  at Module 12 via `G2C_APPLY_SOLUTIONS=01-11` + `./checkpoints.sh`).
