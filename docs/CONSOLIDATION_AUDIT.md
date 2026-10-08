# scINTILLA consolidation audit — 8 October 2026

## Canonical source and preservation

The canonical checkout is `codebase/Code`, on `master`. GitHub stores this
checkout's contents at the repository root, with one `scintilla` package.
`Code_variants` and `Code_fragmentation` are local linked Git worktrees,
not duplicate directories in the remote repository. Their branches and
working directories have been retained.

| Source | Revision | Finding |
| --- | --- | --- |
| Original pipeline | `37aa81a` | All 136 tracked file contents matched the local original checkout. |
| Fragmentation | `8916c0e` | All 137 tracked file contents matched its worktree; this commit is an ancestor of the variants branch. |
| Label-quality variants | `2b0d7b0` | All 141 tracked file contents matched its worktree. |
| Squashed GitHub master | `0807f2f` | Its complete Git tree was identical to the variants tree. |

No original-only or fragmentation-only files required recovery. Compared
with the original, the variants work adds five files and extends three
existing source files. Compared with fragmentation, it adds four files
and extends the CLI registry. The other apparent changes were file modes.

The initial uncommitted changes in the original and fragmentation
worktrees were exclusively executable-bit changes (`100644` to `100755`),
with no source-content edits. These modes were already present in the
squashed remote master. The original checkout was fast-forwarded using
command-local `core.filemode=false`; the repository configuration was not
changed, and no squash commits were replayed.

Every tracked file and every function/class name from all three historical
source trees remains present in the canonical source. This structural
check complements the regression tests; it does not prove equivalence for
every possible input. Original finite-data composite scoring is retained
through `fragmentation_weight=0.0`; the independent reviewer compared 20
cases against the historical implementation and found exact agreement.
Frozen fusion coefficients and variant column definitions were unchanged.

Before consolidation, a complete Git bundle, three tracked/untracked file
archives, binary patches, status records and a SHA-256 manifest were saved
outside the repository in
`scintilla_consolidation_backup_2026-10-08/`. All 414 archived file hashes
were verified, and `git bundle verify` confirmed complete history.

`scverse_plan.md` was already tracked. Its bytes remain identical to the
initial snapshot; it is excluded from the new commit.

## Corrections made during integration review

- Remove stale clustering ranks, cluster assignments and cached default
  quality scores when replacing clustering outputs, including all-failed
  runs. Bound confusion neighbours to the available cells.
- Replace stale classifier predictions and consistency outputs on rerun.
  Apply the training-fitted scaler to full-data consistency predictions.
- Exclude disabled metric arms entirely, so `0 * NaN` cannot contaminate
  ablations. Preserve missing active metrics rather than turning missing
  fragmentation into a favourable contribution.
- Group only observed label categories. Preserve any existing temporary
  observation column and report a missing top-1 confusion prerequisite.
- Require matching cached confusion/fragmentation ranks in the CLI. Run
  PCA when the default representation is absent, and create output
  directories through the existing exporters.
- Declare the YAML dependency required by configuration loading/saving.
  Resolve undefined names in deferred type annotations.
- Update documentation to use `master` and explain original scoring,
  reruns and explicit cache invalidation with `force=True`.

## Verification

Regression tests were run failing before their corresponding fixes.
Evidence and build/install logs are stored with the external backup.

- Existing variants baseline: **118 passed** in both the original
  environment and a dependency-equipped validation environment.
- Final full suite in a clean environment: **138 passed**, no skips.
  This includes a real raw-input CLI run through PCA, clustering,
  supervised consistency, all variant scores, CSVs and saved AnnData.
- Original environment before the final raw-input test was added:
  **136 passed, one skip**, because Scanpy was not installed there.
- Clean package installation and `uv pip check`: successful. The missing
  `PyYAML` dependency was reproduced before adding it.
- `uv build`: source distribution and wheel built successfully.
- Ruff checks for undefined names and syntax (`F821,F822,F823,E9`), and
  `git diff --check`: passed.
- Independent final review: no blocking findings; public positional
  scoring arguments and the explicit score-cache contract preserved.

The initial validation environment inherited an old Dask installation
incompatible with NumPy 2. A clean environment avoided that unrelated
dependency conflict and passed with NumPy 2.2.6, pandas 2.3.3,
scikit-learn 1.7.2 and Scanpy 1.11.5.

This audit validates source retention and tested integration behaviour.
It does not revalidate frozen-model calibration, biological claims,
real-atlas benchmark performance, or every optional backend combination.
