# scINTILLA → scverse: audit, verdict, and route to listing

## Context

`soumickmj/scINTILLA` is a 13,019-line pure-Python package (127 modules, 16 subpackages)
that wraps scanpy and scikit-learn to benchmark and select methods across a whole
scRNA-seq analysis: normalisation, feature selection, dimensionality reduction,
clustering, supervised classification, differential expression, annotation and batch
correction. The question is whether it belongs in scverse, and if so in which tier.

This document records the audit, the verdict, the decisions taken, and the executable
plan. Every finding below was verified directly against the working tree at `bf50df2`.

**Decisions taken (yours, 2026-09-11):**
- Distribution name **`scintilla-py`**, import name stays **`scintilla`**.
- Public API goes **AnnData-first**; the array-based statistics layer stays as a
  documented internal layer.
- **Correctness bugs first**, before any packaging or restructuring work.

---

## Part 1: What the two scverse tiers actually are

`scverse.org` is blocked by this session's egress proxy, so the facts below come from
the registry that *drives* that website,
[`scverse/ecosystem-packages`](https://github.com/scverse/ecosystem-packages), plus the
governance repo and the cookiecutter template.

Every listed package carries a `category` field in its `meta.yaml`. There are three
values, and the website renders them as its three sections:

| `category` | Website section | Members |
|---|---|---|
| `core-datastructure` | Core packages | anndata, mudata, spatialdata |
| `core-framework` | Frameworks | scanpy, muon, squidpy, scvi-tools, scirpy, SnapATAC2, rapids-singlecell, pertpy, decoupler |
| `ecosystem` | Ecosystem | ~130 packages (scib, liana, cell2location, CellRank, moscot, …) |

Source: [`core-order.yml`](https://raw.githubusercontent.com/scverse/ecosystem-packages/main/core-order.yml),
whose header comment reads *"Must list exactly the packages whose `category` is
core-datastructure or core-framework"*.

Ecosystem listing *is* an open process: a pull request adding one
`meta.yaml` file, reviewed against a fixed nine-item checklist.

---

## Part 3: Audit

### 3.1 Gap against the nine mandatory scverse requirements

| # | Requirement | Status |
|---|---|---|
| 1 | OSI-approved licence | **Pass.** Apache-2.0 (`LICENSE`, 201 lines, verbatim). But `pyproject.toml` declares no licence at all, so the built wheel and PyPI page would read "UNKNOWN". |
| 2 | Versioned releases | **Fail.** No git tags, no GitHub releases. |
| 3 | Installable from a standard registry | **Fail.** Not on PyPI, conda-forge or bioconda. `scintilla` is **taken on PyPI** (v0.0.3, an abandoned 2022 Faker-based test-data helper). Hence the `scintilla-py` decision. |
| 4 | Automated tests over a reasonable range of inputs | **Fail.** Zero test files, no `tests/`, no `conftest.py`, no `[tool.pytest]`, no doctests. 13,019 lines of numerically sensitive code with no automated verification. |
| 5 | CI that runs those tests | **Fail.** No `.github/` directory at all. |
| 6 | API documentation | **Partial.** `README.md` (1,421 lines) and `GUIDE.md` (1,741 lines) are unusually thorough prose, but there is no generated API reference, no docs site, no intersphinx inventory. Reviewers explicitly asked Spectra for Sphinx autodoc rather than hand-written README docs. |
| 7 | Uses scverse data structures where appropriate | **Partial, and this is the substantive one.** AnnData is used throughout, but almost none of the conventions around it are followed. See §3.3. |
| 8 | Author agrees to listing | Yours to tick. |
| 9 | Agrees to the code of conduct | Yours to tick. |

Recommended but absent: tutorials (`demo_notebook.ipynb` does not qualify, §3.5),
the cookiecutter template, a community announcement.

### 3.2 Correctness bugs (fix these first, regardless of scverse)

These are not style issues. Each produces a wrong answer silently.

**(a) `transfer_labels` matches genes by column position, not by `var_names`.**
`scintilla/annotation/label_transfer.py:49-59` extracts raw matrices and, when the
shapes differ, fits a PCA on the reference and calls `pca.transform(X_qry)`. There is
no `var_names` intersection anywhere. When the shapes happen to match, labels are
transferred assuming gene *i* in the reference is gene *i* in the query. For a
reference-mapping function this is the worst class of bug: it returns plausible labels
that are meaningless.

**(b) The pseudobulk arm of `benchmark_de_methods` fabricates a degenerate design.**
`scintilla/differential_expression/benchmark.py:77-81` creates one pseudo-sample per
condition (`groups + "_s1"`) when no `sample` column exists, so pseudobulk runs at
n=1 per group. Whatever it reports is not a pseudobulk result, yet it appears in the
leaderboard alongside valid methods.

**(c) `import scintilla` globally hijacks the matplotlib backend.**
`matplotlib.use("Agg")` is called at import time in **14 modules**
(`clustering/benchmark.py:13`, `clustering/hierarchical.py:18`,
`classification/benchmark.py:14`, `classification/visualise.py:12`, and all ten
`visualisation/*.py` including `visualisation/__init__.py:4`). The chain
`scintilla/__init__.py:14` → `clustering.run` → `clustering/__init__.py` →
`clustering/hierarchical.py:18` fires on a bare `import scintilla`. In a Jupyter
kernel this switches matplotlib to the non-interactive Agg backend for the whole
session, after which nothing from any library renders inline. Compounding it,
`classification/visualise.py:17-22` calls `plt.close(fig)` on every figure it returns.

**(d) 16 silent `except Exception: pass` sites inside benchmark loops.**
There are 64 broad `except Exception` handlers across 32 files; 16 swallow entirely
without logging, concentrated exactly where it hurts most:
`clustering/consensus.py:73,83,94,116` (four in one 137-line file),
`preprocessing/benchmark.py:185,200,328`, `classification/run.py:138,143`,
`classification/benchmark.py:50,57`, `clustering/benchmark.py:217`,
`clustering/spectral.py:88`, `evaluation/classification_metrics.py:132`,
`preprocessing/normality.py:85`, `classification/diagnostics.py:55`.
On a benchmarking tool, **a silently dropped method is indistinguishable from a method
that scored poorly**, and the user gets no signal either way. The codebase already
knows the right pattern: there are 32 correct `except ImportError` handlers elsewhere.
`classification/benchmark.py:50,57` in particular use `except Exception` for optional
xgboost/lightgbm imports, so a genuine bug inside `models.py` looks like "not
installed" and the model vanishes from the leaderboard.

**(e) `AnalysisConfig` is largely inert, and two guards are wired backwards.**
Ten of its 24 fields are never read by anything: `random_seed`, `test_size`,
`n_pca_comps`, `verbose`, `bootstrap_ci`, `n_bootstrap`, `scoring_method`,
`auto_pca_components`, `no_info_method`, `mp_sigma_method`. All ten are documented as
live in the YAML that `generate_default_yaml` emits (`analysis_config.py:269-290`), so
users will set them and get no error and no effect. `AnalysisConfig.robust()` promises
five behaviours in its docstring; only `adaptive_resolution` and `auto_eps` actually
take effect. Two specific wiring bugs:

```python
# scintilla/classification/run.py:152-154   (guard is inverted)
if config is not None and not include_shap:
    include_shap = getattr(config, "include_shap", include_shap)
```
`include_shap` defaults to `True`, so `not include_shap` is `False` and the config is
never consulted. `AnalysisConfig.fast()` sets `include_shap=False` precisely to skip
the expensive SHAP step, and is ignored.

```python
# scintilla/feature_selection/benchmark.py:55-56   (sentinel by value)
if config is not None and hasattr(config, "n_features") and n_features == 50:
    n_features = config.n_features
```
A user who explicitly passes `n_features=50` is silently overridden; `n_features=51`
is respected. This inverts the documented precedence rule at `analysis_config.py:53-54`.

**(f) The documented global-seed override does not work.**
`README.md:1141-1145` tells users to set `scintilla.config.RANDOM_SEED = 0`. Thirty-two
modules do `from scintilla.config import RANDOM_SEED`, binding by value at import, and
`scintilla/__init__.py:3` forces that import eagerly. Only two modules
(`preprocessing/normality.py:67`, `classification/diagnostics.py:26`) import inside the
function body and would see the change. So the documented knob works for 2 of 32 call
sites, and silently breaks reproducibility for anyone who follows the README.

**(g) `scintilla/scripts/` ships in no wheel.** The directory has 7 modules / 268 lines
and **no `__init__.py`**, and `[tool.setuptools.packages.find]` (`pyproject.toml:48-50`)
uses `find`, not `find_namespace`. Verified: `setuptools.find_packages('.',
include=['scintilla*'])` does not discover `scintilla.scripts`. Those lines work from a
git checkout and vanish on `pip install`. They also duplicate `cli/commands/` almost
one-for-one (`run_de_analysis.py:20-34` versus `cli/commands/de.py:17-31` share the
same dispatch block) while lacking the `--config`/`--fast` options `cli/` has since
gained. Delete the directory.

### 3.3 API conventions: the substantive rework

**No sparse support. 45 unconditional densification sites, `scipy.sparse` imported once
in the whole package** (`evaluation/batch_metrics.py:69`, and only to test scanpy's
connectivity graph). The idiom
`X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()` appears 45
times, often followed by `.astype(np.float64)`, doubling the already-dense footprint.
The worst case is `eda/summary.py:17-20`, which densifies the entire matrix purely to
compute `sparsity = float(np.mean(X == 0))`. On a 500k × 30k CSR matrix that allocates
tens of GB for a number `1 - adata.X.nnz / (n_obs * n_vars)` gives for free. The
cheapest EDA call in the package is the one most likely to run out of memory.

**Signatures are inconsistent about `adata`.** Four conventions coexist: `adata: AnnData`
(~15 functions), `adata: Union[pd.DataFrame, AnnData]` (~10), `data: Union[...]`
(~35, including both flagship entry points `unsupervised_analysis` at
`clustering/run.py:18` and `supervised_analysis` at `classification/run.py:36`), and
bare arrays with no AnnData at all (~90).

**No `copy`, no `inplace`, no `key_added` anywhere.** `grep` finds zero matches for
`copy: bool` or `inplace`. `key_added` appears twice, both internal pass-throughs to
scanpy (`feature_selection/benchmark.py:179,181`). Aliasing is unpredictable and
undocumented: `run_umap` (`dimensionality_reduction/umap.py:39`), `combat_correct`,
`harmony_correct`, `annotate_by_markers` and all transformations copy defensively;
`run_pca` (`preprocessing/pca.py:46`), `rank_genes_groups`
(`differential_expression/rank_genes.py:42`), `unsupervised_analysis` and
`supervised_analysis` mutate the caller's object. `select_hvg`
(`feature_selection/hvg.py:41,57`) manages both: it mutates the input's `.var` *and*
returns a different filtered copy.

**Output keys are hard-coded and unconfigurable.** `unsupervised_analysis` with defaults
writes **13 new obs columns**, including `scintilla_top1_leiden|res=1.0`-style names with
a pipe and embedded parameters (`clustering/run.py:109-110`), plus
`scintilla_top{n}_confusion` (`:121`) and `scintilla_cluster` (`:127`).
`supervised_analysis` writes up to 28 more (`pred_{model}`, `pred_{model}_confidence`,
`pred_consensus`, `pred_agreement`, `pred_entropy`, `pred_avg_confidence` at
`classification/run.py:237-285`). Two runs at different resolutions cannot coexist.

**`.layers` is never used anywhere.** Transformations replace `.X` wholesale
(`preprocessing/transformations.py:57`), destroying raw counts unless the user copied
first. `.var` is never written. `.uns` carries no provenance: the only scintilla-written
keys are `uns["pca"]["auto_n_comps"]` and `["auto_method"]`.

**Seeding is by fiat, not by API.** `RANDOM_SEED = 42` (`config.py:3`) is used at ~30
sites. Only `kmeans_clustering` exposes a seed parameter. `benchmarking/scalability.py:53`
hard-codes a *second* literal `np.random.default_rng(42)`, and `time_estimator.py` uses
the bare literal `42` in 14 places. The package's own `seed_stability_test`
(`benchmarking/reproducibility.py:13`) requires its `method_fn` to accept
`random_state=`, which almost no scintilla function does. So runs are repeatable at the
default seed but not seed-sweepable, which is precisely the property a benchmarking
package needs.

**`print()` is the logging mechanism.** 86 `print()` calls across 26 files against a
single function-local `logging.getLogger` (`evaluation/clustering_metrics.py:232-235`).
About 49 of the 86 are in library modules, not the CLI:
`classification/run.py` (11), `benchmarking/time_estimator.py` (13),
`classification/benchmark.py` (6). `verbose: bool = True` defaults mean a library user
gets unsolicited stdout unless they pass `verbose=False` at every call. There is no
`scintilla.settings.verbosity` equivalent.

**Dataset-specific defaults leaked from the internal project.**
`config.py:24-34` hard-codes a haematopoietic weighting as a global:

```python
CELL_TYPE_WEIGHTS = {
    'Early GMP': 3.0, 'GMP': 3.0, 'LMPP': 3.0, 'MEP_prog': 3.0, 'MPP': 3.0,
    'ProMono': 3.0, 'Monocyte': 2.0, 'Pre-cDC': 2.0, 'Pre-pDC': 2.0,
}
```

I verified it is **dead code**: zero references outside `config.py`, not exported, not
documented. It is residue from the bone-marrow/AML project the repo was extracted from
("initial push from internal", `ef82cf6`). `CLASSIFIER_SUITE`, `DE_METHODS` and
`BATCH_CORRECTION_WEIGHTS` are likewise dead while being documented as live knobs in
`README.md:1166-1169`. Separately, the package disagrees with itself on the cell-type
column: `classification/visualise.py:167,268,333,399,483` default to `"CellType"`,
while `benchmarking/time_estimator.py:565` and `cli/commands/estimate_time.py:12`
default to `"cell_type"`. `classification/run.py:37` defaults `target_col="target"`,
a scikit-learn-ism no real `.h5ad` carries.

**MuData support is vestigial and mis-annotated.** `load_h5mu` exists
(`io/loaders.py:22-33`) and `auto_detect_format` is annotated `-> ad.AnnData` but
returns a `MuData` for `.h5mu` (`io/loaders.py:62,68-69`). Nothing downstream can
consume one: `ensure_anndata` (`:77-109`) raises `TypeError` on anything that is not
AnnData, ndarray or DataFrame. So `scintilla eda data.h5mu` loads and then crashes one
function later. SpatialData: zero occurrences.

**The README's quickstart shadows scanpy.** It opens `import scintilla as sc`, colliding
with the near-universal `import scanpy as sc`. Use `import scintilla as si` throughout.

### 3.4 Packaging

`pyproject.toml` is 51 lines and `[project]` is 20 of them. Absent: `readme` (so the
PyPI page would be blank despite a 1,421-line README), `license`/`license-files`,
`authors`/`maintainers`, `[project.urls]`, `classifiers`, `keywords`, `dynamic`
version. Version is duplicated by hand at `pyproject.toml:7` and
`scintilla/__init__.py:86`. All 13 core and 15 optional dependencies are **bare names
with no lower bounds**, yet `annotation/marker_based.py:52` calls
`sc.tl.score_genes(..., ctrl_as_ref=False)`, which needs scanpy ≥ 1.10: a clean install
on scanpy 1.9 fails at runtime. `requires-python = ">=3.9"` (Python 3.9 reached end of
life in October 2025; the scverse template now targets ≥ 3.12). No `py.typed`, so the
83% of arguments that *are* annotated are invisible to downstream type checkers. There
are **183 `# noqa: PLC0415` suppressions** but no `[tool.ruff]` config committed, so
the lint state is unreproducible and the markers are unexplained noise.

**Licence caveat.** The package is Apache-2.0, but **`pingouin` is a mandatory runtime
dependency and is GPL-3.0**. Apache-2.0 is one-way compatible *into* GPLv3, so the
combined installed work is effectively governed by GPLv3 terms. `mrmr-selection`
(GPL-3.0) and `leidenalg` (GPL-3.0) are also GPL but sit in the optional `full` extra,
which is the much safer place. Not a blocker for listing (liana itself is GPL-3.0-only)
but worth resolving deliberately rather than by accident.

### 3.5 Docs, notebook, community files

`README.md` (1,421 lines) and `GUIDE.md` (1,741 lines) overlap by ~25% of the README's
substantive content. Both are good material in the wrong place. No badges of any kind.
Three doc bugs found: `GUIDE.md:62` documents `pip install -e ".[test]"` but no `test`
extra exists; `README.md:1160` says `DBSCAN_EPS_RANGE` has 9 values when `config.py:14`
has 5; `README.md:1141-1145` documents the seed override that does not work (§3.2f).

`demo_notebook.ipynb` (10 cells) is not a tutorial and cannot run: no data-loading cell
at all, and both code cells open with `print(f"Total cells: {adata_hvg.shape[0]}")`
against an `adata_hvg` defined nowhere, so it raises `NameError` immediately. All four
code cells have `"outputs": []` and were committed that way. It covers 2 of 16
subpackages. As shipped it is a liability: it is the first thing a prospective user
clicks.

Absent: `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`,
`SECURITY.md`, issue and PR templates, `.pre-commit-config.yaml`, `.readthedocs.yaml`.
No DOI, no Zenodo, no "how to cite": grep for `cite`/`citation` across both docs
returns zero hits, which for an academic tool is the single highest-value omission.

**Clean on the usual leak vectors**, which is worth saying: no absolute or cluster
paths, no institution names, no credentials, no e-mail addresses in the tree, no
ethics or patient identifiers, no deleted-but-recoverable blobs in git history, and the
notebook never contained outputs. `.gitignore` is the unmodified GitHub Python
template (207 lines); it ignores `.pypirc`, which is the one that matters.

---

## Part 4: Route A — Ecosystem listing (the plan)

Roughly 6 to 10 weeks part-time. Phases 1 to 4 are the bulk.

### Phase 0 — Correctness bugs (first, as agreed)

Branch `claude/gallant-goldberg-94eaav`. Small, reviewable commits.

1. **`transfer_labels`**: intersect `reference_adata.var_names` with
   `query_adata.var_names`, subset both to the shared genes in a consistent order, and
   raise a clear error when the overlap is empty or implausibly small. Drop the
   `np.pad` fallback at `annotation/label_transfer.py:57-59` entirely; it papers over
   exactly the mismatch that should be an error.
2. **`_run_pseudobulk`**: when no sample column exists, raise or skip the pseudobulk
   arm with an explicit message rather than fabricating `groups + "_s1"`
   (`differential_expression/benchmark.py:77-81`). A method that cannot run honestly
   must not appear in the leaderboard.
3. **Remove all 14 `matplotlib.use("Agg")` calls.** If a headless default is wanted,
   set it once in `cli/main.py` before any plotting import. Stop calling
   `plt.close(fig)` on returned figures (`classification/visualise.py:17-22`); let the
   caller decide.
4. **The 16 silent handlers**: narrow to the specific exception types expected and emit
   `warnings.warn(f"{method} failed: {exc}")` at minimum. Add a `failed` column or a
   `failures` dict to every benchmark result so a dropped method is visible in the
   output, not just absent from it. Change `classification/benchmark.py:50,57` to
   `except ImportError`.
5. **`AnalysisConfig`**: fix the inverted `include_shap` guard
   (`classification/run.py:152-154`) with a proper `None` sentinel; replace the
   `n_features == 50` value-sentinel (`feature_selection/benchmark.py:55-56`) with
   `n_features: int | None = None`; then either wire up or delete the ten inert fields,
   and regenerate the YAML template so it advertises only what works.
6. **Seeding**: make `RANDOM_SEED` a default value rather than an imported constant.
   Thread `random_state` through every stochastic public function, so
   `seed_stability_test` can actually be pointed at the package's own functions.
   Replace the bare `42` literals in `benchmarking/`.
7. **Delete `scintilla/scripts/`** (unshipped, unreferenced, duplicates `cli/`), and
   delete the dead constants `CELL_TYPE_WEIGHTS`, `CLASSIFIER_SUITE`, `DE_METHODS`,
   `BATCH_CORRECTION_WEIGHTS` from `config.py` along with their README table rows.
   Settle on `cell_type` as the single default column name.
8. **Fix the three doc bugs** (§3.5) so the docs stop lying while we still have them.

### Phase 1 — Restructure onto the scverse template

Because the import name stays `scintilla`, this is far cheaper than a rename: the
package directory moves but no import statement changes.

Generate the template into a scratch directory and port the scaffolding across:

```bash
uvx --with prek cruft create https://github.com/scverse/cookiecutter-scverse
```

Then:

1. `git mv scintilla src/scintilla`. No intra-package import rewrites needed.
2. Rewrite `pyproject.toml` from the template: `build-backend = "hatchling.build"`,
   `name = "scintilla-py"`, `readme`, `license = "Apache-2.0"` +
   `license-files = ["LICENSE"]`, `authors`/`maintainers`, `classifiers`
   (`Intended Audience :: Science/Research`,
   `Topic :: Scientific/Engineering :: Bio-Informatics`,
   `License :: OSI Approved :: Apache Software License`,
   `Development Status :: 3 - Alpha`, the Python versions), `keywords`,
   `urls.Documentation`/`Homepage`/`Source`/`Issues`,
   `requires-python = ">=3.11"`, a dynamic version read from
   `src/scintilla/__init__.py`, lower bounds on every dependency
   (`scanpy>=1.10` is required by `annotation/marker_based.py:52`, plus
   `anndata>=0.10`, `numpy>=1.24`), and `[dependency-groups]` for `dev`/`test`/`doc`.
   Add `[tool.hatch.build.targets.wheel] packages = ["src/scintilla"]`.
3. Move `pingouin`, `plotly`, `seaborn` and `scikit-posthocs` out of the mandatory
   dependencies into extras (`stats`, `viz`), or replace the narrow `pingouin` usage
   with `scipy.stats`/`statsmodels`. This also removes GPL-3.0 from the default install.
4. Note in the README that the PyPI distribution is `scintilla-py` while the import is
   `scintilla`, since the unrelated `scintilla` distribution still exists on PyPI. Both
   installing the same top-level module is the residual risk you accepted; it is low
   (that package is abandoned at 0.0.3) but should be stated rather than discovered.
5. Copy in from the template: `.github/workflows/{test,build,release,autofix}.yaml`,
   `.github/ISSUE_TEMPLATE/`, `.pre-commit-config.yaml`, `.readthedocs.yaml`,
   `.codecov.yaml`, `.editorconfig`, `CHANGELOG.md`, `docs/`, `tests/`, and the
   `[tool.ruff]` block (line-length 120, `lint.pydocstyle.convention = "numpy"`) that
   justifies the 183 existing `# noqa` markers, plus `[tool.coverage]`.
6. Add `src/scintilla/py.typed`, `.cruft.json` (so `cruft update` can track the
   template), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and `CITATION.cff`.
7. Rename the default branch `master` → `main`.
8. Add `results/`, `*.h5ad`, `*.png` to `.gitignore`; the docs examples write to
   `results/` and nothing ignores it.

### Phase 2 — AnnData-first public API

Per your decision: public `pp`/`tl`/`pl` functions take AnnData; the array-based
statistics stay as a documented internal layer (`scintilla.stats`, importable and
tested, but not the headline API).

For every public function in `preprocessing`, `feature_selection`,
`dimensionality_reduction`, `clustering`, `batch_correction`,
`differential_expression`, `annotation`, `classification`, `evaluation`:

- **Signature:** `def f(adata: AnnData, *, key_added: str = "...", copy: bool = False,
  random_state: int | None = 0, ...) -> AnnData | None`. First argument positional and
  named `adata`; everything else keyword-only. Keep `ensure_anndata` internally for
  convenience, but the documented contract is AnnData.
- **Side effects:** write into `.obs`/`.var`/`.obsm`/`.uns`/`.layers`, honour `copy`,
  return `None` in the in-place case. Namespace everything under a single
  `adata.uns["scintilla"]` dict, and record the parameters that produced each result
  there so runs are self-describing.
- **Transformations write a layer**, not `.X` in place, so raw counts survive.
- **Sparsity:** replace all 45 densification sites. Where an algorithm genuinely needs
  dense input (LDA/QDA, Box's M, hierarchical linkage), densify locally on a subsample
  or on a PCA representation and say so in the docstring; otherwise guard with
  `scipy.sparse.issparse`. Fix `eda/summary.py` first: compute sparsity from `.nnz`.
- **Plotting:** fold `visualisation/` and `classification/visualise.py` into
  `scintilla.pl`. Compute functions return data; plotting functions take `adata` plus
  `show`/`save`/`ax` in the scanpy style. `unsupervised_analysis` stops returning a
  `fig` in its result dict.
- **Return shapes:** collapse the five current conventions. Drop the argument-dependent
  union returns in `spectral_clustering`, `hierarchical_clustering` and `pseudobulk_de`
  into separate functions or a stable dataclass.
- **Logging:** a module logger plus `scintilla.settings.verbosity`, replacing the 86
  `print()` calls in library code. `print` stays only in `cli/`.
- **Naming:** flatten `__init__.py`'s 72-name bag into `pp`/`tl`/`pl`/`stats`
  namespaces, and export the currently-unreachable `eda`, `evaluation` and the seven
  hidden `visualisation` modules.
- **Docstrings:** bring the 61 bare one-liners up to numpydoc; ruff `D` rules enforce
  it. Fix `name:` → `name : type` and the free-text `Returns` blocks so numpydoc
  renders. Annotate the `config` parameter (currently unannotated everywhere) with a
  `TYPE_CHECKING` import to dodge the circular import.

Ship this as `0.2.0` with the breaking changes in `CHANGELOG.md`. With no users yet you
can break freely, which is a luxury worth spending now.

### Phase 3 — Tests

Requirement 4 wants coverage of essential functions across a reasonable range of
inputs, and the Spectra review shows reviewers want correctness, not smoke tests.

- `tests/conftest.py`: a small synthetic AnnData (≈300 cells × 200 genes, known cluster
  structure, `batch` and `cell_type` columns) in both dense and CSR variants, plus
  `scanpy.datasets.pbmc68k_reduced()` for an integration test.
- One module per subpackage. For each public function assert: expected keys land in
  expected slots; `copy=True` leaves the input untouched; `copy=False` returns `None`;
  `key_added` is honoured; the same `random_state` gives the same output; dense and
  sparse inputs agree.
- **Correctness anchors** on the parts that are actually novel: effect sizes
  (`cohens_d`, `hedges_g`, `cliffs_delta`) against hand-computed values; bootstrap CIs
  against analytic intervals on normal data; multiple-testing correction against
  `statsmodels`; `gavish_donoho_threshold` and `marchenko_pastur_cutoff` against the
  published formulae on a synthetic low-rank-plus-noise matrix; clustering metrics
  against scikit-learn references on known labels.
- A regression test per Phase 0 bug: gene-name alignment in `transfer_labels`; the
  pseudobulk arm refusing a degenerate design; `import scintilla` leaving
  `matplotlib.get_backend()` untouched; a deliberately failing method appearing as
  failed rather than absent.
- An import test over every symbol in `__init__.py` (a single typo there breaks
  `import scintilla` for everyone).
- `@pytest.mark.slow` on the benchmark sweeps, excluded by default, run nightly.
  `pytest.importorskip` for optional dependencies so the minimal install passes.
- Target ≥70% line coverage to codecov. Below ~50% expect a review comment.

### Phase 4 — Documentation site

- `docs/` from the template: `conf.py`, `index.md`, `api.md`, `contributing.md`,
  `changelog.md`, `references.bib`, `notebooks/`.
- `docs/api.md` with `autosummary` over every public symbol, grouped
  `pp`/`tl`/`pl`/`stats`/`benchmark`. This is what satisfies requirement 6 properly.
- Migrate `README.md` + `GUIDE.md` into narrative docs pages; cut the README to a badge
  row (PyPI, tests, docs, codecov, DOI), a paragraph, install, a ten-line quickstart
  using `import scintilla as si`, and links.
- Replace `demo_notebook.ipynb` with two or three genuinely runnable `myst-nb`
  notebooks executed at build time on `sc.datasets.pbmc3k_processed()`:
  end-to-end pipeline, normalisation and clustering benchmarking with the statistics,
  and the CLI. Executed notebooks double as integration tests.
- Enable readthedocs, autofix.ci and codecov. Add
  `inventory: https://scintilla-py.readthedocs.io/en/latest/objects.inv` to the
  registry entry so other scverse docs can cross-link.

### Phase 5 — Release

1. Dynamic version from a single source; write the `0.2.0` changelog entry.
2. Tag and cut a **GitHub release**. Reviewers check for this specifically, not just PyPI.
3. Publish to PyPI via the template's `release.yaml` using **trusted publishing**
   (configure the publisher on PyPI first; no API token in secrets).
4. Optional but valuable: a conda-forge feedstock via `conda-forge/staged-recipes`,
   after PyPI.
5. Archive on Zenodo for a citable DOI, and fill in `CITATION.cff`.

### Phase 6 — The scverse submission

1. **Talk to the community first.** Post in the scverse Zulip or on
   `discourse.scverse.org` describing the package and asking whether the niche makes
   sense. Cheap, and it turns a cold PR into an expected one. It is also where you
   would learn early if a maintainer thinks it overlaps too much with scanpy.
2. **Fork `scverse/ecosystem-packages`** and add `packages/scintilla-py/meta.yaml`:

```yaml
name: scintilla-py
blurb: Benchmarking-driven method selection for scRNA-seq pipelines
description: |
  scINTILLA benchmarks and selects methods at every stage of a single-cell RNA-seq
  analysis: normalisation, feature selection, dimensionality reduction, clustering,
  classification, differential expression, annotation and batch correction. Method
  comparisons are backed by bootstrap confidence intervals, effect sizes, rank
  aggregation and permutation tests, and results are written back to the AnnData
  object.
project_home: https://github.com/soumickmj/scINTILLA
documentation_home: https://scintilla-py.readthedocs.io/
tutorials_home: https://scintilla-py.readthedocs.io/en/latest/notebooks/
install:
  pypi: scintilla-py
primary_category: scRNA-seq
tags:
  - benchmarking
  - pipeline
  - clustering
  - dimensionality reduction
  - differential expression
  - cell-type annotation
  - data integration
  - preprocessing
license: Apache-2.0
language: Python
contact:
  - soumickmj
inventory: https://scintilla-py.readthedocs.io/en/latest/objects.inv
test_command: pip install ".[test]" && pytest -m "not slow"
category: ecosystem
```

   Validate against
   [`schema.json`](https://github.com/scverse/ecosystem-packages/blob/main/schema.json)
   first. Take `primary_category` and `tags` only from the controlled vocabulary
   (primary categories: Data structures, scRNA-seq, bulk RNA-seq, Spatial, Epigenomics,
   Proteomics, Adaptive immune cell receptor, Multimodal, Imaging, Infrastructure).
   Do not set `version`; it is auto-populated daily from PyPI.
3. **Add `packages/scintilla-py/logo.svg`.** Optional, but every listed package has one.
4. **Open the PR**, copying the mandatory checklist from the registry README into the
   description and ticking each item with evidence: link the CI run, the docs site, the
   GitHub release, the coverage badge. Items 8 and 9 are personal attestations.
5. **Answer review comments promptly.** The failure mode in that repo is not rejection,
   it is submitter silence. Budget a couple of weeks of responsiveness.
6. After merge: request an invitation to the scverse GitHub organisation, and ask for a
   forum tag so users have somewhere to raise questions.

### Phase 7 — A paper (strongly advised)

`decoupler`, `scib` and most listed packages carry a `publications:` DOI.

- **JOSS** is the efficient route: its review criteria are almost exactly the scverse
  checklist, so Phases 1 to 5 do double duty, and it yields a citable DOI quickly.
- **A bioRxiv preprint plus a methods journal** (Bioinformatics Advances, GigaScience)
  if you want to present the benchmarking methodology and results on public datasets as
  a contribution in its own right. Given what is already in `statistical_tests/`, there
  is a genuine paper here, not just a software note.

---

## Part 6: Files that change

| Area | Paths |
|---|---|
| Correctness (Phase 0) | `annotation/label_transfer.py:49-59`, `differential_expression/benchmark.py:77-81`, the 14 `matplotlib.use` sites, the 16 silent handlers, `classification/run.py:152-154`, `feature_selection/benchmark.py:55-56`, `config.py:3,24-46` |
| Deletions | `scintilla/scripts/` (7 files), dead constants in `config.py`, `demo_notebook.ipynb` |
| Package move | `scintilla/**` → `src/scintilla/**` (import name unchanged) |
| Packaging | `pyproject.toml` (full rewrite from the template) |
| API conventions | every public function; representative: `clustering/leiden.py`, `clustering/run.py`, `classification/run.py`, `preprocessing/transformations.py`, `feature_selection/hvg.py`, `dimensionality_reduction/umap.py`, `batch_correction/harmony.py`, `differential_expression/wilcoxon.py`, `annotation/marker_based.py` |
| Sparsity | the 45 `toarray()` sites; `eda/summary.py:17,37,62` first |
| Plotting split | `visualisation/**`, `classification/visualise.py` → `src/scintilla/pl/**` |
| New | `tests/**`, `docs/**`, `.github/**`, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `.codecov.yaml`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `py.typed`, `.cruft.json` |
| Docs | `README.md` (shrink), `GUIDE.md` → `docs/`, new `docs/notebooks/*.ipynb` |
| Registry (separate repo) | fork of `scverse/ecosystem-packages`: `packages/scintilla-py/{meta.yaml,logo.svg}` |

---

## Part 7: Verification

**Phase 0** — each bug gets a failing reproduction first, then the fix:
```bash
python -c "
import matplotlib; b = matplotlib.get_backend()
import scintilla
assert matplotlib.get_backend() == b, 'import still hijacks the backend'
print('backend preserved:', b)"
```
plus a shuffled-`var_names` reference/query pair for `transfer_labels` (labels must be
unchanged by shuffling, and must error on zero overlap), and a benchmark run with a
deliberately broken method (it must appear as failed in the results, not vanish).

**Phase 1**
```bash
uv pip install -e ".[dev,test]"
python -c "import scintilla; print(scintilla.__version__)"
python -c "import scintilla, importlib, pkgutil; [importlib.import_module(m.name) for m in pkgutil.walk_packages(scintilla.__path__, 'scintilla.')]"
python -c "import setuptools" && python -m build && python -c "
import zipfile,glob; w=glob.glob('dist/*.whl')[0]
print([n for n in zipfile.ZipFile(w).namelist() if n.endswith('py.typed') or 'scripts/' in n])"
scintilla --help
prek run -a
```

**Phase 2** — confirm the contract by hand before the tests exist:
```python
import scanpy as sc, scintilla as si
adata = sc.datasets.pbmc68k_reduced()
assert si.tl.leiden_clustering(adata, key_added="my_clusters") is None
assert "my_clusters" in adata.obs
assert si.tl.leiden_clustering(adata, copy=True) is not adata
```
and check sparsity survives: `X` stays CSR through the pipeline, and
`si.eda.dataset_summary` on a 100k-cell CSR matrix stays within a few hundred MB.

**Phase 3**
```bash
pytest -m "not slow" --cov=scintilla --cov-report=term-missing
```
≥70% coverage, green across the full `hatch-test` matrix (lowest and highest supported
Python, plus a pre-release-dependency run).

**Phase 4**
```bash
hatch run docs:build     # sphinx-build -W, warnings are errors
```
then confirm the readthedocs build is green and `objects.inv` is served.

**Phase 5** — install the published artefact into a clean environment and run the
notebooks against it:
```bash
uv venv /tmp/v && uv pip install --python /tmp/v scintilla-py
/tmp/v/bin/python -c "import scintilla; print(scintilla.__version__)"
```

**Phase 6** — run the exact `test_command` from `meta.yaml` in a clean checkout, since
the registry may run it; validate the YAML against `schema.json`.

---