# scINTILLA → scverse: audit, verdict, and route to listing (revision 2)

## Context

`soumickmj/scINTILLA` is a pure-Python package (122 modules, 15 subpackages, 13,738
lines at the time of this revision) that wraps scanpy and scikit-learn to score the
quality of cell-type labels and to benchmark methods across a whole scRNA-seq analysis:
normalisation, feature selection, dimensionality reduction, clustering, supervised
classification, differential expression, annotation and batch correction. The question
is whether it belongs in scverse, and if so in which tier. The answer is the
**Ecosystem** tier (Route A below).

This is **revision 2** of the plan. Revision 1 was written against commit `bf50df2`.
Revision 2 was re-verified against `a6ee153` (the head of `master` when the `scverse`
branch was cut). Every status below was checked directly in the working tree, in the
PyPI JSON API, and in the `scverse/ecosystem-packages` registry.

Status legend used throughout: **Done** (verified fixed in the tree), **Partly**,
**Open**, **Changed** (the plan item itself needed rewriting), **Dropped**.

**Decisions taken (yours):**

- Distribution name **`scintilla-py`**, import name stays **`scintilla`**. (2026-09-11)
- Public API goes **AnnData-first**; the array-based statistics layer stays as a
  documented internal layer. (2026-09-11)
- **Correctness bugs first**, before any packaging or restructuring work. (2026-09-11)
  Outcome: finished on `master`, see Phase 0.
- **Full Phase 2 rework in this bump**, not the additive subset. (2026-10-08)
- **`requires-python >= 3.11`**. (2026-10-08)
- **Authors taken from the bioRxiv preprint**, used in `pyproject.toml`,
  `CITATION.cff` and the registry contact. (2026-10-08)
- Work happens on a new branch **`scverse`** cut from `master`; `master` is not touched
  and only `scverse` is pushed. (2026-10-08)

**Constraint added in revision 2: preprint reproducibility.** The package has a
bioRxiv preprint (doi:10.64898/2026.07.27.740477) and a frozen set of label-quality
scorers (`classification/label_quality_variants.py`, "computed exactly as in the V4
benchmark"). The API rework must not change any number those scorers produce, and the
default names of the `obs` columns the pipeline writes (`scintilla_top{k}_confusion`,
`scintilla_top{k}_fragmentation`, `pred_agreement`, `pred_entropy`, ...) stay the
defaults. They become configurable through `key_added`, not renamed. A numerical
regression test pins this (Phase 3).

---

## What changed since revision 1

| Event | Consequence for the plan |
|---|---|
| Phase 0 (all eight correctness items) landed on `master` | Part 3.2 is now a record, not a to-do list. One residual item remains (§3.2d). |
| `scintilla-py` 0.1.0 published to PyPI on 2026-10-08 (wheel and sdist) | Requirement 3 is met. Phase 5 changes from "first release" to "second release", and trusted publishing replaces the API-token route in `PUBLISHING.md`. |
| `uv.lock`, `PUBLISHING.md`, `MANIFEST.in` added; build still uses setuptools | Phase 1 still moves to hatchling and the `src/` layout, but the lock file now exists and must be regenerated. |
| A `label-quality` pipeline and CLI subcommand was added (`classification/label_quality_variants.py`, `cli/commands/label_quality.py`, `LABEL_QUALITY_VARIANTS.md`) | The package's centre of gravity is now cell-type label quality, not only method benchmarking. The registry blurb, description, tags and keywords change accordingly. This is also the thing that distinguishes it from scanpy. |
| bioRxiv preprint exists, 28 July 2026 | `publications:` goes in `meta.yaml`, `CITATION.cff` cites it, and Phase 7 changes: the preprint is done, so the remaining question is the journal or JOSS route. |
| The registry README now words requirement 6 as "API documentation is provided via a website **or README**" | A hosted docs site is no longer a hard requirement. It stays in the plan because the package has 100+ public functions and an `objects.inv` lets other scverse docs link to it, but it is now a quality goal, not a gate. |
| Test suite grew from 0 to 138 tests (12 files, 2,094 lines) | Requirement 4 is Partly met. Phase 3 shrinks from "write tests" to "organise, extend and measure". |
| `scverse.org` is still blocked by the session egress proxy | Registry facts still come from the `scverse/ecosystem-packages` repository, not the website. |

---

## Part 1: What the scverse tiers are (re-checked)

`scverse.org` is blocked here, so the facts come from the registry that drives the
website, [`scverse/ecosystem-packages`](https://github.com/scverse/ecosystem-packages).
Re-read on 2026-10-08.

| `category` | Website section | Members |
|---|---|---|
| `core-datastructure` | Core packages | anndata, mudata, spatialdata |
| `core-framework` | Frameworks | scanpy, muon, squidpy, scvi-tools, scirpy, decoupler, ... |
| `ecosystem` | Ecosystem | ~130 packages (scib, liana, ...) |

**Valid.** Ecosystem listing is a pull request adding one `meta.yaml` file, reviewed
against a nine-item checklist. The nine items are unchanged except item 6 (above).
Sample entries re-read (`liana`, `scib`, `decoupler`) confirm these `meta.yaml` fields:
`name`, optional `blurb`, `description`, `project_home`, `documentation_home`,
`tutorials_home`, `publications` (list of DOIs), `install.pypi`, `primary_category`,
`tags`, `license`, `language`, `contact` (GitHub handles), optional `logo`,
`test_command`, `category`, optional `inventory`. `version` is auto-populated and must
not be set.

**Open check.** The old path of `schema.json` returns 404 on `main`, so the vocabulary
for `primary_category` and `tags` could only be confirmed from the README (primary
categories) and from existing entries (`benchmarking`, `data integration` and
`functional analysis` are in use). Validate `meta.yaml` against the schema CI in the
registry repository before opening the pull request, and add any missing tag to the
enum in the same PR, as the registry README allows.

---

## Part 3: Audit

### 3.1 Gap against the nine mandatory requirements

| # | Requirement | Revision 1 | Now |
|---|---|---|---|
| 1 | OSI-approved licence | Pass, metadata missing | **Done.** Apache-2.0 in `LICENSE`, `license` and `license-files` in `pyproject.toml`, PyPI shows `Apache-2.0`. |
| 2 | Versioned releases | Fail | **Partly.** PyPI 0.1.0 exists. There are still no git tags and no GitHub release, which reviewers check separately. |
| 3 | Installable from a standard registry | Fail | **Done.** `pip install scintilla-py`. conda-forge is optional. |
| 4 | Automated tests over a reasonable range of inputs | Fail | **Partly.** 138 tests pass in about 15 s. They are regression-driven (one file per fixed bug) and not organised per subpackage; coverage has never been measured; no dense/sparse parametrisation exists. |
| 5 | CI that runs those tests | Fail | **Open.** There is still no `.github/` directory. |
| 6 | API documentation | Partial | **Partly, and now sufficient in letter.** README, GUIDE and `LABEL_QUALITY_VARIANTS.md` are thorough prose, but there is no generated API reference. |
| 7 | Uses scverse data structures where appropriate | Partial | **Partly.** See §3.3; this remains the substantive gap and is Phase 2. |
| 8 | Author agrees to listing | Yours | Yours to tick. |
| 9 | Agrees to the code of conduct | Yours | Yours to tick. |

### 3.2 Correctness bugs (Phase 0): verified status

| Item | Revision 1 finding | Status at `a6ee153` |
|---|---|---|
| (a) `transfer_labels` matched genes by position | wrong labels, silently | **Done.** `annotation/label_transfer.py` intersects `var_names`, raises on empty or under 10 shared genes, no `np.pad` fallback. `tests/test_label_transfer.py`. |
| (b) pseudobulk with n=1 per group | fabricated `groups + "_s1"` | **Done.** `_run_pseudobulk` raises when no `sample` column exists. `tests/test_de_benchmark.py`. |
| (c) 14 `matplotlib.use("Agg")` calls | import hijacks backend | **Done.** Zero calls remain; `plt.close(fig)` on returned figures removed. `tests/test_matplotlib_isolation.py`. |
| (d) 16 silent `except Exception: pass` | dropped methods invisible | **Done, one residual.** 15 of 16 now warn and report failures (`tests/test_visible_benchmark_failures.py`). The sixteenth, `benchmarking/time_estimator.py:_timed_call`, still swallows by design ("failures are handled upstream"); Phase 2 makes it log at debug level. 45 broad handlers remain in total, all reviewed or logged. |
| (e) inert `AnalysisConfig` fields, inverted guards | 10 of 24 fields unread | **Done.** `include_shap` and `n_features` now use `None` sentinels; every field is read by at least one pipeline (`tests/test_analysis_config_seeding.py`, 823 lines). |
| (f) global seed override did not work | README advised a no-op | **Done.** `random_state` is threaded through the public stochastic functions and the README now documents per-call seeding. The constant `RANDOM_SEED` remains as a default value only (no bare `42` in `benchmarking/`). |
| (g) `scintilla/scripts/` shipped in no wheel | 7 dead files | **Done.** Directory deleted. |

### 3.3 API conventions: the substantive rework (still open)

Measured at `a6ee153`:

- **Sparse support: Open.** 45 unconditional `toarray()` sites across 35 modules;
  `scipy.sparse.issparse` is used nowhere. `eda/summary.py` still densifies the whole
  matrix to compute `sparsity = mean(X == 0)`.
- **Signatures: Open.** 19 functions take `adata: ad.AnnData`, 8 take
  `adata: Union[...]`, 27 take `data: Union[pd.DataFrame, ad.AnnData]` and 32 take
  `data: Union[...]`, including both flagship entry points.
- **`copy` / `inplace` / `key_added`: Open.** No `copy` parameter exists; `key_added`
  occurs in two internal pass-throughs. Aliasing remains inconsistent between
  functions (`run_pca` mutates, `run_umap` copies, `select_hvg` does both).
- **Output keys: Open.** `unsupervised_analysis` writes `scintilla_top{k}_<method>|<params>`,
  `scintilla_top{k}_confusion`, `scintilla_top{k}_fragmentation` and `scintilla_cluster`;
  `supervised_analysis` writes `pred_*` columns. These are now load-bearing for the
  label-quality pipeline, hence the preprint-reproducibility constraint above.
- **`.layers`, `.uns` provenance: Open.** Transformations still overwrite `.X`.
- **Seeding: Done** for defaults (see §3.2f); sweepability through `random_state` is
  covered by the Phase 0 tests.
- **Logging: Open.** 63 `print()` calls remain, 33 of them in library modules
  (`time_estimator.py` 13, `classification/run.py` 11, `classification/benchmark.py` 6,
  `preprocessing/benchmark.py` 2, `clustering/benchmark.py` 1). There is no
  `scintilla.settings`.
- **Dataset-specific residue: Done.** `CELL_TYPE_WEIGHTS`, `CLASSIFIER_SUITE`,
  `DE_METHODS`, `BATCH_CORRECTION_WEIGHTS` are deleted; the default label column is
  `cell_type` everywhere (`tests/test_cell_type_defaults.py`).
- **MuData: Open.** `load_h5mu` exists, `auto_detect_format` is still annotated
  `-> ad.AnnData`, and `ensure_anndata` still raises `TypeError` for MuData. Decision for
  this bump: MuData input is accepted by selecting one modality (`modality="rna"`),
  with a clear error when the choice is ambiguous. SpatialData stays out of scope.
- **README quickstart alias: Partly.** `import scintilla as si` is used in the new
  sections, but `import scintilla as sc` still appears in `README.md` (2 sites) and
  `GUIDE.md` (1 site).

### 3.4 Packaging: still open apart from the metadata PyPI needed

Done since revision 1: `readme`, `license`, `license-files`, `keywords`, minimal
`classifiers`, `[project.urls]`, `dependency-groups.dev`, `uv.lock`, `MANIFEST.in`.

Still open: `authors` and `maintainers`; full classifiers (Python versions,
`Development Status`, `License`); dynamic version (it is still duplicated in
`pyproject.toml` and `scintilla/__init__.py`); lower bounds on every dependency
(`scanpy>=1.10` is needed by `annotation/marker_based.py`, which passes `ctrl_as_ref`);
`requires-python` is still `>=3.9` in the published 0.1.0 wheel; no `py.typed`; no
committed `[tool.ruff]` block while 178 `# noqa: PLC0415` markers remain; build
backend is still setuptools with a flat layout.

**Licence caveat: resolved by removal, not by moving.** Revision 1 proposed moving
`pingouin` (GPL-3.0) to an extra. Re-reading the code shows it is used exactly once, in
`benchmarking/reproducibility.py:_compute_icc`, wrapped in `try/except` with an existing
closed-form fallback. ICC(1) is a one-way random-effects ANOVA ratio, so it is
re-implemented in about ten lines of NumPy and cross-checked against `pingouin` in a
test that is skipped when `pingouin` is absent. `pingouin` leaves the dependency list
altogether. `scikit-posthocs` (MIT) is used once in `statistical_tests/dunns.py`,
`plotly` once in `visualisation/sankey_plots.py` and `seaborn` in three plotting
modules; they move to the `stats` and `viz` extras. `mrmr-selection` and `leidenalg`
(GPL-3.0) already sit in the optional `full` extra.

### 3.5 Docs, notebook, community files

- Doc bugs: the `.[test]` extra (GUIDE) and the `DBSCAN_EPS_RANGE` count (README) are
  **Done**; the seed-override advice is **Done**.
- `demo_notebook.ipynb` is **Open and unchanged**: 10 cells, no data-loading cell, both
  code cells start from an undefined `adata_hvg`, no outputs. It is replaced in Phase 4.
- Absent and still **Open**: `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `CITATION.cff`, `SECURITY.md`, issue and PR templates, `.pre-commit-config.yaml`,
  `.readthedocs.yaml`, `.github/`. The preprint now exists, so `CITATION.cff` has
  something to cite.
- Clean on the usual leak vectors (unchanged). New leak to avoid: git author metadata
  carries an institutional email, so no email is written into any file.
- `scverse_plan.md` itself is tracked in git although `.gitignore` lists it. It is kept
  tracked on this branch so its history survives; remove it from the index before
  the registry submission if you would rather not publish the plan.

---

## Part 4: Route A, Ecosystem listing (the plan)

Phase 0 is finished. The `scverse` branch carries Phases 1 to 4 and the release
preparation of Phase 5, as release **0.2.0** (breaking changes are allowed by the
full-rework decision; there are no known downstream users of 0.1.0).

### Phase 0: Correctness bugs. **Done on `master`.**

See §3.2. Residual: make `_timed_call` log the swallowed exception at debug level.

### Phase 1: Restructure onto the scverse template. **Open**

The cookiecutter template itself cannot be generated offline and `cruft` needs the
template repository, so the scaffolding is ported by hand to match what the template
produces; `.cruft.json` is deliberately not added until the project has been generated
once against the real template (a hand-written `.cruft.json` would claim a template
commit that was never applied).

1. `git mv scintilla src/scintilla` (import name unchanged).
2. Rewrite `pyproject.toml`: `hatchling` backend with `hatch-vcs`-free single-source
   version read from `src/scintilla/__init__.py`; `requires-python = ">=3.11"`; authors
   and maintainers from the preprint; full classifiers; lower bounds on every
   dependency; extras `full`, `stats`, `viz`, `test`, `doc`; `[dependency-groups]`;
   `[tool.ruff]` (line length 120, numpy docstring convention) with the
   `PLC0415` suppression moved into configuration and the 178 inline markers removed;
   `[tool.pytest]`, `[tool.coverage]`.
3. Remove `pingouin` (see §3.4), move `scikit-posthocs`, `plotly`, `seaborn` to extras
   with clear `ImportError` messages.
4. Add `.github/workflows/{test,build,release}.yaml`, issue templates, PR template,
   `.pre-commit-config.yaml`, `.readthedocs.yaml`, `.codecov.yaml`, `.editorconfig`.
   The release workflow uses PyPI **trusted publishing** (no token in secrets).
5. Add `src/scintilla/py.typed`, `CHANGELOG.md`, `CONTRIBUTING.md`,
   `CODE_OF_CONDUCT.md` (Contributor Covenant), `SECURITY.md`, `CITATION.cff`.
6. Add `results/`, `*.h5ad`, `*.png` to `.gitignore`.
7. **Not done on this branch (by instruction):** renaming `master` to `main`.

### Phase 2: AnnData-first public API. **Open** (full rework)

Shared infrastructure, one private module `scintilla/_compat.py` plus
`scintilla/_logging.py`:

- `settings.verbosity` and a module logger hierarchy under `scintilla`; `print` stays only
  in `cli/`. Library functions keep their existing `verbose` argument as a shortcut that
  maps to the logger level, so scripts written against 0.1.0 still run.
- `_resolve(adata, copy)` returns the object to work on and whether to return it, so every
  function has the same `copy` semantics. `_get_matrix(adata, layer, dense=False)`
  is the single place where densification happens, with a documented reason when it
  does, and a sparse-preserving path otherwise.
- `_record(adata, key, params)` writes provenance to `adata.uns["scintilla"][key]`.

Per-function contract (applied to every public function in `preprocessing`,
`feature_selection`, `dimensionality_reduction`, `clustering`, `batch_correction`,
`differential_expression`, `annotation`, `classification`, `evaluation`, `eda`):

- First argument `adata: AnnData`, positional. Everything else keyword-only where it can
  be without breaking the recorded test fixtures. `DataFrame` and array input stay
  accepted through `ensure_anndata` (a convenience, no longer the documented contract).
- `copy: bool = False`, `key_added`, `random_state` where applicable. In-place calls
  return `None`; `copy=True` returns a new object. Functions that previously returned
  results tables keep returning them; they gain `key_added`/`uns` storage.
- Transformations write `.layers[key_added]` (default keeps the raw counts safe) instead
  of replacing `.X`; `inplace` replacement stays available as `layer=None`.
- Sparse inputs stay sparse unless the algorithm needs dense arrays (LDA/QDA, Box's M,
  hierarchical linkage), in which case densification is local and documented.
  `eda/summary.py` computes sparsity from `.nnz`.
- Namespaces: `scintilla.pp`, `scintilla.tl`, `scintilla.pl`, `scintilla.stats`,
  `scintilla.benchmark`, `scintilla.eda`, `scintilla.io`. The existing flat names in
  `scintilla/__init__.py` keep working through 0.2.x with a `DeprecationWarning`
  where the old and new behaviour differ.
- Plotting functions take `adata` (or a results object), `show`, `save` and `ax`, and
  return the figure or axes without closing it. `unsupervised_analysis` no longer
  returns a figure in its result dict.
- Return shapes: argument-dependent union returns in `spectral_clustering`,
  `hierarchical_clustering` and `pseudobulk_de` are split into separate functions.
- Docstrings are brought to numpydoc; `config` is annotated through a `TYPE_CHECKING`
  import.
- MuData: `ensure_anndata(data, modality=...)` accepts a `MuData` and selects one
  modality; `auto_detect_format` is annotated correctly.
- **Reproducibility guard:** default `key_added` values reproduce the existing `obs`
  column names and the label-quality outputs byte for byte (Phase 3 pins this).

### Phase 3: Tests. **Partly done, extend**

- Keep the 138 existing regression tests (they are the Phase 0 acceptance tests).
- Add `tests/conftest.py` with a synthetic AnnData (300 cells × 200 genes, known cluster
  structure, `batch` and `cell_type` columns) in dense and CSR variants.
- Add one test module per subpackage asserting the Phase 2 contract: expected keys land
  in expected slots; `copy=True` leaves the input untouched; `key_added` is honoured;
  same `random_state` gives the same output; dense and sparse inputs agree.
- Correctness anchors: effect sizes against hand-computed values; bootstrap intervals
  against analytic intervals; multiple-testing correction against `statsmodels`;
  Gavish-Donoho and Marchenko-Pastur against the published formulae; ICC(1) against
  `pingouin` when installed.
- Reproducibility pin: label-quality outputs on a fixed synthetic dataset compared with
  values recorded from the 0.1.0 code before the rework.
- An import test over every public symbol, and a test that `import scintilla` leaves the
  matplotlib backend untouched (already present).
- `@pytest.mark.slow` on benchmark sweeps; `pytest.importorskip` for optional
  dependencies. Target at least 70 % line coverage.

### Phase 4: Documentation site. **Open**

`docs/` with `conf.py`, `index.md`, `api.md` (autosummary grouped `pp`/`tl`/`pl`/`stats`/
`benchmark`), `contributing.md`, `changelog.md`, `references.bib`, `notebooks/`.
README is cut to badges, one paragraph, install, a ten-line quickstart and links;
`README.md` and `GUIDE.md` content moves into docs pages. `demo_notebook.ipynb` is
replaced by runnable `myst-nb` notebooks (the label-quality workflow on
`sc.datasets.pbmc3k_processed()`, plus method benchmarking). Docs build with warnings as
errors. readthedocs, codecov and autofix.ci are switched on by the repository owner
after merge (they need account access).

### Phase 5: Release. **Changed**

1. Single-source version, `0.2.0`, detailed `CHANGELOG.md` entry. **Done on this branch.**
2. Tag `v0.2.0` and cut a GitHub release after the branch is merged. Reviewers check
   for this specifically. **Yours** (needs repository rights).
3. Configure the trusted publisher on PyPI for `release.yaml`, then publish 0.2.0.
   **Yours.** `PUBLISHING.md` is rewritten around trusted publishing.
4. Optional: conda-forge feedstock after PyPI.
5. Zenodo archive for a citable software DOI, then add it to `CITATION.cff`.

### Phase 6: The scverse submission. **Open, unchanged apart from content**

1. Post on Zulip or discourse.scverse.org first, describing the label-quality angle.
2. Fork `scverse/ecosystem-packages` and add `packages/scintilla-py/meta.yaml`:

```yaml
name: scintilla-py
blurb: Per-label quality scores and method benchmarking for scRNA-seq annotations
description: |
  scINTILLA scores how learnable and internally consistent each cell-type label in a
  single-cell RNA-seq dataset is, by combining an unsupervised arm (a panel of
  clustering algorithms, neighbourhood confusion and fragmentation) with a supervised
  arm (up to twelve classifiers, prediction agreement, entropy and confidence).
  It also benchmarks methods for normalisation, feature selection, dimensionality
  reduction, differential expression, annotation and batch correction, with bootstrap
  confidence intervals, effect sizes, rank aggregation and permutation tests. Results
  are written back to the AnnData object.
project_home: https://github.com/soumickmj/scINTILLA
documentation_home: https://scintilla-py.readthedocs.io/
tutorials_home: https://scintilla-py.readthedocs.io/en/latest/notebooks/
publications:
  - 10.64898/2026.07.27.740477
install:
  pypi: scintilla-py
primary_category: scRNA-seq
tags:
  - benchmarking
  - clustering
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

   `pipeline`, `differential expression`, `cell-type annotation` and
   `dimensionality reduction` from revision 1 are dropped from the draft because they
   could not be confirmed as existing tags; add them to the registry enum in the same PR
   only if you want them. `publications` may need to be switched to the journal DOI once
   the preprint is published. Do not set `version`. Add `logo.svg` (field `logo: logo.svg`)
   if you have one.
3. Open the PR with the nine-item checklist copied in and each item evidenced (CI run,
   docs site, GitHub release, coverage badge). Items 8 and 9 are personal attestations.
4. Answer review comments promptly; the failure mode is submitter silence.
5. After merge, request an invitation to the scverse GitHub organisation.

### Phase 7: Paper. **Changed**

The preprint is posted. Remaining decision: submit it to a methods journal, and/or write
a short JOSS paper focused on the software (JOSS review criteria overlap almost
completely with the scverse checklist, so Phases 1 to 5 do double duty). Update
`publications` and `CITATION.cff` when either is accepted.

---

## Part 6: Files that change on the `scverse` branch

| Area | Paths |
|---|---|
| Package move | `scintilla/**` → `src/scintilla/**` |
| Packaging | `pyproject.toml` (rewrite), `uv.lock` (regenerated), `MANIFEST.in` (removed), `PUBLISHING.md` (rewritten) |
| Infrastructure | `src/scintilla/_compat.py`, `src/scintilla/_logging.py`, `src/scintilla/settings.py` |
| Namespaces | `src/scintilla/{pp,tl,pl,stats,benchmark}/__init__.py` |
| API conventions | every public function listed under Phase 2; representative: `clustering/run.py`, `classification/run.py`, `preprocessing/transformations.py`, `feature_selection/hvg.py`, `dimensionality_reduction/umap.py`, `batch_correction/*.py`, `differential_expression/*.py`, `annotation/*.py`, `eda/summary.py` |
| Licence | `benchmarking/reproducibility.py` (ICC), dependency list |
| New | `tests/**`, `docs/**`, `.github/**`, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `.codecov.yaml`, `.editorconfig`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `py.typed` |
| Docs | `README.md` (shrink), `GUIDE.md` → `docs/`, `demo_notebook.ipynb` → `docs/notebooks/` |
| Registry (separate repo, yours) | fork of `scverse/ecosystem-packages`: `packages/scintilla-py/{meta.yaml,logo.svg}` |

---

## Part 7: Verification

**Phase 1**
```bash
uv sync --extra full --extra test
python -c "import scintilla; print(scintilla.__version__)"
python -c "import scintilla, importlib, pkgutil; [importlib.import_module(m.name) for m in pkgutil.walk_packages(scintilla.__path__, 'scintilla.')]"
uv build && python -c "
import zipfile,glob; w=glob.glob('dist/*.whl')[0]
print([n for n in zipfile.ZipFile(w).namelist() if n.endswith('py.typed')])"
twine check --strict dist/*
scintilla --help
ruff check . && ruff format --check .
```

**Phase 2** (contract, by hand before the tests exist)
```python
import scanpy as sc, scintilla as si
adata = sc.datasets.pbmc68k_reduced()
assert si.tl.leiden(adata, key_added="my_clusters") is None
assert "my_clusters" in adata.obs
assert si.tl.leiden(adata, copy=True) is not adata
```
and `si.eda.dataset_summary` on a 100k-cell CSR matrix stays within a few hundred MB.

**Phase 3**
```bash
pytest -m "not slow" --cov=scintilla --cov-report=term-missing
```

**Phase 4**
```bash
sphinx-build -W -b html docs docs/_build
```

**Phase 5** (after you publish) install the published artefact into a clean environment
and run the notebooks against it.

**Phase 6** run the exact `test_command` from `meta.yaml` in a clean checkout and
validate the YAML with the registry's schema CI.
