# Changelog

All notable changes to `scintilla-py` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/) (with the usual caveat that 0.x minor releases may break the API).

## [Unreleased]

## [0.2.1] - 2026-10-09

A follow-up to 0.2.0 that finishes the scverse listing work (author metadata, registry entry, dependency
hygiene), fixes problems that the first run of the GitHub workflows turned up, and records a systematic
comparison of 0.2.x against 0.1.0 (see [Equivalence with 0.1.0](#equivalence-with-010)).
**One change affects downstream code:** two CSV column names changed (see *Changed*). No numerical
result of any analysis function changed relative to 0.2.0, except the embedding benchmark described under
*Fixed*, which now agrees with 0.1.0 again.

### Added

- **`scverse-registry/`**: the `meta.yaml` for the entry in
  [`scverse/ecosystem-packages`](https://github.com/scverse/ecosystem-packages), the upstream
  `schema.json` it is validated against, and a README that says how to submit it. `tests/test_registry_entry.py`
  validates the entry against that schema on every test run and checks that it agrees with `pyproject.toml`
  (name, licence, PyPI name, URLs) and with the DOI in `CITATION.cff`, so the entry cannot drift out of date.
- **`.github/dependabot.yml`**: weekly, grouped updates for the `uv` lock file and for the GitHub Actions
  used by the workflows. Semver-major updates are left for a person to decide.
- **`settings.dense_warning_gb`** (default 4.0): a `UserWarning` is issued before a sparse matrix is converted to
  a dense array larger than this many GiB. The message names the size, the reason (for instance
  "classifier training") and the usual remedy (`use_rep="X_pca"` or `supervised_use_rep="X_pca"`). Raise
  the threshold, or set it to `float("inf")`, to silence it on a machine with plenty of memory.
- The release workflow refuses a tag that is not `v` followed by the package version (a release tagged `v1`
  would have published 0.2.1). An empty `.nojekyll` file keeps GitHub Pages from running Jekyll on the
  repository, which is not used for the documentation (Read the Docs builds it).
- The workflows can now be started by hand (`workflow_dispatch`), which is how a branch can be checked
  before a pull request is opened.

### Changed

- **Authors** are now given by full name in `pyproject.toml`, `CITATION.cff` (including `preferred-citation`)
  and the documentation, in the order of the bioRxiv preprint: Sina Kanannejad, Noemi Bongiorni, Elisa
  Nordera, Sara Redaelli, Irene Rusconi, Rachele Zanin, Alice Giustacchini and Soumick Chatterjee.
- **Breaking for anything that reads the label-quality CSV files by column name.** The two silhouette
  variants had a doubled underscore (`scintilla_composite__silhouette`,
  `scintilla_composite_frag__silhouette`). The names are now `scintilla_composite_silhouette` and
  `scintilla_composite_frag_silhouette`, in line with every other `scintilla_*` column. The values are
  unchanged (the pinned baseline in `tests/data/label_quality_baseline.csv` was renamed, not recomputed).
  Rename the columns in old result files with
  `df.rename(columns=lambda c: c.replace("__silhouette", "_silhouette"))`.
- `get_matrix` (the single place where sparse data is made dense) now converts the dtype while the matrix is
  still sparse and allocates one dense copy instead of two.
- Test dependencies have lower bounds (`pytest-cov>=4`, `jsonschema>=4.18`); without them the
  `lowest-direct` CI job resolved `jsonschema` to version 0.2, which is Python 2 era code and cannot be built.
- **Read the Docs build.** The docs environment is installed without `louvain`
  (`uv sync ... --no-install-package louvain`). Read the Docs has no wheel of `louvain` 0.8.2 for its
  Python 3.12 image, so uv compiled it from source, which needs CMake, and the build failed. The tutorials do
  not use Louvain. The GitHub `Docs` job uses the same command so that it catches this class of failure.
- The GitHub Actions used by the workflows moved to their current major versions (`actions/checkout` 7,
  `astral-sh/setup-uv` 7, `codecov/codecov-action` 7), which also removes the Node.js 20 deprecation warning.
- Dependabot now opens at most one pull request a month for each ecosystem, instead of weekly. It no longer
  tries to move `leidenalg` and `python-igraph`: `louvain` 0.8.2, the latest release, needs
  `python-igraph<0.12`, whereas `leidenalg` 0.12 needs `python-igraph>=1`, so the lock file cannot hold the
  newest of both and Dependabot's `uv` job failed on every run.
- The documentation address is now `https://scintilla.readthedocs.io/`, matching the import name and the
  repository (the PyPI distribution stays `scintilla-py`). The README badge, the project URLs in
  `pyproject.toml` and the registry entry were updated. The 0.2.0 release on PyPI still links to the
  old `scintilla-py.readthedocs.io` address, so that address should redirect permanently to the new one.
- The documentation no longer cross-links `mudata` through intersphinx: its inventory moved and neither the old
  nor the new address served an `objects.inv`, which failed the strict docs build. `MuData` now renders as plain text.
- `scverse_plan.md`, the working document for this listing, was removed from the repository, together
  with its entry in `.gitignore`.
- **Documentation.**
  - The README now covers installation (pip and uv, with the optional methods listed), a quick start for
    the label-quality score and for the step-by-step analysis, the presets, the namespaces and the
    command line, so that it can be read on its own. The long-form reference stays in the documentation.
  - The tutorial *Choosing methods* gained the explanation of the clustering and classifier benchmarks
    that previously lived only in the 0.1 demonstration notebook, together with a classifier benchmark
    run on the example dataset. The manual gained a section on memory and large datasets.
  - The label-quality page no longer refers to a Git checkout of `master`, to the first PyPI release, or to
    paths on the authors' computing cluster.
  - Headings are in sentence case, dashes in running text were replaced by ordinary punctuation, and two
    links that pointed at files on `master` now point at the matching documentation pages.

### Removed

- `CODE_OF_CONDUCT.md`. The scverse listing asks the author to agree to scverse's own code of conduct,
  which is a tick-box in the registry pull request and not a file in the repository, and GitHub's
  community checklist treats a project-level file as optional. It can be added back at any time.
- `PUBLISHING.md`. The release steps now form the "Releasing" section of `CONTRIBUTING.md`, which is also
  part of the documentation, and the note about a `setuptools<82` pin that it still carried was dropped.
  Both files remain in the Git history.

### Fixed

- **Louvain with setuptools 82 or later.** The `louvain` package (0.8.2, the latest) imports `pkg_resources`,
  which setuptools 82 removed, so `scintilla.tl.louvain` failed with `ModuleNotFoundError` on a current
  Python environment (it also failed that way in 0.1.0). 0.2.0 worked around this with a
  `setuptools<82` pin in the `full` extra. That pin made the lock file carry setuptools 81, which has a
  published advisory (GHSA-h35f-9h28-mq5c), and the Dependabot proposal to move it to `<84` would have installed
  a setuptools without `pkg_resources` and broken Louvain again. `louvain` is now imported through a
  small stand-in for the two `pkg_resources` names it uses (`get_distribution` and `DistributionNotFound`),
  which exists only while the import runs and is removed afterwards, so nothing else in the process
  sees a fake `pkg_resources`. The pin is gone, and a subprocess test checks both the import and the clean-up.
  The stale Dependabot branch for the `<84` bump should be closed.
- **`benchmark_embeddings` embedded a float64 copy of the representation** (0.2.0 only), whereas 0.1.0
  embedded it as stored and used float64 only to compute trustworthiness. For a float32 `X_pca` this
  changed the UMAP and t-SNE coordinates in the last digits and the scores by up to 0.003. It now
  embeds the stored representation again and agrees with 0.1.0 exactly; `tests/test_contract_embeddings_clustering.py`
  has a regression test.
- The pinned label-quality baseline test used a relative tolerance of 1e-6, tighter than the float32 PCA and
  silhouette it checks; one silhouette value differed by 3e-6 on a different BLAS build. The tolerance is now 1e-5
  (the recorded values are unchanged).
- Import ordering in one test module that the current Ruff release rejects (CI lint job).
- The documentation build failed in CI because the `mudata` intersphinx inventory returned HTTP 404; see *Changed*.

### Known limitations

- **What still densifies, and whether that matters.** Dense conversion happens in: the supervised pipeline
  and classifier benchmark (on `X`), label transfer, the per-gene tests (ANOVA, Kruskal-Wallis, Dunn,
  Box's M, differential expression), ComBat, feature selection and PCA loadings, the transformation
  benchmark, normality checks, and DBSCAN, HDBSCAN, hierarchical and consensus clustering when they are
  run on `X` rather than on a representation. In every case the cost is memory of
  `n_cells x n_genes x 8` bytes (500,000 cells and 30,000 genes would need about 120 GB), and it is a
  cost of scale, not a correctness problem: results are identical to the sparse path. Most of these
  steps need a dense array for a reason, either because the method works across all genes at once
  (scaling, ComBat, covariance matrices, SHAP) or because scikit-learn / SciPy build a dense array
  internally anyway (pairwise distances for linkage and DBSCAN are n x n regardless of input format).
  Graph-based clustering, embeddings and label quality from a representation already work from
  `obsm["X_pca"]` and stay small. For large data pass `use_rep="X_pca"` to clustering and
  `supervised_use_rep="X_pca"` to `label_quality` (or `use_rep` to `supervised_analysis`); the latter is not the default because it changes the
  scores relative to the preprint (they then describe a PCA-reduced problem). 0.2.1 adds the warning and
  the single-copy allocation. Making individual tests sparse-native (the per-gene tests are the obvious
  candidates) is possible but needs separate numerical validation and is left for a later release.

### Equivalence with 0.1.0

0.2.1 was compared with 0.1.0 on synthetic count data (four cell types, two batches, a few mislabelled
cells), using the same environment, the same seeds and the same input, once with 160 cells and 80 genes
(seed 3) and once with 240 cells (seed 11). The harness called, in both versions, PCA (including the
Gavish-Donoho and Marchenko-Pastur rules and the variance threshold), all 14 transformations (v2 both
through the transformation functions and through the public `pp` API, which writes a layer), normality
checks, highly variable gene selection (Seurat v3 and Cell Ranger), UMAP, t-SNE and diffusion maps (with
and without a stored representation), k-means (plain, spherical, bisecting), hierarchical, DBSCAN, HDBSCAN,
spectral (single and grid), Leiden, Louvain and consensus clustering, the clustering benchmark, the full
unsupervised and supervised pipelines (default and robust configuration, dense and sparse input), the
classifier benchmark under cross-validation, hold-out and the .632+ bootstrap, all label-quality
variants, differential expression (Wilcoxon, t-test, permutation, pseudobulk, `rank_genes_groups`,
marker genes), over-representation analysis, marker-based annotation and label transfer, the feature
selection benchmark, mutual information, PCA loadings and HVG sensitivity, batch correction (ComBat,
Harmony, Scanorama and the batch benchmark), the statistics module (bootstrap intervals, paired and
permutation tests, McNemar, effect sizes, p-value correction, rank aggregation, ANOVA, Kruskal-Wallis,
Box's M, Dunn), clustering metrics, exploratory summaries, the transformation and embedding benchmarks
and the seed-stability test.

**Result: 98 of 102 comparable results (seed 3) and 97 of 101 (seed 11) are identical to 0.1.0 within
1e-6. All remaining differences are intended and are bug fixes or additions made in 0.2.0:**

| Result | 0.1.0 | 0.2.x | Reason |
|---|---|---|---|
| `benchmark_batch_correction` scores | Identical batch ASW and bio-conservation for ComBat, Harmony and Scanorama | Different per method, plus an `embed_key` column | 0.1.0 scored the uncorrected `X_pca` for every method; see 0.2.0 *Fixed*. The old leaderboards were wrong, not merely different. |
| `marchenko_pastur_cutoff(sigma_method="trimmed_mean")` and PCA with `mp_sigma_method="trimmed_mean"` | 42 components retained by the cutoff; 28 or 32 in the PCA | 9; 8 or 16 | 0.1.0 under-estimated the noise level; see 0.2.0 *Fixed*. The default `"median"` rule is identical. |
| `annotate_by_markers(method="threshold")` | Labels only | Same labels, plus the per-type scores in `obsm` | Addition; the labels are identical. |

Differences that are not numerical, and therefore not visible to the comparison: 0.1.0 returned new
objects and 0.2.x writes in place (or returns a copy with `copy=True`); outputs go to `layers`,
`obsm` and `uns` under the `key_added` you choose instead of replacing `X`; 0.1.0 left a stray `leiden`
column in `obs`; and the two silhouette column names changed as described above. All of these are in the
migration table under 0.2.0. Louvain is the one function that did not run at all in 0.1.0 on a current
setuptools (see *Fixed*); it was compared with a `pkg_resources` stand-in provided to 0.1.0.

## [0.2.0] - 2026-10-08

This release prepares scINTILLA for listing in the [scverse](https://scverse.org) ecosystem. It
turns the package into a scanpy-style library: results are written into the `AnnData` under a name
you control, nothing leaks into the object that you did not ask for, sparse data is not densified
needlessly, library code no longer prints, and the whole thing is tested, documented and built by CI.
**It contains breaking changes**, listed under [Migrating from 0.1](#migrating-from-01).

**The label-quality numbers are unchanged.** `tests/test_label_quality_baseline.py` pins the
per-label scores and the default `obs` column names recorded from 0.1.0 on a fixed synthetic
dataset; the scores match to 1e-6. The only difference in the columns is that the stray `leiden`
column that the clustering benchmark used to leave in `adata.obs` is gone (see *Fixed*).

### Added

**API**

- Scanpy-style namespaces `scintilla.pp` (preprocessing), `scintilla.tl` (tools), `scintilla.pl`
  (plotting), `scintilla.stats` (the array-level statistics layer), `scintilla.benchmark`,
  `scintilla.eda` and `scintilla.io`, plus `scintilla.settings`. The 0.1 flat names
  (`si.run_pca`, `si.unsupervised_analysis`, ...) remain as aliases of the same functions.
- `tl.label_quality(adata, cell_type_col)`: one call from an `AnnData` to the per-label scores of
  every variant. It runs the unsupervised and supervised arms unless their columns are already in
  `adata.obs`. The `scintilla label-quality` command now delegates to it.
- `pp.select_features(adata, target_col, method=...)` flags selected genes in `var` for
  mutual information, PCA loadings, Boruta and mRMR.
- `tl.pseudobulk_by_celltype` and `tl.spectral_grid_search`, which replace the argument-dependent
  return types of `pseudobulk_de(cell_type_col=...)` and `spectral_clustering(n_clusters=None)`.
- A common contract for every function that writes to an `AnnData`:
  `f(adata, ..., *, layer=None, key_added=..., copy=False)`. With `copy=False` the object is modified
  in place and `None` is returned (or the documented result object); `copy=True` returns a modified
  copy and leaves the input untouched. `random_state` is accepted wherever there is randomness.
  `DataFrame` and array input is still converted, in which case the converted `AnnData` is returned.
- Provenance: the parameters behind each result are recorded in `adata.uns["scintilla"][<key>]`, keyed
  by the layer, `obsm` entry, `var` column or `obs` column that holds the result. Records survive
  `write_h5ad` (parameters that are `None` are omitted, and only h5ad-safe values are stored).
- `key_added` for the two pipelines, as a column prefix (default `"scintilla"` and `"pred"`, which
  reproduces the 0.1 names), and `use_rep`/`key_added` for the array-level clustering algorithms
  (`kmeans`, `dbscan`, `hdbscan`, `hierarchical`, `spectral`), which also accept SciPy sparse matrices.
- MuData: `ensure_anndata(data, modality=...)` and the `eda` functions accept a `MuData` together with
  `modality`; the modality may be omitted only if there is exactly one, and the error says so
  otherwise. Analysis functions called with a single-modality `MuData` work on that modality in place.
  The CLI gained `--modality`, so `scintilla eda data.h5mu --modality rna` works (before, `.h5mu` input
  loaded and then crashed one function later). SpatialData is not supported.
- Logging: all progress goes through the `scintilla` logger. `scintilla.settings.verbosity` (`"error"`,
  `"warning"` default, `"info"`, `"debug"`) controls it globally and the existing `verbose=` arguments
  control it for one call. `AnalysisConfig.verbose` now defaults to `None` (follow the setting).
- Plot functions accept `show` and `save` and never close the figure they return. New
  `pl.clustering_benchmark` and `pl.classification_benchmark` draw the tables that the benchmarks
  used to plot themselves.
- CLI: `--quiet`, `--modality`, and a headless Matplotlib backend that is set once in the CLI entry
  point rather than on import.
- `numpydoc` `Parameters` sections on every public function (enforced by `ruff` rule set `D`),
  `py.typed`, and a `TYPE_CHECKING` annotation of the `config` argument.

**Project**

- Sphinx documentation (`docs/`): API reference grouped by namespace, two tutorials executed at build
  time on scanpy's `pbmc68k_reduced`, the long-form guide, manual and label-quality pages (examples
  updated to the 0.2 API), a bibliography, and this changelog. Builds with warnings as errors.
- GitHub Actions: tests on the oldest and newest allowed dependency versions, build and `twine check`,
  docs build, and a trusted-publishing release workflow. Issue and pull-request templates,
  `pre-commit`, Read the Docs, Codecov and EditorConfig configuration.
- `CITATION.cff` (citing the bioRxiv preprint, doi:10.64898/2026.07.27.740477), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`; a short `README.md` with badges and a quick start.
- 356 tests in 26 files (up from 138 in 12), 82 % line coverage (including the `slow` integration test). They cover the API contract of each
  area (including dense and CSR parity and fixed-seed reproducibility), statistical anchors against
  hand-computed values and published formulae, every CLI sub-command, every benchmark estimator, and
  an integration test on `pbmc68k_reduced` (marked `slow`).

### Changed

**Packaging**

- `src/` layout and the `hatchling` build backend; the version is defined in one place
  (`scintilla/__init__.py`).
- `requires-python` is now `>=3.11` (was `>=3.9`).
- Lower bounds on every dependency, checked in CI by installing the oldest allowed versions:
  `anndata>=0.10`, `scanpy>=1.10.3` (the first release with `score_genes(ctrl_as_ref=...)`),
  `numpy>=1.26`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `matplotlib>=3.8`,
  `seaborn>=0.13`, `statsmodels>=0.14`.
- `plotly` and `scikit-posthocs` moved from the core dependencies to the `viz` and `stats` extras
  (both are imported lazily, with a clear `ImportError`); the `full` extra includes them and now also
  `scikit-misc`, which scanpy's default `seurat_v3` HVG flavour needs but was never declared.
- Full classifiers, authors (from the preprint), maintainers and project URLs; `[tool.ruff]`,
  `[tool.pytest]` and `[tool.coverage]` are committed, and the 178 inline `# noqa: PLC0415` markers
  are gone. `uv.lock` is regenerated.

**Behaviour** (all breaking; see the migration table)

- `pp.pca`, `tl.umap`, `tl.tsne`, `tl.diffmap`, `tl.draw_graph`, `tl.combat`, `tl.harmony`,
  `tl.scanorama`, `tl.bbknn`, `pp.highly_variable_genes`, `tl.annotate_by_markers` and
  `tl.transfer_labels` modify their input in place and return `None` (they used to return a copy, or
  a mix of in-place mutation and a returned object).
- Transformations write `adata.layers[key_added]` instead of replacing `adata.X`, so raw counts
  survive. The pure form (`pp.TRANSFORM_REGISTRY[name](adata) -> AnnData`) is unchanged and is what
  the benchmarks use. Transformations that change the set of variables (`*_hvg*`, `glm_pca_transform`)
  return a new `AnnData`.
- Embeddings, Leiden and Louvain compute on a stand-in object that holds only the representation, so
  the caller's object no longer receives a neighbour graph, a PCA or a cluster column. When the
  requested representation is missing a temporary PCA is used and not stored.
- Benchmarks return data only. `benchmark_clustering_methods` returns `(results_df, labels_dict)`,
  `benchmark_models_comprehensive` returns `results_df`, `hierarchical_scipy` returns
  `(labels, Z, cpcc)` and `unsupervised_analysis` no longer returns `"fig"`. Draw them with `si.pl`.
- `annotate_by_markers` stores the per-type scores in `obsm["<key_added>_scores"]` (a DataFrame)
  instead of one `score_<type>` column per cell type in `obs`.
- `rank_genes_groups` no longer writes `uns["rank_genes_groups"]` unless `key_added` is given.
- `select_hvg` flags `var[key_added]` and returns `None`; use `subset=True` for the filtered copy it
  used to return.
- The two-group DE tests (`wilcoxon`, `ttest`, `permutation`) densify only the cells of the two
  groups being compared.
- `benchmark_batch_correction` evaluates each method in the representation that holds its own output
  (see *Fixed*) and reports it in a new `embed_key` column.
- Library code is silent by default. Progress bars and messages that 0.1 printed unconditionally now
  appear only with `verbose=True` or `settings.verbosity = "info"`; failures are reported as warnings.
- `benchmarking.reproducibility.seed_stability_test`: `icc_method` defaults to `"anova"` (see
  *Removed*); `"pingouin"` is accepted as an alias and gives the same value.
- First arguments named `data` that take an `AnnData` are now named `adata`; code that passed them by
  keyword must be updated.

### Fixed

- **Batch-correction benchmark scored the wrong embedding.** `benchmark_batch_correction` computed
  batch ASW and bio-conservation on `obsm["X_pca"]` whenever the input had one, which is the
  *uncorrected* embedding, and never looked at Harmony's `X_pca_harmony`. Every method with an
  `X_pca` in the input therefore received the same score. Each method is now scored in its own output
  (ComBat: a PCA of the corrected layer; Harmony: `X_pca_harmony`; Scanorama: `X_scanorama`; BBKNN: a
  spectral embedding of the batch-balanced graph). **Leaderboards from 0.1 for this benchmark should
  be recomputed.**
- **`marchenko_pastur_cutoff(sigma_method="trimmed_mean")`** divided the bulk mean of the squared
  singular values by `1 + gamma`, under-estimating the noise level and retaining far too many
  components (32 instead of 5 on a rank-5 test matrix). The default `"median"` rule was correct and is
  unchanged. This only affected `auto_components="marchenko_pastur"` with `mp_sigma_method="trimmed_mean"`.
- **Plot functions failed on matplotlib 3.9 and later** (`plt.cm.get_cmap` was removed): PCA, embedding,
  batch and label-quality plots.
- **Leaked state.** Leiden, Louvain and the embedding functions used to modify the caller's `AnnData`
  as a side effect (a neighbour graph, `obs["leiden"]`, `obs["louvain"]`, a PCA). `obs["leiden"]` was
  even part of the "default" columns of the label-quality pipeline by accident. `rank_genes_groups` and
  `annotate_by_markers` also wrote scanpy bookkeeping and `score_*` columns into `obs`/`uns`.
- `draw_graph` returned ForceAtlas2 in its documentation but stored the Fruchterman-Reingold layout
  when `fa2-modified` was missing; the layout actually used is now recorded in `uns["scintilla"]`.
- `tsne` fell back to scikit-learn on *any* exception, hiding real errors; it now falls back only when
  scanpy is unavailable. `select_hvg(method="pearson_residuals")` likewise caught every exception.
- `benchmark-all` swallowed the failure of a stage unless `--verbose` was given; failures now always go
  to stderr. A failing SHAP step or a failing per-model prediction in `supervised_analysis` is now a
  warning instead of silent.
- `ensure_anndata` rejected `MuData` with a confusing `TypeError` one function after loading;
  `auto_detect_format` is annotated correctly.

### Performance and memory

- `eda.dataset_summary` computes sparsity from the number of stored non-zeros; on CSR input it no
  longer allocates the dense matrix (this was the cheapest call in the package and the most likely to
  run out of memory). `expressed_genes` and `mean_expression_by_group` are sparse-safe.
- `log_shift_size_factor`, `arcsinh_transform`, `log_alpha_transform`, `log_cpm_transform` and
  `normalise_scran` are zero-preserving and keep CSR input sparse (the dense and sparse paths agree to
  rounding error and are tested against each other).
- PCA, HVG selection, marker scoring, `rank_genes_groups`, k-means, DBSCAN and spectral clustering
  accept sparse input. The 45 unconditional `toarray()` sites of 0.1 are gone: every access to an
  expression matrix goes through `get_matrix`, 15 of which keep matrices sparse and 34 of which
  densify through that one helper, logging the reason at debug level, because the algorithm needs a
  dense array (LDA/QDA, Box's M, hierarchical linkage, most scikit-learn classifiers, SHAP). These are
  documented, not hidden.

### Removed

- The **`pingouin`** dependency, and with it GPL-3.0 from the default install. It was used once, for
  ICC(1,1) in `seed_stability_test`; this is now a closed-form one-way ANOVA estimator, covered by a
  test that compares it with `pingouin` when that is installed.
- `demo_notebook.ipynb` (it could not run: `adata_hvg` was never defined and no cell had outputs); the
  tutorials in `docs/notebooks/` replace it.
- `MANIFEST.in` (hatchling selects the sdist contents), and support for Python 3.9 and 3.10.

### Deprecated

- `pseudobulk_de(..., cell_type_col=...)` (returns a dict; use `pseudobulk_de_by_celltype`),
  `spectral_clustering(n_clusters=None)` (use `spectral_grid_search`) and
  `hierarchical_clustering(mode="scipy")` (use `hierarchical_scipy`) still work and emit a
  `DeprecationWarning`.

### Known limitations

- The label-quality columns keep their 0.1 names by default so that published analyses reproduce;
  `key_added` renames them for the pipelines, but `compute_label_quality_variants` and the label-quality
  plots read the default `scintilla_*` and `pred_*` names.
- Some algorithms need a dense matrix (see *Performance and memory*); very large datasets should be
  analysed on a representation (`use_rep`) rather than on `X`.
- SpatialData is not supported. The package is not yet on conda-forge, and has no GitHub release or
  software DOI until 0.2.0 is tagged and archived.

### Migrating from 0.1

| 0.1 | 0.2 |
|---|---|
| `adata = run_pca(adata, n_comps=30)` | `si.pp.pca(adata, n_comps=30)` (in place; `copy=True` to get a copy) |
| `adata2 = log_cpm_transform(adata)` (new object, `X` replaced) | `si.pp.log_cpm_transform(adata)` then `adata.layers["log_cpm_transform"]`, or `si.pp.TRANSFORM_REGISTRY["log_cpm_transform"](adata)` for the old behaviour |
| `adata = run_umap(adata, ...)` (also `run_tsne`, `run_diffusion_map`, `run_force_directed`) | `si.tl.umap(adata, ...)` (also `tsne`, `diffmap`, `draw_graph`); keys `X_umap`, `X_tsne`, `X_diffmap`, `X_draw_graph_fa` |
| `adata_hvg = select_hvg(adata, n_top_genes=2000)` | `adata_hvg = si.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)`, or without `subset` to only flag `var["highly_variable"]` |
| `adata_c = combat_correct(adata, "batch")` (corrected `X`) | `si.tl.combat(adata, "batch")` then `adata.layers["combat"]` |
| `harmony_correct`, `scanorama_correct`, `bbknn_correct` returned a copy | `si.tl.harmony`, `si.tl.scanorama`, `si.tl.bbknn` modify in place (`copy=True` for a copy) |
| `adata_q = transfer_labels(ref, query, "cell_type")` | `si.tl.transfer_labels(ref, query, "cell_type")` (in place; `copy=True` for a copy) |
| `adata = annotate_by_markers(adata, markers)`; columns `score_<type>` | `si.tl.annotate_by_markers(adata, markers)`; scores in `adata.obsm["predicted_cell_type_scores"]` |
| `results, labels, fig = benchmark_clustering_methods(...)` | `results, labels = si.benchmark.benchmark_clustering(...)`; `si.pl.clustering_benchmark(results)` |
| `results, fig = benchmark_models_comprehensive(...)` | `results = si.benchmark.benchmark_classifiers(...)`; `si.pl.classification_benchmark(results)` |
| `result["fig"]` from `unsupervised_analysis` | `si.pl.clustering_benchmark(result["results_df"])` |
| `pseudobulk_de(..., cell_type_col="ct")` (a dict) | `si.tl.pseudobulk_by_celltype(adata, cond, sample, "ct")` |
| `spectral_clustering(X)` without `n_clusters` (a grid search) | `si.tl.spectral_grid_search(X)` |
| `labels, Z, cpcc, fig = hierarchical_scipy(...)` | `labels, Z, cpcc = hierarchical_scipy(...)`; `si.pl.dendrogram(Z)` |
| `batch_asw(adata_corrected, "batch")` after Harmony | `batch_asw(adata, "batch", embed_key="X_pca_harmony")` |
| `import scintilla as sc` | `import scintilla as si` (avoids clashing with `import scanpy as sc`) |
| progress printed by default | `si.settings.verbosity = "info"` or `verbose=True` |
| `pip install scintilla-py` on Python 3.9 or 3.10 | upgrade Python to 3.11 or newer |

## [0.1.0] - 2026-10-08

First release on PyPI (`scintilla-py`; the import name is `scintilla`). Contained the correctness fixes
of the pre-release audit (label transfer by gene name, no fabricated pseudobulk replicates, no
import-time Matplotlib backend change, visible benchmark failures, a working `AnalysisConfig`, per-call
seeding), the per-label quality variants and the `scintilla label-quality` command.

[Unreleased]: https://github.com/soumickmj/scINTILLA/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/soumickmj/scINTILLA/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/soumickmj/scINTILLA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/soumickmj/scINTILLA/releases/tag/v0.1.0
