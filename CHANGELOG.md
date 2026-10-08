# Changelog

All notable changes to `scintilla-py` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/) (with the usual caveat that 0.x minor releases may break the API).

## [Unreleased]

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

[Unreleased]: https://github.com/soumickmj/scINTILLA/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/soumickmj/scINTILLA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/soumickmj/scINTILLA/releases/tag/v0.1.0
