# scINTILLA

**scINTILLA** - *Single-Cell INTegrated Inference, Labelling, and Landscape Analysis*

scINTILLA is an end-to-end single-cell RNA-seq analysis pipeline that automates data ingestion, preprocessing, normalisation benchmarking, feature selection, dimensionality reduction, unsupervised clustering, supervised classification, differential expression, cell-type annotation, batch correction, and comprehensive benchmarking - all through a unified Python API and a rich command-line interface (CLI).

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data I/O](#data-io)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Preprocessing & Normalisation](#preprocessing--normalisation)
6. [Feature Selection](#feature-selection)
7. [Dimensionality Reduction](#dimensionality-reduction)
8. [Clustering](#clustering)
9. [Classification](#classification)
10. [Differential Expression](#differential-expression)
11. [Cell-Type Annotation](#cell-type-annotation)
12. [Batch Correction](#batch-correction)
13. [Benchmarking Utilities](#benchmarking-utilities)
14. [Robust Statistics](#robust-statistics)
15. [CLI Reference](#cli-reference)
16. [Configuration & AnalysisConfig](#configuration--analysisconfig)
17. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)

---

## Installation

### Minimal install (core dependencies)

```bash
pip install .
```

### Full install (enables all optional features)

Install with all optional dependencies (UMAP, HDBSCAN, Harmony, BBKNN, Scanorama, XGBoost, LightGBM, Boruta, mRMR, SHAP, and more):

```bash
pip install ".[full]"
```

### Requirements

| Dependency | Purpose |
|---|---|
| `numpy`, `pandas`, `scipy` | Numerical computing |
| `scikit-learn` | ML models, metrics |
| `anndata` | AnnData data structure |
| `scanpy` | Single-cell utilities |
| `statsmodels` | Statistical tests |
| `matplotlib`, `seaborn`, `plotly` | Visualisation |
| `scikit-posthocs`, `pingouin` | Post-hoc statistical tests |
| `umap-learn` *(optional)* | UMAP embedding |
| `hdbscan` *(optional)* | HDBSCAN clustering |
| `harmonypy` *(optional)* | Harmony batch correction |
| `bbknn` *(optional)* | BBKNN batch correction |
| `scanorama` *(optional)* | Scanorama batch correction |
| `xgboost`, `lightgbm` *(optional)* | Gradient-boosted trees |
| `Boruta` *(optional)* | Boruta feature selection |
| `mrmr-selection` *(optional)* | mRMR feature selection |
| `shap` *(optional)* | SHAP feature importance |
| `psutil` *(optional)* | Memory profiling |
| `kneed` *(optional)* | Elbow detection for adaptive DBSCAN eps |

---

## Quick Start

```python
import scintilla as sc

# 1. Load data
adata = sc.load_h5ad("data/pbmc3k.h5ad")

# 2. Preprocess: normalise with the best-performing transform
adata_norm = sc.TRANSFORM_REGISTRY["log_shift_size_factor"](adata)

# 3. Dimensionality reduction
from scintilla.preprocessing.pca import run_pca
adata_pca = run_pca(adata_norm, n_comps=30)

# 4. Cluster (best labels are automatically stored in adata.obs["scintilla_cluster"])
result = sc.unsupervised_analysis(adata_pca, cell_type_col="cell_type")
print("Best clustering method:", result["best_method"])
print(adata_pca.obs["scintilla_cluster"].value_counts())

# 5. Classify
cls_result = sc.supervised_analysis(adata_pca, target_col="cell_type")
print("Best classifier:", cls_result["best_model_name"])
```

For a quick exploration run, use the `fast` preset via the `AnalysisConfig` system:

```python
from scintilla import AnalysisConfig

cfg = AnalysisConfig.fast()   # kmeans + leiden only; LogReg, RF, kNN; no SHAP
result = sc.unsupervised_analysis(adata_pca, cell_type_col="cell_type", config=cfg)
cls_result = sc.supervised_analysis(adata_pca, target_col="cell_type", config=cfg)
```

For publication-quality results with bootstrap CIs, rank aggregation, and adaptive methods:

```python
cfg = AnalysisConfig.robust()  # bootstrap CIs, Borda scoring, adaptive PCA & resolution, permutation no-info rate, trimmed-mean MP sigma²
result = sc.unsupervised_analysis(adata_pca, cell_type_col="cell_type", config=cfg)
cls_result = sc.supervised_analysis(adata_pca, target_col="cell_type", config=cfg)
```

Or via the CLI (full pipeline in one command):

```bash
scintilla run-all data/pbmc3k.h5ad --target-col cell_type --output-dir results/

# Use the fast preset to skip slow methods
scintilla run-all data/pbmc3k.h5ad --target-col cell_type --fast --output-dir results/

# Load a custom config file (generated via `scintilla generate-config`)
scintilla run-all data/pbmc3k.h5ad --target-col cell_type --config my_config.yaml --output-dir results/
```

---

## Data I/O

### Supported formats

| Format | Extension | Notes |
|---|---|---|
| HDF5 AnnData | `.h5ad` | Native AnnData format |
| CSV / TSV | `.csv`, `.tsv`, `.txt` | Cells × genes matrix |
| MuData | `.h5mu` | Requires `pip install mudata` |

### Python API

```python
from scintilla.io.loaders import load_h5ad, load_csv, auto_detect_format, ensure_anndata
from scintilla.io.exporters import save_anndata, save_results_csv, save_results_json

# Load from .h5ad
adata = load_h5ad("data/my_data.h5ad")

# Load from CSV (cells as rows, genes as columns)
adata = load_csv("data/counts.csv")

# Load from CSV where genes are rows (transpose automatically)
adata = load_csv("data/counts_genes_rows.csv", transpose=True)

# Auto-detect format (recommended)
adata = auto_detect_format("data/my_data.h5ad")  # or .csv / .tsv

# Convert a pandas DataFrame to AnnData
import pandas as pd
df = pd.read_csv("data/counts.csv", index_col=0)
adata = ensure_anndata(df)

# Save results
save_anndata(adata, "output/processed.h5ad")
save_results_csv(results_df, "output/results.csv")
save_results_json({"best_model": "RF", "accuracy": 0.95}, "output/summary.json")
```

### Parameters

**`load_csv`**

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Path to the CSV file |
| `index_col` | `0` | Column to use as index |
| `transpose` | `False` | Transpose so cells are rows |
| `**kwargs` | - | Passed to `pandas.read_csv` |

**`ensure_anndata`**

| Parameter | Default | Description |
|---|---|---|
| `data` | required | `pd.DataFrame`, `np.ndarray`, or `AnnData` |
| `target_col` | `None` | Column to keep in `obs` even if numeric |

---

## Exploratory Data Analysis (EDA)

### Python API

```python
from scintilla.eda.summary import dataset_summary

summary = dataset_summary(adata)
print(summary)
# Keys: n_cells, n_genes, sparsity, obs_columns, var_columns,
#        cell_type_counts (if present), ...
```

### CLI

```bash
# Print summary to stdout
scintilla eda data/pbmc3k.h5ad

# Group by a cell-type column and save to JSON
scintilla eda data/pbmc3k.h5ad --group-col cell_type --output results/eda_summary.json
```

**`eda` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input file (`.h5ad`, `.csv`) |
| `--group-col` | `None` | Column in `obs` to group cells by |
| `--output` | `None` | Save summary as JSON to this path |

---

## Preprocessing & Normalisation

scINTILLA provides 14 built-in data transformations plus an automated benchmarking framework that selects the best one for your dataset.

### Available transforms

| Key | Description |
|---|---|
| `log_shift_size_factor` | log(x / size_factor + 1); standard scRNA-seq normalisation |
| `arcsinh_transform` | arcsinh(α·x), default α=0.05; common for CyTOF/CITE-seq |
| `log_alpha_transform` | log(α·x + 1), default α=0.05 |
| `log_cpm_transform` | log(CPM + 1); counts per million normalisation |
| `log_shift_scale_by_std` | log1p then divide each gene by its standard deviation |
| `log_shift_size_factor_hvg` | `log_shift_size_factor` then top 35% most variable genes |
| `log_shift_size_factor_z` | `log_shift_size_factor` then z-score per gene |
| `log_shift_hvg_z` | log-shift + HVG selection + z-score |
| `normalise_scran` | Scran-style geometric mean deconvolution |
| `normalise_tmm` | TMM normalisation (Robinson & Oshlack, 2010) |
| `box_cox_transform` | Box-Cox per gene |
| `pearson_residuals_transform` | Analytic Pearson residuals (uses scanpy if available) |
| `glm_pca_transform` | Poisson GLM-PCA approximation via TruncatedSVD |
| `sanity_transform` | Bayesian estimation approximation (SANITY-like) |

### Python API

```python
from scintilla.preprocessing.transformations import TRANSFORM_REGISTRY, get_all_transformations

# List all available transforms
print(list(TRANSFORM_REGISTRY.keys()))

# Apply a specific transform
transform_fn = TRANSFORM_REGISTRY["log_shift_size_factor"]
adata_norm = transform_fn(adata)

# Apply arcsinh with custom alpha
from scintilla.preprocessing.transformations import arcsinh_transform
adata_norm = arcsinh_transform(adata, alpha=0.1)

# Apply Pearson residuals with custom theta
from scintilla.preprocessing.transformations import pearson_residuals_transform
adata_norm = pearson_residuals_transform(adata, theta=50.0)

# Apply GLM-PCA approximation with custom dimensions
from scintilla.preprocessing.transformations import glm_pca_transform
adata_pca = glm_pca_transform(adata, n_components=30)
```

### Normalisation benchmarking

Automatically rank all transforms for your dataset using a composite score of kNN overlap, silhouette score, normality, and PCA variance preservation:

```python
from scintilla.preprocessing.benchmark import benchmark_transformations

results_df, best_name, best_adata = benchmark_transformations(adata, verbose=True)
print(f"Best transform: {best_name}")
print(results_df[["transform", "status", "composite_score"]].head(10))
```

Every transform gets a row. One that raised carries `status="failed"` and a
`failure_reason` rather than disappearing from the leaderboard. If *every*
transform fails, `best_name` and `best_adata` are `None` and `results_df` still
lists each failure, so check `best_name is not None` before using the result.

**Score weights** (configurable in `scintilla/config.py`):

| Metric | Weight |
|---|---|
| kNN overlap | 0.35 |
| PCA variance preservation | 0.30 |
| Silhouette score | 0.25 |
| Shapiro-Wilk normality | 0.05 |
| Anderson-Darling normality | 0.05 |

### CLI

```bash
# Benchmark all transforms and show composite scores
scintilla normalise data/pbmc3k.h5ad

# Save leaderboard to CSV
scintilla normalise data/pbmc3k.h5ad --output results/norm_benchmark.csv --verbose

# Apply a specific transform and save output
scintilla preprocess data/pbmc3k.h5ad --transform log_cpm_transform --output preprocessed.h5ad
```

**`normalise` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input file |
| `--output` | `None` | Save benchmark results CSV |
| `--verbose` | `False` | Print progress |

**`preprocess` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input file |
| `--transform` | `log_shift_size_factor` | Name of the transform to apply |
| `--output` | `preprocessed.h5ad` | Output `.h5ad` path |

### Normality checking

```python
from scintilla.preprocessing.normality import check_normality

is_normal, report = check_normality(adata)
print("Data is normal:", is_normal)
print(report)
```

**`check_normality` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `sample_size` | `500` | Cells subsampled per gene for Shapiro-Wilk test |
| `random_state` | `42` | Seed for the Shapiro-Wilk subsample |

> Genes whose Shapiro-Wilk test cannot be computed count as **not passing** and
> are listed in the report under `shapiro_failed_features`, so the pass fraction
> is always taken over every gene rather than over the ones that happened to
> succeed.
| `threshold` | `0.3` | Fraction of genes that must pass to call data "normal" |

> **Note:** The subsample size is intentionally small. At large *n* (e.g. 5 000), the Shapiro-Wilk test has extreme statistical power and rejects normality even for well-normalised scRNA-seq data, leading to false negatives.

### PCA

```python
from scintilla.preprocessing.pca import run_pca

adata = run_pca(adata, n_comps=30)
# PCA coordinates stored in adata.obsm["X_pca"]
```

---

## Feature Selection

scINTILLA provides six feature-selection strategies plus a benchmarking wrapper that compares them by downstream classification accuracy.

### Methods

| Method | Function | Notes |
|---|---|---|
| PCA loadings | `extract_top_genes_per_pc` | Top genes contributing to each PC |
| Highly variable genes (Seurat v3) | `select_hvg(method="seurat_v3")` | HVG via Seurat v3 |
| HVG (Pearson residuals) | `select_hvg(method="pearson_residuals")` | HVG via scanpy |
| Mutual information | `mi_feature_selection` | MI between each gene and labels |
| Boruta | `boruta_selection` | Random-forest shadow-feature selection |
| mRMR | `mrmr_selection` | Minimum-Redundancy Maximum-Relevance |

### Python API

```python
from scintilla.feature_selection import (
    select_hvg,
    mi_feature_selection,
    boruta_selection,
    mrmr_selection,
)
from scintilla.feature_selection.pca_loadings import (
    extract_top_genes_per_pc,
    build_reduced_dataset,
    validate_reduced_set,
)

# Highly variable genes (Seurat v3 default)
adata_hvg = select_hvg(adata, n_top_genes=2000, method="seurat_v3")

# PCA-loadings-based selection
gene_list = extract_top_genes_per_pc(adata, n_per_pc=9)
adata_reduced = build_reduced_dataset(adata, gene_list)

# The three supervised selectors below work on arrays and return indices or a
# boolean mask, so pull X and y out of the AnnData first.
X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
y = adata.obs["cell_type"].values

# Mutual information -> (selected_indices, scores)
idx, mi_scores = mi_feature_selection(X, y, n_features=50)
selected_genes = adata.var_names[idx]

# Boruta -> (support_mask, ranking); needs the Boruta package
mask, ranking = boruta_selection(X, y, max_iter=100)
selected_genes = adata.var_names[mask]

# mRMR -> (selected_indices, scores); needs the mrmr-selection package
idx, mrmr_scores = mrmr_selection(X, y, n_features=50)
selected_genes = adata.var_names[idx]
```

### Benchmarking feature selection

```python
from scintilla.feature_selection.benchmark import benchmark_feature_selection

# Default runs only fast methods; opt in to boruta/mrmr explicitly
results_df = benchmark_feature_selection(
    adata,
    target_col="cell_type",
    # methods defaults to ["pca_loadings", "mutual_information"]
    # add "boruta" or "mrmr" for richer (but slower) comparison:
    methods=["pca_loadings", "mutual_information", "boruta", "mrmr"],
    n_features=50,
)
print(results_df)
```

**`benchmark_feature_selection` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `target_col` | required | Column in `obs` with class labels |
| `methods` | `None` | Methods to benchmark; `None` uses `["pca_loadings", "mutual_information"]` |
| `n_features` | `None` | Features to select per method; `None` uses `config.n_features`, else `50` |
| `test_size` | `None` | Hold-out fraction; `None` uses `config.test_size`, else `0.2` |
| `random_state` | `None` | Seed; `None` uses `config.random_seed`, else `42` |
| `config` | `None` | `AnalysisConfig` supplying the defaults above |

> **Performance note:** `boruta` and `mrmr` are not in the default set because they are significantly slower on large datasets (>10 000 cells). Add them explicitly when you have time, or adjust via `AnalysisConfig`.

### CLI

```bash
# PCA-loadings selection (default)
scintilla feature-select data/pbmc3k.h5ad --n-per-pc 9 --output genes.csv
```

**`feature-select` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input `.h5ad` file |
| `--n-per-pc` | `9` | Top genes to extract per PC |
| `--output` | `None` | Save gene list to CSV |

---

## Dimensionality Reduction

### Methods

| Method | Function | Key parameter |
|---|---|---|
| UMAP | `run_umap` | `n_neighbors`, `min_dist` |
| t-SNE | `run_tsne` | `perplexity` |
| Diffusion Map | `run_diffusion_map` | `n_comps` |
| Force-directed graph | `run_force_directed` | `layout` |

### Python API

```python
from scintilla.dimensionality_reduction import run_umap, run_tsne, run_diffusion_map
from scintilla.dimensionality_reduction.force_directed import run_force_directed

# Run UMAP (embedding stored in adata.obsm["X_umap"])
adata = run_umap(adata, use_rep="X_pca", n_neighbors=15, min_dist=0.5)

# Run t-SNE (embedding stored in adata.obsm["X_tsne"])
adata = run_tsne(adata, use_rep="X_pca", perplexity=30)

# Run Diffusion Map (embedding stored in adata.obsm["X_diffmap"])
adata = run_diffusion_map(adata, use_rep="X_pca", n_comps=10)

# Force-directed layout
adata = run_force_directed(adata)
```

### Embedding benchmarking

```python
from scintilla.dimensionality_reduction.benchmark import benchmark_embeddings

results_df = benchmark_embeddings(adata, use_rep="X_pca")
print(results_df)
```

### CLI

```bash
# UMAP (default)
scintilla reduce data/pbmc3k.h5ad --method umap --use-rep X_pca --output embedded.h5ad

# t-SNE
scintilla reduce data/pbmc3k.h5ad --method tsne --output embedded_tsne.h5ad

# Diffusion Map
scintilla reduce data/pbmc3k.h5ad --method diffmap --output embedded_dm.h5ad
```

**`reduce` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input `.h5ad` file |
| `--method` | `umap` | Embedding method: `umap`, `tsne`, `diffmap` |
| `--use-rep` | `X_pca` | Key in `adata.obsm` to use as input |
| `--output` | `None` | Save output `.h5ad`; otherwise print `obsm` keys |
| `--verbose` | `False` | Print progress |

---

## Clustering

scINTILLA benchmarks multiple unsupervised clustering algorithms against known cell-type labels using the Adjusted Rand Index (ARI).

### Algorithms

| Algorithm | Notes |
|---|---|
| K-Means | Grid search over `init` strategies; includes spherical and bisecting variants |
| Agglomerative / Hierarchical | Grid over distance metrics × linkage methods |
| DBSCAN | Grid over `eps` values × distance metrics |
| HDBSCAN | Grid over `min_cluster_size` × `min_samples` |
| Leiden | Resolution sweep (0.1–3.0) |
| Louvain | Resolution sweep (0.1–3.0) |
| Spectral | Grid over number of clusters |
| Consensus | Ensemble of the above algorithms |

### Python API

```python
from scintilla import unsupervised_analysis

result = unsupervised_analysis(
    adata,
    cell_type_col="cell_type",   # ground-truth column for ARI evaluation
    n_clusters=None,              # defaults to number of unique cell types
    run_pca_first=True,           # run PCA before clustering
    n_pca_comps=30,               # PCA components
    store_labels=True,            # write best labels to adata.obs["scintilla_cluster"]
    config=None,                  # optional AnalysisConfig (selects methods/grids)
    verbose=True,
)

print("Best method:", result["best_method"])
print(result["results_df"].head(10))
print(adata.obs["scintilla_cluster"].value_counts())   # best cluster labels

# Visualise with the returned figure
result["fig"].show()
```

**`unsupervised_analysis` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `cell_type_col` | required | Column in `obs` with true cell-type labels |
| `n_clusters` | `None` | Number of clusters; inferred from `cell_type_col` if `None` |
| `run_pca_first` | `True` | Run PCA before all clustering algorithms |
| `use_rep` | `None` | `obsm` key to cluster on when `run_pca_first=False` |
| `n_pca_comps` | `None` | PCA components when `run_pca_first=True`; `None` uses `config.n_pca_comps`, else `30` |
| `store_labels` | `True` | Write best cluster labels to `adata.obs["scintilla_cluster"]` |
| `n_jobs` | `1` | Run method groups in parallel when `> 1` |
| `auto_pca_components` | omitted | Adaptive component count: `"gavish_donoho"`, `"marchenko_pastur"`, or `None` to disable. Omit to use `config.auto_pca_components` |
| `mp_sigma_method` | omitted | Noise-scale estimator for Marchenko-Pastur: `"median"`, `"trimmed_mean"`. Omit to use `config.mp_sigma_method` |
| `random_state` | `None` | Seed for PCA and every stochastic clustering method; `None` uses `config.random_seed`, else `42` |
| `config` | `None` | `AnalysisConfig` controlling which methods run and their grids |
| `verbose` | `None` | Print progress; `None` uses `config.verbose`, else `True` |

> `auto_pca_components` and `mp_sigma_method` distinguish "argument omitted" from
> an explicit `None`. Passing `auto_pca_components=None` disables adaptive
> selection even when the config enables it; omitting the argument defers to the
> config.

**Return values**

| Key | Type | Description |
|---|---|---|
| `adata` | AnnData | Input data (with PCA if `run_pca_first=True`; with `scintilla_cluster` if `store_labels=True`) |
| `results_df` | DataFrame | Columns: `method`, `params`, `ari`, `ami`, `n_clusters`, `noise_fraction`, `status`, `failure_reason` |
| `labels_dict` | dict | `{method_key: labels_array}` for every tested combination |
| `fig` | Figure | Matplotlib figure with ARI comparison |
| `best_method` | str | Name of the method with the highest ARI |

### Low-level clustering API

```python
from scintilla.clustering.kmeans import kmeans_clustering
from scintilla.clustering.leiden import leiden_clustering
from scintilla.clustering.hdbscan import hdbscan_clustering

labels, centers, inertia = kmeans_clustering(X, n_clusters=8)
labels = leiden_clustering(adata, resolution=0.5, use_rep="X_pca")
hdbscan_df, labels = hdbscan_clustering(X, min_cluster_size_range=[20])
```

### CLI

```bash
# Benchmark all clustering methods
scintilla cluster data/pbmc3k.h5ad --cell-type-col cell_type

# Specify number of clusters and save results
scintilla cluster data/pbmc3k.h5ad \
    --cell-type-col cell_type \
    --n-clusters 8 \
    --output results/clustering.csv \
    --verbose
```

**`cluster` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input `.h5ad` |
| `--cell-type-col` | `cell_type` | Column in `obs` with true cell-type labels |
| `--n-clusters` | `None` | Number of clusters (defaults to unique cell types) |
| `--output` | `None` | Save results DataFrame to CSV |
| `--config` | `None` | Path to a YAML config file (see `generate-config`) |
| `--fast` | `False` | Use `AnalysisConfig.fast()` preset (overrides `--config`) |
| `--verbose` | `False` | Print progress |

---

## Classification

scINTILLA trains and evaluates up to 12 classifiers and returns the one with the highest macro-F1 score, optionally with SHAP feature importances.

### Classifiers

| Short name | Model |
|---|---|
| `LogReg` | Logistic Regression |
| `RF` | Random Forest |
| `SVM` | SVM (RBF kernel) |
| `MLP` | Multi-layer Perceptron |
| `LDA` | Linear Discriminant Analysis |
| `QDA` | Quadratic Discriminant Analysis |
| `kNN` | k-Nearest Neighbours |
| `GradientBoosting` | Gradient Boosting |
| `NaiveBayes` | Gaussian Naive Bayes |
| `StackingEnsemble` | Stacking Ensemble |
| `XGBoost` | XGBoost *(optional)* |
| `LightGBM` | LightGBM *(optional)* |

### Python API

```python
from scintilla import supervised_analysis

result = supervised_analysis(
    adata,
    target_col="cell_type",   # column in obs with labels
    normality=None,            # None = auto-detect; True/False = override
    test_size=0.2,             # fraction of data for test set
    include_shap=True,         # compute SHAP feature importances
    models=None,               # None = all classifiers; or e.g. ["LogReg", "RF", "kNN"]
    config=None,               # optional AnalysisConfig
    verbose=True,
)

print("Best model:", result["best_model_name"])
print("All results:")
for name, res in result["all_results"].items():
    m = res["metrics"]
    print(f"  {name}: acc={m['accuracy']:.3f}  f1={m['f1']:.3f}")
```

**`supervised_analysis` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `target_col` | `"cell_type"` | Column in `obs` with class labels |
| `use_rep` | `None` | `obsm` key to classify on instead of `X` |
| `scale` | `False` | Standardise features before training |
| `normality` | `None` | `None` = auto-detect via `check_normality`; `True`/`False` = override |
| `test_size` | `None` | Hold-out fraction; `None` uses `config.test_size`, else `0.2` |
| `include_shap` | `None` | Compute SHAP importances; `None` uses `config.include_shap`, else `True` |
| `check_consistency` | `False` | Write per-model predictions and consensus columns to `obs` |
| `models` | `None` | Explicit list of classifier short names to run; `None` = all available |
| `n_jobs` | `1` | Train models in parallel when `> 1` |
| `random_state` | `None` | Seed for the split and every stochastic model; `None` uses `config.random_seed`, else `42` |
| `config` | `None` | `AnalysisConfig` overriding `classifiers` and `include_shap` |
| `verbose` | `None` | Print progress; `None` uses `config.verbose`, else `True` |

> `XGBoost` and `LightGBM` are offered only when their package is installed.
> When it is not, the model is absent from `all_results` rather than present
> with a failure — an uninstalled backend is not a failed method.

> **Performance note:** `GradientBoosting` uses `HistGradientBoostingClassifier` internally which is orders of magnitude faster than the legacy `GradientBoostingClassifier` for large datasets (native support for missing values, multi-core computation).

**Return values**

| Key | Type | Description |
|---|---|---|
| `best_model` | sklearn model | Fitted best model |
| `best_model_name` | str | Name of the best model |
| `all_results` | dict | `{model_name: {"model": ..., "metrics": {...}}}`. A model that raised is kept with `status="failed"`, `failure_reason`, `model=None` and an empty `metrics`, so a dropped model is visible rather than merely absent |
| `feature_importances` | dict or None | SHAP values per feature |
| `normality` | bool | Whether data passed normality test |

### Comprehensive benchmarking

```python
from scintilla.classification.benchmark import benchmark_models_comprehensive

results_df, fig = benchmark_models_comprehensive(
    adata,
    target_col="cell_type",
    verbose=True,
)
print(results_df[["model", "space", "accuracy", "f1"]].head(20))
```

### CLI

```bash
# Classify with all models, print best
scintilla classify data/pbmc3k.h5ad --target-col cell_type

# Skip SHAP and save summary JSON
scintilla classify data/pbmc3k.h5ad \
    --target-col cell_type \
    --no-shap \
    --output results/classification.json \
    --verbose
```

**`classify` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input file |
| `--target-col` | `cell_type` | Column in `obs` with class labels |
| `--no-shap` | `False` | Skip SHAP feature importance computation |
| `--output` | `None` | Save summary as JSON |
| `--config` | `None` | Path to a YAML config file (see `generate-config`) |
| `--fast` | `False` | Use `AnalysisConfig.fast()` preset (overrides `--config`) |
| `--verbose` | `False` | Print progress |

---

## Differential Expression

scINTILLA supports four DE methods plus rank-genes-groups analysis from scanpy.

### Methods

| Method | Function | Notes |
|---|---|---|
| Wilcoxon rank-sum | `wilcoxon_de` | Non-parametric, robust |
| Welch's t-test | `ttest_de` | Parametric, unequal variance |
| Permutation test | `permutation_de` | Permutation-based p-values |
| Pseudobulk DESeq2 | `pseudobulk_de` | Bulk-like approach per sample |
| Rank genes groups | `rank_genes_groups` | scanpy wrapper, one-vs-rest |

### Python API

```python
from scintilla import wilcoxon_de, ttest_de, permutation_de, rank_genes_groups
from scintilla import volcano_plot_data, filter_de_genes

# Wilcoxon rank-sum test between two groups
de_df = wilcoxon_de(adata, group_col="cell_type", group1="Monocyte", group2="T Cell")
print(de_df.head(20))
# Columns: gene, statistic, pvalue, pvalue_adj, log2fc, rank_biserial, cliffs_delta

# Welch's t-test
de_df = ttest_de(adata, group_col="cell_type", group1="Monocyte", group2="T Cell")

# Permutation test
de_df = permutation_de(adata, group_col="cell_type", group1="Monocyte", group2="T Cell")

# Pseudobulk: aggregates per sample, so it needs at least two biological
# samples per condition and raises ValueError otherwise.
de_df = pseudobulk_de(adata, condition_col="condition", sample_col="sample_id")

# scanpy rank-genes-groups (one-vs-rest across all groups)
de_df = rank_genes_groups(adata, groupby="cell_type", method="wilcoxon", n_genes=50)

# method="logreg" ranks genes by model coefficient. Scanpy does not compute
# p-values or fold changes for it, so `pval`, `pval_adj` and `logfoldchange`
# come back as NaN while `gene` and `score` are populated as usual.
de_df = rank_genes_groups(adata, groupby="cell_type", method="logreg", n_genes=50)

# Filter significant genes
sig_df = filter_de_genes(de_df, alpha=0.05, log2fc_threshold=1.0)

# Filter by effect size (rank-biserial for Wilcoxon, Cohen's d for t-test)
sig_df = filter_de_genes(de_df, alpha=0.05, log2fc_threshold=1.0,
                         effect_size_col="rank_biserial", effect_size_threshold=0.3)

# Prepare data for volcano plot
vp_data = volcano_plot_data(de_df)
```

### CLI

```bash
# Wilcoxon test between two groups
scintilla de data/pbmc3k.h5ad \
    --group-col cell_type \
    --group1 Monocyte \
    --group2 "T Cell" \
    --method wilcoxon \
    --output results/de_results.csv

# t-test
scintilla de data/pbmc3k.h5ad \
    --group-col cell_type --group1 Monocyte --group2 "T Cell" \
    --method ttest

# Permutation test
scintilla de data/pbmc3k.h5ad \
    --group-col cell_type --group1 Monocyte --group2 "T Cell" \
    --method permutation
```

**`de` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input `.h5ad` |
| `--group-col` | required | Column in `obs` defining groups |
| `--group1` | required | First group name |
| `--group2` | required | Second group name |
| `--method` | `wilcoxon` | DE method: `wilcoxon`, `ttest`, `permutation` |
| `--output` | `None` | Save results to CSV |
| `--verbose` | `False` | Print progress |

---

## Cell-Type Annotation

### Strategies

| Strategy | Function | Notes |
|---|---|---|
| Marker-based annotation | `annotate_by_markers` | Score clusters against a marker dict |
| Find marker genes | `find_marker_genes` | Rank genes per cluster |
| Over-representation analysis | `ora_test` | Fisher's exact test against gene sets |
| Label transfer | `transfer_labels` | Transfer labels from a reference AnnData |

### Python API

```python
from scintilla import annotate_by_markers, find_marker_genes, ora_test, transfer_labels

# Define marker genes per cell type
markers = {
    "Monocyte": ["CD14", "LYZ", "CST3"],
    "T Cell":   ["CD3D", "CD3E", "IL7R"],
    "NK Cell":  ["GNLY", "NKG7"],
    "B Cell":   ["MS4A1", "CD79A"],
}

# Score clusters and assign cell types
adata = annotate_by_markers(adata, marker_dict=markers)
print(adata.obs["predicted_cell_type"].value_counts())

# Find top marker genes per cluster
markers_df = find_marker_genes(
    adata,
    groupby="leiden",
    method="wilcoxon",  # DE method used for ranking
    n_genes=50,
)
print(markers_df.head(20))

# Over-representation analysis
ora_results = ora_test(
    gene_list=["CD14", "LYZ", "CST3", "FCN1"],
    background=list(adata.var_names),
    gene_sets={"Myeloid": ["CD14", "LYZ", "CST3"], "Lymphoid": ["CD3D", "CD19"]},
)

# Transfer labels from a reference dataset
# Genes are matched by name, not by column position: the reference and query
# are intersected on `var_names` and subsetted in reference order. At least 10
# shared genes are required, and any precomputed `X_pca` is ignored because
# embeddings fitted on different datasets are not comparable.
adata_query = transfer_labels(
    reference_adata=adata_ref,
    query_adata=adata_query,
    label_col="cell_type",
)
```

### CLI

```bash
# Find marker genes per cluster (Leiden by default)
scintilla annotate data/pbmc3k_clustered.h5ad \
    --groupby leiden \
    --method wilcoxon \
    --n-genes 50 \
    --output results/annotated.h5ad

# Save annotated AnnData
scintilla annotate data/pbmc3k_clustered.h5ad \
    --groupby leiden --output annotated.h5ad
```

**`annotate` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input `.h5ad` |
| `--groupby` | `leiden` | Column for grouping (e.g., cluster labels) |
| `--method` | `wilcoxon` | DE method for ranking genes |
| `--n-genes` | `50` | Number of top genes per group |
| `--output` | `None` | Save annotated `.h5ad` |
| `--verbose` | `False` | Print progress |

---

## Batch Correction

scINTILLA benchmarks four batch correction strategies and selects the best one by batch-mixing ASW.

### Methods

| Method | Notes |
|---|---|
| ComBat | Linear regression-based correction; fast |
| Harmony | PCA-space iterative correction |
| BBKNN | Graph-based batch correction (no corrected expression) |
| Scanorama | Panoramic stitching across batches |

### Python API

```python
from scintilla import combat_correct, batch_asw, benchmark_batch_correction

# Apply ComBat correction
adata_corrected = combat_correct(adata, batch_key="batch")

# Compute batch ASW metric (higher = better batch mixing)
asw = batch_asw(adata_corrected, batch_key="batch", label_key="cell_type")
print(f"Batch ASW: {asw:.3f}")

# Benchmark all methods
result = benchmark_batch_correction(
    adata,
    batch_key="batch",
    label_key="cell_type",       # optional: used for bio-conservation score
    methods=["combat", "harmony", "bbknn", "scanorama"],  # default: all
    n_pcs=30,
)
print("Best method:", result["best_method"])
print(result["leaderboard"])
```

**Return values of `benchmark_batch_correction`**

| Key | Type | Description |
|---|---|---|
| `leaderboard` | DataFrame | Columns: `method`, `batch_asw`, `status` |
| `best_method` | str | Method with the best batch ASW |
| `corrected_adatas` | dict | `{method: corrected AnnData}` |

### CLI

```bash
# Benchmark all batch correction methods
scintilla batch-correct data/pbmc3k.h5ad --batch-key batch --label-key cell_type

# Save leaderboard
scintilla batch-correct data/pbmc3k.h5ad \
    --batch-key batch \
    --label-key cell_type \
    --methods combat harmony \
    --output results/batch_correction/ \
    --verbose
```

**`batch-correct` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input `.h5ad` |
| `--batch-key` | required | Column in `obs` with batch labels |
| `--label-key` | `None` | Column in `obs` with cell-type labels |
| `--methods` | all four | Subset of `combat harmony bbknn scanorama` |
| `--output` | `None` | Directory for output CSV |
| `--verbose` | `False` | Print progress |

---

## Benchmarking Utilities

### Method profiling

```python
from scintilla import profile_method, BenchmarkResult

def my_analysis(adata):
    from scintilla.clustering.kmeans import kmeans_clustering
    X = adata.obsm["X_pca"]
    return kmeans_clustering(X, n_clusters=8)

# profile_method is a decorator: it wraps a callable, and the wrapper returns
# BenchmarkResult(result, elapsed_seconds, peak_memory_mb).
result: BenchmarkResult = profile_method(my_analysis)(adata)
print(f"Wall time: {result.elapsed_seconds:.2f}s")
print(f"Peak memory: {result.peak_memory_mb:.1f} MB")
```

### Scalability sweep

```python
from scintilla import scalability_sweep

def analysis_fn(adata):
    from scintilla.clustering.kmeans import kmeans_clustering
    return kmeans_clustering(adata.obsm["X_pca"], n_clusters=8)

# Signature is (method_fn, adata, ...); sizes are given as fractions of the
# input, not absolute cell counts.
results_df = scalability_sweep(
    analysis_fn,
    adata,
    fractions=[0.1, 0.25, 0.5, 1.0],
)
print(results_df[["fraction", "n_cells", "elapsed_seconds", "peak_memory_mb"]])
```

### Seed-stability test

```python
from scintilla import seed_stability_test

# method_fn is called as method_fn(data, random_state=seed); metric_fn reduces
# its return value to the single number whose stability is measured.
from scintilla.clustering.kmeans import kmeans_clustering

stability_df = seed_stability_test(
    method_fn=lambda data, random_state: kmeans_clustering(
        data.obsm["X_pca"], n_clusters=8, random_state=random_state
    )[0],
    data=adata,
    metric_fn=lambda labels: float(len(set(labels))),
    n_seeds=5,
    compute_icc=True,       # proper ICC(1,1) via pingouin (default)
    icc_method="pingouin",  # or "legacy" for original approximate formula
)
print(stability_df)
print(f"ICC: {stability_df.attrs['icc']:.3f}")
```

### Benchmark report

```python
from scintilla import BenchmarkReport

report = BenchmarkReport()
report.add_result("clustering", "Leiden_0.5", {"ari": 0.82})
report.add_result("clustering", "KMeans_8",   {"ari": 0.71})
report.add_result("classification", "RF",     {"accuracy": 0.94})

print(report.summary_table())
report.export("results/benchmark_report/")  # saves CSV + JSON summary
```

### Full benchmark CLI

```bash
# Run full clustering + classification benchmark
scintilla benchmark-all data/pbmc3k.h5ad \
    --cell-type-col cell_type \
    --output-dir results/full_benchmark/ \
    --verbose
```

**`benchmark-all` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Path to input `.h5ad` |
| `--cell-type-col` | `cell_type` | Column with ground-truth cell-type labels |
| `--output-dir` | `benchmark_output` | Directory for results |
| `--verbose` | `False` | Print progress |

---

## Robust Statistics

scINTILLA includes a comprehensive suite of robust statistical methods that can be enabled across all benchmarking modules. These are collected in `scintilla.statistical_tests` and surfaced as top-level imports.

### Key capabilities

| Feature | Description |
|---|---|
| **BCa bootstrap CIs** | Non-parametric confidence intervals for any metric (`method="bca"` or `"percentile"` fast-path for large *n*) |
| **.632+ bootstrap** | Bias-corrected performance estimator with analytical no-information rate (`no_info_method="analytical"`) or averaged permutations |
| **Rank aggregation** | Borda count for combining multiple metric rankings |
| **Effect sizes** | Cohen's d, Hedges' g, rank-biserial, Cliff's delta - always included in DE output |
| **Adaptive PCA** | Gavish-Donoho / Marchenko-Pastur thresholds for data-driven component selection (warns when γ > 0.8) |
| **Adaptive resolution** | NVI-stability-based Leiden/Louvain resolution selection (normalise by `log(n)` or `max(H(A), H(B))`) |
| **Auto DBSCAN eps** | Kneedle elbow on k-distance curve |
| **Permutation tests** | Pairwise method comparison via permutation or McNemar's test |
| **ICC** | Intraclass correlation coefficient for seed stability (via `pingouin` for proper ICC(1,1) or `"legacy"` mode) |
| **Hardened bootstrap** | `bootstrap_clustering_metrics` handles rare-cluster dropout gracefully with discard-rate logging |

### Quick example

```python
from scintilla import AnalysisConfig

# Enable all robust features at once
cfg = AnalysisConfig.robust()
# → bootstrap_ci=True, scoring_method="borda", auto_pca_components="gavish_donoho",
#   adaptive_resolution=True, auto_eps=True

result = sc.unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
```

Or enable individual features:

```python
from scintilla import bca_bootstrap_ci, borda_count, rank_biserial, cohens_d
from scintilla import gavish_donoho_threshold, adaptive_resolution_search
from scintilla import pairwise_method_comparison, bio_conservation_score

# BCa CI with percentile fast-path (faster for large datasets)
ci = bca_bootstrap_ci(data, stat_fn=np.mean, B=2000, method="percentile")

# .632+ with analytical no-information rate (exact, zero-variance)
from scintilla import dot632plus_bootstrap
result = dot632plus_bootstrap(
    model, X, y, metric_fn, B=200,
    no_info_method="analytical",  # or "permutation" with n_permutations=50
)
```

See the [User Guide](GUIDE.md#robust-statistics) for detailed API documentation of all statistical functions.

---

## CLI Reference

### Overview

```
scintilla <subcommand> [options]
```

| Subcommand | Description |
|---|---|
| `eda` | Exploratory data analysis & summary statistics |
| `preprocess` | Apply a normalisation transform |
| `normalise` | Benchmark all normalisation transforms |
| `feature-select` | PCA-loadings feature selection |
| `reduce` | Dimensionality reduction (UMAP / t-SNE / Diffusion Map) |
| `cluster` | Benchmark unsupervised clustering methods |
| `classify` | Benchmark supervised classifiers |
| `de` | Differential expression analysis |
| `annotate` | Find marker genes per cluster |
| `batch-correct` | Benchmark batch correction methods |
| `run-all` | Full pipeline (clustering + classification) |
| `benchmark-all` | Full clustering + classification benchmark |
| `generate-config` | Write a commented default YAML config to disk |

### Getting help

```bash
# Top-level help
scintilla --help

# Subcommand help
scintilla cluster --help
scintilla classify --help
```

---

## Configuration & AnalysisConfig

### Global constants (`scintilla/config.py`)

These constants provide package defaults. Configure an analysis with
`AnalysisConfig` or an individual function's public arguments rather than
mutating `scintilla.config` after import: the constants are read once, when
each module is imported, so reassigning `scintilla.config.RANDOM_SEED`
afterwards has no effect on code that already imported it.

| Constant | Default | Description |
|---|---|---|
| `RANDOM_SEED` | `42` | Default random seed |
| `DEFAULT_TEST_SIZE` | `0.2` | Train/test split fraction |
| `DEFAULT_N_PCA_COMPS` | `30` | Default PCA components |
| `DEFAULT_VARIANCE_THRESHOLD` | `0.80` | Variance explained threshold |
| `DISTANCE_METRICS` | 7 metrics | Distance metrics for clustering grid search |
| `LINKAGE_METHODS` | 4 methods | Linkage methods for hierarchical clustering |
| `DBSCAN_EPS_RANGE` | 5 values | eps grid for DBSCAN |
| `LEIDEN_RESOLUTIONS` | 8 values | Resolution sweep for Leiden / Louvain |
| `HDBSCAN_MIN_CLUSTER_SIZE_RANGE` | `[10, 20, 50]` | `min_cluster_size` grid for HDBSCAN |
| `HDBSCAN_MIN_SAMPLES_RANGE` | `[None, 5]` | `min_samples` grid for HDBSCAN |
| `SPECTRAL_N_CLUSTERS_RANGE` | 9 values | Cluster count grid for Spectral |
| `FEATURE_SELECTION_METHODS` | 6 methods | All feature-selection strategies |
| `TRANSFORMATION_BENCHMARK_WEIGHTS` | 5 weights | Composite-score weights |

### Reproducibility and seeding

Every stochastic public function takes a `random_state` argument, and every
pipeline entry point forwards its own `random_state` down to the methods it
calls. This is the supported way to control seeding — it replaces the old
advice to reassign `scintilla.config.RANDOM_SEED`, which never worked for the
modules that had already imported the constant.

```python
import scintilla as si

# Seed one call.
result = si.unsupervised_analysis(adata, cell_type_col="cell_type", random_state=0)

# Seed a whole analysis through the config.
from scintilla import AnalysisConfig
cfg = AnalysisConfig(random_seed=0)
result = si.unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)

# Sweep seeds to check a result is not an artefact of one draw.
aris = [
    si.unsupervised_analysis(
        adata, cell_type_col="cell_type", random_state=seed, verbose=False
    )["results_df"]["ari"].max()
    for seed in range(10)
]
```

Precedence is the same everywhere: **an explicit argument wins, then
`config`, then the historical default**. Passing `random_state=None` is not a
way to request "unseeded" — it means "not supplied", so the config value or
the default applies.

Functions exposing `random_state` include `run_pca`, `run_umap`, `run_tsne`,
`run_diffusion_map`, `leiden_clustering`, `louvain_clustering`,
`spectral_clustering`, `consensus_clustering`, `check_normality`,
`rank_genes_groups`, `find_marker_genes`, `transfer_labels`,
`harmony_correct`, `bbknn_correct`, `scanorama_correct`,
`mi_feature_selection`, `boruta_selection`, `mrmr_selection`,
`benchmark_transformations`, `benchmark_feature_selection`,
`benchmark_clustering_methods`, `benchmark_batch_correction`,
`benchmark_models_comprehensive`, `unsupervised_analysis` and
`supervised_analysis`.

> **Note on comparing against pre-0.2 results.** Several functions previously
> relied on the scanpy/scikit-learn default seed rather than scINTILLA's, and
> one silhouette computation was genuinely unseeded. Seeding them consistently
> changes their numbers: `run_umap`, `run_tsne`, `run_diffusion_map`,
> `run_force_directed`, `hvg_sensitivity_analysis`, `graph_connectivity`, the
> internal PCA inside `harmony_correct`/`bbknn_correct`, and the silhouette
> subsample in `benchmark_transformations` for datasets above 1 000 cells.
> Clustering, classification, differential expression and feature selection are
> unaffected — they already used seed 42.

---

### AnalysisConfig

`AnalysisConfig` is a dataclass that lets you control exactly which methods run in each pipeline step and override key hyperparameter grids - all in one place. It can be used programmatically or stored as a YAML file.

#### Presets

```python
from scintilla import AnalysisConfig

# All methods, all grids (default pipeline behaviour)
cfg = AnalysisConfig.default()

# Fast preset: only quick methods, no SHAP
cfg = AnalysisConfig.fast()

# Robust preset: bootstrap CIs, rank aggregation, adaptive methods
cfg = AnalysisConfig.robust()
```

**`AnalysisConfig` fields**

| Field | Default (full) | Fast preset | Description |
|---|---|---|---|
| `clustering_methods` | all 8 algorithms | `["kmeans","leiden"]` | Which clustering algorithms to run |
| `leiden_resolutions` | `[0.1,0.3,0.5,0.8,1.0,1.5,2.0,3.0]` | `[0.5,1.0]` | Resolution sweep for Leiden |
| `louvain_resolutions` | `[0.1,0.3,0.5,0.8,1.0,1.5,2.0,3.0]` | `[0.5,1.0]` | Resolution sweep for Louvain |
| `hdbscan_min_cluster_sizes` | `[10,20,50]` | `[20]` | HDBSCAN min_cluster_size grid |
| `hdbscan_min_samples` | `[None,5]` | `[None]` | HDBSCAN min_samples grid |
| `spectral_n_clusters_range` | `[2..10]` | `[]` | Spectral clustering k grid |
| `classifiers` | all 10 classifiers | `["LogReg","RF","kNN"]` | Classifiers to train |
| `include_shap` | `True` | `False` | Compute SHAP importances for best model |
| `feature_selection_methods` | `["pca_loadings","mutual_information"]` | same | Methods for feature-selection benchmark |
| `n_features` | `50` | `50` | Features to select per method |
| `random_seed` | `42` | `42` | Random seed |
| `test_size` | `0.2` | `0.2` | Train/test split fraction |
| `n_pca_comps` | `30` | `30` | PCA components |
| `cv_folds` | `5` | `3` | Cross-validation folds for classification |
| `verbose` | `True` | `True` | Verbose output |
| `bootstrap_ci` | `False` | `False` | Enable BCa bootstrap confidence intervals |
| `n_bootstrap` | `2000` | `2000` | Number of bootstrap resamples |
| `scoring_method` | `"weighted"` | `"weighted"` | Transformation scoring (`"weighted"`/`"borda"`); batch correction treats `"weighted"` as its `"single_metric"` default |
| `auto_pca_components` | `None` | `None` | Adaptive PCA: `"gavish_donoho"` or `"marchenko_pastur"` |
| `adaptive_resolution` | `False` | `False` | NVI-based adaptive resolution for Leiden/Louvain |
| `auto_eps` | `False` | `False` | Data-driven DBSCAN eps estimation |
| `classification_estimator` | `"cv"` | `"cv"` | `"cv"`, `"holdout"`, or `"bootstrap_632plus"` |
| `no_info_method` | `"analytical"` | `"analytical"` | No-information rate for .632+: `"analytical"` or `"permutation"` |
| `mp_sigma_method` | `"median"` | `"median"` | MP noise-variance estimation: `"median"` or `"trimmed_mean"` |

#### Using a config in Python

```python
from scintilla import AnalysisConfig, unsupervised_analysis, supervised_analysis

# Programmatic override
cfg = AnalysisConfig(
    clustering_methods=["kmeans", "leiden", "louvain"],
    leiden_resolutions=[0.3, 0.5, 1.0],
    classifiers=["LogReg", "RF", "SVM"],
    include_shap=False,
)

result   = unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
cls_result = supervised_analysis(adata, target_col="cell_type", config=cfg)
```

#### YAML workflow

```bash
# Step 1 – generate a fully commented template
scintilla generate-config --output my_config.yaml

# Step 2 – edit the file to enable/disable methods
# Step 3 – run any pipeline using the config
scintilla run-all data.h5ad --target-col cell_type --config my_config.yaml --output-dir results/
```

In Python:

```python
cfg = AnalysisConfig.from_yaml("my_config.yaml")
result = unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
cfg.to_yaml("my_config_v2.yaml")   # serialise back
```

**`generate-config` options**

| Option | Default | Description |
|---|---|---|
| `--output` | `scintilla_config.yaml` | Path to write the template YAML |

#### Priority rules

Explicit keyword arguments always win over config values:

```python
# config says classifiers=["LogReg","RF"], but models= overrides it
cfg = AnalysisConfig.fast()
result = supervised_analysis(adata, target_col="cell_type",
                              models=["SVM", "MLP"], config=cfg)
```

---

## Full Pipeline Walkthrough

This end-to-end example shows a typical scINTILLA workflow starting from a raw count matrix.

### 1. Load data

```python
import scintilla as sc
from scintilla.io.loaders import auto_detect_format
from scintilla.preprocessing.pca import run_pca

# Load h5ad (or CSV/TSV)
adata = auto_detect_format("data/pbmc3k_raw.h5ad")
print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
```

### 2. EDA

```python
from scintilla.eda.summary import dataset_summary
import json

summary = dataset_summary(adata)
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
```

### 3. Benchmark and apply the best normalisation

```python
from scintilla.preprocessing.benchmark import benchmark_transformations

results_df, best_name, adata_norm = benchmark_transformations(adata, verbose=True)
print(f"Best normalisation: {best_name}")
print(results_df[["transform", "composite_score"]].head(5).to_string(index=False))
```

### 4. Reduce dimensions

```python
# PCA (required for most downstream steps)
adata_pca = run_pca(adata_norm, n_comps=30)

# UMAP for visualisation
from scintilla.dimensionality_reduction import run_umap
adata_pca = run_umap(adata_pca, use_rep="X_pca")
```

### 5. Select features

```python
from scintilla.feature_selection import select_hvg

adata_hvg = select_hvg(adata_pca, n_top_genes=2000)
```

### 6. Cluster

```python
result = sc.unsupervised_analysis(
    adata_hvg,
    cell_type_col="cell_type",
    run_pca_first=False,  # PCA already done
    store_labels=True,    # best labels → adata_hvg.obs["scintilla_cluster"]
    verbose=True,
)
print("Best clustering method:", result["best_method"])
result["fig"].savefig("results/clustering_ari.png", dpi=150)

# Best labels are already in adata_hvg.obs["scintilla_cluster"]
print(adata_hvg.obs["scintilla_cluster"].value_counts())
```

### 7. Classify

```python
cls_result = sc.supervised_analysis(
    adata_hvg,
    target_col="cell_type",
    include_shap=True,
    verbose=True,
)
print(f"Best classifier: {cls_result['best_model_name']}")
for name, res in cls_result["all_results"].items():
    m = res["metrics"]
    print(f"  {name:20s}  acc={m.get('accuracy',0):.3f}  f1={m.get('f1',0):.3f}")
```

### 8. Differential expression

```python
from scintilla import wilcoxon_de, filter_de_genes

de_df = wilcoxon_de(adata_hvg, group_col="cell_type", group1="Monocyte", group2="T Cell")
sig_df = filter_de_genes(de_df, alpha=0.05, log2fc_threshold=1.0)
print(f"Significant DE genes: {len(sig_df)}")
print(sig_df.head(10))
```

### 9. Annotate cell types

```python
from scintilla import find_marker_genes

markers_df = find_marker_genes(adata_hvg, groupby="leiden", n_genes=20)
print(markers_df.head(40))
```

### 10. Batch correction (if multi-batch)

```python
from scintilla import benchmark_batch_correction

bc_result = benchmark_batch_correction(
    adata_hvg,
    batch_key="batch",
    label_key="cell_type",
)
print("Best batch correction:", bc_result["best_method"])
adata_corrected = bc_result["corrected_adatas"][bc_result["best_method"]]
```

### 11. Save all results

```python
from scintilla.io.exporters import save_anndata, save_results_csv, save_results_json

save_anndata(adata_corrected, "results/final_annotated.h5ad")
save_results_csv(result["results_df"], "results/clustering_benchmark.csv")
save_results_json(
    {"best_classifier": cls_result["best_model_name"],
     "best_clustering": result["best_method"]},
    "results/summary.json",
)
```

### Equivalent one-liner via CLI

```bash
scintilla run-all data/pbmc3k_raw.h5ad \
    --target-col cell_type \
    --output-dir results/ \
    --verbose
```

---

## License

See [LICENSE](LICENSE) for details.
