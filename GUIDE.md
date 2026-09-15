# scINTILLA User Guide

This guide covers every module in scINTILLA in depth: what it does, when to use it, all parameters with their defaults, and worked examples.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Environments](#installation--environments)
3. [Data I/O](#data-io)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Preprocessing & Normalisation](#preprocessing--normalisation)
6. [Feature Selection](#feature-selection)
7. [Dimensionality Reduction](#dimensionality-reduction)
8. [Unsupervised Clustering](#unsupervised-clustering)
9. [Supervised Classification](#supervised-classification)
10. [Differential Expression](#differential-expression)
11. [Cell-Type Annotation](#cell-type-annotation)
12. [Batch Correction](#batch-correction)
13. [Benchmarking Utilities](#benchmarking-utilities)
14. [AnalysisConfig System](#analysisconfig-system)
15. [Robust Statistics](#robust-statistics)
16. [CLI Reference](#cli-reference)
17. [Performance Tips](#performance-tips)
18. [Full Worked Example](#full-worked-example)

---

## Overview

scINTILLA (*Single-Cell INTegrated Inference, Labelling, and Landscape Analysis*) is an end-to-end Python pipeline for single-cell RNA-seq (scRNA-seq) analysis built on top of AnnData. It wraps the most common analysis steps — normalisation, feature selection, clustering, classification, differential expression, annotation, and batch correction — under a unified API and CLI.

**Key design principles:**

- All functions accept and return `AnnData` objects.
- Every benchmark module returns a leaderboard DataFrame and (where applicable) a best AnnData.
- The `AnalysisConfig` dataclass lets you swap method sets and hyperparameter grids without changing code.
- The CLI exposes every major operation as a subcommand so pipelines can be scripted without Python.

---

## Installation & Environments

### Core install

```bash
pip install .
```

### Full install (all optional packages)

```bash
pip install ".[full]"
```

This installs optional extras including: `umap-learn`, `hdbscan`, `harmonypy`, `bbknn`, `scanorama`, `xgboost`, `lightgbm`, `Boruta`, `mrmr-selection`, `shap`, `psutil`, `kneed`.

### Editable development install

```bash
pip install -e .
```

---

## Data I/O

### Loading data

```python
from scintilla.io.loaders import load_h5ad, load_csv, auto_detect_format, ensure_anndata

# Native AnnData HDF5
adata = load_h5ad("data/pbmc3k.h5ad")

# CSV where cells are rows, genes are columns
adata = load_csv("data/counts.csv")

# CSV where genes are rows (transpose)
adata = load_csv("data/counts_genes_rows.csv", transpose=True)

# Auto-detect format (.h5ad, .csv, .tsv, .txt, .h5mu)
adata = auto_detect_format("data/pbmc3k.h5ad")

# Convert a DataFrame to AnnData
import pandas as pd
df = pd.read_csv("data/counts.csv", index_col=0)
adata = ensure_anndata(df)
```

**`load_csv` parameters**

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Path to file |
| `index_col` | `0` | Column used as the row index (cell barcodes) |
| `transpose` | `False` | Transpose so cells are rows |
| `**kwargs` | — | Forwarded to `pandas.read_csv` |

**`ensure_anndata` parameters**

| Parameter | Default | Description |
|---|---|---|
| `data` | required | `pd.DataFrame`, `np.ndarray`, or `AnnData` |
| `target_col` | `None` | Column to retain in `obs` even when numeric |

### Saving data

```python
from scintilla.io.exporters import save_anndata, save_results_csv, save_results_json

save_anndata(adata, "output/processed.h5ad")
save_results_csv(results_df, "output/results.csv")
save_results_json({"best_model": "RF", "accuracy": 0.95}, "output/summary.json")
```

### Supported formats

| Format | Extensions |
|---|---|
| HDF5 AnnData | `.h5ad` |
| CSV / TSV / TXT | `.csv`, `.tsv`, `.txt` |
| MuData | `.h5mu` (requires `pip install mudata`) |

---

## Exploratory Data Analysis (EDA)

### Python API

```python
from scintilla.eda.summary import dataset_summary

summary = dataset_summary(adata)
```

The returned dictionary contains: `n_cells`, `n_genes`, `sparsity`, `obs_columns`, `var_columns`, and (if present in `obs`) `cell_type_counts`.

### CLI

```bash
scintilla eda data/pbmc3k.h5ad
scintilla eda data/pbmc3k.h5ad --group-col cell_type --output results/eda.json
```

**`eda` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input file (`.h5ad`, `.csv`) |
| `--group-col` | `None` | `obs` column to group by |
| `--output` | `None` | Save summary as JSON |

---

## Preprocessing & Normalisation

### Available Transforms

scINTILLA ships 14 normalisation strategies accessible via `TRANSFORM_REGISTRY`:

| Key | Description |
|---|---|
| `log_shift_size_factor` | log(x / size_factor + 1) — standard scRNA-seq |
| `arcsinh_transform` | arcsinh(α·x), α=0.05 — common for CyTOF / CITE-seq |
| `log_alpha_transform` | log(α·x + 1), α=0.05 |
| `log_cpm_transform` | log(CPM + 1) — counts per million |
| `log_shift_scale_by_std` | log1p then divide each gene by its standard deviation |
| `log_shift_size_factor_hvg` | log-shift then top 35% most-variable genes |
| `log_shift_size_factor_z` | log-shift then z-score per gene |
| `log_shift_hvg_z` | log-shift + HVG + z-score |
| `normalise_scran` | Scran-style geometric-mean deconvolution |
| `normalise_tmm` | TMM normalisation (Robinson & Oshlack 2010) |
| `box_cox_transform` | Box-Cox per gene (parallelised over genes) |
| `pearson_residuals_transform` | Analytic Pearson residuals |
| `glm_pca_transform` | Poisson GLM-PCA via TruncatedSVD |
| `sanity_transform` | Bayesian-estimation approximation (SANITY-like) |

### Applying a transform

```python
from scintilla.preprocessing.transformations import TRANSFORM_REGISTRY, arcsinh_transform

# Via the registry
adata_norm = TRANSFORM_REGISTRY["log_shift_size_factor"](adata)

# With custom parameters
adata_norm = arcsinh_transform(adata, alpha=0.1)
```

**Parameterised transforms**

| Function | Parameter | Default | Description |
|---|---|---|---|
| `arcsinh_transform` | `alpha` | `0.05` | Scaling factor |
| `log_alpha_transform` | `alpha` | `0.05` | Scaling factor |
| `pearson_residuals_transform` | `theta` | `100.0` | Overdispersion parameter |
| `glm_pca_transform` | `n_components` | `50` | Latent dimensions |
| `box_cox_transform` | `lambda_val` | `None` | Box-Cox λ (estimated per gene if `None`) |

### Benchmarking normalisation

Automatically ranks all transforms using a composite score of kNN overlap, PCA variance preservation, silhouette score, and normality.

```python
from scintilla.preprocessing.benchmark import benchmark_transformations

results_df, best_name, best_adata = benchmark_transformations(adata, verbose=True)
print(f"Best transform: {best_name}")
print(results_df[["transform", "status", "composite_score"]])
```

Every transform gets a row. One that raised carries `status="failed"` and a
`failure_reason` instead of vanishing. When *all* of them fail, `best_name` and
`best_adata` are `None` and `results_df` still lists each failure — check
`best_name is not None` before using the returned object.

**`benchmark_transformations` parameters**

Arguments left at `None` fall back to `config`, then to the historical default
shown in brackets.

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `transformations` | `None` | `{name: callable}` to benchmark; `None` uses the built-in set |
| `weights` | `None` | Composite-score weights; `None` uses `TRANSFORMATION_BENCHMARK_WEIGHTS` |
| `n_pca_components` | `None` | PCs for the metric space [`config.n_pca_comps`, else `20`] |
| `max_k` | `None` | Neighbourhood size for the kNN-overlap metric |
| `cell_type_col` | `None` | Ground-truth labels for the silhouette metric |
| `verbose` | `None` | Print progress [`config.verbose`, else `True`] |
| `scoring_method` | `None` | `"weighted"` (weighted sum) or `"borda"` (rank aggregation); unknown values raise `ValueError` [`config.scoring_method`, else `"weighted"`] |
| `bootstrap_ci` | `None` | Add BCa bootstrap 95 % confidence intervals [`config.bootstrap_ci`, else `False`] |
| `n_bootstrap` | `None` | Bootstrap resamples when `bootstrap_ci=True` [`config.n_bootstrap`, else `200`] |
| `random_state` | `None` | Seed for PCA, KMeans and the silhouette subsample [`config.random_seed`, else `42`] |
| `config` | `None` | `AnalysisConfig` supplying the defaults above |

> Passing a default `AnalysisConfig` changes two of these from the function's
> own defaults: `n_pca_components` becomes 30 and `n_bootstrap` becomes 2000.

> **Performance note:** When `bootstrap_ci=True` each bootstrap replicate re-applies every transformation and re-runs PCA and kNN.  For datasets with >10 000 cells this can be extremely slow.  Consider sub-sampling first or using `verbose=True` to monitor progress.

**Score weights** (used when `scoring_method="weighted"`; editable in `scintilla/config.py` under `TRANSFORMATION_BENCHMARK_WEIGHTS`):

| Metric | Weight |
|---|---|
| kNN overlap | 0.35 |
| PCA variance preservation | 0.30 |
| Silhouette score | 0.25 |
| Shapiro-Wilk normality | 0.05 |
| Anderson-Darling normality | 0.05 |

When `scoring_method="borda"`, the transform ranking is determined by Borda count across all individual metrics instead of a single weighted score. This eliminates sensitivity to arbitrary weight choices.

### Normality testing

```python
from scintilla.preprocessing.normality import check_normality

is_normal, report = check_normality(adata, sample_size=500, threshold=0.3)
```

**`check_normality` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `sample_size` | `500` | Cells subsampled per gene for Shapiro-Wilk |
| `alpha` | `0.05` | Significance level for the per-gene tests |
| `threshold` | `0.3` | Minimum fraction of genes that must pass to call data "normal" |
| `n_replicates` | `1` | Repeat the subsampled test this many times and take the median pass fraction |
| `correction` | `None` | Multiple-testing correction: `"bh"` for Benjamini-Hochberg FDR control |
| `random_state` | `42` | Seed for the per-gene subsample |

> Genes whose Shapiro-Wilk test cannot be computed count as **not passing**, and
> the pass fraction is taken over every gene rather than only the ones that
> succeeded. One warning names the failure and the report carries
> `shapiro_failure_count`, `shapiro_failure_events` and
> `shapiro_failed_features`.

> At large *n* the Shapiro-Wilk test gains extreme statistical power and will reject even well-normalised scRNA-seq data; subsampling to 500 cells per gene gives a practically meaningful test. When testing many genes, pass `correction="bh"` to apply Benjamini-Hochberg FDR correction to the per-gene p-values.

### PCA

```python
from scintilla.preprocessing.pca import run_pca

adata = run_pca(adata, n_comps=30)
# Result stored in adata.obsm["X_pca"]

# Adaptive component selection — let the data decide how many components to keep
adata = run_pca(adata, n_comps=50, auto_components="gavish_donoho")
# Components trimmed to the Gavish-Donoho optimal threshold
```

**`run_pca` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `n_comps` | `30` | Number of PCA components (upper bound when `auto_components` is set) |
| `variance_threshold` | `None` | Trim components to this cumulative explained-variance fraction |
| `auto_components` | `None` | `"gavish_donoho"` or `"marchenko_pastur"` for data-driven component selection |
| `mp_sigma_method` | `"median"` | Noise-variance estimation for Marchenko-Pastur: `"median"` (fast, slightly conservative) or `"trimmed_mean"` (iterative, more accurate) |
| `random_state` | `42` | Seed for the PCA solver |

When `auto_components` is set, PCA is initially computed with `n_comps` components, then trimmed to the optimal number determined by the chosen method:
- **Gavish-Donoho**: Applies the universal singular-value threshold $\omega(\beta)\cdot\sigma$ to discard noise components.
- **Marchenko-Pastur**: Retains components whose eigenvalues exceed the upper edge of the Marchenko-Pastur distribution.  The noise variance σ² can be estimated via the median eigenvalue (`mp_sigma_method="median"`, default) or an iterative trimmed mean of bulk eigenvalues (`mp_sigma_method="trimmed_mean"`), which is more accurate when the signal-to-noise ratio is moderate.

> **Note:** When $p/n > 0.8$ (i.e. the number of genes is close to the number of cells), the Marchenko-Pastur noise-variance estimate becomes unreliable because $(1 - \sqrt{\gamma})^2 \to 0$.  In this regime a warning is emitted and the **Gavish-Donoho** method is recommended instead.

---

## Feature Selection

### Methods

| Method | Short name | Function | Notes |
|---|---|---|---|
| PCA loadings | `pca_loadings` | `extract_top_genes_per_pc` | Top genes per principal component |
| Highly variable genes (Seurat v3) | — | `select_hvg(method="seurat_v3")` | Standard HVG selection |
| HVG (Pearson residuals) | — | `select_hvg(method="pearson_residuals")` | — |
| Mutual information | `mutual_information` | `mi_feature_selection` | MI between each gene and labels |
| Boruta | `boruta` | `boruta_selection` | Shadow-feature RF selection |
| mRMR | `mrmr` | `mrmr_selection` | Min-Redundancy Max-Relevance |

### PCA loadings

```python
from scintilla.feature_selection.pca_loadings import (
    extract_top_genes_per_pc,
    build_reduced_dataset,
    validate_reduced_set,
)

gene_list = extract_top_genes_per_pc(adata, n_per_pc=9)
adata_reduced = build_reduced_dataset(adata, gene_list)
```

**`extract_top_genes_per_pc` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData (must have `obsm["X_pca"]`) |
| `n_pcs` | `None` | Principal components to draw genes from; `None` uses all available |
| `n_per_pc` | `9` | Top genes to extract per PC |
| `random_state` | `42` | Seed for the PCA computed when `obsm["X_pca"]` is missing |

### Highly variable genes (HVG)

```python
from scintilla.feature_selection import select_hvg

adata_hvg = select_hvg(adata, n_top_genes=2000, method="seurat_v3")
```

**`select_hvg` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `n_top_genes` | `2000` | Number of HVGs to select |
| `span` | `0.3` | Loess span used by the `seurat_v3` flavour |
| `method` | `"seurat_v3"` | Method: `"seurat_v3"` or `"pearson_residuals"` |

### Mutual information

```python
from scintilla.feature_selection import mi_feature_selection

import numpy as np

X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
y = adata.obs["cell_type"].values
idx, scores = mi_feature_selection(X, y, n_features=50)
genes = adata.var_names[idx]
```

**`mi_feature_selection` parameters**

These low-level selectors operate on arrays, not on AnnData, and return
`(selected_indices, scores)`.

| Parameter | Default | Description |
|---|---|---|
| `X` | required | `(n_cells, n_genes)` expression matrix |
| `y` | required | `(n_cells,)` class labels |
| `n_features` | `100` | Number of top genes to return |
| `random_state` | `42` | Seed for the mutual-information estimator |

### Boruta

```python
from scintilla.feature_selection import boruta_selection

mask, ranking = boruta_selection(X, y, max_iter=20, n_estimators=50)
genes = adata.var_names[mask]
```

**`boruta_selection` parameters**

Returns `(support_mask, ranking)`.

| Parameter | Default | Description |
|---|---|---|
| `X` | required | `(n_cells, n_genes)` expression matrix |
| `y` | required | `(n_cells,)` class labels |
| `n_estimators` | `50` | Trees in the internal Random Forest |
| `max_iter` | `20` | Maximum shadow-feature iterations |
| `random_state` | `42` | Seed for the internal Random Forest |

> Requires `pip install Boruta`. Use with caution on datasets >10 000 cells — the algorithm is O(n_estimators × n_iter).

### mRMR

```python
from scintilla.feature_selection import mrmr_selection

idx, scores = mrmr_selection(X, y, n_features=50)
genes = adata.var_names[idx]
```

**`mrmr_selection` parameters**

Returns `(selected_indices, scores)`.

| Parameter | Default | Description |
|---|---|---|
| `X` | required | `(n_cells, n_genes)` expression matrix |
| `y` | required | `(n_cells,)` class labels |
| `n_features` | `50` | Number of features to select |
| `random_state` | `42` | Seed for the subsample and MI estimator |

> Internally subsamples to 5 000 cells when the dataset is larger, and pre-bins all features once before the greedy selection loop.

### Benchmarking feature selection

```python
from scintilla.feature_selection.benchmark import benchmark_feature_selection

results_df = benchmark_feature_selection(
    adata,
    target_col="cell_type",
    methods=["pca_loadings", "mutual_information"],   # default
    n_features=50,
)
```

**`benchmark_feature_selection` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `target_col` | required | Column in `obs` with class labels |
| `methods` | `None` | Methods to compare; `None` uses `["pca_loadings", "mutual_information"]` |
| `n_features` | `None` | Features selected per method [`config.n_features`, else `50`] |
| `test_size` | `None` | Hold-out fraction for the downstream classifier [`config.test_size`, else `0.2`] |
| `random_state` | `None` | Seed for the split, PCA and every selector [`config.random_seed`, else `42`] |
| `config` | `None` | `AnalysisConfig` supplying the defaults above |

Available method keys: `"pca_loadings"`, `"mutual_information"`, `"boruta"`, `"mrmr"`.

### HVG sensitivity analysis

Sweep over multiple values of `n_top_genes` for HVG selection and evaluate the downstream impact on clustering:

```python
from scintilla.feature_selection.benchmark import hvg_sensitivity_analysis

sensitivity_df = hvg_sensitivity_analysis(
    adata,
    target_col="cell_type",
    n_top_genes_values=[500, 1000, 2000, 3000, 5000],
    clustering_method="leiden",
    resolution=0.5,
)
# Returns DataFrame with columns: n_top_genes, ari, silhouette, n_clusters, n_hvg_genes
```

**`hvg_sensitivity_analysis` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `target_col` | required | Ground-truth label column |
| `n_top_genes_values` | `None` | List of HVG counts to sweep; `None` uses a built-in grid |
| `clustering_method` | `"leiden"` | Clustering algorithm to evaluate |
| `resolution` | `1.0` | Resolution for Leiden/Louvain |
| `random_state` | `42` | Seed for PCA, neighbours and the clustering step |

### CLI

```bash
scintilla feature-select data/pbmc3k.h5ad --n-per-pc 9 --output genes.csv
```

**`feature-select` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input `.h5ad` |
| `--n-per-pc` | `9` | Top genes per PC |
| `--output` | `None` | Save gene list to CSV |
| `--config` | `None` | Path to YAML config |
| `--fast` | `False` | Use fast preset |

---

## Dimensionality Reduction

### Methods

| Method | Function | Output key |
|---|---|---|
| UMAP | `run_umap` | `adata.obsm["X_umap"]` |
| t-SNE | `run_tsne` | `adata.obsm["X_tsne"]` |
| Diffusion Map | `run_diffusion_map` | `adata.obsm["X_diffmap"]` |
| Force-directed graph | `run_force_directed` | `adata.obsm["X_draw_graph_fa"]` |

### Python API

```python
from scintilla.dimensionality_reduction import run_umap, run_tsne, run_diffusion_map
from scintilla.dimensionality_reduction.force_directed import run_force_directed

adata = run_umap(adata, use_rep="X_pca", n_neighbors=15, min_dist=0.5)
adata = run_tsne(adata, use_rep="X_pca", perplexity=30)
adata = run_diffusion_map(adata, use_rep="X_pca", n_comps=10)
adata = run_force_directed(adata)
```

**`run_umap` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `use_rep` | `"X_pca"` | Key in `obsm` to use as input |
| `n_neighbors` | `15` | Number of neighbours for the kNN graph |
| `min_dist` | `0.5` | Minimum distance between points in the embedding |
| `n_components` | `2` | Output dimensions |
| `random_state` | `42` | Seed for the fallback PCA, the kNN graph and UMAP |

**`run_tsne` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `use_rep` | `"X_pca"` | Key in `obsm` to use as input |
| `perplexity` | `30.0` | t-SNE perplexity |
| `n_components` | `2` | Output dimensions |
| `random_state` | `42` | Seed for the fallback PCA and t-SNE |

**`run_diffusion_map` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `use_rep` | `"X_pca"` | Key in `obsm` to use as input |
| `n_comps` | `10` | Number of diffusion components |
| `random_state` | `42` | Seed for the fallback PCA and the kNN graph |

### Benchmarking embeddings

```python
from scintilla.dimensionality_reduction.benchmark import benchmark_embeddings

results_df = benchmark_embeddings(adata, use_rep="X_pca")
```

### CLI

```bash
scintilla reduce data/pbmc3k.h5ad --method umap --use-rep X_pca --output embedded.h5ad
scintilla reduce data/pbmc3k.h5ad --method tsne --output embedded_tsne.h5ad
scintilla reduce data/pbmc3k.h5ad --method diffmap --output embedded_dm.h5ad
```

**`reduce` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input `.h5ad` |
| `--method` | `umap` | `umap`, `tsne`, or `diffmap` |
| `--use-rep` | `X_pca` | Key in `obsm` |
| `--output` | `None` | Save output `.h5ad` |
| `--verbose` | `False` | Print progress |

---

## Unsupervised Clustering

### Algorithms

scINTILLA benchmarks up to 9 clustering algorithms (exact set controlled by `AnalysisConfig.clustering_methods`):

| Key | Algorithm | Grid |
|---|---|---|
| `kmeans` | K-Means | init strategies |
| `hierarchical` | Agglomerative / Hierarchical | distance metrics × linkage |
| `dbscan` | DBSCAN | eps values × distance metrics |
| `hdbscan` | HDBSCAN | `min_cluster_size` × `min_samples` |
| `leiden` | Leiden | resolution sweep |
| `louvain` | Louvain | resolution sweep |
| `spectral` | Spectral | number of clusters |
| `consensus` | Consensus | ensemble of the above |

### High-level API

```python
from scintilla import unsupervised_analysis

result = unsupervised_analysis(
    adata,
    cell_type_col="cell_type",
    n_clusters=None,        # auto-inferred from cell_type_col
    run_pca_first=True,
    n_pca_comps=30,
    store_labels=True,      # writes best labels → adata.obs["scintilla_cluster"]
    config=None,            # optional AnalysisConfig
    verbose=True,
)
```

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `cell_type_col` | required | Ground-truth label column for ARI scoring |
| `n_clusters` | `None` | Target cluster count; inferred if `None` |
| `run_pca_first` | `True` | Run PCA before all algorithms |
| `n_pca_comps` | `30` | PCA components |
| `store_labels` | `True` | Write best labels to `adata.obs["scintilla_cluster"]` |
| `config` | `None` | `AnalysisConfig` for method/grid selection |
| `verbose` | `True` | Print progress |

**Return values**

| Key | Type | Description |
|---|---|---|
| `adata` | AnnData | Input data + PCA + `scintilla_cluster` (if `store_labels=True`) |
| `results_df` | DataFrame | `method`, `params`, `ari`, `ami`, `n_clusters`, `noise_fraction`, `status`, `failure_reason` |
| `labels_dict` | dict | `{key: labels_array}` for every configuration tested |
| `fig` | Figure | ARI comparison bar chart |
| `best_method` | str | Method with the highest ARI |

**Reading `status`.** Every method and grid point that was attempted keeps a
row, so a dropped method is never merely absent:

| `status` | Meaning |
|---|---|
| `"ok"` | The method ran; metric columns are populated |
| `"failed"` | The method raised. `failure_reason` holds the message, metrics are `NaN`, and one warning was emitted |
| `"skipped"` | The combination was never attempted — an invalid parameter pair such as `ward` linkage with a non-Euclidean metric, or an optional backend that is not installed. No warning is emitted, because this is not an error |

Rows without a valid `ari` are excluded from the best-method choice and from
the figure, but remain in `results_df`.

### Low-level APIs

```python
from scintilla.clustering.kmeans import kmeans_clustering
from scintilla.clustering.leiden import leiden_clustering
from scintilla.clustering.louvain import louvain_clustering
from scintilla.clustering.hdbscan import hdbscan_clustering
from scintilla.clustering.dbscan import dbscan_clustering, estimate_eps
from scintilla.clustering.hierarchical import hierarchical_clustering
from scintilla.clustering.spectral import spectral_clustering

labels, centers, inertia = kmeans_clustering(X, n_clusters=8)
labels = leiden_clustering(adata, resolution=0.5, use_rep="X_pca")
labels = louvain_clustering(adata, resolution=0.5, use_rep="X_pca")
hdbscan_df, labels = hdbscan_clustering(X, min_cluster_size_range=[20], min_samples_range=[5])
labels = dbscan_clustering(X, eps=0.5, metric="euclidean")
labels, linkage_matrix = hierarchical_clustering(X, n_clusters=8, metric="euclidean", linkage="ward")
labels = spectral_clustering(X, n_clusters=8)

# Data-driven DBSCAN eps estimation
data_driven_eps = estimate_eps(X, min_samples=5)
labels = dbscan_clustering(X, eps=data_driven_eps)
```

**`hdbscan_clustering` parameters**

Sweeps a grid and returns `(results_df, best_labels)`.

| Parameter | Default | Description |
|---|---|---|
| `data` | required | Data matrix (cells × features) or AnnData |
| `min_cluster_size_range` | `None` | Grid of minimum cluster sizes; `None` uses `HDBSCAN_MIN_CLUSTER_SIZE_RANGE` |
| `min_samples_range` | `None` | Grid of `min_samples` values; `None` uses `HDBSCAN_MIN_SAMPLES_RANGE` |
| `metric` | `"euclidean"` | Distance metric |

`results_df` carries one row per grid point with `status` and, when a grid
point raised, a `failure_reason` plus `NaN` cluster counts — a failed fit is
never reported as an all-noise result.

### Consensus clustering

```python
from scintilla.clustering.consensus import consensus_clustering

co_matrix, labels, stability = consensus_clustering(
    adata, methods=["kmeans", "leiden"], n_runs_per_method=5
)
```

**`consensus_clustering` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | AnnData object |
| `methods` | `None` | Any of `"kmeans"`, `"leiden"`, `"spectral"`; `None` uses `["kmeans"]`. Unknown names raise `ValueError` rather than contributing nothing |
| `n_runs_per_method` | `5` | Parameter variations per method |
| `resolution_range` | `None` | Leiden resolutions; `None` uses `LEIDEN_RESOLUTIONS` |
| `random_state` | `42` | Seed forwarded to every clustering run |

**Return values**

| Position | Type | Description |
|---|---|---|
| `consensus_matrix` | `np.ndarray` | `(n_cells, n_cells)` co-clustering frequency in `[0, 1]` |
| `consensus_labels` | `np.ndarray` | Labels from the hierarchical cut of the consensus matrix |
| `stability_scores` | `dict` | One `{method: float}` entry per requested method — the mean pairwise ARI between that method's runs, or `nan` when fewer than two runs succeeded |

When any run fails, `stability_scores` gains the reserved key `"failures"`
mapping `{method: first_error_message}`; it is absent when everything
succeeded. Because method names are validated, `"failures"` can never collide
with a method entry, so iterate per-method scores as:

```python
failures = stability.get("failures", {})
scores = {k: v for k, v in stability.items() if k != "failures"}
```

### Benchmark API

```python
from scintilla.clustering.benchmark import benchmark_clustering_methods

results_df, labels_dict, fig = benchmark_clustering_methods(
    adata,
    cell_type_col="cell_type",
    n_clusters=None,
    config=None,
    verbose=True,
)

# With robust statistics options
results_df, labels_dict, fig = benchmark_clustering_methods(
    adata,
    cell_type_col="cell_type",
    adaptive_resolution=True,      # NVI-based resolution selection for Leiden/Louvain
    resolution_selection="nvi",    # selection criterion: "nvi" or "ari" (default)
    auto_eps=True,                 # data-driven DBSCAN eps via kneedle elbow
    verbose=True,
)
```

**Additional `benchmark_clustering_methods` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adaptive_resolution` | `False` | Enable adaptive resolution search for Leiden/Louvain |
| `resolution_selection` | `"ari"` | Resolution selection criterion: `"ari"` (best ARI) or `"nvi"` (most stable NVI) |
| `auto_eps` | `False` | Auto-estimate DBSCAN `eps` via k-distance kneedle elbow |

### Bootstrap clustering metrics

```python
from scintilla.evaluation.clustering_metrics import (
    comprehensive_clustering_metrics,
    bootstrap_clustering_metrics,
)

# Both take the embedding as well as the labels, because internal metrics
# (silhouette, Calinski-Harabasz) need the coordinates.
X = adata.obsm["X_pca"]

# Standard metrics
metrics = comprehensive_clustering_metrics(X, labels_true, labels_pred)

# With bootstrap CIs
metrics = comprehensive_clustering_metrics(
    X, labels_true, labels_pred, bootstrap_ci=True
)
# metrics["ari_ci_lower"], metrics["ari_ci_upper"], etc.

# Or use the standalone bootstrap function
ci_df = bootstrap_clustering_metrics(X, labels_true, labels_pred, B=2000, alpha=0.05)
```

> **Note:** `bootstrap_clustering_metrics` wraps each per-resample metric computation in a `try/except` to handle rare-cluster dropout (where a bootstrap resample happens to exclude all representatives of a small cluster). Failed resamples are recorded as NaN and a warning is emitted when the discard rate exceeds 10 %.

### CLI

```bash
scintilla cluster data/pbmc3k.h5ad --cell-type-col cell_type --output results/clustering.csv
scintilla cluster data/pbmc3k.h5ad --cell-type-col cell_type --config my_config.yaml
scintilla cluster data/pbmc3k.h5ad --cell-type-col cell_type --fast
```

**`cluster` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input `.h5ad` |
| `--cell-type-col` | `cell_type` | Ground-truth label column |
| `--n-clusters` | `None` | Target cluster count |
| `--output` | `None` | Save leaderboard CSV |
| `--config` | `None` | Path to YAML config |
| `--fast` | `False` | Use fast preset (kmeans + leiden only) |
| `--verbose` | `False` | Print progress |

---

## Supervised Classification

### Classifiers

| Short name | Model | Notes |
|---|---|---|
| `LogReg` | Logistic Regression | — |
| `RF` | Random Forest | 100 estimators (sklearn default) |
| `SVM` | SVM (RBF kernel) | — |
| `MLP` | Multi-layer Perceptron | — |
| `LDA` | Linear Discriminant Analysis | — |
| `QDA` | Quadratic Discriminant Analysis | — |
| `kNN` | k-Nearest Neighbours | — |
| `GradientBoosting` | Histogram Gradient Boosting | Multi-core, handles missing values |
| `NaiveBayes` | Gaussian Naive Bayes | — |
| `StackingEnsemble` | Stacking Ensemble | RF + LogReg; 2-fold CV |
| `XGBoost` | XGBoost | Requires `xgboost` |
| `LightGBM` | LightGBM | Requires `lightgbm` |

> `GradientBoosting` uses `HistGradientBoostingClassifier` internally (sklearn ≥ 1.0), which is orders of magnitude faster than the legacy `GradientBoostingClassifier` on large datasets.

### High-level API

```python
from scintilla import supervised_analysis

result = supervised_analysis(
    adata,
    target_col="cell_type",
    normality=None,       # None = auto-detect; True / False = override
    test_size=0.2,
    include_shap=True,
    models=None,          # None = all classifiers; or list of short names
    config=None,          # optional AnalysisConfig
    verbose=True,
)
```

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `target_col` | required | Column in `obs` with class labels |
| `normality` | `None` | `None` = run `check_normality`; `True`/`False` = override |
| `test_size` | `0.2` | Fraction held out for evaluation |
| `include_shap` | `True` | Compute SHAP importances for the best model |
| `models` | `None` | Explicit list of short names; `None` = all classifiers |
| `config` | `None` | `AnalysisConfig` (overridden by explicit `models`/`include_shap`) |
| `verbose` | `True` | Print progress |

**Return values**

| Key | Type | Description |
|---|---|---|
| `best_model` | sklearn estimator | Fitted best model |
| `best_model_name` | str | Short name of the best model |
| `all_results` | dict | `{name: {"model": ..., "metrics": {...}}}` |
| `feature_importances` | dict or None | SHAP values keyed by feature name |
| `normality` | bool | Whether data passed normality test |

**Metrics in `all_results[name]["metrics"]`:** `accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `confusion_matrix`.

### Running specific classifiers

```python
# Run just three fast classifiers
result = supervised_analysis(
    adata, target_col="cell_type",
    models=["LogReg", "RF", "kNN"],
    include_shap=False,
)
```

### Individual classifier functions

```python
from scintilla.classification.models import (
    logistic_regression_classification,
    random_forest_classification,
    svm_classification,
    mlp_classification,
    knn_classification,
    gradient_boosting_classification,
    stacking_ensemble_classification,
)

model, metrics = logistic_regression_classification(X_train, X_test, y_train, y_test)
model, metrics = random_forest_classification(X_train, X_test, y_train, y_test)
```

### Comprehensive benchmarking

```python
from scintilla.classification.benchmark import benchmark_models_comprehensive

results_df, fig = benchmark_models_comprehensive(
    adata,
    target_col="cell_type",
    verbose=True,
)
print(results_df[["model", "space", "accuracy", "f1"]].head(20))

# With BCa bootstrap confidence intervals
results_df, fig = benchmark_models_comprehensive(
    adata,
    target_col="cell_type",
    bootstrap_ci=True,          # add 95 % BCa CIs for each metric
    n_bootstrap=2000,           # number of bootstrap resamples
    verbose=True,
)
# results_df now includes: accuracy_ci_lower, accuracy_ci_upper, f1_ci_lower, etc.

# .632+ bootstrap estimator (avoids train/test leakage bias)
results_df, fig = benchmark_models_comprehensive(
    adata,
    target_col="cell_type",
    estimator="bootstrap_632plus",  # "cv" (default), "holdout", or "bootstrap_632plus"
    verbose=True,
)
```

**Additional `benchmark_models_comprehensive` parameters**

| Parameter | Default | Description |
|---|---|---|
| `bootstrap_ci` | `False` | Compute BCa bootstrap 95 % confidence intervals for each metric |
| `n_bootstrap` | `2000` | Number of bootstrap resamples |
| `estimator` | `"cv"` | Performance estimator: `"cv"` (stratified K-fold), `"holdout"` (single split), or `"bootstrap_632plus"` (.632+ bootstrap) |
| `no_info_method` | `"analytical"` | No-information rate for .632+: `"analytical"` (exact for accuracy) or `"permutation"` (correct for any metric) |

### Pairwise classifier comparison

Perform a paired bootstrap test to determine whether the performance difference between two classifiers is statistically significant:

```python
from scintilla.classification.benchmark import compare_classifiers

result_df = compare_classifiers(
    y_true,
    predictions={"RF": y_pred_rf, "SVM": y_pred_svm},
    metric_fn=lambda yt, yp: (yt == yp).mean(),
    B=2000,
)
# result_df columns: model_a, model_b, diff, ci_low, ci_high, p_value
# result_df["p_value"].iloc[0] < 0.05 → significantly different performance
```

### Influential cell detection

Identify cells whose inclusion/exclusion disproportionately affects model performance, using jackknife-after-bootstrap:

```python
from scintilla.classification.diagnostics import influential_cells

influence_scores = influential_cells(
    y_true, y_pred, metric_fn=lambda yt, yp: (yt == yp).mean(), B=2000
)
# Returns array of per-cell influence scores; high-influence cells may be mislabelled
```

### SHAP feature importance

```python
from scintilla.classification.feature_importance import compute_shap_importance

shap_dict = compute_shap_importance(model, X_test, feature_names=list(adata.var_names))
print(sorted(shap_dict.items(), key=lambda x: -x[1])[:20])
```

### CLI

```bash
scintilla classify data/pbmc3k.h5ad --target-col cell_type
scintilla classify data/pbmc3k.h5ad --target-col cell_type --no-shap --output results/cls.json
scintilla classify data/pbmc3k.h5ad --target-col cell_type --fast
scintilla classify data/pbmc3k.h5ad --target-col cell_type --config my_config.yaml
```

**`classify` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input file |
| `--target-col` | `cell_type` | Class label column |
| `--no-shap` | `False` | Skip SHAP |
| `--output` | `None` | Save summary JSON |
| `--config` | `None` | Path to YAML config |
| `--fast` | `False` | Fast preset (LogReg, RF, kNN; no SHAP) |
| `--verbose` | `False` | Print progress |

---

## Differential Expression

### Methods

| Method | Function | Notes |
|---|---|---|
| Wilcoxon rank-sum | `wilcoxon_de` | Non-parametric, robust |
| Welch's t-test | `ttest_de` | Parametric, unequal variance |
| Permutation test | `permutation_de` | Permutation-based p-values |
| Pseudobulk | `pseudobulk_de` | Aggregates per sample; DESeq2-like |
| Rank genes groups | `rank_genes_groups` | scanpy wrapper, one-vs-rest |

### Python API

```python
from scintilla import wilcoxon_de, ttest_de, permutation_de, rank_genes_groups
from scintilla import volcano_plot_data, filter_de_genes

# Two-group Wilcoxon
de_df = wilcoxon_de(adata, group_col="cell_type", group1="Monocyte", group2="T Cell")

# Filter to significant genes
sig_df = filter_de_genes(de_df, alpha=0.05, log2fc_threshold=1.0)
```

**Shared parameters for `wilcoxon_de`, `ttest_de`, `permutation_de`**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `group_col` | required | Column in `obs` defining groups |
| `group1` | required | First group label |
| `group2` | required | Second group label |
| `pseudocount` | `0.01` | Added to group means before log₂ ratio: `log2((mean1 + pc) / (mean2 + pc))` |

> **Pseudocount convention:** scINTILLA uses a default pseudocount of **0.01**, which differs from **DESeq2** (0.5) and **Seurat** (1.0).  The smaller value better preserves fold-change magnitude for single-cell data where post-normalisation means are typically in the 0–5 range, but it means scINTILLA fold changes are **not directly comparable** with those tools.  Set `pseudocount=0.5` or `pseudocount=1.0` if you need compatible values.

**`permutation_de` extra parameters**

| Parameter | Default | Description |
|---|---|---|
| `n_permutations` | `1000` | Number of permutations for null distribution |
| `standardise` | `False` | Standardise features before permuting |
| `random_state` | `42` | Seed for the permutation draws |

**`pseudobulk_de` parameters**

Unlike the two-group tests above, this aggregates every sample and compares the
two levels found in `condition_col`; it does not take `group1`/`group2`.

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `condition_col` | required | Column in `obs` defining the two conditions |
| `sample_col` | required | Column in `obs` defining biological replicates |
| `cell_type_col` | `None` | When given, run per cell type and return `{cell_type: DataFrame}` |
| `alpha` | `0.05` | Significance level for the adjusted p-values |
| `pseudocount` | `0.01` | Added to group means before the log₂ ratio |

> **Replication is required, not approximated.** Pseudobulk needs at least two
> biological samples per condition and raises `ValueError` otherwise. It does
> **not** fall back to a cell-level test: treating cells as replicates conflates
> technical with biological variation and inflates false discoveries. In the
> per-cell-type form, a cell type that cannot be tested warns and returns an
> empty frame carrying `status="failed"` and `failure_reason` in its `.attrs`,
> so it is never silently indistinguishable from "no significant genes".

**`filter_de_genes` parameters**

| Parameter | Default | Description |
|---|---|---|
| `de_results` | required | DE results DataFrame |
| `log2fc_col` | `"log2fc"` | Column holding log₂ fold changes |
| `pval_col` | `"p_adjusted"` | Column holding adjusted p-values |
| `log2fc_threshold` | `1.0` | Absolute log₂ fold-change threshold |
| `alpha` | `0.05` | Adjusted p-value threshold |
| `effect_size_col` | `None` | Column name to filter on (e.g. `"cohens_d"`, `"rank_biserial"`) |
| `effect_size_threshold` | `0.3` | Minimum absolute effect-size value when `effect_size_col` is set |

**Output columns:**

| Column | Returned by | Description |
|---|---|---|
| `gene` | all methods | Gene identifier |
| `statistic` | all methods | Test statistic |
| `pvalue` | all methods | Raw p-value |
| `pvalue_adj` | all methods | BH-adjusted p-value |
| `log2fc` | all methods | Log₂ fold-change |
| `rank_biserial` | `wilcoxon_de` | Rank-biserial correlation effect size (−1 to +1) |
| `cliffs_delta` | `wilcoxon_de` | Cliff's delta non-parametric effect size |
| `cohens_d` | `ttest_de` | Cohen's d standardised effect size |
| `hedges_g` | `ttest_de` | Hedges' g (bias-corrected Cohen's d) |

> **Sign convention:** All directional effect-size columns (`log2fc`, `rank_biserial`, `cliffs_delta`, `cohens_d`, `hedges_g`) are **positive when group 1 > group 2** and negative when group 1 < group 2.  This is consistent across the Wilcoxon and t-test modules.

Effect sizes are always computed alongside the DE test at negligible extra cost. Use them with `filter_de_genes(de_df, effect_size_col="cohens_d", effect_size_threshold=0.5)` for biologically meaningful filtering.

### CLI

```bash
scintilla de data/pbmc3k.h5ad \
    --group-col cell_type --group1 Monocyte --group2 "T Cell" \
    --method wilcoxon --output results/de.csv
```

**`de` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input `.h5ad` |
| `--group-col` | required | Group column |
| `--group1` | required | First group |
| `--group2` | required | Second group |
| `--method` | `wilcoxon` | `wilcoxon`, `ttest`, `permutation` |
| `--output` | `None` | Save CSV |
| `--verbose` | `False` | Print progress |

---

## Cell-Type Annotation

### Strategies

| Function | Strategy |
|---|---|
| `annotate_by_markers` | Score clusters against a user-supplied marker dict |
| `find_marker_genes` | Rank genes per cluster via DE |
| `ora_test` | Over-representation analysis (Fisher's exact test) |
| `transfer_labels` | KNN-based label transfer from a reference AnnData |

### Python API

```python
from scintilla import annotate_by_markers, find_marker_genes, ora_test, transfer_labels

# Marker-based annotation
markers = {
    "Monocyte": ["CD14", "LYZ", "CST3"],
    "T Cell":   ["CD3D", "CD3E", "IL7R"],
    "B Cell":   ["MS4A1", "CD79A"],
}
adata = annotate_by_markers(adata, marker_dict=markers)
print(adata.obs["predicted_cell_type"].value_counts())

# Find top marker genes per cluster
markers_df = find_marker_genes(adata, groupby="leiden", n_genes=50, method="wilcoxon")

# Over-representation analysis
ora_results = ora_test(
    gene_list=["CD14", "LYZ", "CST3"],
    background=list(adata.var_names),
    gene_sets={"Myeloid": ["CD14", "LYZ"], "Lymphoid": ["CD3D"]},
)

# Label transfer from reference
# Genes are matched by name (>= 10 shared genes required), not by position;
# any precomputed X_pca is ignored as it is not comparable across datasets.
adata_query = transfer_labels(
    reference_adata=adata_ref, query_adata=adata_query, label_col="cell_type"
)
```

**`find_marker_genes` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `groupby` | required | Column in `obs` with cluster labels |
| `method` | `"wilcoxon"` | DE method for ranking (`"wilcoxon"`, `"t-test"`, `"logreg"`) |
| `n_genes` | `50` | Top genes to report per group |
| `random_state` | `42` | Seed used when `method="logreg"` |

> `method="logreg"` ranks by model coefficient. Scanpy computes no p-values or
> fold changes for it, so those columns are `NaN`.

### CLI

```bash
scintilla annotate data/pbmc3k_clustered.h5ad \
    --groupby leiden --n-genes 50 --output results/annotated.h5ad
```

**`annotate` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input `.h5ad` |
| `--groupby` | `leiden` | Cluster label column |
| `--method` | `wilcoxon` | DE method |
| `--n-genes` | `50` | Genes per group |
| `--output` | `None` | Save annotated `.h5ad` |
| `--verbose` | `False` | Print progress |

---

## Batch Correction

### Methods

| Key | Method | Notes |
|---|---|---|
| `combat` | ComBat | Linear regression; fast |
| `harmony` | Harmony | PCA-space iterative; requires `harmonypy` |
| `bbknn` | BBKNN | Graph-based; requires `bbknn` |
| `scanorama` | Scanorama | Panoramic stitching; requires `scanorama` |

### Python API

```python
from scintilla import combat_correct, batch_asw, benchmark_batch_correction

# Apply ComBat
adata_corrected = combat_correct(adata, batch_key="batch")

# Evaluate batch mixing
asw = batch_asw(adata_corrected, batch_key="batch", label_key="cell_type")

# Benchmark all methods
result = benchmark_batch_correction(
    adata,
    batch_key="batch",
    label_key="cell_type",
    methods=["combat", "harmony", "bbknn", "scanorama"],
    n_pcs=30,
)
print("Best:", result["best_method"])
```

**`benchmark_batch_correction` parameters**

| Parameter | Default | Description |
|---|---|---|
| `adata` | required | Input AnnData |
| `batch_key` | required | Column in `obs` with batch labels |
| `label_key` | `None` | Column in `obs` with cell-type labels (for bio-conservation score) |
| `methods` | `None` | List of method keys to benchmark; `None` uses all four. An empty list raises `ValueError` |
| `n_pcs` | `None` | PCA components used when required by the method [`config.n_pca_comps`, else `30`] |
| `scoring_method` | `None` | `"single_metric"` (batch ASW), `"rank_aggregate"` (Borda count across batch ASW + bio-conservation), or `"pareto"` (Pareto frontier flagging). The shared-config spellings `"weighted"` and `"borda"` map to `"single_metric"` and `"rank_aggregate"`; anything else raises `ValueError` [`config.scoring_method`, else `"single_metric"`] |
| `random_state` | `None` | Seed forwarded to Harmony, BBKNN, Scanorama and their PCA [`config.random_seed`, else `42`] |
| `config` | `None` | `AnalysisConfig` supplying the defaults above |

**Return values**

| Key | Type | Description |
|---|---|---|
| `leaderboard` | DataFrame | `method`, `batch_asw`, `bio_conservation` (when `label_key` set), `status`, `pareto_optimal` (when `scoring_method="pareto"`) |
| `best_method` | str | Method with best score |
| `corrected_adatas` | dict | `{method: corrected AnnData}` |

When `scoring_method="rank_aggregate"` and `label_key` is provided, the bio-conservation score (silhouette on cell-type labels in PCA space) is computed alongside batch ASW. The final ranking uses Borda count across both metrics, providing a balanced view of batch mixing and biological signal preservation.

When `scoring_method="pareto"`, the leaderboard includes a `pareto_optimal` boolean column flagging methods on the Pareto frontier of batch ASW vs. bio-conservation.

### Bio-conservation score

```python
from scintilla.batch_correction.metrics import bio_conservation_score

score = bio_conservation_score(adata, label_key="cell_type", embed_key="X_pca")
# Returns silhouette score of cell-type labels in embedding space [−1, 1]
```

### CLI

```bash
scintilla batch-correct data/pbmc3k.h5ad --batch-key batch --label-key cell_type
scintilla batch-correct data/pbmc3k.h5ad --batch-key batch --methods combat harmony \
    --output results/batch_correction/
```

**`batch-correct` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input `.h5ad` |
| `--batch-key` | required | Batch label column |
| `--label-key` | `None` | Cell-type label column |
| `--methods` | all four | Subset of `combat harmony bbknn scanorama` |
| `--output` | `None` | Output directory |
| `--verbose` | `False` | Print progress |

---

## Benchmarking Utilities

### Profiling a single function

```python
from scintilla import profile_method

def my_fn(adata):
    from scintilla.clustering.kmeans import kmeans_clustering
    return kmeans_clustering(adata.obsm["X_pca"], n_clusters=8)

# profile_method is a decorator: wrap the callable, then call the wrapper.
result = profile_method(my_fn)(adata)
print(f"Wall time: {result.wall_time:.2f}s")
print(f"Peak memory: {result.peak_memory_mb:.1f} MB")
```

### Scalability sweep

```python
from scintilla import scalability_sweep

# Signature is (method_fn, adata, ...); sizes are fractions of the input.
results_df = scalability_sweep(
    my_fn,
    adata,
    fractions=[0.1, 0.25, 0.5, 1.0],
)
print(results_df[["fraction", "n_cells", "elapsed_seconds", "peak_memory_mb"]])

# With multi-seed runs and bootstrap confidence intervals
results_df = scalability_sweep(
    my_fn,
    adata,
    cell_counts=[1000, 5000, 10000, 50000],
    n_seeds=5,              # run each cell count with 5 seeds
    bootstrap_ci=True,      # add CI bands for wall_time
    n_bootstrap=2000,
)
# results_df includes: wall_time_mean, wall_time_ci_lower, wall_time_ci_upper
```

**Additional `scalability_sweep` parameters**

| Parameter | Default | Description |
|---|---|---|
| `n_seeds` | `1` | Number of random seeds per cell count |
| `bootstrap_ci` | `False` | Compute BCa bootstrap CIs for timing metrics |
| `n_bootstrap` | `2000` | Number of bootstrap resamples |

### Seed-stability test

```python
from scintilla import seed_stability_test

# method_fn is called as method_fn(data, random_state=seed); metric_fn reduces
# the return value to the number whose stability is measured.
stability_df = seed_stability_test(
    method_fn=lambda data, random_state: my_fn(data),
    data=adata,
    metric_fn=lambda out: float(len(set(out[0]))),
    n_seeds=5,
)

# With bootstrap CIs and intraclass correlation coefficient
stability_df = seed_stability_test(
    method_fn=lambda data, random_state: my_fn(data),
    data=adata,
    metric_fn=lambda out: float(len(set(out[0]))),
    n_seeds=5,
    bootstrap_ci=True,      # BCa CIs for the mean metric across seeds
    n_bootstrap=2000,
    compute_icc=True,       # ICC(1,1) stored in stability_df.attrs["icc"]
    icc_method="pingouin",  # proper split-half ICC via pingouin (default)
)
print(f"ICC: {stability_df.attrs['icc']:.3f}")
```

**Additional `seed_stability_test` parameters**

| Parameter | Default | Description |
|---|---|---|
| `bootstrap_ci` | `False` | Compute BCa bootstrap CIs for the mean metric |
| `n_bootstrap` | `2000` | Number of bootstrap resamples |
| `compute_icc` | `False` | Compute ICC(1,1) across seeds (stored in `df.attrs["icc"]`) |
| `icc_method` | `"pingouin"` | ICC computation method: `"pingouin"` (proper split-half ICC via pingouin) or `"legacy"` (approximate formula) |

### Pairwise method comparison

Statistically compare pairs of methods using permutation tests, McNemar's test, or paired bootstrap:

```python
from scintilla import pairwise_method_comparison

comparison_df = pairwise_method_comparison(
    results={"RF": scores_rf, "SVM": scores_svm, "kNN": scores_knn},
    metric_col="accuracy",
    method_col="method",
    test="permutation",     # "permutation", "mcnemar", or "paired_bootstrap"
)
# Returns DataFrame with method_a, method_b, p_value, significant columns
```

### Benchmark report

```python
from scintilla import BenchmarkReport

report = BenchmarkReport()
report.add_result("clustering", "Leiden_0.5", {"ari": 0.82})
report.add_result("classification", "RF",     {"accuracy": 0.94})

print(report.summary_table())
report.export("results/benchmark_report/")
```

---

## AnalysisConfig System

`AnalysisConfig` is the central configuration object for scINTILLA. It controls which methods are included in each pipeline step and what hyperparameter grids are used, without requiring you to pass many individual parameters.

### Creating a config

```python
from scintilla import AnalysisConfig

# All methods (default behaviour)
cfg = AnalysisConfig.default()

# Fast preset: only quick methods, no SHAP
cfg = AnalysisConfig.fast()

# Robust preset: enables all statistical enhancements
cfg = AnalysisConfig.robust()

# Custom config
cfg = AnalysisConfig(
    clustering_methods=["kmeans", "leiden"],
    leiden_resolutions=[0.3, 0.5, 1.0],
    classifiers=["LogReg", "RF", "SVM"],
    include_shap=False,
    n_features=30,
)
```

### All fields

| Field | Type | Default | Fast preset | Robust preset | Description |
|---|---|---|---|---|---|
| `clustering_methods` | `List[str]` | all 8 | `["kmeans","leiden"]` | all 8 | Algorithms to run during clustering benchmark |
| `leiden_resolutions` | `List[float]` | `[0.1,0.3,0.5,0.8,1.0,1.5,2.0,3.0]` | `[0.5,1.0]` | default | Resolution parameter grid for Leiden |
| `louvain_resolutions` | `List[float]` | `[0.1,0.3,0.5,0.8,1.0,1.5,2.0,3.0]` | `[0.5,1.0]` | default | Resolution parameter grid for Louvain |
| `hdbscan_min_cluster_sizes` | `List[int]` | `[10,20,50]` | `[20]` | default | `min_cluster_size` grid for HDBSCAN |
| `hdbscan_min_samples` | `list` | `[None,5]` | `[None]` | default | `min_samples` grid for HDBSCAN |
| `spectral_n_clusters_range` | `List[int]` | `[2..10]` | `[]` | default | Number-of-clusters grid for Spectral |
| `classifiers` | `List[str]` | all 10 | `["LogReg","RF","kNN"]` | all 10 | Classifiers to train |
| `include_shap` | `bool` | `True` | `False` | `True` | Compute SHAP for best model |
| `feature_selection_methods` | `List[str]` | `["pca_loadings","mutual_information"]` | same | same | Methods for feature-selection benchmark |
| `n_features` | `int` | `50` | `50` | `50` | Features to select per method |
| `random_seed` | `int` | `42` | `42` | `42` | Default seed forwarded as `random_state` to every stochastic step |
| `test_size` | `float` | `0.2` | `0.2` | `0.2` | Train/test split fraction |
| `n_pca_comps` | `int` | `30` | `30` | `30` | PCA components |
| `cv_folds` | `int` | `5` | `3` | `5` | Number of cross-validation folds for classification |
| `verbose` | `bool` | `True` | `True` | `True` | Verbose output |
| `bootstrap_ci` | `bool` | `False` | `False` | `True` | Enable BCa bootstrap confidence intervals across benchmarks |
| `n_bootstrap` | `int` | `2000` | `2000` | `2000` | Bootstrap resamples |
| `scoring_method` | `str` | `"weighted"` | `"weighted"` | `"borda"` | Transformation scoring: `"weighted"`/`"borda"`; batch scoring maps `"weighted"` to `"single_metric"` and `"borda"` to `"rank_aggregate"`, while also accepting `"pareto"` |
| `auto_pca_components` | `str\|None` | `None` | `None` | `"gavish_donoho"` | Adaptive PCA: `"gavish_donoho"` or `"marchenko_pastur"` |
| `adaptive_resolution` | `bool` | `False` | `False` | `True` | NVI-based adaptive resolution search for Leiden/Louvain |
| `auto_eps` | `bool` | `False` | `False` | `True` | Data-driven DBSCAN eps estimation |
| `classification_estimator` | `str` | `"cv"` | `"cv"` | `"cv"` | `"cv"`, `"holdout"`, or `"bootstrap_632plus"` |
| `no_info_method` | `str` | `"analytical"` | `"analytical"` | `"permutation"` | No-information rate for .632+: `"analytical"` (exact for accuracy) or `"permutation"` |
| `mp_sigma_method` | `str` | `"median"` | `"median"` | `"trimmed_mean"` | MP noise-variance estimation: `"median"` or `"trimmed_mean"` |

**Available clustering method keys:** `kmeans`, `hierarchical`, `dbscan`, `leiden`, `louvain`, `hdbscan`, `spectral`, `consensus`

**Available classifier keys:** `LogReg`, `RF`, `SVM`, `MLP`, `LDA`, `QDA`, `kNN`, `GradientBoosting`, `NaiveBayes`, `StackingEnsemble`, `XGBoost`, `LightGBM`

`XGBoost` and `LightGBM` are offered only when their package is installed.
Without it the model is simply absent from the results rather than present with
a failure, which keeps "not installed" distinct from a genuine error.

**Available feature-selection keys:** `pca_loadings`, `mutual_information`, `boruta`, `mrmr`

### Passing a config to pipeline functions

```python
result   = unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
cls_result = supervised_analysis(adata, target_col="cell_type", config=cfg)
fs_results = benchmark_feature_selection(adata, target_col="cell_type", config=cfg)
```

**Priority rule:** Explicit keyword arguments always override config values,
which in turn override the historical per-function default.

Most such arguments default to `None`, which means "not supplied" rather than a
literal `None`: passing `verbose=None` defers to `config.verbose`. Two arguments
of `unsupervised_analysis` — `auto_pca_components` and `mp_sigma_method` — need
`None` itself to stay meaningful (it disables adaptive PCA), so they use a
private omission sentinel instead. Omit them to defer to the config; pass
`auto_pca_components=None` to force adaptive selection off even when the config
enables it.

### Reproducibility and seeding

Set the seed through `random_state` on any single call, or through
`AnalysisConfig(random_seed=...)` for a whole analysis. Reassigning
`scintilla.config.RANDOM_SEED` after import does **not** work — the constant is
read once at import time by each module that uses it.

```python
import scintilla as si
from scintilla import AnalysisConfig

# One call.
result = si.unsupervised_analysis(adata, cell_type_col="cell_type", random_state=0)

# A whole analysis.
cfg = AnalysisConfig(random_seed=0)
result = si.unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
```

Because every stochastic public function now accepts `random_state`, the
package's own `seed_stability_test` can be pointed at them directly:

```python
from scintilla.benchmarking.reproducibility import seed_stability_test

stability = seed_stability_test(
    method_fn=lambda data, random_state: si.unsupervised_analysis(
        data, cell_type_col="cell_type", random_state=random_state, verbose=False
    )["results_df"]["ari"].max(),
    data=adata,
    metric_fn=float,
    n_seeds=10,
)
```

> **Comparing against pre-0.2 results.** Several functions previously used the
> scanpy/scikit-learn default seed rather than scINTILLA's, and one silhouette
> computation was unseeded. Seeding them consistently changes their numbers:
> `run_umap`, `run_tsne`, `run_diffusion_map`, `run_force_directed`,
> `hvg_sensitivity_analysis`, `graph_connectivity`, the internal PCA inside
> `harmony_correct`/`bbknn_correct`, and the silhouette subsample in
> `benchmark_transformations` above 1 000 cells. Clustering, classification,
> differential expression and feature selection are unaffected — they already
> used seed 42.

```python
# config says classifiers=["LogReg","RF"], but models= takes precedence
result = supervised_analysis(adata, target_col="cell_type",
                              models=["SVM", "MLP"], config=cfg)
```

### YAML workflow

```bash
# Generate a commented template
scintilla generate-config --output my_config.yaml

# Edit the file, then use it
scintilla cluster data.h5ad --cell-type-col cell_type --config my_config.yaml
scintilla classify data.h5ad --target-col cell_type --config my_config.yaml
scintilla run-all data.h5ad --target-col cell_type --config my_config.yaml --output-dir out/
```

In Python:

```python
from scintilla import AnalysisConfig, generate_default_yaml

# Write template
generate_default_yaml("my_config.yaml")

# Load
cfg = AnalysisConfig.from_yaml("my_config.yaml")

# Modify and save
cfg2 = cfg.copy(leiden_resolutions=[0.5, 1.0], include_shap=False)
cfg2.to_yaml("my_config_fast.yaml")
```

### `copy` with overrides

```python
cfg_no_shap = cfg.copy(include_shap=False)
cfg_fast_cluster = cfg.copy(leiden_resolutions=[0.5, 1.0], clustering_methods=["leiden"])
```

---

## Robust Statistics

scINTILLA includes a suite of robust statistical utilities that can be enabled across all benchmark modules. These are collected in the `scintilla.statistical_tests` package.

### Bootstrap confidence intervals

BCa (bias-corrected and accelerated) bootstrap confidence intervals provide non-parametric uncertainty estimates for any metric:

```python
from scintilla import bca_bootstrap_ci, bootstrap_metric_ci

# BCa CI on raw data (default: full BCa with jackknife).
# Returns a dict with "point", "ci_low" and "ci_high".
ci = bca_bootstrap_ci(data, stat_fn=np.mean, B=2000, alpha=0.05)
ci_lower, ci_upper = ci["ci_low"], ci["ci_high"]

# Percentile bootstrap — faster for large n (> 5 000), skips the O(n) jackknife
ci = bca_bootstrap_ci(
    data, stat_fn=np.mean, B=2000, method="percentile",
)

# Convenience wrapper for a metric computed on (y_true, y_pred)
ci = bootstrap_metric_ci(
    y_true, y_pred, metric_fn=lambda yt, yp: (yt == yp).mean(), B=2000
)
ci_lower, ci_upper = ci["ci_low"], ci["ci_high"]
```

**`bca_bootstrap_ci` parameters**

| Parameter | Default | Description |
|---|---|---|
| `data` | required | 1-D or 2-D array (resampled along axis 0) |
| `stat_fn` | required | Callable reducing the resampled data to one number |
| `stat_fn` | required | Function returning a scalar statistic |
| `B` | `2000` | Number of bootstrap replicates |
| `alpha` | `0.05` | Significance level (default 0.05 → 95 % CI) |
| `seed` | `42` | Random seed |
| `method` | `"bca"` | `"bca"` (full BCa with jackknife) or `"percentile"` (skip jackknife, faster for large *n*) |

### .632+ bootstrap estimator

The .632+ bootstrap corrects for both overfitting and no-information-rate bias:

```python
from scintilla import dot632plus_bootstrap

# Default: analytical no-information rate (exact, zero variance)
estimate = dot632plus_bootstrap(model, X, y, metric_fn, B=200)

# Permutation-based no-information rate (averaged over 50 permutations)
estimate = dot632plus_bootstrap(
    model, X, y, metric_fn, B=200,
    no_info_method="permutation", n_permutations=50,
)
```

**`dot632plus_bootstrap` parameters**

| Parameter | Default | Description |
|---|---|---|
| `model_cls` | required | Unfitted scikit-learn estimator |
| `X` | required | Feature matrix |
| `y` | required | Labels |
| `metric_fn` | required | `metric_fn(y_true, y_pred) → float` (higher is better) |
| `B` | `200` | Bootstrap iterations |
| `seed` | `42` | Random seed |
| `no_info_method` | `"analytical"` | `"analytical"` (sum of squared class proportions — exact for accuracy) or `"permutation"` (average over *n_permutations* random permutations — correct for any metric) |
| `n_permutations` | `50` | Number of permutations when `no_info_method="permutation"` |

> **Note:** The analytical no-information rate `sum(p_k²)` is the exact no-information accuracy but is only an approximation for other metrics (e.g. macro-F1).  On imbalanced datasets with non-accuracy metrics, prefer `no_info_method="permutation"` for an unbiased estimate.  A `UserWarning` is emitted when using `"analytical"` with a non-accuracy metric.

### Paired bootstrap test

Test whether two methods differ significantly:

```python
from scintilla import paired_bootstrap_test

# Compares two sets of predictions against shared ground truth and returns a
# dict with "diff", "ci_low", "ci_high" and "p_value".
result = paired_bootstrap_test(
    y_true, pred_a, pred_b, metric_fn=lambda yt, yp: (yt == yp).mean(), B=10000
)
p_value = result["p_value"]
```

### Rank aggregation

Combine multiple metric rankings into a consensus ranking:

```python
from scintilla import borda_count, rank_aggregate

ranking = borda_count(scores_df, higher_is_better={"ari": True, "runtime": False})
# Or use the generic interface
ranking = rank_aggregate(scores_df, method="borda")
```

### Effect sizes

Standardised effect-size measures for DE and group comparisons:

```python
from scintilla import rank_biserial, cohens_d, hedges_g, cliffs_delta

r = rank_biserial(U=1500, n1=50, n2=60)   # from Mann-Whitney U
d = cohens_d(group1_values, group2_values)
g = hedges_g(group1_values, group2_values)  # bias-corrected
delta = cliffs_delta(group1_values, group2_values)
```

### Adaptive PCA thresholds

Data-driven selection of the number of PCA components:

```python
from scintilla import gavish_donoho_threshold, marchenko_pastur_cutoff

# Gavish-Donoho: universal singular-value threshold
k = gavish_donoho_threshold(singular_values, n_samples, n_features)

# Marchenko-Pastur: noise eigenvalue upper edge
# ⚠️ emits a warning when p/n > 0.8 — consider Gavish-Donoho in that regime
k = marchenko_pastur_cutoff(eigenvalues, n_samples, n_features)

# Use trimmed-mean sigma² estimation for better accuracy:
k = marchenko_pastur_cutoff(
    eigenvalues, n_samples, n_features, sigma_method="trimmed_mean"
)
```

### Adaptive resolution search

Find the optimal Leiden resolution by maximising NVI stability across a grid:

```python
from scintilla import adaptive_resolution_search

# Takes callables, not an AnnData: run_fn(resolution) -> labels, and
# metric_fn(labels) -> score. Returns a dict with "best_resolution",
# "best_score" and "all_scores".
from scintilla.clustering.leiden import leiden_clustering
from sklearn.metrics import adjusted_rand_score

search = adaptive_resolution_search(
    run_fn=lambda res: leiden_clustering(adata, resolution=res),
    metric_fn=lambda labels: adjusted_rand_score(
        adata.obs["cell_type"].values, labels
    ),
    coarse_grid=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5],
)
best_resolution = search["best_resolution"]
```

### NVI normalisation

`nvi_stability` accepts a `normalise` parameter to control how the Variation of Information is normalised:

```python
from scintilla.statistical_tests.adaptive import nvi_stability

# Default: normalise by log(n) — comparable across dataset sizes
result = nvi_stability(labels_at_resolutions, normalise="log_n")

# Max-entropy: normalise by max(H(A), H(B)) — comparable across cluster counts
result = nvi_stability(labels_at_resolutions, normalise="max_entropy")
```

| Parameter | Default | Description |
|---|---|---|
| `labels_at_resolutions` | required | List of `(resolution, labels)` tuples |
| `normalise` | `"log_n"` | `"log_n"` (divide VI by log *n*) or `"max_entropy"` (divide VI by max(H(A), H(B))) |

> Use `"max_entropy"` when comparing clusterings with very different numbers of clusters, since raw VI tends to grow with cluster count even when the clusterings are equally stable.

### Permutation tests and McNemar's test

```python
from scintilla import permutation_test_methods, mcnemar_test

# Returns a dict with "observed_diff" and "p_value".
result = permutation_test_methods(
    y_true, pred_a, pred_b,
    metric_fn=lambda yt, yp: (yt == yp).mean(),
    n_permutations=10000,
)
p_value = result["p_value"]
p_value = mcnemar_test(y_true, y_pred_a, y_pred_b)
```

### Enabling robust statistics globally

Use the `robust()` preset to enable all statistical enhancements at once:

```python
from scintilla import AnalysisConfig

cfg = AnalysisConfig.robust()
# Sets: bootstrap_ci=True, scoring_method="borda", auto_pca_components="gavish_donoho",
#        adaptive_resolution=True, auto_eps=True, no_info_method="permutation",
#        mp_sigma_method="trimmed_mean"

result = sc.unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
```

---

## CLI Reference

### Quick reference

```
scintilla <subcommand> [options]
scintilla --help
scintilla <subcommand> --help
```

| Subcommand | Description |
|---|---|
| `eda` | Dataset summary statistics |
| `preprocess` | Apply a single normalisation transform |
| `normalise` | Benchmark all normalisation transforms |
| `feature-select` | PCA-loadings feature selection |
| `reduce` | Dimensionality reduction |
| `cluster` | Benchmark clustering algorithms |
| `classify` | Benchmark classifiers |
| `de` | Differential expression (two groups) |
| `annotate` | Find marker genes per cluster |
| `batch-correct` | Benchmark batch correction methods |
| `run-all` | Full pipeline (clustering + classification) |
| `benchmark-all` | Full clustering + classification benchmark |
| `generate-config` | Write a commented config template to YAML |

### `run-all`

```bash
scintilla run-all data.h5ad --target-col cell_type --output-dir results/
scintilla run-all data.h5ad --target-col cell_type --fast
scintilla run-all data.h5ad --target-col cell_type --config my_config.yaml --output-dir results/
```

**`run-all` options**

| Option | Default | Description |
|---|---|---|
| `input` | required | Input `.h5ad` |
| `--target-col` | `cell_type` | Class label column |
| `--output-dir` | `results` | Output directory |
| `--config` | `None` | Path to YAML config |
| `--fast` | `False` | Fast preset |
| `--verbose` | `False` | Print progress |

### `generate-config`

```bash
scintilla generate-config
scintilla generate-config --output custom_analysis.yaml
```

**`generate-config` options**

| Option | Default | Description |
|---|---|---|
| `--output` | `scintilla_config.yaml` | Output YAML path |

---

## Performance Tips

1. **Use `AnalysisConfig.fast()`** for exploratory runs. The fast preset runs only `kmeans` + `leiden` for clustering and `LogReg`, `RF`, `kNN` for classification — reducing runtime by 5–10×.

2. **Skip SHAP for large datasets** — SHAP computation scales poorly: `include_shap=False` or `--no-shap`.

3. **Limit classifiers** via `models=["LogReg", "RF"]` or the config. `StackingEnsemble` and `SVM` are particularly slow at >50 000 cells.

4. **Boruta and mRMR feature selection** are not in the default benchmark. Add them explicitly with `methods=["pca_loadings","mutual_information","boruta","mrmr"]` only when you need them.

5. **HDBSCAN** uses all CPU cores (`n_jobs=-1`) and a reduced grid `[10, 20, 50] × [None, 5]` (6 combinations vs the previous 18). If you need a finer grid, override via `AnalysisConfig(hdbscan_min_cluster_sizes=[5,10,15,20,30,50], hdbscan_min_samples=[None,5,10])`.

6. **Box-Cox transform** runs gene-level computation in parallel across all cores. On machines with many cores this is already fast; no further tuning needed.

7. **Normality testing subsamples per gene** to 500 cells by default. If you observe unexpected normality calls, adjust via `check_normality(adata, sample_size=200, threshold=0.2)`.

8. **PCA once** — set `run_pca_first=False` in `unsupervised_analysis` if you already ran `run_pca` upstream.

9. **Bootstrap CIs are resampling-intensive.** The default `n_bootstrap=2000` is a good balance of accuracy and speed. Reduce to 500 for quick exploratory runs; increase to 10 000+ for publication-quality intervals.

10. **`AnalysisConfig.robust()`** enables all statistical enhancements (bootstrap CIs, Borda scoring, adaptive PCA, adaptive resolution, auto eps, permutation-based no-information rate, trimmed-mean MP sigma²). This increases runtime but provides much richer output. Use `default()` or `fast()` for exploratory work and `robust()` for final analyses.

---

## Full Worked Example

```python
import scintilla as sc
from scintilla import AnalysisConfig
from scintilla.io.loaders import auto_detect_format
from scintilla.preprocessing.pca import run_pca
from scintilla.dimensionality_reduction import run_umap
from scintilla.feature_selection import select_hvg
from scintilla.eda.summary import dataset_summary

# ── 1. Load ─────────────────────────────────────────────────────────
adata = auto_detect_format("data/pbmc3k_raw.h5ad")
print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")

# ── 2. EDA ───────────────────────────────────────────────────────────
summary = dataset_summary(adata)
print(f"Sparsity: {summary['sparsity']:.1%}")

# ── 3. Normalise (benchmark + apply best) ───────────────────────────
from scintilla.preprocessing.benchmark import benchmark_transformations
results_df, best_name, adata_norm = benchmark_transformations(adata, verbose=True)
print(f"Best normalisation: {best_name}")

# ── 4. PCA + UMAP ────────────────────────────────────────────────────
adata_pca = run_pca(adata_norm, n_comps=30)
adata_pca  = run_umap(adata_pca, use_rep="X_pca")

# ── 5. Feature selection ─────────────────────────────────────────────
adata_hvg = select_hvg(adata_pca, n_top_genes=2000)

# ── 6. Cluster ───────────────────────────────────────────────────────
# Use a custom config: only leiden + kmeans, with a tight resolution grid
cfg = AnalysisConfig(
    clustering_methods=["kmeans", "leiden"],
    leiden_resolutions=[0.3, 0.5, 0.8, 1.0],
    include_shap=False,
)

cluster_result = sc.unsupervised_analysis(
    adata_hvg,
    cell_type_col="cell_type",
    run_pca_first=False,
    store_labels=True,    # → adata_hvg.obs["scintilla_cluster"]
    config=cfg,
    verbose=True,
)
print("Best clustering:", cluster_result["best_method"])

# ── 7. Classify ───────────────────────────────────────────────────────
cls_result = sc.supervised_analysis(
    adata_hvg,
    target_col="cell_type",
    config=cfg,    # uses LogReg, RF, kNN (set in previous cfg); no SHAP
    verbose=True,
)
print(f"Best classifier: {cls_result['best_model_name']}")

# ── 8. Differential expression ────────────────────────────────────────
from scintilla import wilcoxon_de, filter_de_genes
de_df = wilcoxon_de(adata_hvg, group_col="cell_type", group1="Monocyte", group2="T Cell")
sig_df = filter_de_genes(de_df, alpha=0.05, log2fc_threshold=1.0)
print(f"Significant DE genes: {len(sig_df)}")

# ── 9. Annotate ───────────────────────────────────────────────────────
from scintilla import find_marker_genes
markers_df = find_marker_genes(adata_hvg, groupby="scintilla_cluster", n_genes=20)

# ── 10. Batch correction (if applicable) ─────────────────────────────
from scintilla import benchmark_batch_correction
bc_result = benchmark_batch_correction(
    adata_hvg,
    batch_key="batch",
    label_key="cell_type",
)
print("Best batch correction:", bc_result["best_method"])
adata_final = bc_result["corrected_adatas"][bc_result["best_method"]]

# ── 11. Save ─────────────────────────────────────────────────────────
from scintilla.io.exporters import save_anndata, save_results_csv, save_results_json
save_anndata(adata_final, "results/final.h5ad")
save_results_csv(cluster_result["results_df"], "results/clustering.csv")
save_results_json(
    {"best_cluster": cluster_result["best_method"],
     "best_classifier": cls_result["best_model_name"]},
    "results/summary.json",
)
```

### Equivalent CLI pipeline

```bash
# Step 1 – generate a config template and edit it
scintilla generate-config --output analysis.yaml

# Step 2 – run everything
scintilla run-all data/pbmc3k_raw.h5ad \
    --target-col cell_type \
    --config analysis.yaml \
    --output-dir results/ \
    --verbose
```
