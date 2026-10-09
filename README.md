# scINTILLA

[![PyPI](https://img.shields.io/pypi/v/scintilla-py.svg)](https://pypi.org/project/scintilla-py/)
[![Tests](https://github.com/soumickmj/scINTILLA/actions/workflows/test.yaml/badge.svg)](https://github.com/soumickmj/scINTILLA/actions/workflows/test.yaml)
[![Docs](https://readthedocs.org/projects/scintilla/badge/?version=latest)](https://scintilla.readthedocs.io/)
[![Codecov](https://codecov.io/gh/soumickmj/scINTILLA/graph/badge.svg)](https://codecov.io/gh/soumickmj/scINTILLA)
[![Preprint](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.07.27.740477-b31b1b.svg)](https://doi.org/10.64898/2026.07.27.740477)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)

**Single-Cell INTegrated Inference, Labelling, and Landscape Analysis.**

Cell-type labels in a single-cell RNA-seq dataset are rarely checked. scINTILLA scores each label by how
well it can be recovered from the expression data. An unsupervised arm clusters the cells with a panel of
algorithms and measures neighbourhood confusion and fragmentation; a supervised arm trains classifiers and
measures their agreement, entropy and confidence. A low score marks a label that hides several
populations or has been applied inconsistently, which makes it a good place to look first.

Around that core the package benchmarks the methods used at each stage of an analysis (normalisation,
feature selection, dimensionality reduction, clustering, classification, differential expression,
annotation and batch correction), reporting bootstrap intervals, effect sizes and rank aggregation so
that you can tell whether a difference between two methods is real.

scINTILLA works on [AnnData](https://anndata.readthedocs.io) and follows the
[scanpy](https://scanpy.readthedocs.io) conventions (`pp`, `tl`, `pl`, `key_added`, `copy`,
`random_state`), so it fits into an existing scverse workflow. The method is described in the
[preprint](https://doi.org/10.64898/2026.07.27.740477).

## Installation

The package is published on PyPI as **`scintilla-py`**. The import name and the command-line tool are
both `scintilla`; the PyPI project called `scintilla` is an unrelated package. Python 3.11 or newer is
required.

```bash
pip install scintilla-py            # core
pip install "scintilla-py[full]"    # with the optional methods listed below
```

With [uv](https://docs.astral.sh/uv/), either add it to an analysis project or install into an
existing environment:

```bash
uv add "scintilla-py[full]"
uv pip install "scintilla-py[full]"
```

To work on the code, clone the repository and let uv build the environment from the lock file:

```bash
git clone https://github.com/soumickmj/scINTILLA.git
cd scINTILLA
uv sync --locked --extra full --extra test
uv run pytest
```

### Optional methods

The core install covers PCA, the main clustering methods (k-means, hierarchical, DBSCAN, spectral),
the classifiers in scikit-learn, differential expression, annotation, and ComBat. The `full` extra adds
the methods that need a further package. A benchmark leaves out any method whose package is missing.

| Package | What it enables |
|---|---|
| `leidenalg`, `python-igraph` | Leiden clustering and graph layouts |
| `louvain` | Louvain clustering |
| `hdbscan` | HDBSCAN clustering |
| `umap-learn` | UMAP |
| `harmonypy`, `bbknn`, `scanorama` | Harmony, BBKNN and Scanorama batch correction |
| `xgboost`, `lightgbm` | Gradient-boosted classifiers |
| `shap` | SHAP feature importance |
| `Boruta`, `mrmr-selection` | Boruta and mRMR feature selection |
| `scikit-misc` | The Seurat v3 highly variable gene method (the default) |
| `scikit-posthocs` | Dunn's post-hoc test (also the `stats` extra) |
| `mudata` | Reading `.h5mu` files |
| `kneed`, `psutil` | Automatic DBSCAN `eps` and memory profiling |

## Quick start

The shortest route is the label-quality score, shown here on a small example dataset that ships with
scanpy:

```python
import scanpy as sc
import scintilla as si

adata = sc.datasets.pbmc68k_reduced()
scores = si.tl.label_quality(adata, cell_type_col="bulk_labels", fast=True)
scores["scintilla_composite_frag_silhouette"].nsmallest(5)   # the labels to review first
```

The result has one row per label. Every score is oriented so that higher means a better label, and
[the documentation](https://scintilla.readthedocs.io/en/latest/label_quality.html) explains what each
variant measures and how it was validated. The same analysis runs from the command line:

```bash
scintilla label-quality data.h5ad --cell-type-col cell_type --output scores.csv
```

### A fuller analysis

The individual steps can also be run one at a time. Raw counts stay in `adata.X`; each transformation
writes a layer, and everything else is stored under a name you choose with `key_added`:

```python
adata = si.io.load_h5ad("data/pbmc3k.h5ad")                 # or .csv / .h5mu via si.io.auto_detect_format

si.pp.log_shift_size_factor(adata)                          # writes adata.layers["log_shift_size_factor"]
adata.X = adata.layers["log_shift_size_factor"]
si.pp.pca(adata, n_comps=30)                                # adata.obsm["X_pca"]

clustering = si.tl.unsupervised_analysis(adata, cell_type_col="cell_type")
print(clustering["best_method"])                            # best labels: adata.obs["scintilla_cluster"]

classification = si.tl.supervised_analysis(adata, target_col="cell_type")
print(classification["best_model_name"])
```

Three presets control how much work is done. `AnalysisConfig.fast()` runs a few quick methods,
`AnalysisConfig()` is the default panel, and `AnalysisConfig.robust()` adds bootstrap confidence
intervals, rank aggregation and adaptive settings for publication-quality comparisons:

```python
cfg = si.AnalysisConfig.robust()
clustering = si.tl.unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg)
```

A configuration can be written to YAML with `scintilla generate-config` and passed back with
`--config`. The whole pipeline is also available as a single command:

```bash
scintilla run-all data/pbmc3k.h5ad --target-col cell_type --output-dir results/
scintilla run-all data/pbmc3k.h5ad --target-col cell_type --fast --output-dir results/
```

## What is in the package

| Namespace | Contents |
|---|---|
| `si.pp` | Fourteen normalisations and transformations, PCA (with Gavish-Donoho and Marchenko-Pastur rules for the number of components), highly variable genes, batch correction (ComBat, Harmony, BBKNN, Scanorama) |
| `si.tl` | k-means, hierarchical, DBSCAN, HDBSCAN, spectral, Leiden, Louvain and consensus clustering; UMAP, t-SNE, diffusion maps and force-directed layouts; classifiers; label quality; differential expression (Wilcoxon, t, permutation, pseudobulk, scanpy's `rank_genes_groups`); marker-based annotation, label transfer and over-representation analysis |
| `si.pl` | Plots for every benchmark and for label quality |
| `si.benchmark` | Benchmarks for normalisation, clustering, classifiers, feature selection, embeddings, differential expression and batch correction, plus seed-stability and scalability tests |
| `si.stats` | Bootstrap and permutation tests, effect sizes, multiple-testing correction, rank aggregation, ANOVA, Kruskal-Wallis, Box's M and Dunn's test |
| `si.eda` and `si.io` | Dataset summaries; reading and writing `.h5ad`, CSV and `.h5mu` |

Every analysis function accepts a dense or a sparse matrix. A few steps cannot avoid building a dense
array (some classifiers, the per-gene tests and hierarchical clustering among them); scINTILLA warns before
allocating more than `si.settings.dense_warning_gb` (4 GiB by default), and the usual remedy is to work on
a PCA representation with `use_rep="X_pca"`. The [manual](https://scintilla.readthedocs.io/en/latest/manual.html)
says which functions are affected.

## Documentation

* [Tutorials](https://scintilla.readthedocs.io/en/latest/notebooks/index.html), executed on a scanpy example dataset: judging label quality, and choosing between clustering methods and classifiers
* [User guide](https://scintilla.readthedocs.io/en/latest/guide.html), a task-by-task walkthrough
* [Manual](https://scintilla.readthedocs.io/en/latest/manual.html), the full reference to each stage, the command-line interface and `AnalysisConfig`
* [API reference](https://scintilla.readthedocs.io/en/latest/api.html)
* [What each label-quality score means](https://scintilla.readthedocs.io/en/latest/label_quality.html)
* [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## Citation

If you use scINTILLA, please cite the preprint (see also [`CITATION.cff`](CITATION.cff)):

> Kanannejad S, Bongiorni N, Nordera E, Redaelli S, Rusconi I, Zanin R, Giustacchini A, Chatterjee S.
> *scINTILLA: Single-Cell Integrated Inference, Labelling, and Landscape Analysis for Cell-Type
> Annotation Quality Assessment.* bioRxiv (2026). doi:10.64898/2026.07.27.740477

## Licence

Apache-2.0. See [LICENSE](LICENSE).
