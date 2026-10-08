# scINTILLA

[![PyPI](https://img.shields.io/pypi/v/scintilla-py.svg)](https://pypi.org/project/scintilla-py/)
[![Tests](https://github.com/soumickmj/scINTILLA/actions/workflows/test.yaml/badge.svg)](https://github.com/soumickmj/scINTILLA/actions/workflows/test.yaml)
[![Docs](https://readthedocs.org/projects/scintilla-py/badge/?version=latest)](https://scintilla-py.readthedocs.io/)
[![Codecov](https://codecov.io/gh/soumickmj/scINTILLA/graph/badge.svg)](https://codecov.io/gh/soumickmj/scINTILLA)
[![Preprint](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.07.27.740477-b31b1b.svg)](https://doi.org/10.64898/2026.07.27.740477)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)

**Single-Cell INTegrated Inference, Labelling, and Landscape Analysis.**

scINTILLA scores how learnable and internally consistent each cell-type label in a single-cell
RNA-seq dataset is, by combining an unsupervised arm (a panel of clustering algorithms,
neighbourhood confusion and fragmentation) with a supervised arm (classifier agreement, entropy
and confidence). A low score flags a label that hides heterogeneity or is applied inconsistently.
It also benchmarks the methods used at each stage of an analysis (normalisation, feature selection,
dimensionality reduction, clustering, classification, differential expression, annotation and batch
correction) with bootstrap intervals, effect sizes and rank aggregation.

It works on [AnnData](https://anndata.readthedocs.io), follows the [scanpy](https://scanpy.readthedocs.io)
conventions (`pp`, `tl`, `pl`, `key_added`, `copy`, `random_state`) and is built for the
[scverse](https://scverse.org) ecosystem.

## Installation

The distribution is `scintilla-py`; the import name and the command-line tool are `scintilla`
(the PyPI project called `scintilla` is unrelated). Python 3.11 or newer is required.

```bash
pip install scintilla-py            # core
pip install "scintilla-py[full]"    # + Leiden, Harmony, XGBoost, SHAP, MuData, ...
```

## Quick start

```python
import scanpy as sc
import scintilla as si

adata = sc.datasets.pbmc68k_reduced()
scores = si.tl.label_quality(adata, cell_type_col="bulk_labels", fast=True)
scores["scintilla_composite_frag__silhouette"].nsmallest(5)   # the labels to review first
```

Every score is oriented so that higher means a better label. The same analysis is available from
the command line:

```bash
scintilla label-quality data.h5ad --cell-type-col cell_type --output scores.csv
```

## Documentation

* [Tutorials](https://scintilla-py.readthedocs.io/en/latest/notebooks/index.html) executed on a scanpy example dataset
* [API reference](https://scintilla-py.readthedocs.io/en/latest/api.html)
* [What each score means](https://scintilla-py.readthedocs.io/en/latest/label_quality.html)
* [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## Citation

If you use scINTILLA, please cite the preprint (see also [`CITATION.cff`](CITATION.cff)):

> Kanannejad S, Bongiorni N, Nordera E, Redaelli S, Rusconi I, Zanin R, Giustacchini A, Chatterjee S.
> *scINTILLA: Single-Cell Integrated Inference, Labelling, and Landscape Analysis for Cell-Type
> Annotation Quality Assessment.* bioRxiv (2026). doi:10.64898/2026.07.27.740477
