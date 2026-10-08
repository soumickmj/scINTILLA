# scINTILLA

**Single-Cell INTegrated Inference, Labelling, and Landscape Analysis**

scINTILLA scores how learnable and internally consistent each cell-type label in a single-cell
RNA-seq dataset is, and benchmarks the methods used to get there. It works on
{class}`~anndata.AnnData`, follows the {mod}`scanpy` conventions and is built for the
[scverse](https://scverse.org) ecosystem.

```python
import scintilla as si

scores = si.tl.label_quality(adata, cell_type_col="cell_type")   # one row per label, higher = better
scores["scintilla_composite_frag__silhouette"].nsmallest(5)      # the five labels to review first
```

The method is described in {cite:t}`Kanannejad2026`.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Tutorials
:link: notebooks/index
:link-type: doc
Label quality and method benchmarking, executed on a scanpy example dataset.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
Every public function, grouped as `pp`, `tl`, `pl`, `stats` and `benchmark`.
:::

:::{grid-item-card} Label-quality variants
:link: label_quality
:link-type: doc
What each score means, and how it was validated.
:::
::::

## Installation

The distribution is **`scintilla-py`**; the import name and the command-line tool are
`scintilla`. (The PyPI project called `scintilla` is unrelated.)

```bash
pip install scintilla-py            # core
pip install "scintilla-py[full]"    # + Leiden, Harmony, XGBoost, SHAP, MuData, ...
```

Python 3.11 or newer is required.

```{toctree}
:hidden:
:maxdepth: 2

notebooks/index
api
label_quality
guide
manual
changelog
contributing
references
```
