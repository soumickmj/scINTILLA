---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Which of my cell-type labels can I trust?

Large atlases come with cell-type labels whose quality is rarely examined. scINTILLA scores every
label by how well a panel of clustering algorithms and classifiers can recover it
({cite:t}`Kanannejad2026`). Low-scoring labels are the ones worth a second look.

This notebook runs the whole workflow on scanpy's `pbmc68k_reduced` example (700 cells), which has
coarse FACS-based labels in `obs["bulk_labels"]`.

```{code-cell} ipython3
import scanpy as sc
import scintilla as si

si.settings.verbosity = "warning"   # set to "info" to follow the progress of each arm

adata = sc.datasets.pbmc68k_reduced()
adata.obs["bulk_labels"].value_counts()
```

## One call

`si.tl.label_quality` runs the unsupervised arm (clustering algorithms, neighbourhood confusion,
fragmentation) and the supervised arm (classifier agreement, entropy, confidence), then scores each
label with every variant described in {doc}`../label_quality`. Every score is oriented so that
**higher means a better label**.

```{code-cell} ipython3
scores = si.tl.label_quality(adata, cell_type_col="bulk_labels", fast=True, random_state=0)
scores[["scintilla_composite", "scintilla_composite_frag__silhouette", "silhouette"]].round(3).sort_values(
    "scintilla_composite_frag__silhouette"
)
```

The labels to review first are the ones at the top of this table. `review_ranks` turns the scores
into ranks, where 1 means most suspicious:

```{code-cell} ipython3
si.tl.review_ranks(scores)["scintilla_composite_frag__silhouette"].sort_values().head(5)
```

## What was written to the object

Both arms leave per-cell columns in `adata.obs`, so you can look at *which cells* make a label
look doubtful, not only at the label as a whole. Nothing else is added: no neighbour graph, no
stray cluster column.

```{code-cell} ipython3
[c for c in adata.obs.columns if c.startswith(("scintilla_", "pred_"))][:8]
```

```{code-cell} ipython3
fig = si.pl.celltype_label_quality(adata, cell_type_col="bulk_labels")
```

## Reproducibility

All stochastic steps take `random_state`, and the call is deterministic for a fixed seed:

```{code-cell} ipython3
again = si.tl.label_quality(adata.copy(), cell_type_col="bulk_labels", fast=True, random_state=0)
(again.round(8) == scores.round(8)).all().all()
```
