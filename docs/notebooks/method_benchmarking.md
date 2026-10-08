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

# Choosing methods: clustering, seeds and differential expression

scINTILLA's benchmarks return data, never figures, and a method that fails stays in the table with
`status="failed"` and a reason. Drawing is a separate step in `si.pl`.

```{code-cell} ipython3
import scanpy as sc
import scintilla as si

adata = sc.datasets.pbmc68k_reduced()
config = si.AnalysisConfig.fast().copy(
    clustering_methods=["kmeans", "hierarchical", "leiden"], leiden_resolutions=[0.3, 0.6, 1.0]
)
```

## Benchmark clustering methods against the labels

```{code-cell} ipython3
table, labels = si.benchmark.benchmark_clustering(
    adata, cell_type_col="bulk_labels", config=config, random_state=0
)
table.sort_values("ari", ascending=False)[["method", "params", "ari", "ami", "n_clusters", "status"]].head(8)
```

```{code-cell} ipython3
fig = si.pl.clustering_benchmark(table)
```

## Are the differences real? Seed stability

A method that wins on one seed and loses on the next is not a winner. `seed_stability_test` repeats a
method over seeds and reports the spread, an intraclass correlation and a bootstrap interval.

```{code-cell} ipython3
def kmeans_labels(adata, random_state):
    return si.tl.kmeans(adata.obsm["X_pca"], n_clusters=8, random_state=random_state)[0]

def ari(labels):
    return si.stats.adjusted_rand_index(adata.obs["bulk_labels"].to_numpy(), labels)

stability = si.benchmark.seed_stability_test(
    kmeans_labels, adata, ari, n_seeds=10, bootstrap_ci=True, n_bootstrap=500, compute_icc=True
)
stability["metric_value"].describe().round(3)
```

```{code-cell} ipython3
{k: round(v, 3) for k, v in stability.attrs.items() if k in {"ci_low", "ci_high", "icc"}}
```

## Differential expression with effect sizes

The two-group tests only densify the two groups being compared, and report effect sizes next to the
p-values.

```{code-cell} ipython3
de = si.tl.wilcoxon(adata, "bulk_labels", "CD14+ Monocyte", "CD19+ B")
de.sort_values("p_adjusted").head(5)[["gene", "log2fc", "p_adjusted", "rank_biserial", "cliffs_delta"]]
```

```{code-cell} ipython3
fig = si.pl.volcano(de)
```
