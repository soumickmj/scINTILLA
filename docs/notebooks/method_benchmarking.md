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

# Choosing methods: clustering, classifiers, seeds and differential expression

There is rarely one obviously right clustering algorithm or classifier for a dataset, so scINTILLA
lets you run a panel of them against the labels you already have and compare the results. The
benchmarks return tables, never figures, and a method that fails stays in the table with
`status="failed"` and a reason. Drawing is a separate step in `si.pl`.

```{code-cell} ipython3
import scanpy as sc
import scintilla as si

adata = sc.datasets.pbmc68k_reduced()
config = si.AnalysisConfig.fast().copy(
    clustering_methods=["kmeans", "hierarchical", "leiden"],
    leiden_resolutions=[0.3, 0.6, 1.0],
    classifiers=["LogReg", "RF", "kNN"],
)
```

## Which clustering method recovers the labels?

`benchmark_clustering` asks which algorithm, with which distance metric and parameters, reproduces
a set of known labels best. It takes the labels from `adata.obs[cell_type_col]`, counts how many
there are, and clusters a representation of the cells (by default the PCA in `obsm["X_pca"]`,
which removes noise and suits distance-based algorithms). The methods are:

* **k-means**, in three forms. The standard form uses Euclidean distance with either k-means++ or
  random seeding. Spherical k-means normalises each cell to unit length first, which makes it a
  clustering by cosine similarity and often suits expression data, where the direction of a profile
  matters more than its size. Bisecting k-means splits the data from the top down and copes better
  when clusters differ a lot in size.
* **Hierarchical clustering**, tried over a grid of distance metrics (Euclidean, cosine, Manhattan
  and others) and linkages. Ward linkage minimises within-cluster variance and needs Euclidean
  distance; complete and average linkage use the largest or the mean distance between members.
* **DBSCAN** finds dense regions and does not need the number of clusters in advance. The benchmark
  sweeps its `eps` (the largest distance at which two cells count as neighbours) for each metric.
* **Leiden and Louvain** build a nearest-neighbour graph and sweep a resolution parameter. A higher
  resolution gives more, smaller clusters.

Each setting is scored with the adjusted Rand index (ARI), which is 1 for a perfect match with the
labels and about 0 for a random assignment. Because it is corrected for chance, an algorithm cannot
score well merely by producing many clusters. The adjusted mutual information (AMI) is reported
alongside it.

```{code-cell} ipython3
table, labels = si.benchmark.benchmark_clustering(
    adata, cell_type_col="bulk_labels", config=config, random_state=0
)
table.sort_values("ari", ascending=False)[["method", "params", "ari", "ami", "n_clusters", "status"]].head(8)
```

`table` is the leaderboard, one row per method and setting. `labels` holds the cluster assignment
of every cell for every setting, keyed by method.

```{code-cell} ipython3
fig = si.pl.clustering_benchmark(table)
```

```{admonition} Large datasets
Hierarchical clustering builds an all-pairs distance matrix, so it becomes slow and memory-hungry
well before the other methods do. Above roughly 10,000 cells, benchmark a random subsample
(`sc.pp.subsample(adata, n_obs=10000, copy=True)`) and apply the method you pick to the full data.
```

## Which classifier, and in which space?

`benchmark_classifiers` does the same for supervised models. Every classifier is evaluated twice,
once on the genes themselves ("Gene" space) and once on the principal components ("PCA" space), so
that you can see whether the extra step helps. By default it uses stratified five-fold
cross-validation, so each cell is predicted by a model that never saw it, and the folds keep the
cell-type proportions of the whole dataset. The PCA is fitted on the training cells of each fold only,
so nothing leaks from the cells being predicted.

The panel is logistic regression (a sensible baseline), random forest (good at non-linear
relationships between genes), a support vector machine with an RBF kernel (flexible class
boundaries), a small neural network, linear and quadratic discriminant analysis (which assume
particular distributions), k-nearest neighbours, scikit-learn's gradient boosting and a stacking
ensemble, plus XGBoost and LightGBM when they are installed. `config.classifiers` chooses which of
them run. For every model and space the table holds accuracy, balanced accuracy,
precision, recall, F1, the false discovery and false negative rates, Matthews correlation and the
area under the ROC and precision-recall curves, each with its spread over the folds.

```{code-cell} ipython3
results = si.benchmark.benchmark_classifiers(
    adata, target_col="bulk_labels", config=config, random_state=0, verbose=False
)
results[["space", "model", "status", "mean_accuracy", "se_accuracy", "mean_f1", "mean_mcc"]].round(3)
```

A model can fail legitimately. Quadratic discriminant analysis, for instance, cannot estimate a
covariance matrix for a cell type that has fewer cells than there are features, and says so in
`failure_reason` instead of disappearing from the table.

```{code-cell} ipython3
fig = si.pl.classification_benchmark(results)
```

## Are the differences real? Seed stability

A method that wins on one seed and loses on the next is not a winner. `seed_stability_test` repeats
a method over several seeds and reports the spread, an intraclass correlation and a bootstrap
interval.

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
