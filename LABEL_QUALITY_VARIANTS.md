# scINTILLA label-quality variants: how to run them and how to read the scores

scINTILLA gives every cell-type label in a dataset a **quality score**. A low score means "this label may be wrong: have a look". The benchmark tested several versions (variants) of this score. This page explains what each variant is, how to compute all of them with one command, and how to read the output.

Branch: `feature/label-quality-variants` (built on `feature/fragmentation-score`). `master` is unchanged.

---

## 1. The short version

```bash
git checkout feature/label-quality-variants
pip install -e ".[full]"

scintilla label-quality my_data.h5ad \
    --cell-type-col cell_type \
    --use-rep X_pca \
    --output my_scores.csv
```

You get two files:

| File | What is in it |
| --- | --- |
| `my_scores.csv` | One row per label, one column per variant. **Higher = better label.** |
| `my_scores_ranks.csv` | The same table as ranks. **1 = lowest score = review this label first.** |

If you only look at one column, use **`scintilla_composite_frag__silhouette`**. It did best overall in the benchmark (section 4).

---

## 2. The variants

All of them come out of the same run. You do not need to run anything twice.

### Main variants (use these)

| Column | Plain-language description |
| --- | --- |
| `scintilla_composite` | **Original scINTILLA**, the published score. It combines two things: whether a label's cells mix with other labels in the clusterings ("confusion"), and whether classifiers agree, are confident and have low entropy on that label ("supervised consistency"). |
| `scintilla_composite_frag` | **scINTILLA + fragmentation.** The original score plus a third ingredient, *fragmentation*: whether one label is spread over several clusters. This catches merged or contaminated labels, which the original score mostly misses. Confusion, supervised consistency and fragmentation each carry about one third of the weight. |
| `scintilla_composite_frag__silhouette` | **scINTILLA + fragmentation, combined with Silhouette.** Within the dataset, each label gets a percentile rank for the fragmentation score and another for Silhouette, and the two are averaged. Best overall in the benchmark. |
| `scintilla_composite__silhouette` | The same rank average, using the original score in place of the fragmentation score. |
| `fusion_add` | **Learned fusion** of the original scINTILLA rank and the Silhouette rank. It is a logistic regression fitted once on the development perturbations and then frozen; it is never refitted on your data. |
| `fusion_log` | Same as `fusion_add`, plus an interaction term between the two ranks. |

### Ablations and components (for diagnosis, not for routine use)

These show *why* a label scored low.

| Column | What it isolates |
| --- | --- |
| `scintilla_unsup_only` | Confusion only (no classifiers, no fragmentation) |
| `scintilla_sup_only` | Classifier consistency only (agreement, entropy, confidence) |
| `scintilla_unsup_frag_only` | Confusion + fragmentation, no classifiers |
| `scintilla_confusion_top1` | Confusion in the single best clustering |
| `scintilla_confusion_mean` | Confusion averaged over the six best clusterings |
| `scintilla_fragmentation_mean` | Fragmentation averaged over the six best clusterings |
| `scintilla_pred_agreement` | How often the classifiers agree with each other |
| `scintilla_pred_entropy` | Classifier uncertainty, sign-flipped so that high = good |
| `scintilla_pred_avg_confidence` | Mean classifier confidence |
| `silhouette` | Per-label mean Silhouette on the embedding. This is an **input** to the fusions, not a scINTILLA variant. |

The benchmark report also lists `scintilla_composite_old`. That was a check run inside the fragmentation experiment: the original score recomputed with fragmentation switched off. In a single run it is identical to `scintilla_composite`, so it has no separate column.

---

## 3. How to read the scores

1. **Higher = better, for every column.** Components where "high = bad" (confusion, fragmentation, entropy) are sign-flipped, so you never have to remember which way a column points.
2. **Scores are relative to one dataset.** The composites are min–max normalised across the labels of that dataset, and the fusions use percentile ranks within it. A value of 0.4 in one atlas and 0.4 in another do **not** mean the same thing. Compare labels **within** a dataset; do not compare raw values across datasets.
3. **Use the ranks to decide what to review.** In `*_ranks.csv`, rank 1 is the most suspicious label. In practice, review the bottom 3 labels, or the bottom 10%.
4. **What the range of each column means:**
   - `scintilla_composite`, `scintilla_composite_frag` and the `*_only` ablations: between 0 and 1.
   - `*__silhouette`: between 0 and 1. It is an average of two percentile ranks, so 0.5 is typical, and close to 0 means both methods put the label at the bottom.
   - `fusion_add` and `fusion_log`: unbounded numbers. They are the negative of the model's "this label is damaged" log-odds, so negative values point to damage. The models were trained with balanced classes, so **do not** read them as calibrated probabilities. Use them for ranking.
   - Raw components (`scintilla_confusion_*`, `scintilla_fragmentation_mean`, `scintilla_pred_*`): kept in their natural units. Fragmentation of −0.5 means the label is split roughly evenly across two clusters.
5. **Low scores tell you what kind of problem a label might have.** Compare the component columns of a flagged label:

   | Pattern for a low-scoring label | Likely problem |
   | --- | --- |
   | Low `scintilla_fragmentation_mean` (label spread over clusters) | Two populations merged under one name, or contamination |
   | Low `scintilla_sup_only` / `scintilla_pred_agreement`, fragmentation fine | Label is not separable from a neighbour: possible over-splitting |
   | Low `scintilla_confusion_*` | The label's cells sit among cells with other labels |
   | Low `silhouette` only | Label is diffuse in the embedding; check this with the other columns |

6. **A low score is not proof of an error.** It is a ranked list of labels to look at. Some biologically real states (transitional cells, cycling cells) will always score low.

---

## 4. How well each variant did (V4 benchmark)

These are the benchmark results from `results_v4/SCINTILLA_FINAL_DETAILED_REPORT.md`. The setup was five atlases with 150 synthetic perturbations (merge / noise / split, 10 replicates each). AUROC measures how well the score ranks damaged labels below intact ones (0.5 = chance, 1 = perfect). "Balanced" gives merge, noise and split equal weight. Recall@3 is the share of damaged labels found if you review the 3 lowest-scoring labels.

| Variant | Balanced AUROC | Merge | Noise | Split | AP | Recall@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scintilla_composite_frag__silhouette` | **0.874** | 0.779 | 0.925 | 0.917 | 0.564 | 0.389 |
| `fusion_add` | 0.871 | 0.782 | 0.930 | 0.901 | 0.566 | 0.400 |
| `fusion_log` | 0.870 | 0.774 | 0.931 | 0.905 | 0.557 | 0.411 |
| `scintilla_composite__silhouette` | 0.824 | 0.625 | 0.889 | 0.957 | 0.518 | 0.363 |
| `scintilla_composite_frag` | 0.816 | 0.641 | 0.867 | 0.941 | 0.490 | 0.322 |
| `scintilla_unsup_frag_only` | 0.810 | 0.707 | 0.879 | 0.845 | 0.464 | 0.317 |
| `scintilla_fragmentation_mean` | 0.749 | 0.859 | 0.886 | **0.501** | 0.448 | 0.356 |
| `scintilla_pred_agreement` | 0.726 | 0.431 | 0.765 | 0.981 | 0.473 | 0.256 |
| `scintilla_pred_avg_confidence` | 0.720 | 0.479 | 0.703 | 0.978 | 0.432 | 0.178 |
| `scintilla_sup_only` | 0.710 | 0.431 | 0.716 | **0.984** | 0.458 | 0.261 |
| `scintilla_composite` (original) | 0.709 | **0.420** | 0.737 | 0.970 | 0.409 | 0.206 |
| `scintilla_pred_entropy` | 0.695 | 0.431 | 0.687 | 0.968 | 0.452 | 0.233 |
| `scintilla_unsup_only` | 0.680 | 0.435 | 0.719 | 0.886 | 0.309 | 0.106 |
| `scintilla_confusion_mean` | 0.680 | 0.435 | 0.719 | 0.886 | 0.309 | 0.106 |
| `scintilla_confusion_top1` | 0.678 | 0.428 | 0.727 | 0.879 | 0.306 | 0.122 |
| *reference: Silhouette alone* | 0.863 | 0.852 | 0.913 | 0.824 | 0.558 | 0.411 |

What this means in practice:

- **Original scINTILLA is excellent at over-split labels (split 0.97) and close to blind to merged labels (merge 0.42, below chance).**
- **Fragmentation fixes most of that** (merge 0.42 → 0.64, balanced 0.71 → 0.82).
- **Combined with Silhouette it is the best overall (0.874),** but its edge over Silhouette alone is small and not statistically reliable: +0.017, 95% interval −0.006 to +0.043. The honest summary is "complementary to Silhouette", not "better than Silhouette".
- **Fragmentation alone cannot see over-splitting** (split 0.50, chance level). Do not use it on its own.

---

## 5. Running it

### 5.1 Command line

```bash
scintilla label-quality INPUT.h5ad --output OUT.csv [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--cell-type-col` | `cell_type` | Column in `adata.obs` holding the labels to judge |
| `--use-rep` | `X_pca` | Embedding used for clustering and Silhouette. If it is `X_pca` and missing, PCA is computed. |
| `--supervised-use-rep` | none (expression matrix) | Embedding the classifiers train on |
| `--fast` | off | Fewer clustering methods and classifiers. Much quicker, but **not** what the benchmark used. |
| `--config` | none | scINTILLA YAML config (`scintilla generate-config` writes one) |
| `--seed` | from config | Random seed |
| `--n-jobs` | 1 | CPU cores |
| `--rerun` | off | Redo the clustering and classification even if their results are already in the file |
| `--save-h5ad` | none | Also save the analysed AnnData, so you can rescore later without rerunning |

The command checks `adata.obs` first. If the clustering and classifier outputs are already there (for example from `--save-h5ad`), it skips straight to scoring, which takes seconds. Otherwise it runs both analyses. With the default preset that took **10–20 minutes for 20,000 cells on 16 CPUs** in the benchmark.

To match the benchmark settings exactly (default preset, seed 0, shared 30-PC embedding, classifiers on HVG expression):

```bash
scintilla label-quality base.h5ad --cell-type-col label --use-rep X_emb --seed 0 --n-jobs 16 --output scores.csv
```

### 5.2 Python

```python
import scintilla as sc
from scintilla.classification.label_quality_variants import (
    compute_label_quality_variants, review_ranks, VARIANTS)

# 1. the two standard analyses (skip them if already done)
res = sc.unsupervised_analysis(adata, cell_type_col="cell_type", use_rep="X_pca", run_pca_first=False)
adata = res["adata"]
sc.supervised_analysis(adata, target_col="cell_type", check_consistency=True, include_shap=False)

# 2. every variant, one row per label
scores = compute_label_quality_variants(adata, cell_type_col="cell_type", use_rep="X_pca")
ranks = review_ranks(scores)           # 1 = most suspicious

# the 5 labels to look at first, by the best overall variant
print(scores["scintilla_composite_frag__silhouette"].nsmallest(5))
print(VARIANTS)                         # one-line description of every column
```

Requirements: the clustering step must come from this branch (or from `feature/fragmentation-score`), because older code does not write the `scintilla_top*_fragmentation` columns. The function stops with a clear error if they are missing.

---

## 6. Ready-made scores for the five benchmark atlases

The command above was run on the original (unperturbed) labels of the five benchmark atlases. It used the same cells, embedding, preset and seed as the benchmark:

```
/ssu/gassu/shared/soumick/sina/results_label_quality_variants/
    <atlas>_label_quality_variants.csv        # scores, higher = better
    <atlas>_label_quality_variants_ranks.csv  # 1 = review first
    run_base_atlases.sh                       # exact SLURM job that produced them
    logs/                                     # job logs and the git commit used
```

The atlases are `brain`, `eye`, `heoca`, `hnoca` and `lung`. Section 7 describes the check against the benchmark's own saved scores.

---

## 7. Check that the code reproduces the benchmark

- **Unit tests** (`tests/test_label_quality_variants.py`) check that the composites equal `compute_label_quality_score` with the right weights. They also check that the rank fusions and learned fusions follow the V4 formulas exactly, and that rank 1 is the lowest score.
- **Real data:** on the five base atlases, the original composite, its ablations, the components and Silhouette were compared with the benchmark's saved per-label scores (`results/base_scores_<atlas>.csv`). The outcome is recorded in `results_label_quality_variants/VERIFICATION.md`.
- The frozen learned-fusion coefficients are copied verbatim from the benchmark's `frozen_fusion.json` (SHA-256 `e781a9f9…`).

The benchmark *evaluation* (AUROC, AP, Recall@k across the perturbations) is not part of this package. It lives in the separate benchmarking scripts (`Baselines/analyse_v4.py`, run through `launch_v4.sh`).
