# API reference

All functions follow one contract: the first argument is an {class}`~anndata.AnnData`;
results are written to `.obs`, `.var`, `.obsm`, `.layers` or `.uns["scintilla"]` under a name
set by `key_added`; `copy=False` modifies the object in place and returns `None`, while
`copy=True` returns a modified copy; stochastic functions take `random_state`. See the
{doc}`changelog` for how this differs from 0.1.

```{eval-rst}
.. currentmodule:: scintilla

Settings
--------

.. autosummary::
   :toctree: generated

   settings

Preprocessing: ``pp``
---------------------

.. autosummary::
   :toctree: generated

   pp.pca
   pp.highly_variable_genes
   pp.select_features
   pp.check_normality
   pp.aggregate_by_patient_celltype
   pp.log_shift_size_factor
   pp.log_cpm_transform
   pp.log_alpha_transform
   pp.arcsinh_transform
   pp.log_shift_scale_by_std
   pp.log_shift_size_factor_z
   pp.log_shift_size_factor_hvg
   pp.log_shift_hvg_z
   pp.normalise_scran
   pp.normalise_tmm
   pp.box_cox_transform
   pp.pearson_residuals_transform
   pp.glm_pca_transform
   pp.sanity_transform

Tools: ``tl``
-------------

Embeddings and clustering
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   tl.umap
   tl.tsne
   tl.diffmap
   tl.draw_graph
   tl.leiden
   tl.louvain
   tl.kmeans
   tl.hierarchical
   tl.spectral
   tl.spectral_grid_search
   tl.dbscan
   tl.estimate_eps
   tl.hdbscan
   tl.consensus

Label quality and pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   tl.label_quality
   tl.compute_label_quality_variants
   tl.review_ranks
   tl.unsupervised_analysis
   tl.supervised_analysis

Differential expression and annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   tl.wilcoxon
   tl.ttest
   tl.permutation
   tl.pseudobulk
   tl.pseudobulk_by_celltype
   tl.rank_genes_groups
   tl.filter_de_genes
   tl.volcano_plot_data
   tl.annotate_by_markers
   tl.transfer_labels
   tl.find_marker_genes
   tl.ora_test

Batch correction
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   tl.combat
   tl.harmony
   tl.bbknn
   tl.scanorama

Plotting: ``pl``
----------------

.. autosummary::
   :toctree: generated

   pl.clustering_benchmark
   pl.classification_benchmark
   pl.classifier_comparison
   pl.celltype_label_quality
   pl.celltype_confusion
   pl.celltype_confusion_network
   pl.metric_correlation
   pl.embedding
   pl.compare_embeddings
   pl.pca_2d
   pl.pca_3d_multiview
   pl.cumulative_variance
   pl.volcano
   pl.ma
   pl.boxplot_genes_by_group
   pl.expression_heatmap
   pl.sankey
   pl.dendrogram
   pl.batch_correction_comparison
   pl.before_after_distribution

Benchmarks: ``benchmark``
-------------------------

.. autosummary::
   :toctree: generated

   benchmark.benchmark_transformations
   benchmark.benchmark_clustering
   benchmark.benchmark_classifiers
   benchmark.benchmark_feature_selection
   benchmark.hvg_sensitivity_analysis
   benchmark.benchmark_embeddings
   benchmark.benchmark_de
   benchmark.benchmark_batch_correction
   benchmark.estimate_benchmark_time
   benchmark.print_time_budget
   benchmark.seed_stability_test
   benchmark.scalability_sweep
   benchmark.pairwise_method_comparison
   benchmark.profile_method
   benchmark.BenchmarkReport

Exploratory data analysis and I/O: ``eda``, ``io``
--------------------------------------------------

.. autosummary::
   :toctree: generated

   eda.dataset_summary
   eda.expressed_genes
   eda.sample_counts
   eda.mean_expression_by_group
   io.load_h5ad
   io.load_csv
   io.auto_detect_format
   io.ensure_anndata
   io.save_anndata
   io.save_results_csv
   io.save_results_json

Configuration
-------------

.. autosummary::
   :toctree: generated

   AnalysisConfig
   generate_default_yaml

Statistics layer: ``stats``
---------------------------

Array-level functions that the AnnData-level tools are built on. They take NumPy arrays or
pandas objects.

.. autosummary::
   :toctree: generated

   stats.bca_bootstrap_ci
   stats.bootstrap_metric_ci
   stats.paired_bootstrap_test
   stats.dot632plus_bootstrap
   stats.jackknife_after_bootstrap
   stats.cohens_d
   stats.hedges_g
   stats.cliffs_delta
   stats.rank_biserial
   stats.correct_pvalues
   stats.borda_count
   stats.rank_aggregate
   stats.gavish_donoho_threshold
   stats.marchenko_pastur_cutoff
   stats.kneedle_elbow
   stats.adaptive_resolution_search
   stats.nvi_stability
   stats.permutation_test_methods
   stats.mcnemar_test
   stats.anova_per_gene
   stats.kruskal_per_gene
   stats.dunn_posthoc
   stats.box_m_test
   stats.adjusted_rand_index
   stats.normalised_mutual_info
   stats.silhouette
   stats.comprehensive_clustering_metrics
   stats.bootstrap_clustering_metrics
   stats.calculate_metrics
   stats.trustworthiness
```
