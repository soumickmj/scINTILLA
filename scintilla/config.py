"""Global configuration constants for scintilla."""

RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_PCA_COMPS = 30
DEFAULT_CV_FOLDS = 5
DEFAULT_BOOTSTRAP_B = 2000
DEFAULT_BOOTSTRAP_ALPHA = 0.05
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.3
DEFAULT_VARIANCE_THRESHOLD = 0.80

DISTANCE_METRICS = ['euclidean', 'cosine', 'manhattan', 'correlation', 'chebyshev', 'canberra', 'braycurtis']
LINKAGE_METHODS = ['complete', 'average', 'ward', 'single']
DBSCAN_EPS_RANGE = [0.3, 0.5, 1.0, 2.0, 5.0]
DBSCAN_MIN_SAMPLES_RANGE = [2, 3, 4, 5]
LEIDEN_RESOLUTIONS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
TRANSFORMATION_BENCHMARK_WEIGHTS = {
    'knn_overlap': 0.35,
    'silhouette': 0.25,
    'shapiro': 0.05,
    'anderson': 0.05,
    'pca_preservation': 0.30,
}
HDBSCAN_MIN_CLUSTER_SIZE_RANGE = [10, 20, 50]
HDBSCAN_MIN_SAMPLES_RANGE = [None, 5]
LOUVAIN_RESOLUTIONS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
SPECTRAL_N_CLUSTERS_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
FEATURE_SELECTION_METHODS = [
    'pca_loadings', 'hvg_seurat_v3', 'hvg_pearson_residuals',
    'mutual_information', 'boruta', 'mrmr',
]
