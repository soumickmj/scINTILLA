"""Script: run the full scINTILLA pipeline."""

import argparse
import os
from scintilla.io.loaders import auto_detect_format
from scintilla.preprocessing.benchmark import benchmark_transformations
from scintilla.preprocessing.pca import run_pca
from scintilla.clustering.run import unsupervised_analysis
from scintilla.classification.run import supervised_analysis
from scintilla.io.exporters import save_results_csv, save_results_json, save_anndata


def main():
    parser = argparse.ArgumentParser(description="Run full scINTILLA pipeline")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--target-col", required=True, help="Target/cell-type column")
    parser.add_argument("--output-dir", default="pipeline_results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    adata = auto_detect_format(args.input)

    print("Benchmarking normalizations...")
    results_norm, best_transform, adata_t = benchmark_transformations(adata, verbose=args.verbose)
    save_results_csv(results_norm, os.path.join(args.output_dir, "normalization_benchmark.csv"))
    print(f"  Best transform: {best_transform}")

    print("Running PCA...")
    adata_t = run_pca(adata_t)

    # Copy obs from original so metadata (e.g. cell_type) is available
    for col in adata.obs.columns:
        adata_t.obs[col] = adata.obs[col].values

    save_anndata(adata_t, os.path.join(args.output_dir, "preprocessed.h5ad"))

    print("Running clustering...")
    clust_result = unsupervised_analysis(
        adata_t, cell_type_col=args.target_col, run_pca_first=False, verbose=args.verbose
    )
    save_results_csv(clust_result["results_df"], os.path.join(args.output_dir, "clustering_results.csv"))
    print(f"  Best method: {clust_result['best_method']}")

    print("Running classification...")
    cls_result = supervised_analysis(adata_t, target_col=args.target_col, verbose=args.verbose)
    print(f"  Best classifier: {cls_result['best_model_name']}")
    summary = {
        "best_transform": best_transform,
        "best_cluster_method": clust_result["best_method"],
        "best_classifier": cls_result["best_model_name"],
        "normality": cls_result["normality"],
    }
    save_results_json(summary, os.path.join(args.output_dir, "pipeline_summary.json"))
    print(f"Pipeline complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
