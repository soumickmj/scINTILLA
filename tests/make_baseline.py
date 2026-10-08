"""Regenerate the label-quality numerical baseline in ``tests/data``.

Only run this deliberately, when a change is *meant* to move the label-quality
numbers (or an upstream dependency changed them) and the changelog says so::

    python tests/make_baseline.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _synthetic import make_labelled_blobs  # noqa: E402

from scintilla.analysis_config import AnalysisConfig  # noqa: E402
from scintilla.classification.label_quality_variants import compute_label_quality_variants  # noqa: E402
from scintilla.classification.run import supervised_analysis  # noqa: E402
from scintilla.clustering.run import unsupervised_analysis  # noqa: E402

DATA = Path(__file__).parent / "data"


def run_pipeline():
    adata = make_labelled_blobs()
    cfg = AnalysisConfig.fast().copy(
        include_shap=False, leiden_resolutions=[0.5, 1.0], clustering_methods=["kmeans", "leiden"]
    )
    unsupervised_analysis(adata, cell_type_col="cell_type", config=cfg, random_state=0, verbose=False)
    supervised_analysis(
        adata, target_col="cell_type", check_consistency=True, include_shap=False,
        config=cfg, random_state=0, verbose=False,
    )
    return adata, compute_label_quality_variants(adata, cell_type_col="cell_type")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    adata, scores = run_pipeline()
    scores.to_csv(DATA / "label_quality_baseline.csv")
    (DATA / "label_quality_obs_columns.json").write_text(json.dumps(sorted(adata.obs.columns), indent=1) + "\n")
    print(scores.round(4))
