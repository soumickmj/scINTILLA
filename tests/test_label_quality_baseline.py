"""Pin the label-quality outputs recorded from the 0.1.0 code.

The scorers are published (bioRxiv 10.64898/2026.07.27.740477), so the API rework
must not move their numbers or the default ``obs`` column names. If an upstream
dependency legitimately changes them, regenerate with ``tests/make_baseline.py``
and record it in the changelog.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
pytest.importorskip("leidenalg")

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def pipeline_output():
    from make_baseline import run_pipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline()


def test_default_obs_column_names_are_unchanged(pipeline_output):
    adata, _ = pipeline_output
    assert sorted(adata.obs.columns) == json.loads((DATA / "label_quality_obs_columns.json").read_text())


def test_label_quality_scores_match_recorded_baseline(pipeline_output):
    _, scores = pipeline_output
    expected = pd.read_csv(DATA / "label_quality_baseline.csv", index_col=0)
    assert list(scores.columns) == list(expected.columns)
    assert [str(i) for i in scores.index] == [str(i) for i in expected.index]
    # The PCA and the silhouette are float32, so another BLAS build moves the last digits (seen: 3e-6
    # relative on one silhouette value). A real change in a scorer is many orders larger.
    np.testing.assert_allclose(scores.to_numpy(float), expected.to_numpy(float), rtol=1e-5, atol=1e-9)
