"""Regression tests for visible normality-probe failures."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scintilla.preprocessing import normality


def test_shapiro_failures_count_against_all_features_and_are_reported(monkeypatch) -> None:
    """Catch failed genes disappearing from normality pass-fraction denominator."""
    adata = ad.AnnData(
        X=np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0], [4.0, 10.0]]),
        var=pd.DataFrame(index=["healthy", "broken"]),
    )

    monkeypatch.setattr(
        normality.stats,
        "anderson",
        lambda *_args, **_kwargs: SimpleNamespace(
            statistic=0.1,
            critical_values=np.array([0.5, 0.5, 0.5]),
        ),
    )

    def fake_shapiro(values):
        if np.all(values == 10.0):
            raise ValueError("constant input")
        return 0.9, 0.5

    monkeypatch.setattr(normality.stats, "shapiro", fake_shapiro)

    with pytest.warns(UserWarning, match="Shapiro-Wilk failed for 1 of 2 features"):
        _, report = normality.check_normality(adata)

    assert report["shapiro_pass_fraction"] == 0.5
    assert report["shapiro_failure_count"] == 1
    assert report["shapiro_failure_events"] == 1
    assert report["shapiro_failed_features"] == ["broken"]
