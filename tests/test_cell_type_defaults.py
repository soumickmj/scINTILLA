"""Executable contracts for the canonical cell-type column default."""

import argparse
import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scintilla.classification.run import supervised_analysis
from scintilla.classification.visualise import compute_label_quality_score


def _classification_adata(label_column: str = "cell_type") -> ad.AnnData:
    """Return a separable two-class dataset with a named label column."""
    adata = ad.AnnData(
        X=np.array([
            [0.0, 0.1], [0.1, 0.0], [0.2, 0.1], [0.0, 0.2],
            [5.0, 5.1], [5.1, 5.0], [5.2, 5.1], [5.0, 5.2],
        ]),
        obs=pd.DataFrame({label_column: ["alpha"] * 4 + ["beta"] * 4}),
    )
    adata.obs["scintilla_top_confusion"] = [0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.8, 0.9]
    return adata


def test_supervised_analysis_defaults_to_the_canonical_cell_type_column() -> None:
    """Catch the legacy ``target`` default rejecting standard AnnData labels."""
    result = supervised_analysis(
        _classification_adata(),
        normality=True,
        models=["LogReg"],
        include_shap=False,
        verbose=False,
    )

    assert result["best_model_name"] == "LogReg"


def test_label_quality_defaults_to_the_canonical_cell_type_column() -> None:
    """Catch visualization helpers still looking for the legacy ``CellType`` name."""
    adata = _classification_adata()

    quality = compute_label_quality_score(adata)

    assert set(quality.index) == {"alpha", "beta"}
    assert "scintilla_label_quality" in adata.obs


@pytest.mark.parametrize(
    ("module_name", "destination"),
    [
        ("classify", "target_col"),
        ("cluster", "cell_type_col"),
        ("run_all", "target_col"),
    ],
)
def test_cli_label_arguments_default_to_the_canonical_cell_type_column(
    module_name: str, destination: str,
) -> None:
    """Catch CLI subcommands requiring a label name despite the public default."""
    parser = argparse.ArgumentParser()
    command = importlib.import_module(f"scintilla.cli.commands.{module_name}")
    command.add_args(parser)

    args = parser.parse_args(["input.h5ad"])

    assert getattr(args, destination) == "cell_type"
