"""Regression tests for matplotlib backend ownership and figure lifecycle."""

from __future__ import annotations

import subprocess
import sys
import types
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt


def test_import_scintilla_preserves_preselected_matplotlib_backend() -> None:
    """Importing the package must not overwrite the caller's backend choice."""
    package_root = Path(__file__).resolve().parents[1]
    script = """
import sys
import types
anndata_stub = types.ModuleType("anndata")
anndata_stub.AnnData = object
sys.modules["anndata"] = anndata_stub
import matplotlib
matplotlib.use("svg")
import scintilla
print(matplotlib.get_backend())
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().lower() == "svg"


def test_classifier_comparison_returns_open_figures() -> None:
    """Callers can inspect and close every figure returned by the helper."""
    module_path = Path(__file__).resolve().parents[1] / "scintilla/classification/visualise.py"
    module_spec = importlib.util.spec_from_file_location(
        "_scintilla_visualise_test", module_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    anndata_stub = types.ModuleType("anndata")
    previous_anndata = sys.modules.get("anndata")
    sys.modules["anndata"] = anndata_stub
    try:
        module_spec.loader.exec_module(module)
    finally:
        if previous_anndata is None:
            sys.modules.pop("anndata", None)
        else:
            sys.modules["anndata"] = previous_anndata

    metrics = {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.85,
        "f1": 0.82,
        "fdr": 0.1,
        "fnr": 0.15,
        "auroc": 0.92,
        "auprc": 0.91,
        "mcc": 0.8,
        "cohen_kappa": 0.79,
        "balanced_accuracy": 0.87,
        "confusion_matrix": [[9, 1], [2, 8]],
    }

    try:
        figures = module.plot_classifier_comparison(
            {"all_results": {"model": {"metrics": metrics}}, "best_model_name": "model"}
        )

        assert figures
        assert all(plt.fignum_exists(fig.number) for fig in figures.values())
    finally:
        plt.close("all")
