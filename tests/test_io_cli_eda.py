"""I/O, MuData handling, EDA on sparse data, the settings/logging system and the CLI."""

from __future__ import annotations

import json
import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scintilla as si

# ── EDA ─────────────────────────────────────────────────────────────────


def test_dataset_summary_matches_between_dense_and_sparse(adata_dense, adata_csr):
    dense, sp = si.eda.dataset_summary(adata_dense), si.eda.dataset_summary(adata_csr)
    assert dense.keys() == sp.keys()
    for key in ("n_cells", "n_genes", "obs_columns", "var_columns"):
        assert dense[key] == sp[key]
    for key in ("total_counts", "mean_counts_per_cell", "sparsity"):
        assert dense[key] == pytest.approx(sp[key], rel=1e-9)


def test_eda_never_densifies_a_sparse_matrix(adata_csr, monkeypatch):
    def boom(self, *a, **k):
        raise AssertionError("the matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", boom)
    summary = si.eda.dataset_summary(adata_csr)
    assert 0.0 <= summary["sparsity"] <= 1.0
    assert len(si.eda.expressed_genes(adata_csr, min_cells=1)) > 0


def test_mean_expression_by_group_only_densifies_the_small_result(adata_csr, monkeypatch):
    original = sparse.csr_matrix.toarray
    sizes = []

    def spy(self, *a, **k):
        sizes.append(self.shape)
        return original(self, *a, **k)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", spy)
    assert si.eda.mean_expression_by_group(adata_csr, "cell_type", ["gene0", "gene1"]).shape == (4, 2)
    assert sizes == [(4, 2)]  # groups x selected genes, never cells x genes


def test_sparsity_is_computed_from_stored_non_zeros():
    X = sparse.csr_matrix(np.array([[0, 0, 1.0, 0], [0, 0, 0, 0]]))
    assert si.eda.dataset_summary(ad.AnnData(X=X))["sparsity"] == pytest.approx(7 / 8)


def test_mean_expression_by_group_matches_pandas(adata_both):
    df = pd.DataFrame(np.asarray(adata_both.X.todense() if sparse.issparse(adata_both.X) else adata_both.X))
    df["g"] = adata_both.obs.cell_type.to_numpy()
    expected = df.groupby("g").mean()
    got = si.eda.mean_expression_by_group(adata_both, "cell_type")
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-5)
    assert list(got.index) == list(expected.index)


def test_sample_counts_and_validation(adata_dense):
    assert si.eda.sample_counts(adata_dense, "cell_type").tolist() == [75] * 4
    with pytest.raises(KeyError):
        si.eda.sample_counts(adata_dense, "missing")


# ── I/O and MuData ──────────────────────────────────────────────────────


def test_h5ad_round_trip_and_exporters(adata_dense, tmp_path):
    si.io.save_anndata(adata_dense, tmp_path / "a" / "x.h5ad")
    back = si.io.load_h5ad(tmp_path / "a" / "x.h5ad")
    assert back.shape == adata_dense.shape
    si.io.save_results_csv(pd.DataFrame({"a": [1]}), tmp_path / "r" / "t.csv")
    si.io.save_results_json({"x": np.float32(1.5), "y": np.arange(3)}, tmp_path / "r" / "t.json")
    assert json.loads((tmp_path / "r" / "t.json").read_text()) == {"x": 1.5, "y": [0, 1, 2]}
    with pytest.raises(FileNotFoundError):
        si.io.load_h5ad(tmp_path / "missing.h5ad")


def test_load_csv_and_unsupported_format(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"g1": [1.0, 2.0], "g2": [3.0, 4.0]}, index=["c1", "c2"]).to_csv(path)
    assert si.io.load_csv(path).shape == (2, 2)
    assert si.io.auto_detect_format(path).shape == (2, 2)
    with pytest.raises(ValueError, match="Unsupported"):
        si.io.auto_detect_format(tmp_path / "x.xyz")


def test_ensure_anndata_converts_dataframes_and_arrays_without_copying_anndata(adata_dense):
    assert si.io.ensure_anndata(adata_dense) is adata_dense
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": ["x", "y"]}, index=[10, 11])
    out = si.io.ensure_anndata(df, target_col="label")
    assert out.shape == (2, 2) and list(out.obs_names) == ["10", "11"] and list(out.obs["label"]) == ["x", "y"]
    assert si.io.ensure_anndata(np.ones((3, 2))).shape == (3, 2)
    with pytest.raises(TypeError, match="Cannot convert"):
        si.io.ensure_anndata(object())


def test_mudata_input_needs_a_modality_when_ambiguous(adata_dense):
    md = pytest.importorskip("mudata")
    mdata = md.MuData({"rna": adata_dense.copy(), "adt": adata_dense[:, :10].copy()})
    with pytest.raises(ValueError, match="pass modality"):
        si.io.ensure_anndata(mdata)
    with pytest.raises(KeyError, match="Modality"):
        si.io.ensure_anndata(mdata, modality="atac")
    assert si.io.ensure_anndata(mdata, modality="adt").n_vars == 10
    assert si.eda.dataset_summary(mdata, modality="rna")["n_genes"] == adata_dense.n_vars
    only = md.MuData({"rna": adata_dense.copy()})
    assert si.io.ensure_anndata(only).shape == adata_dense.shape


def test_mudata_inplace_tools_modify_the_modality_in_place(adata_dense):
    md = pytest.importorskip("mudata")
    mdata = md.MuData({"rna": adata_dense.copy()})
    assert si.pp.pca(mdata, n_comps=5) is None
    assert "X_pca" in mdata.mod["rna"].obsm


# ── settings and logging ────────────────────────────────────────────────


def test_settings_verbosity_maps_to_the_logger():
    from scintilla._logging import logger

    previous = si.settings.verbosity
    try:
        si.settings.verbosity = "debug"
        assert logger.level == logging.DEBUG and si.settings.verbosity == "debug"
        si.settings.verbosity = "info"
        assert si.settings.verbosity == "info"
        with pytest.raises(ValueError, match="verbosity must be one of"):
            si.settings.verbosity = "loud"
    finally:
        si.settings.verbosity = previous
    assert "verbosity" in repr(si.settings)


def test_library_is_silent_by_default(adata_logged, capsys):
    si.tl.umap(adata_logged, random_state=0)
    si.pp.log_shift_size_factor(adata_logged)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_verbose_argument_enables_progress_for_one_call_only(adata_logged, caplog):
    from scintilla._logging import logger

    config = si.AnalysisConfig.fast().copy(include_shap=False, classifiers=["LogReg"])
    assert logger.level == logging.WARNING
    with caplog.at_level(logging.INFO, logger="scintilla"):
        si.tl.supervised_analysis(adata_logged, "cell_type", config=config, verbose=True)
    assert "accuracy=" in caplog.text
    assert logger.level == logging.WARNING  # restored


def test_no_print_in_library_code_outside_the_cli_and_time_budget():
    import ast
    from pathlib import Path

    root = Path(si.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("cli/") or rel == "benchmarking/time_estimator.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_print_time_budget_is_the_documented_exception(capsys):
    table = pd.DataFrame(
        {"method": ["a", "b"], "estimated_seconds": [5.0, 500.0], "estimated_time": ["5 s", "8 min"], "stage": ["x", "x"]}
    )
    kept = si.benchmark.print_time_budget(table, max_seconds=60)
    assert list(kept["method"]) == ["a"] and "Methods within budget: 1 / 2" in capsys.readouterr().out


# ── CLI ─────────────────────────────────────────────────────────────────


@pytest.fixture
def h5ad(adata_dense, tmp_path):
    path = tmp_path / "toy.h5ad"
    adata_dense.write_h5ad(path)
    return path


def test_cli_eda_and_generate_config(h5ad, tmp_path, capsys):
    from scintilla.cli.main import main

    main(["eda", str(h5ad), "--output", str(tmp_path / "eda.json")])
    assert '"n_cells": 300' in capsys.readouterr().out
    main(["generate-config", "--output", str(tmp_path / "cfg.yaml")])
    cfg = si.AnalysisConfig.from_yaml(str(tmp_path / "cfg.yaml"))
    assert cfg.verbose is None and cfg.n_features == 50


def test_cli_reduce_runs_in_place_and_saves(h5ad, tmp_path):
    from scintilla.cli.main import main

    out = tmp_path / "umap.h5ad"
    main(["reduce", str(h5ad), "--method", "umap", "--output", str(out), "--quiet"])
    assert "X_umap" in ad.read_h5ad(out).obsm


def test_cli_requires_a_modality_for_multimodal_input(adata_dense, tmp_path):
    md = pytest.importorskip("mudata")
    from scintilla.cli.main import main

    path = tmp_path / "toy.h5mu"
    md.MuData({"rna": adata_dense.copy(), "adt": adata_dense.copy()}).write_h5mu(path)
    with pytest.raises(ValueError, match="pass modality"):
        main(["eda", str(path), "--quiet"])
    main(["eda", str(path), "--modality", "rna", "--quiet"])


def test_cli_label_quality_writes_scores_and_ranks(h5ad, tmp_path):
    pytest.importorskip("leidenalg")
    from scintilla.cli.main import main

    out = tmp_path / "scores.csv"
    main(["label-quality", str(h5ad), "--fast", "--seed", "0", "--output", str(out), "--quiet"])
    scores = pd.read_csv(out, index_col=0)
    assert len(scores) == 4 and "scintilla_composite" in scores.columns
    assert (tmp_path / "scores_ranks.csv").exists()


def test_cli_help_does_not_raise(capsys):
    from scintilla.cli.main import main

    with pytest.raises(SystemExit):
        main(["--help"])
    assert "label-quality" in capsys.readouterr().out
