"""Every CLI sub-command runs end to end on a small file."""

from __future__ import annotations

import json

import anndata as ad
import pandas as pd
import pytest

from scintilla.cli.main import main

pytest.importorskip("leidenalg")


@pytest.fixture
def h5ad(adata_dense, tmp_path):
    small = adata_dense[::3].copy()
    path = tmp_path / "toy.h5ad"
    small.write_h5ad(path)
    return str(path)


@pytest.fixture
def config(tmp_path):
    from scintilla import AnalysisConfig

    path = tmp_path / "cfg.yaml"
    AnalysisConfig.fast().copy(
        clustering_methods=["kmeans"], classifiers=["LogReg"], include_shap=False, feature_selection_methods=["mutual_information"]
    ).to_yaml(str(path))
    return str(path)


def test_normalise_and_preprocess(h5ad, tmp_path, capsys):
    main(["normalise", h5ad, "--output", str(tmp_path / "norm.csv"), "--quiet"])
    assert "Best transformation" in capsys.readouterr().out
    main(["preprocess", h5ad, "--transform", "log_cpm_transform", "--output", str(tmp_path / "pre.h5ad"), "--quiet"])
    assert ad.read_h5ad(tmp_path / "pre.h5ad").n_obs == 100


def test_cluster_classify_and_run_all(h5ad, config, tmp_path, capsys):
    main(["cluster", h5ad, "--cell-type-col", "cell_type", "--config", config, "--output", str(tmp_path / "c.csv"), "--quiet"])
    assert "Best method" in capsys.readouterr().out and (tmp_path / "c.csv").exists()
    main(["classify", h5ad, "--target-col", "cell_type", "--config", config, "--output", str(tmp_path / "k.json"), "--quiet"])
    assert "adata" not in json.loads((tmp_path / "k.json").read_text())
    main(["run-all", h5ad, "--target-col", "cell_type", "--config", config, "--output-dir", str(tmp_path / "all"), "--quiet"])
    assert (tmp_path / "all" / "clustering_results.csv").exists()


def test_de_annotate_feature_select_and_eda(h5ad, tmp_path, capsys):
    for method in ("wilcoxon", "ttest", "permutation"):
        main(["de", h5ad, "--group-col", "cell_type", "--group1", "type0", "--group2", "type1", "--method", method,
              "--output", str(tmp_path / f"{method}.csv"), "--quiet"])
        assert len(pd.read_csv(tmp_path / f"{method}.csv")) == 200
    main(["annotate", h5ad, "--groupby", "cell_type", "--n-genes", "5", "--quiet"])
    assert "type0" in capsys.readouterr().out
    main(["feature-select", h5ad, "--output", str(tmp_path / "genes.csv"), "--quiet"])
    assert len(pd.read_csv(tmp_path / "genes.csv")) > 0


def test_batch_correct_and_estimate_time(h5ad, config, tmp_path, capsys):
    main(["batch-correct", h5ad, "--batch-key", "batch", "--label-key", "cell_type", "--output", str(tmp_path / "bc"), "--quiet"])
    assert "Best method" in capsys.readouterr().out
    main(["estimate-time", h5ad, "--config", config, "--output", str(tmp_path / "t.csv"), "--stages", "clustering", "--quiet"])
    assert (tmp_path / "t.csv").exists()


def test_benchmark_all(h5ad, config, tmp_path, capsys):
    main(["benchmark-all", h5ad, "--cell-type-col", "cell_type", "--output-dir", str(tmp_path / "ba"), "--quiet"])
    assert "Benchmark results saved" in capsys.readouterr().out
