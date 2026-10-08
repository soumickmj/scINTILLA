"""Regression coverage for classifier inputs and stale consistency metrics."""

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scintilla.classification.run import supervised_analysis


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    labels = np.repeat(["a", "b"], 20)
    X[labels == "b"] += 1.0
    return ad.AnnData(X=X, obs=pd.DataFrame({"cell_type": labels},
                                          index=[f"c{i}" for i in range(40)]))


def test_consistency_uses_scaler_fitted_on_classifier_training_cells():
    x = _data()
    result = supervised_analysis(x, scale=True, normality=True, models=["LogReg"],
                                 test_size=0.5, random_state=0, include_shap=False,
                                 check_consistency=True, verbose=False)
    X_train, _, _, _ = train_test_split(x.X, x.obs.cell_type.to_numpy(),
                                       test_size=0.5, random_state=0, stratify=x.obs.cell_type)
    transformed = StandardScaler().fit(X_train).transform(x.X)
    expected = result["best_model"].predict_proba(transformed).max(axis=1)
    np.testing.assert_allclose(x.obs.pred_LogReg_confidence, expected)


def test_rerun_without_successful_classifiers_clears_previous_consistency():
    x = _data()
    for col in ["pred_agreement", "pred_entropy", "pred_avg_confidence", "scintilla_label_quality"]:
        x.obs[col] = 0.1
    x.obs["pred_consensus"] = "old"
    x.obs["pred_RF"] = "old"
    x.obs["pred_RF_confidence"] = 0.1
    x.obs["pred_user_metadata"] = "keep"
    result = supervised_analysis(x, normality=True, models=[], include_shap=False,
                                 check_consistency=True, verbose=False)
    assert result["best_model"] is None
    assert not any(c in x.obs for c in ["pred_agreement", "pred_entropy", "pred_avg_confidence",
                                      "pred_consensus", "pred_RF", "pred_RF_confidence",
                                      "scintilla_label_quality"])
    assert (x.obs.pred_user_metadata == "keep").all()
