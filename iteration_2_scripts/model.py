"""Model training, uncertainty, saving, and prediction utilities."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    RDKit_AVAILABLE,
    add_rdkit_features,
    align_to_training_features,
    prepare_feature_table,
)

warnings.filterwarnings("ignore", category=UserWarning)


def make_base_model(random_state: int = 0) -> Pipeline:
    """
    Simple interpretable classifier for P(bind)
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_calibrated_model(random_state: int = 0, calibration_cv: int = 3) -> CalibratedClassifierCV:
    """
    Return a calibrated classifier so outputs are probability-like
    """
    base_model = make_base_model(random_state=random_state)
    return CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=calibration_cv)


def evaluate_predictions(y_true: pd.Series, p: np.ndarray) -> Dict[str, float]:
    """
    Evaluate probability predictions
    """
    metrics = {
        "n": int(len(y_true)),
        "n_active": int(np.sum(y_true == 1)),
        "n_decoy": int(np.sum(y_true == 0)),
        "active_rate": float(np.mean(y_true)),
        "brier_score": float(brier_score_loss(y_true, p)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p))
        metrics["average_precision"] = float(average_precision_score(y_true, p))
    return metrics


def bootstrap_probability_uncertainty(
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 100,
    random_state: int = 42,
    calibration_cv: int = 3,
) -> Tuple[pd.DataFrame, List[CalibratedClassifierCV]]:
    """
    Fit bootstrapped calibrated models and return per-training-row summaries
    plus the fitted models themselves.

    The fitted models are retained so they can be re-applied to new molecules
    at predict time, turning the P(bind) SD into a propagatable uncertainty
    rather than a retrospective number on the training table
    """
    rng = np.random.default_rng(random_state)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_arr = y.to_numpy()
    active_idx = np.where(y_arr == 1)[0]
    decoy_idx = np.where(y_arr == 0)[0]

    if len(active_idx) < calibration_cv or len(decoy_idx) < calibration_cv:
        raise ValueError(
            "Not enough active or decoy examples for calibrated bootstrap "
            "Reduce calibration_cv or add more examples"
        )

    preds = []
    models: List[CalibratedClassifierCV] = []
    for i in range(n_boot):
        boot_active = rng.choice(active_idx, size=len(active_idx), replace=True)
        boot_decoy = rng.choice(decoy_idx, size=len(decoy_idx), replace=True)
        boot_idx = np.concatenate([boot_active, boot_decoy])
        rng.shuffle(boot_idx)

        model = make_calibrated_model(
            random_state=random_state + i,
            calibration_cv=calibration_cv,
        )
        model.fit(X.iloc[boot_idx], y.iloc[boot_idx])
        preds.append(model.predict_proba(X)[:, 1])
        models.append(model)

    pred_arr = np.vstack(preds)
    summary = pd.DataFrame(
        {
            "p_binding_bootstrap_mean": pred_arr.mean(axis=0),
            "p_binding_bootstrap_std": pred_arr.std(axis=0, ddof=1),
            "p_binding_ci_low": np.quantile(pred_arr, 0.025, axis=0),
            "p_binding_ci_high": np.quantile(pred_arr, 0.975, axis=0),
        }
    )
    return summary, models


def train_receptor_model(
    df: pd.DataFrame,
    receptor_name: str,
    feature_groups: Iterable[str],
    out_dir: str | Path,
    n_boot: int = 100,
    test_size: float = 0.25,
    random_state: int = 42,
    calibration_cv: int = 3,
    oof_cv: int = 5,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train one receptor-specific P(bind) model and save all artifacts

    Saves to out_dir:
        model.joblib [Final model + bootstrap models + metadata]
        predictions.csv [Out-of-fold predictions for labeled rows]
        metrics.json [Performance + config metadata]
        config.json [Training config]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df = df.dropna(subset=["label"]).copy()
    model_df["label"] = model_df["label"].astype(int)

    model_df, feature_columns = prepare_feature_table(model_df, feature_groups=feature_groups)
    X = model_df[feature_columns].copy()
    y = model_df["label"].copy()

    # Held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    split_model = make_calibrated_model(random_state=random_state, calibration_cv=calibration_cv)
    split_model.fit(X_train, y_train)
    p_test = split_model.predict_proba(X_test)[:, 1]

    # Out-of-fold predictions for labeled rows (replaces in-sample)
    skf = StratifiedKFold(n_splits=oof_cv, shuffle=True, random_state=random_state)
    oof_template = make_calibrated_model(random_state=random_state, calibration_cv=calibration_cv)
    p_oof = cross_val_predict(oof_template, X, y, cv=skf, method="predict_proba")[:, 1]

    # Final model trained on all labels — used to score unseen compounds
    final_model = make_calibrated_model(random_state=random_state, calibration_cv=calibration_cv)
    final_model.fit(X, y)

    # Bootstrap models — saved in the saved_model for predict-time uncertainty
    bootstrap_models: List[CalibratedClassifierCV] = []
    if n_boot and n_boot > 0:
        _, bootstrap_models = bootstrap_probability_uncertainty(
            X, y,
            n_boot=n_boot,
            random_state=random_state,
            calibration_cv=calibration_cv,
        )

    # Training-row output uses OOF predictions, not in-sample
    output = model_df.copy()
    output["p_binding"] = p_oof
    if bootstrap_models:
        boot_preds = np.vstack([m.predict_proba(X)[:, 1] for m in bootstrap_models])
        output["p_binding_bootstrap_mean"] = boot_preds.mean(axis=0)
        output["p_binding_bootstrap_std"] = boot_preds.std(axis=0, ddof=1)
        output["p_binding_ci_low"] = np.quantile(boot_preds, 0.025, axis=0)
        output["p_binding_ci_high"] = np.quantile(boot_preds, 0.975, axis=0)

    metrics: Dict = {
        "receptor_name": receptor_name,
        "feature_groups": list(feature_groups),
        "feature_columns": feature_columns,
        "rdkit_available": RDKit_AVAILABLE,
        "n_bootstrap_models": int(n_boot),
        "oof_cv": oof_cv,
        "test_set": evaluate_predictions(y_test, p_test),
        "oof_full_data": evaluate_predictions(y, p_oof),
        "notes": [
            "predictions.csv p_binding is out-of-fold, not in-sample",
            "binding_energy_error_pose is meaningful only when n_poses > 1",
            "Bootstrap models are saved in model.joblib",
        ],
    }

    saved_model = {
        "model": final_model,
        "bootstrap_models": bootstrap_models,
        "receptor_name": receptor_name,
        "feature_groups": list(feature_groups),
        "feature_columns": feature_columns,
        "rdkit_available_at_training": RDKit_AVAILABLE,
    }

    joblib.dump(saved_model, out_dir / "model.joblib")
    output.to_csv(out_dir / "predictions.csv", index=False)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "config.json").open("w") as f:
        json.dump(
            {
                "receptor_name": receptor_name,
                "feature_groups": list(feature_groups),
                "feature_columns": feature_columns,
                "test_size": test_size,
                "random_state": random_state,
                "calibration_cv": calibration_cv,
                "oof_cv": oof_cv,
                "n_boot": n_boot,
            },
            f,
            indent=2,
        )

    return output, metrics


def load_model(model_dir: str | Path) -> Dict:
    """
    Load a saved receptor-specific model artifact
    """
    return joblib.load(Path(model_dir) / "model.joblib")


def predict_pbind(model_dir: str | Path, records: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the coupling payload for new compounds

    Input
    -----
    Required column:
        smiles
    Recommended columns:
        Same docking score columns used by the receptor model (e.g. Total Energy)
        Missing optional columns are filled with NaN and handled by the saved
        pipeline's imputer. If pose-aggregate columns (e.g. binding_energy_error_pose)
        are present, they're surfaced as binding_energy_error

    Output
    ------
    DataFrame with the same rows as `records`, plus:
        receptor_name
        binding_probability Bootstrap mean if bootstraps were saved, else the final-model probability
        binding_probability_std SD across bootstraps; NaN if no bootstraps saved
        binding_energy From input Total Energy if present
        binding_energy_error From input pose-spread if present, else NaN
    """
    saved_model = load_model(model_dir)
    final_model: CalibratedClassifierCV = saved_model["model"]
    bootstrap_models: List[CalibratedClassifierCV] = saved_model.get("bootstrap_models", []) or []
    feature_columns: List[str] = saved_model["feature_columns"]

    df = add_rdkit_features(records.copy())
    X = align_to_training_features(df, feature_columns)

    # P(bind): prefer bootstrap mean (matched to the SD), fall back to final model
    if bootstrap_models:
        boot_p = np.vstack([m.predict_proba(X)[:, 1] for m in bootstrap_models])
        p_mean = boot_p.mean(axis=0)
        p_std = boot_p.std(axis=0, ddof=1)
    else:
        p_mean = final_model.predict_proba(X)[:, 1]
        p_std = np.full_like(p_mean, np.nan)

    # Energy fields from input
    if "Total Energy" in records.columns:
        binding_energy = pd.to_numeric(records["Total Energy"], errors="coerce").to_numpy()
    else:
        binding_energy = np.full(len(records), np.nan)
    if "binding_energy_error_pose" in records.columns:
        binding_energy_error = pd.to_numeric(
            records["binding_energy_error_pose"], errors="coerce"
        ).to_numpy()
    else:
        binding_energy_error = np.full(len(records), np.nan)

    out = records.copy()
    out["receptor_name"] = saved_model["receptor_name"]
    out["binding_probability"] = p_mean
    out["binding_probability_std"] = p_std
    out["binding_energy"] = binding_energy
    out["binding_energy_error"] = binding_energy_error

    return out
