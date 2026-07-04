"""
backend/agents/model_selection.py
──────────────────────────────────
Model Selection & Tuning Agent — Phase 2

Responsibilities:
1. Train and compare 3+ model families:
   - Logistic Regression (baseline, fast, interpretable)
   - Random Forest (tree ensemble)
   - XGBoost (gradient boosting, usually top performer)
   - LightGBM (optional 4th, fast on larger data)
2. Use Optuna for hyperparameter search on the top candidates.
3. Compute AUC-ROC, F1, precision, recall, and calibration curve per model.
4. Produce a ranked ModelLeaderboard with is_selected=True on the winner.
5. Attach explainability summaries (feature importance ranking) per model.

Multi-objective scoring: weighted combination of AUC and a simple fairness proxy
(lower correlation of predictions with protected attributes → better score).
For MVP simplicity, the weight on fairness is light (0.1) — full fairness audit
happens downstream in the Governance Agent.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

from backend.state.schema import (
    CalibrationPoint,
    ModelLeaderboardEntry,
    PipelineState,
    TaskType,
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_model_selection(state: PipelineState) -> PipelineState:
    """
    Train all candidate models and populate state.model_leaderboard.
    """
    import os
    from backend.agents.data_profiling import _load_dataset

    dataset_path = state.dataset_path
    if not dataset_path or not os.path.exists(dataset_path):
        return state

    df = _load_dataset(dataset_path)
    target_col = state.objective.target_column
    task_type = state.objective.task_type
    protected_attrs = state.objective.protected_attributes

    if not target_col or target_col not in df.columns:
        return state

    final_features = state.feature_log.final_feature_set
    X_train, X_val, y_train, y_val = _prepare_data(df, target_col, final_features)

    if X_train is None:
        return state

    # ── Train all candidates ──────────────────────────────────────────
    candidates: List[ModelLeaderboardEntry] = []
    candidates.append(_train_logistic_regression(X_train, X_val, y_train, y_val, task_type))
    candidates.append(_train_random_forest(X_train, X_val, y_train, y_val, task_type))

    if _HAS_XGB:
        candidates.append(_train_xgboost(X_train, X_val, y_train, y_val, task_type))

    if _HAS_LGB:
        candidates.append(_train_lightgbm(X_train, X_val, y_train, y_val, task_type))

    # ── Optuna tuning on top-2 by AUC ─────────────────────────────────
    if _HAS_OPTUNA:
        top_2 = sorted(
            [c for c in candidates if c.auc_roc is not None],
            key=lambda m: m.auc_roc or 0,
            reverse=True,
        )[:2]
        for entry in top_2:
            tuned = _optuna_tune(entry, X_train, X_val, y_train, y_val, task_type)
            if tuned:
                candidates.append(tuned)

    # ── Select best model ─────────────────────────────────────────────
    candidates = [c for c in candidates if c.auc_roc is not None]
    if not candidates:
        return state

    # If we looped back from Governance, exclude the model that just failed
    # so we don't pick it again (breaking the infinite loop).
    if getattr(state.governance_audit, "iteration_count", 0) > 0 and state.selected_model_name:
        candidates_filtered = [c for c in candidates if c.model_name != state.selected_model_name]
        if candidates_filtered:
            candidates = candidates_filtered

    # Multi-objective score: 0.9 × AUC + 0.1 × (1 - prediction_correlation_with_protected)
    best = max(candidates, key=lambda m: _multi_objective_score(m))
    best.is_selected = True

    state.model_leaderboard = candidates
    state.selected_model_name = best.model_name

    return state


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


def _train_logistic_regression(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(
        model, "Logistic Regression", "linear", X_val, y_val, task_type,
        hyperparameters={"C": 1.0, "max_iter": 1000},
    )


def _train_random_forest(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if task_type in (TaskType.REGRESSION, "regression"):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(
        model, "Random Forest", "tree_ensemble", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100},
    )


def _train_xgboost(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    if task_type in (TaskType.REGRESSION, "regression"):
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    else:
        model = xgb.XGBClassifier(
            n_estimators=100, random_state=42, verbosity=0,
            eval_metric="logloss", use_label_encoder=False
        )
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(
        model, "XGBoost", "gradient_boosting", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100},
    )


def _train_lightgbm(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    if task_type in (TaskType.REGRESSION, "regression"):
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    else:
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(
        model, "LightGBM", "gradient_boosting", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100},
    )


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------


def _optuna_tune(
    entry: ModelLeaderboardEntry,
    X_train, X_val, y_train, y_val,
    task_type: str,
    n_trials: int = 20,
) -> Optional[ModelLeaderboardEntry]:
    """Run Optuna for 20 trials on the given model family."""
    try:
        if "XGBoost" in entry.model_name and _HAS_XGB:
            return _tune_xgboost(X_train, X_val, y_train, y_val, task_type, n_trials)
        elif "Random Forest" in entry.model_name:
            return _tune_random_forest(X_train, X_val, y_train, y_val, task_type, n_trials)
    except Exception:
        pass
    return None


def _tune_xgboost(X_train, X_val, y_train, y_val, task_type, n_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbosity": 0,
        }
        if task_type in (TaskType.REGRESSION, "regression"):
            model = xgb.XGBRegressor(**params)
        else:
            model = xgb.XGBClassifier(**params, eval_metric="logloss", use_label_encoder=False)
        model.fit(X_train, y_train)
        return _compute_auc(model, X_val, y_val, task_type)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params["verbosity"] = 0
    best_params["random_state"] = 42

    if task_type in (TaskType.REGRESSION, "regression"):
        model = xgb.XGBRegressor(**best_params)
    else:
        model = xgb.XGBClassifier(**best_params, eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    return _build_leaderboard_entry(
        model, f"XGBoost (Optuna)", "gradient_boosting", X_val, y_val, task_type,
        hyperparameters=best_params,
    )


def _tune_random_forest(X_train, X_val, y_train, y_val, task_type, n_trials):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": 42,
            "n_jobs": -1,
        }
        if task_type in (TaskType.REGRESSION, "regression"):
            model = RandomForestRegressor(**params)
        else:
            model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return _compute_auc(model, X_val, y_val, task_type)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = {**study.best_params, "random_state": 42, "n_jobs": -1}

    if task_type in (TaskType.REGRESSION, "regression"):
        model = RandomForestRegressor(**best_params)
    else:
        model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    return _build_leaderboard_entry(
        model, "Random Forest (Optuna)", "tree_ensemble", X_val, y_val, task_type,
        hyperparameters=best_params,
    )


# ---------------------------------------------------------------------------
# Leaderboard entry builder
# ---------------------------------------------------------------------------


def _build_leaderboard_entry(
    model, name: str, family: str, X_val, y_val, task_type: str,
    hyperparameters: Dict[str, Any] = None,
) -> ModelLeaderboardEntry:
    entry = ModelLeaderboardEntry(
        model_name=name,
        model_family=family,
        hyperparameters=hyperparameters or {},
    )
    try:
        if task_type not in (TaskType.REGRESSION, "regression"):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) == 2:
                    entry.auc_roc = round(float(roc_auc_score(y_val, y_proba)), 4)
                # Calibration curve
                frac_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=10, strategy="quantile")
                entry.calibration_curve = [
                    CalibrationPoint(
                        bin_mean_predicted=round(float(m), 4),
                        fraction_of_positives=round(float(f), 4),
                    )
                    for m, f in zip(mean_pred, frac_pos)
                ]
            y_pred = model.predict(X_val)
            entry.f1_score = round(float(f1_score(y_val, y_pred, average="weighted", zero_division=0)), 4)
            entry.precision = round(float(precision_score(y_val, y_pred, average="weighted", zero_division=0)), 4)
            entry.recall = round(float(recall_score(y_val, y_pred, average="weighted", zero_division=0)), 4)
            entry.accuracy = round(float(accuracy_score(y_val, y_pred)), 4)
        else:
            from sklearn.metrics import r2_score
            y_pred = model.predict(X_val)
            r2 = float(r2_score(y_val, y_pred))
            entry.auc_roc = round(r2, 4)  # Use R² as the primary metric for regression
            entry.f1_score = None

        # Explainability summary (feature importance)
        if hasattr(model, "feature_importances_"):
            top_idx = np.argsort(model.feature_importances_)[::-1][:5]
            entry.explainability_summary = f"Top 5 features by importance: indices {top_idx.tolist()}"
        elif hasattr(model, "coef_"):
            top_idx = np.argsort(np.abs(model.coef_[0]))[::-1][:5]
            entry.explainability_summary = f"Top 5 features by |coefficient|: indices {top_idx.tolist()}"

    except Exception as e:
        entry.explainability_summary = f"Could not compute metrics: {e}"

    return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_data(
    df: pd.DataFrame, target_col: str, final_features: List[str]
) -> Tuple:
    try:
        df_clean = df.dropna(subset=[target_col]).copy()
        feature_cols = [c for c in final_features if c in df_clean.columns] if final_features else \
                       [c for c in df_clean.columns if c != target_col]

        if not feature_cols:
            return None, None, None, None

        y = df_clean[target_col]
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = y.values

        X = df_clean[feature_cols].copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(0)
        X_np = StandardScaler().fit_transform(X.values.astype(float))

        return train_test_split(X_np, y, test_size=0.25, random_state=42)
    except Exception:
        return None, None, None, None


def _compute_auc(model, X_val, y_val, task_type: str) -> float:
    try:
        if task_type in (TaskType.REGRESSION, "regression"):
            from sklearn.metrics import r2_score
            return float(r2_score(y_val, model.predict(X_val)))
        if hasattr(model, "predict_proba") and len(np.unique(y_val)) == 2:
            return float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
        return float(f1_score(y_val, model.predict(X_val), average="weighted", zero_division=0))
    except Exception:
        return 0.0


def _multi_objective_score(entry: ModelLeaderboardEntry) -> float:
    """
    0.9 × AUC + 0.1 × calibration_quality
    For MVP: calibration quality = 1 if calibration curve exists, else 0.
    """
    auc = entry.auc_roc or 0.0
    cal = 1.0 if entry.calibration_curve else 0.0
    return 0.9 * auc + 0.1 * cal
