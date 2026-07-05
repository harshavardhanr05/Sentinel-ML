"""
backend/agents/cost_awareness.py
──────────────────────────────────
Cost-Awareness Agent — Phase 2

Responsibilities:
1. Before model selection runs full tuning, time a single quick fit on a
   small sample of the data.
2. Extrapolate estimated wall-clock time for a full run with N Optuna trials.
3. Attach cost estimates to state.cost_estimates and to each leaderboard entry.

This runs BEFORE model_selection so the checkpoint card can show
cost-vs-performance trade-offs to the user.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

from backend.state.schema import PipelineState, TaskType
from backend.state.store import log_step_and_broadcast_sync

# Sample size for timing estimate
COST_SAMPLE_SIZE = 500
OPTUNA_TRIALS_ESTIMATE = 20  # Number of Optuna trials to estimate for


def run_cost_awareness(state: PipelineState) -> PipelineState:
    """
    Estimate compute cost for each candidate model family before full tuning.
    Populates state.cost_estimates with per-model timing estimates.
    """
    log_step_and_broadcast_sync(state, "cost_awareness", "Cost Estimation Started", "Profiling model training times on a data sample to project full tuning costs.")
    import os
    from backend.agents.data_profiling import _load_dataset

    dataset_path = state.dataset_path
    if not dataset_path or not os.path.exists(dataset_path):
        return state

    df = _load_dataset(dataset_path)
    target_col = state.objective.target_column
    task_type = state.objective.task_type

    if not target_col or target_col not in df.columns:
        return state

    X_sample, y_sample = _prepare_sample(df, target_col, state.feature_log.final_feature_set)
    if X_sample is None:
        return state

    n_full = len(df)
    estimates: Dict[str, Any] = {}

    # ── Estimate each candidate family ────────────────────────────────
    families = {
        "Logistic Regression": _estimate_lr,
        "Random Forest": _estimate_rf,
    }

    try:
        import xgboost as xgb
        families["XGBoost"] = _estimate_xgb
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        families["LightGBM"] = _estimate_lgb
    except ImportError:
        pass

    for name, estimator_fn in families.items():
        try:
            single_fit_sec = estimator_fn(X_sample, y_sample)
            # Scale to full dataset size (roughly linear for most models)
            scale_factor = (n_full / len(X_sample)) ** 0.8  # sub-linear scaling
            full_fit_sec = single_fit_sec * scale_factor
            optuna_total_sec = full_fit_sec * OPTUNA_TRIALS_ESTIMATE

            estimates[name] = {
                "single_fit_seconds": round(single_fit_sec, 2),
                "full_dataset_fit_seconds": round(full_fit_sec, 2),
                "optuna_20_trials_seconds": round(optuna_total_sec, 2),
                "human_readable": _format_duration(optuna_total_sec),
                "sample_size_used": len(X_sample),
                "full_dataset_size": n_full,
            }
        except Exception as e:
            estimates[name] = {"error": str(e)}

    state.cost_estimates = estimates
    
    log_step_and_broadcast_sync(state, "cost_awareness", "Cost Estimation Complete", f"Estimated training time for {len(estimates)} model families based on N={OPTUNA_TRIALS_ESTIMATE} Optuna trials.")
    return state


# ---------------------------------------------------------------------------
# Per-model quick-fit timers
# ---------------------------------------------------------------------------


def _time_fit(model, X: np.ndarray, y: np.ndarray) -> float:
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def _estimate_lr(X: np.ndarray, y: np.ndarray) -> float:
    return _time_fit(LogisticRegression(max_iter=200, random_state=42), X, y)


def _estimate_rf(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.ensemble import RandomForestClassifier
    return _time_fit(RandomForestClassifier(n_estimators=50, random_state=42), X, y)


def _estimate_xgb(X: np.ndarray, y: np.ndarray) -> float:
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=50, verbosity=0, random_state=42,
                               eval_metric="logloss", use_label_encoder=False)
    return _time_fit(model, X, y)


def _estimate_lgb(X: np.ndarray, y: np.ndarray) -> float:
    import lightgbm as lgb
    return _time_fit(lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1), X, y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_sample(
    df: pd.DataFrame, target_col: str, final_features: List[str]
) -> tuple:
    try:
        df_clean = df.dropna(subset=[target_col]).copy()
        n = min(COST_SAMPLE_SIZE, len(df_clean))
        df_sample = df_clean.sample(n=n, random_state=42)

        feature_cols = [c for c in final_features if c in df_sample.columns] if final_features else \
                       [c for c in df_sample.columns if c != target_col]

        if not feature_cols:
            return None, None

        y = df_sample[target_col]
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = y.values

        X = df_sample[feature_cols].copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(0)
        X_np = StandardScaler().fit_transform(X.values.astype(float))

        return X_np, y
    except Exception:
        return None, None


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"~{seconds:.0f}s"
    elif seconds < 3600:
        return f"~{seconds / 60:.1f}min"
    else:
        return f"~{seconds / 3600:.1f}hr"
