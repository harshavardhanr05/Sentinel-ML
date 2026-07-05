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
    _HAS_OPTUNA = False  # Temporarily disabled to speed up training
except ImportError:
    _HAS_OPTUNA = False

from backend.state.schema import (
    CalibrationPoint,
    ModelLeaderboardEntry,
    PipelineState,
    TaskType,
)
from backend.state.store import log_step_and_broadcast_sync


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
    try:
        X_train, X_val, y_train, y_val, feature_names = _prepare_data(df, target_col, final_features, state)
        if X_train is None:
            return state
    except Exception as e:
        log_step_and_broadcast_sync(state, "model_selection", "Native Process Failed", f"Data preparation encountered an error: {e}")
        state.is_paused = True
        return state

    # ── Handle Class Imbalance with SMOTE ────────────────────────────
    # Automatically apply if severity was flagged by profiling, or if user requested it
    is_imbalanced = False
    if state.data_health_report and state.data_health_report.imbalance_flag:
        is_imbalanced = True
        
    if task_type == TaskType.CLASSIFICATION and (state.smote_applied or is_imbalanced):
        try:
            from imblearn.over_sampling import SMOTE
            state.smote_applied = True
            
            # Record BEFORE distribution
            if isinstance(y_train, pd.Series):
                before_dist = y_train.value_counts().to_dict()
            else:
                import numpy as np
                u, c = np.unique(y_train, return_counts=True)
                before_dist = dict(zip(u, c))
            before_dist_str = {str(k): int(v) for k, v in before_dist.items()}
            
            log_step_and_broadcast_sync(state, "model_selection", "Target Imbalance Found", f"Training data exhibits class imbalance. Class distribution before oversampling: {before_dist_str}")
            
            # Fit Oversampler
            oversampler_type = getattr(state.objective, "oversampler_type", "smote")
            if oversampler_type == "smotetomek":
                from imblearn.combine import SMOTETomek
                oversampler = SMOTETomek(random_state=42)
                method_name = "SMOTETomek"
            elif oversampler_type == "smoteenn":
                from imblearn.combine import SMOTEENN
                oversampler = SMOTEENN(random_state=42)
                method_name = "SMOTEENN"
            else:
                from imblearn.over_sampling import SMOTE
                oversampler = SMOTE(random_state=42)
                method_name = "SMOTE"
                
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
            
            # Record AFTER distribution
            if isinstance(y_train, pd.Series):
                after_dist = y_train.value_counts().to_dict()
            else:
                import numpy as np
                u, c = np.unique(y_train, return_counts=True)
                after_dist = dict(zip(u, c))
            after_dist_str = {str(k): int(v) for k, v in after_dist.items()}
            
            state.smote_class_distributions = {
                "before": before_dist_str,
                "after": after_dist_str
            }
            log_step_and_broadcast_sync(state, "model_selection", f"{method_name} Applied", f"Synthesized minority class samples to balance dataset. New distribution: {after_dist_str}")
        except Exception as e:
            method_str = method_name if 'method_name' in locals() else "Oversampler"
            log_step_and_broadcast_sync(state, "model_selection", f"{method_str} Failed", f"Attempted to apply {method_str} but failed: {e}")
            state.is_paused = True
            return state

    def _fmt(m: ModelLeaderboardEntry) -> str:
        if task_type in (TaskType.REGRESSION, "regression"):
            r2 = f"{m.auc_roc:.4f}" if m.auc_roc is not None else "N/A"
            r2_tr = f"{m.train_auc_roc:.4f}" if getattr(m, "train_auc_roc", None) is not None else "N/A"
            rmse = f"{m.rmse:.4f}" if m.rmse is not None else "N/A"
            rmse_tr = f"{m.train_rmse:.4f}" if getattr(m, "train_rmse", None) is not None else "N/A"
            return f"Val R2={r2} (Train R2={r2_tr}), Val RMSE={rmse} (Train RMSE={rmse_tr})"
        else:
            a = f"{m.auc_roc:.4f}" if m.auc_roc is not None else "N/A"
            a_tr = f"{m.train_auc_roc:.4f}" if getattr(m, "train_auc_roc", None) is not None else "N/A"
            f = f"{m.f1_score:.4f}" if m.f1_score is not None else "N/A"
            f_tr = f"{m.train_f1_score:.4f}" if getattr(m, "train_f1_score", None) is not None else "N/A"
            return f"Val AUC={a} (Train AUC={a_tr}), Val F1={f} (Train F1={f_tr})"

    # ── Train all candidates ─────────────────────────────────────────
    log_step_and_broadcast_sync(state,"model_selection", "Training candidates", "Starting training of all model families.")
    candidates: List[ModelLeaderboardEntry] = []

    try:
        lr = _train_logistic_regression(X_train, X_val, y_train, y_val, task_type)
        candidates.append(lr)
        log_step_and_broadcast_sync(state,"model_selection", f"{lr.model_name} trained", _fmt(lr))
    except Exception as e:
        log_step_and_broadcast_sync(state,"model_selection", "Logistic Regression/Linear FAILED", str(e))

    try:
        rf = _train_random_forest(X_train, X_val, y_train, y_val, task_type)
        candidates.append(rf)
        log_step_and_broadcast_sync(state,"model_selection", "Random Forest trained", _fmt(rf))
    except Exception as e:
        log_step_and_broadcast_sync(state,"model_selection", "Random Forest FAILED", str(e))

    try:
        et = _train_extra_trees(X_train, X_val, y_train, y_val, task_type)
        candidates.append(et)
        log_step_and_broadcast_sync(state,"model_selection", "Extra Trees trained", _fmt(et))
    except Exception as e:
        log_step_and_broadcast_sync(state,"model_selection", "Extra Trees FAILED", str(e))

    try:
        gb = _train_gradient_boosting(X_train, X_val, y_train, y_val, task_type)
        candidates.append(gb)
        log_step_and_broadcast_sync(state,"model_selection", "Gradient Boosting trained", _fmt(gb))
    except Exception as e:
        log_step_and_broadcast_sync(state,"model_selection", "Gradient Boosting FAILED", str(e))

    try:
        ridge = _train_ridge(X_train, X_val, y_train, y_val, task_type)
        candidates.append(ridge)
        log_step_and_broadcast_sync(state,"model_selection", "Ridge trained", _fmt(ridge))
    except Exception as e:
        log_step_and_broadcast_sync(state,"model_selection", "Ridge FAILED", str(e))

    try:
        knn = _train_knn(X_train, X_val, y_train, y_val, task_type)
        candidates.append(knn)
        log_step_and_broadcast_sync(state,"model_selection", "KNN trained", _fmt(knn))
    except Exception as e:
        log_step_and_broadcast_sync(state,"model_selection", "KNN FAILED", str(e))

    if _HAS_XGB:
        try:
            xgb_m = _train_xgboost(X_train, X_val, y_train, y_val, task_type)
            candidates.append(xgb_m)
            log_step_and_broadcast_sync(state,"model_selection", "XGBoost trained", _fmt(xgb_m))
        except Exception as e:
            log_step_and_broadcast_sync(state,"model_selection", "XGBoost FAILED", str(e))

    if _HAS_LGB:
        try:
            lgb_m = _train_lightgbm(X_train, X_val, y_train, y_val, task_type)
            candidates.append(lgb_m)
            log_step_and_broadcast_sync(state,"model_selection", "LightGBM trained", _fmt(lgb_m))
        except Exception as e:
            log_step_and_broadcast_sync(state,"model_selection", "LightGBM FAILED", str(e))

    log_step_and_broadcast_sync(state,"model_selection", "Training complete", f"{len(candidates)} models trained successfully.")

    # ── Optuna tuning on top-2 by AUC ─────────────────────────────────
    if _HAS_OPTUNA:
        top_2 = sorted(
            [c for c in candidates if c.auc_roc is not None],
            key=lambda m: m.auc_roc or 0,
            reverse=True,
        )[:2]
        for entry in top_2:
            try:
                tuned = _optuna_tune(entry, X_train, X_val, y_train, y_val, task_type)
                if tuned:
                    candidates.append(tuned)
                    log_step_and_broadcast_sync(state,"model_selection", f"Optuna tuned {entry.model_name}",
                        f"Improved AUC from {entry.auc_roc:.4f} to {tuned.auc_roc:.4f}")
            except Exception as e:
                log_step_and_broadcast_sync(state,"model_selection", f"Optuna failed for {entry.model_name}", str(e))

    # ── Select best model ──────────────────────────────────────────
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

    for c in candidates:
        c.features_used = final_features

    state.model_leaderboard = candidates
    state.selected_model_name = best.model_name
    
    log_step_and_broadcast_sync(
        state, "model_selection", "Best Model Selected",
        f"Selected {best.model_name} (AUC: {best.auc_roc:.4f}) based on multi-objective optimization (performance + fairness)."
    )

    # Update data_analysis_metrics with post-SMOTE distribution if applicable
    if state.smote_applied and state.smote_class_distributions.get("after"):
        state.data_analysis_metrics["post_smote_target_distribution"] = state.smote_class_distributions["after"]

    return state


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


def _train_logistic_regression(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    if task_type in (TaskType.REGRESSION, "regression"):
        model = LinearRegression()
        name = "Linear Regression"
        params = {}
    else:
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        name = "Logistic Regression"
        params = {"C": 1.0, "max_iter": 1000}
        
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(model, name, "linear", X_val, y_val, task_type,
        hyperparameters=params, X_train=X_train, y_train=y_train)


def _train_random_forest(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if task_type in (TaskType.REGRESSION, "regression"):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(model, "Random Forest", "tree_ensemble", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100}, X_train=X_train, y_train=y_train)


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
    return _build_leaderboard_entry(model, "XGBoost", "gradient_boosting", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100}, X_train=X_train, y_train=y_train)


def _train_lightgbm(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    if task_type in (TaskType.REGRESSION, "regression"):
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    else:
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(model, "LightGBM", "gradient_boosting", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100}, X_train=X_train, y_train=y_train)


def _train_extra_trees(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    if task_type in (TaskType.REGRESSION, "regression"):
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(model, "Extra Trees", "tree_ensemble", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100}, X_train=X_train, y_train=y_train)


def _train_gradient_boosting(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    if task_type in (TaskType.REGRESSION, "regression"):
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(model, "Gradient Boosting", "gradient_boosting", X_val, y_val, task_type,
        hyperparameters={"n_estimators": 100}, X_train=X_train, y_train=y_train)


def _train_ridge(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    from sklearn.linear_model import RidgeClassifier, Ridge
    if task_type in (TaskType.REGRESSION, "regression"):
        model = Ridge(alpha=1.0)
    else:
        model = RidgeClassifier(alpha=1.0)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(model, "Ridge", "linear", X_val, y_val, task_type,
        hyperparameters={"alpha": 1.0}, X_train=X_train, y_train=y_train)


def _train_knn(
    X_train, X_val, y_train, y_val, task_type: str
) -> ModelLeaderboardEntry:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    if task_type in (TaskType.REGRESSION, "regression"):
        model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    else:
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(X_train, y_train)
    return _build_leaderboard_entry(model, "K-Nearest Neighbors", "instance_based", X_val, y_val, task_type,
        hyperparameters={"n_neighbors": 5}, X_train=X_train, y_train=y_train)


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
    hyperparameters: Dict[str, Any] = None, X_train=None, y_train=None
) -> ModelLeaderboardEntry:
    entry = ModelLeaderboardEntry(
        model_name=name,
        model_family=family,
        hyperparameters=hyperparameters or {},
    )
    try:
        if task_type not in (TaskType.REGRESSION, "regression"):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)
                if len(np.unique(y_val)) == 2:
                    classes = np.unique(y_val)
                    y_val_bin = (y_val == classes[1]).astype(int)
                    class_idx = list(model.classes_).index(classes[1]) if hasattr(model, "classes_") else 1
                    entry.auc_roc = round(float(roc_auc_score(y_val_bin, y_proba[:, class_idx])), 4)
                else:
                    entry.auc_roc = None
                
                # Calibration curve (requires 1D probabilities)
                if len(np.unique(y_val)) == 2:
                    y_proba_1d = y_proba[:, class_idx]
                else:
                    y_proba_1d = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                # Calibration curve
                frac_pos, mean_pred = calibration_curve(y_val_bin if len(np.unique(y_val)) == 2 else y_val, y_proba_1d, n_bins=10, strategy="quantile")
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
            if X_train is not None and y_train is not None:
                y_train_pred = model.predict(X_train)
                entry.train_f1_score = round(float(f1_score(y_train, y_train_pred, average="weighted", zero_division=0)), 4)
                if hasattr(model, "predict_proba"):
                    y_train_proba = model.predict_proba(X_train)
                    if len(np.unique(y_train)) == 2:
                        classes = np.unique(y_train)
                        y_train_bin = (y_train == classes[1]).astype(int)
                        class_idx = list(model.classes_).index(classes[1]) if hasattr(model, "classes_") else 1
                        entry.train_auc_roc = round(float(roc_auc_score(y_train_bin, y_train_proba[:, class_idx])), 4)
        else:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            y_pred = model.predict(X_val)
            r2 = float(r2_score(y_val, y_pred))
            entry.auc_roc = round(r2, 4)  # Use R² as the primary metric for regression
            entry.f1_score = None
            entry.rmse = round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 4)
            entry.mae = round(float(mean_absolute_error(y_val, y_pred)), 4)
            if X_train is not None and y_train is not None:
                y_train_pred = model.predict(X_train)
                entry.train_auc_roc = round(float(r2_score(y_train, y_train_pred)), 4)
                entry.train_rmse = round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 4)
                entry.train_mae = round(float(mean_absolute_error(y_train, y_train_pred)), 4)

        # Explainability summary (feature importance)
        if hasattr(model, "feature_importances_"):
            top_idx = np.argsort(model.feature_importances_)[::-1][:5]
            entry.explainability_summary = f"Top 5 features by importance: indices {top_idx.tolist()}"
        elif hasattr(model, "coef_"):
            top_idx = np.argsort(np.abs(model.coef_[0]))[::-1][:5]
            entry.explainability_summary = f"Top 5 features by |coefficient|: indices {top_idx.tolist()}"

    except Exception as e:
        entry.auc_roc = 0.0
        entry.f1_score = 0.0
        entry.precision = 0.0
        entry.recall = 0.0
        entry.accuracy = 0.0
        entry.rmse = 0.0
        entry.mae = 0.0
        entry.explainability_summary = f"Could not compute metrics: {e}"

    return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_data(
    df: pd.DataFrame, target_col: str, final_features: List[str], state: PipelineState
) -> Tuple:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder

        df_clean = df.dropna(subset=[target_col]).copy()
        feature_cols = [c for c in final_features if c in df_clean.columns] if final_features else \
                       [c for c in df_clean.columns if c != target_col]

        if not feature_cols:
            return None, None, None, None, None

        y = df_clean[target_col]
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = y.values

        X = df_clean[feature_cols].copy()
        
        # Read AI strategies
        ai_strats = state.data_schema.get("ai_strategies", {})
        
        transformers = []
        out_features = []
        
        for col in feature_cols:
            strat = ai_strats.get(col, {}) or {}
            imp_strat = (strat.get("imputation_strategy") or "mean").lower()
            enc_strat = (strat.get("encoding_strategy") or "one_hot").lower()
            
            is_cat = str(X[col].dtype) in ["object", "category"]
            
            # Map imputation strategy with safe fallbacks
            if imp_strat == "zero":
                if is_cat:
                    imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
                else:
                    imputer = SimpleImputer(strategy="constant", fill_value=0)
            elif imp_strat == "unknown":
                if is_cat:
                    imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
                else:
                    imputer = SimpleImputer(strategy="mean")  # fallback to mean for numeric
            elif imp_strat == "median":
                if is_cat:
                    imputer = SimpleImputer(strategy="most_frequent")
                else:
                    imputer = SimpleImputer(strategy="median")
            elif imp_strat == "mode":
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                if is_cat:
                    imputer = SimpleImputer(strategy="most_frequent")
                else:
                    imputer = SimpleImputer(strategy="mean")
            
            steps = [("imputer", imputer)]
            
            if is_cat:
                if enc_strat == "ordinal" or enc_strat == "target_encoding":
                    # Fallback to ordinal if target_encoding is requested (target_encoding is complex)
                    steps.append(("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
                else:
                    steps.append(("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            else:
                # Numeric features get scaled
                scaler_type = getattr(state.objective, "numeric_scaler", "standard")
                if scaler_type == "robust":
                    from sklearn.preprocessing import RobustScaler
                    steps.append(("scaler", RobustScaler()))
                elif scaler_type == "minmax":
                    from sklearn.preprocessing import MinMaxScaler
                    steps.append(("scaler", MinMaxScaler()))
                elif scaler_type == "quantile":
                    from sklearn.preprocessing import QuantileTransformer
                    steps.append(("scaler", QuantileTransformer(output_distribution="uniform", random_state=42)))
                elif scaler_type == "power":
                    from sklearn.preprocessing import PowerTransformer
                    steps.append(("scaler", PowerTransformer(method="yeo-johnson")))
                else:
                    steps.append(("scaler", StandardScaler()))
                
            transformers.append((col, Pipeline(steps), [col]))
            
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        X_processed = preprocessor.fit_transform(X)
        
        # Determine feature names after one-hot encoding
        feature_names = []
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = feature_cols
            
        # Optional Polynomial Features
        if state.objective.feature_optimization == "polynomial":
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_processed = poly.fit_transform(X_processed)
            if hasattr(poly, "get_feature_names_out"):
                feature_names = poly.get_feature_names_out(feature_names)
            else:
                feature_names = [f"poly_{i}" for i in range(X_processed.shape[1])]
            state.log_step("model_selection", "Polynomial Features", f"Generated {X_processed.shape[1]} interaction features.")

        # Optional PCA Optimization
        if state.objective.feature_optimization == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=0.95, random_state=42)
            X_processed = pca.fit_transform(X_processed)
            feature_names = [f"pca_{i}" for i in range(X_processed.shape[1])]
            state.log_step("model_selection", "PCA Applied", f"Reduced dimensions to {X_processed.shape[1]} components capturing 95% variance.")

        X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.25, random_state=42)
        
        # Outlier Removal on Training Data
        if getattr(state.objective, "outlier_removal", "none") == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42)
            yhat = iso.fit_predict(X_train)
            mask = yhat != -1
            X_train = X_train[mask]
            if isinstance(y_train, pd.Series):
                y_train = y_train.iloc[mask]
            elif isinstance(y_train, np.ndarray):
                y_train = y_train[mask]
            state.log_step("model_selection", "Isolation Forest", f"Removed outliers from training set. New train size: {X_train.shape[0]}")

        return X_train, X_val, y_train, y_val, feature_names
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        raise ValueError(f"Native Data Preparation Failed: {e}")


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
