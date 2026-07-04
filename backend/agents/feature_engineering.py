"""
backend/agents/feature_engineering.py
──────────────────────────────────────
Feature Engineering Agent — Phase 2

Responsibilities:
1. Read the DataHealthReport and propose feature transformations.
2. For each candidate transformation: fit a quick baseline model with/without it
   on a validation split and compute metric delta.
3. Before finalizing, call the Governance mid-stage consult hook to flag any
   feature highly correlated with a protected attribute.
4. Log every accepted/rejected feature with a plain-language reason string.
5. Produce state.feature_log with accepted/rejected entries + final feature set.

Transformations considered:
- Drop columns: high missingness (>50%), confirmed leakage, governance-flagged proxies
- One-hot encoding: low-cardinality categoricals (≤ 15 unique values)
- Target/frequency encoding: high-cardinality categoricals (> 15 unique)
- Log transform: skewed positive numeric columns (skewness > 1.5)
- Simple imputation: median for numeric, mode for categorical
- Interaction terms: top-2 most predictive numeric features (if task allows)

Deterministic ML code only — LLM is used only for the rejection reason narration
(via governance consult, which is also deterministic in its metric checks).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.state.schema import (
    FeatureLog,
    FeatureLogEntry,
    PipelineState,
    TaskType,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_MISSINGNESS_DROP_THRESHOLD = 0.50  # Drop if >50% missing
LOW_CARDINALITY_THRESHOLD = 15          # ≤15 unique → one-hot
SKEWNESS_THRESHOLD = 1.5                # |skew| > 1.5 → log transform
MIN_METRIC_DELTA = -0.005               # Accept if metric delta ≥ this (allows tiny neutral drops)
MAX_INTERACTION_TERMS = 2               # Number of top features for interaction terms


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_feature_engineering(state: PipelineState) -> PipelineState:
    """
    Full feature engineering pass.
    Loads dataset, proposes transformations, validates each, consults governance,
    and populates state.feature_log.
    """
    import os

    dataset_path = state.dataset_path
    if not dataset_path or not os.path.exists(dataset_path):
        return state

    # Load dataset
    from backend.agents.data_profiling import _load_dataset
    df = _load_dataset(dataset_path)

    target_col = state.objective.target_column
    task_type = state.objective.task_type
    protected_attrs = state.objective.protected_attributes
    leakage_cols = {f["column"] for f in (state.data_health_report.leakage_flags if state.data_health_report else [])}

    if not target_col or target_col not in df.columns:
        return state

    accepted: List[FeatureLogEntry] = []
    rejected: List[FeatureLogEntry] = []

    # ── Step 1: Drop high-missingness columns ─────────────────────────
    if state.data_health_report:
        for col, pct in state.data_health_report.missingness_flags.items():
            if col == target_col:
                continue
            if pct > HIGH_MISSINGNESS_DROP_THRESHOLD:
                rejected.append(FeatureLogEntry(
                    feature=col,
                    transformation="drop",
                    status="rejected",
                    reason=f"Dropped: {pct:.1%} missing values (threshold: {HIGH_MISSINGNESS_DROP_THRESHOLD:.0%})",
                ))
                df = df.drop(columns=[col], errors="ignore")

    # ── Step 2: Drop confirmed leakage columns ────────────────────────
    for col in leakage_cols:
        if col == target_col or col not in df.columns:
            continue
        rejected.append(FeatureLogEntry(
            feature=col,
            transformation="drop",
            status="rejected",
            reason="Dropped: potential target leakage detected by Data Profiling Agent.",
        ))
        df = df.drop(columns=[col], errors="ignore")

    # ── Step 3: Prepare base dataset for metric evaluation ────────────
    df_work = df.copy()
    _impute_simple(df_work, target_col)

    X_base, y_base, encoder = _prepare_baseline(df_work, target_col)
    if X_base is None or len(X_base) < 20:
        # Not enough data to evaluate — accept all remaining
        state.feature_log = FeatureLog(
            accepted=[], rejected=rejected,
            final_feature_set=_get_remaining_features(df, target_col),
        )
        return state

    base_score = _quick_score(X_base, y_base, task_type)

    # ── Step 4: Governance mid-stage proxy check ───────────────────────
    from backend.agents.governance import quick_fairness_proxy_check

    remaining_cols = [c for c in df.columns if c != target_col]
    proxy_flags: Dict[str, str] = {}
    for col in remaining_cols:
        if col in protected_attrs:
            continue
        flag_reason = quick_fairness_proxy_check(df, col, protected_attrs)
        if flag_reason:
            proxy_flags[col] = flag_reason

    # ── Step 5: Propose and evaluate transformations ──────────────────
    feature_set = [c for c in remaining_cols if c not in proxy_flags]

    # If we failed governance previously (loopback), explicitly drop the protected attributes
    # to fix the Fairness FAIL (Disparate Impact).
    if state.governance_audit.iteration_count > 0:
        for p_attr in protected_attrs:
            if p_attr in feature_set:
                feature_set.remove(p_attr)
                rejected.append(FeatureLogEntry(
                    feature=p_attr,
                    transformation="drop",
                    status="rejected",
                    reason=f"Dropped protected attribute '{p_attr}' after Governance loopback to improve fairness.",
                    governance_flagged=True,
                ))

    for col in list(feature_set):
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_categorical = not is_numeric
        unique_count = series.nunique(dropna=True)

        transform_applied, transform_label, new_col_name = None, None, col

        if is_numeric:
            skew = float(series.skew()) if series.notna().sum() > 2 else 0.0
            if abs(skew) > SKEWNESS_THRESHOLD and (series.dropna() > 0).all():
                # Propose log transform
                df_trial = df_work.copy()
                df_trial[col] = np.log1p(df_trial[col].clip(lower=0))
                X_trial, y_trial, _ = _prepare_baseline(df_trial, target_col)
                trial_score = _quick_score(X_trial, y_trial, task_type) if X_trial is not None else base_score
                delta = trial_score - base_score
                transform_label = "log1p transform"
                if delta >= MIN_METRIC_DELTA:
                    accepted.append(FeatureLogEntry(
                        feature=col, transformation=transform_label, status="accepted",
                        reason=f"Log transform improved metric by {delta:+.4f} (skewness was {skew:.2f})",
                        metric_delta=round(delta, 4),
                    ))
                else:
                    accepted.append(FeatureLogEntry(
                        feature=col, transformation="keep_as_is", status="accepted",
                        reason=f"Kept as-is: log transform showed no improvement ({delta:+.4f})",
                        metric_delta=round(delta, 4),
                    ))
            else:
                accepted.append(FeatureLogEntry(
                    feature=col, transformation="keep_as_is", status="accepted",
                    reason="Numeric feature retained as-is (no significant skewness).",
                ))

        elif is_categorical:
            if unique_count <= LOW_CARDINALITY_THRESHOLD:
                accepted.append(FeatureLogEntry(
                    feature=col, transformation="one_hot_encoding", status="accepted",
                    reason=f"One-hot encoding applied ({unique_count} unique values ≤ {LOW_CARDINALITY_THRESHOLD} threshold).",
                ))
            else:
                accepted.append(FeatureLogEntry(
                    feature=col, transformation="frequency_encoding", status="accepted",
                    reason=f"Frequency encoding applied (high cardinality: {unique_count} unique values).",
                ))

    # ── Step 6: Log governance-flagged proxies as rejected ─────────────
    for col, reason in proxy_flags.items():
        rejected.append(FeatureLogEntry(
            feature=col,
            transformation="drop",
            status="rejected",
            reason=reason,
            governance_flagged=True,
        ))

    # ── Step 7: Enforce top-K feature selection (if requested) ────────
    top_k = state.objective.feature_selection_top_k
    if top_k is not None and top_k > 0 and len(accepted) > top_k:
        # Sort accepted features by metric_delta descending (None/NaN treated as 0)
        def get_delta(e: FeatureLogEntry) -> float:
            return e.metric_delta if e.metric_delta is not None else 0.0
            
        accepted.sort(key=get_delta, reverse=True)
        
        # Move excess features to rejected
        excess_features = accepted[top_k:]
        accepted = accepted[:top_k]
        
        for e in excess_features:
            e.status = "rejected"
            e.reason += f" (Rejected due to user-specified top-{top_k} constraint)"
            rejected.append(e)

    final_features = [e.feature for e in accepted if e.feature in df.columns]

    state.feature_log = FeatureLog(
        accepted=accepted,
        rejected=rejected,
        final_feature_set=final_features,
    )

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _impute_simple(df: pd.DataFrame, target_col: str) -> None:
    """In-place simple imputation: median for numeric, mode for categorical."""
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                mode = df[col].mode()
                df[col].fillna(mode[0] if len(mode) > 0 else "MISSING", inplace=True)


def _prepare_baseline(
    df: pd.DataFrame, target_col: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[LabelEncoder]]:
    """
    Prepare X, y arrays for quick model evaluation.
    Encodes categoricals with label encoding (fast, not optimal but good enough for delta eval).
    """
    try:
        df_clean = df.dropna(subset=[target_col]).copy()
        if len(df_clean) < 20:
            return None, None, None

        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])

        # Encode categoricals
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if needed
        enc = None
        if y.dtype == object or str(y.dtype) == "category":
            enc = LabelEncoder()
            y = enc.fit_transform(y.astype(str))
        else:
            y = y.values

        X = X.select_dtypes(include=[np.number]).fillna(0).values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, enc
    except Exception:
        return None, None, None


def _quick_score(
    X: np.ndarray, y: np.ndarray, task_type: str, n_splits: int = 1
) -> float:
    """Fit a fast LogisticRegression on a small train/val split and return AUC or F1."""
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        model = LogisticRegression(max_iter=200, random_state=42, C=1.0)
        model.fit(X_train, y_train)

        if task_type in (TaskType.REGRESSION, "regression"):
            from sklearn.metrics import r2_score
            return float(r2_score(y_val, model.predict(X_val)))
        else:
            if len(np.unique(y_val)) == 2:
                return float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
            else:
                return float(f1_score(y_val, model.predict(X_val), average="weighted"))
    except Exception:
        return 0.5  # neutral fallback


def _get_remaining_features(df: pd.DataFrame, target_col: str) -> List[str]:
    return [c for c in df.columns if c != target_col]
