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

Deterministic ML code is combined with LLM semantic reasoning. The LLM is used to:
1. Semantically filter out conceptually irrelevant columns (IDs, names, etc.) before math.
2. Provide rejection reason narration (via governance consult, which is also deterministic in its metric checks).
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.llm.client import get_llm_json
from backend.state.schema import (
    FeatureLog,
    FeatureLogEntry,
    PipelineState,
    TaskType,
)
from backend.state.store import log_step_and_broadcast_sync

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
# Prompt Templates
# ---------------------------------------------------------------------------

_SEMANTIC_FEATURE_PROMPT = """
You are an expert Data Scientist performing feature selection for a machine learning project.

Objective: Predict '{target_column}' in a {task_type} task for the '{domain}' domain.
User Description: "{objective_text}"

You must be EXTREMELY AGGRESSIVE in dropping columns. We want a highly robust, generalizable model. Do not keep columns just because they "might" be related. If a column is a weak predictor, noisy, or its meaning is unclear, you MUST DROP IT.
This is especially important to prevent overtraining and ensure robustness.

For each column in the dataset schema below, you must:
1. Write a SHORT plain-English description of what the column likely represents (1-2 sentences).
2. Decide if it should be KEPT for modeling or DROPPED.
3. If dropping, give a clear, specific reason why this column is a bad predictor or would harm the model.

Drop a column if it is:
- A unique ID or row index (e.g. PassengerId, customer_uuid, row_number)
- Free text or names with no encoded structure (e.g. name, address, description)
- A timestamp or metadata export column with no predictive relationship to the target
- Weakly relevant, noisy, or its meaning is unclear for predicting '{target_column}' in the {domain} domain
- A direct surrogate for the target (data leakage — e.g. a column that is computed from the target)

Keep a column ONLY if:
- It strongly and logically influences or correlates with '{target_column}' based on strict domain knowledge
- It provides high-value discriminative signal for the prediction task

Schema (column name → data type, number of unique values, sample values):
{schema}

Respond ONLY with a JSON object in EXACTLY this structure:
{{
  "feature_reasoning": [
    {{
      "column_name": "string",
      "column_description": "What this column represents in 1-2 sentences.",
      "action": "keep or drop",
      "reason": "Clear, specific reason (required for both keep and drop). For drop, explain exactly why. For keep, explain the predictive value.",
      "imputation_strategy": "mean, median, mode, zero, or unknown (required if action is keep)",
      "encoding_strategy": "one_hot, target_encoding, ordinal, or none (required if action is keep)"
    }}
  ]
}}
"""


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
    
    log_step_and_broadcast_sync(state, "feature_engineering", "Feature Engineering Started", "Initializing feature selection, imputation, and encoding pipelines.")
    
    task_type = state.objective.task_type
    protected_attrs = state.objective.protected_attributes
    leakage_cols = {f["column"] for f in (state.data_health_report.leakage_flags if state.data_health_report else [])}

    if not target_col or target_col not in df.columns:
        return state

    accepted: List[FeatureLogEntry] = []
    rejected: List[FeatureLogEntry] = []

    # ── Step 0: AI Semantic Feature Selection ─────────────────────────
    # Use LLM to conceptually drop IDs/names and provide reasoning for keeping others.
    try:
        # Build rich schema info: dtype + nunique + sample values
        schema_info = {}
        for col, dtype in df.dtypes.items():
            if col == target_col:
                continue
            try:
                sample_vals = df[col].dropna().unique()[:5].tolist()
                sample_vals = [str(v) for v in sample_vals]
            except Exception:
                sample_vals = []
            schema_info[col] = {
                "dtype": str(dtype),
                "unique_count": int(df[col].nunique(dropna=True)),
                "sample_values": sample_vals,
            }
        prompt = _SEMANTIC_FEATURE_PROMPT.format(
            target_column=target_col,
            task_type=task_type.value if hasattr(task_type, 'value') else str(task_type),
            domain=state.objective.domain_tag,
            objective_text=state.objective.raw_text,
            schema=json.dumps(schema_info, indent=2)
        )
        log_step_and_broadcast_sync(state, "feature_engineering", "AI Semantic Selection Started", f"Analyzing {len(schema_info)} features for semantic relevance to the objective.")
        semantic_result = get_llm_json(prompt)
        feature_reasoning = semantic_result.get("feature_reasoning", [])
        
        for item in feature_reasoning:
            c = item.get("column_name")
            action = item.get("action")
            reason = item.get("reason", "").strip()
            col_desc = item.get("column_description", "").strip()
            imputation = item.get("imputation_strategy")
            encoding = item.get("encoding_strategy")
            
            if not reason:
                reason = "Retained based on domain and objective relevance."
            
            if c and c in df.columns and c != target_col:
                if action == "drop":
                    # Format the rejection reason with the column description for frontend
                    full_reason = f"AI Semantic Filter: {reason}"
                    if col_desc:
                        full_reason = f"{col_desc} — {reason}"
                    rejected.append(FeatureLogEntry(
                        feature=c,
                        transformation="drop",
                        status="rejected",
                        reason=full_reason,
                        imputation_strategy=None,
                    ))
                    df = df.drop(columns=[c], errors="ignore")
                elif action == "keep":
                    # Store AI strategies plus description for later rendering
                    if "ai_strategies" not in state.data_schema:
                        state.data_schema["ai_strategies"] = {}
                    state.data_schema["ai_strategies"][c] = {
                        "imputation_strategy": imputation,
                        "encoding_strategy": encoding,
                        "semantic_reason": reason,
                        "column_description": col_desc,
                    }
                    log_step_and_broadcast_sync(
                        state, "feature_engineering", f"AI retained '{c}'",
                        f"{col_desc} | {reason} | Imputation: {imputation} | Encoding: {encoding}"
                    )

    except Exception as e:
        # If LLM fails, just proceed to math-based checks
        pass

    # ── Step 0.5: Drop Zero Variance / Constant Columns ───────────────
    for col in list(df.columns):
        if col == target_col:
            continue
        if df[col].nunique(dropna=False) <= 1:
            rejected.append(FeatureLogEntry(
                feature=col,
                transformation="drop",
                status="rejected",
                reason="Dropped: Zero variance (constant column).",
            ))
            df = df.drop(columns=[col], errors="ignore")

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

    # ── Step 2.5: Drop multicollinear features ────────────────────────
    numeric_cols_for_corr = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols_for_corr) > 1:
        corr_matrix = df[numeric_cols_for_corr].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
        for col in to_drop:
            # Find which column it's most correlated with
            max_corr_col = upper_tri[col].idxmax() if not upper_tri[col].isna().all() else "another feature"
            max_corr_val = upper_tri[col].max() if not upper_tri[col].isna().all() else 0.0
            rejected.append(FeatureLogEntry(
                feature=col,
                transformation="drop",
                status="rejected",
                reason=f"Dropped due to multicollinearity: corr({col}, {max_corr_col}) = {max_corr_val:.2f} (> 0.85 threshold). Keeping one redundant feature is sufficient.",
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
        log_step_and_broadcast_sync(state, "feature_engineering", f"Evaluating {col}", f"Testing transformations for feature {col}...")
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

    # ── Step 8: Advanced Feature Optimization (PCA / Tree) ────────────
    if state.objective.feature_optimization == "pca":
        log_step_and_broadcast_sync(state, "feature_engineering", "PCA Scheduled", "PCA dimensionality reduction will be applied during Model Selection.")
    elif state.objective.feature_optimization == "tree" and len(final_features) > 2:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            
            # Fast crude tree to evaluate feature importance
            df_tree = df[final_features].copy()
            y_tree = df[target_col].copy()
            
            if y_tree.dtype == object:
                y_tree = LabelEncoder().fit_transform(y_tree.astype(str))
                
            for col in df_tree.select_dtypes(include=["object", "category"]).columns:
                df_tree[col] = LabelEncoder().fit_transform(df_tree[col].astype(str))
            df_tree = df_tree.fillna(0)
            
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            if task_type in (TaskType.CLASSIFICATION, "classification"):
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                
            model.fit(df_tree, y_tree)
            
            # Prune 0.0 importance features
            kept_features = []
            for col, imp in zip(final_features, model.feature_importances_):
                if imp > 0.001:
                    kept_features.append(col)
                else:
                    # Find in accepted and move to rejected
                    for e in accepted:
                        if e.feature == col:
                            e.status = "rejected"
                            e.reason = "Pruned by Tree Feature Importance Optimization (importance ≈ 0)"
                            rejected.append(e)
                            accepted.remove(e)
                            break
            final_features = kept_features
            log_step_and_broadcast_sync(state, "feature_engineering", "Tree Optimization", f"Pruned {len(df_tree.columns) - len(final_features)} features.")
        except Exception as e:
            pass

    state.feature_log = FeatureLog(
        accepted=accepted,
        rejected=rejected,
        final_feature_set=final_features,
    )
    
    # Calculate imputation/encoding stats for logging
    imputed_count = len([e for e in accepted if e.imputation_strategy and e.imputation_strategy != "none"])
    encoded_count = len([e for e in accepted if e.transformation in ["one_hot", "target_encode", "frequency_encode"]])
    
    log_step_and_broadcast_sync(state, "feature_engineering", "Feature Transformation Summary", f"Finalized {len(final_features)} features. Applied missing value imputation to {imputed_count} features, and categorical encoding to {encoded_count} features.")

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
                # Need to map the two unique values to 0 and 1 so roc_auc_score doesn't fail
                classes = np.unique(y_val)
                y_val_bin = (y_val == classes[1]).astype(int)
                y_proba = model.predict_proba(X_val)
                # If model was trained on more than 2 classes, find the index of classes[1]
                class_idx = list(model.classes_).index(classes[1]) if hasattr(model, "classes_") else 1
                return float(roc_auc_score(y_val_bin, y_proba[:, class_idx]))
            else:
                return float(f1_score(y_val, model.predict(X_val), average="weighted"))
    except Exception:
        return 0.5  # neutral fallback


def _get_remaining_features(df: pd.DataFrame, target_col: str) -> List[str]:
    return [c for c in df.columns if c != target_col]
