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
You are a Principal Data Scientist and Lead Machine Learning Architect performing feature selection.

Objective: Predict '{target_column}' in a {task_type} task for the '{domain}' domain.
User Description: "{objective_text}"

Your goal is to build a robust, generalizable model by filtering out features that lead to overfitting, noise, or data leakage, while preserving high-value causal predictors.

### Deep Reasoning Process Requirement:
This is one of the most crucial parts of the ML system. You must perform an in-depth, adaptive reasoning process tailored to the specific dataset domain.

#### Self-Reflective Constraint Generation:
You are an expert Data Scientist. Analyze the objective and the raw data, and derive the strict fairness, data leakage, and operational constraints organically based purely on the context. Do not apply arbitrary rules.

Schema (column name → data type, number of unique values, sample values):
{schema}

Raw Data (First 10 rows):
{raw_data}

Respond ONLY with a valid JSON object in EXACTLY this structure (no markdown wrappers). 
By generating `step_0_domain_analysis` and `step_1_data_dictionary` first, you force yourself to deeply understand the dataset and legal constraints before making selection decisions in `step_2`.

{{
  "step_0_domain_analysis": {{
    "identified_domain": "Determine the exact industry/context based on the data.",
    "ethical_and_bias_constraints": "Dynamically generate constraints organically based on the domain. Does this domain require banning geographic proxies, or is location a fundamental driver?",
    "data_leakage_risks": "Identify what 'future information' looks like in this specific context.",
    "key_predictive_drivers": "Causal factors for this domain."
  }},
  "step_1_data_dictionary": [
    {{
      "column_name": "string",
      "inferred_description": "Extensive description of what this column actually represents in the real world based on the schema and samples."
    }}
  ],
  "step_2_feature_selection": [
    {{
      "column_name": "string",
      "chain_of_thought_reasoning": "Step 1: The user description explicitly states [QUOTE]. Step 2: The dataset schema shows this column is [TYPE/SAMPLES]. Step 3: Based strictly on the description and schema, does this violate any of the derived constraints?",
      "action": "keep" | "drop",
      "reason": "Final technical justification for keeping or dropping.",
      "imputation_strategy": "mean" | "median" | "mode" | "zero" | "unknown" (required if action is keep, otherwise null),
      "encoding_strategy": "one_hot" | "target_encoding" | "ordinal" | "none" (required if action is keep, otherwise null)
    }}
  ]
}}

CRITICAL REQUIREMENT: Both arrays MUST contain an entry for EVERY SINGLE COLUMN listed in the schema. Do not skip or omit any column. If a column is omitted, it will be considered useless and automatically dropped.
"""

_POST_CHECK_PROMPT = """
You are an AI Failsafe mechanism in an automated machine learning pipeline.
Objective: Predict '{target_column}' in a {task_type} task for the '{domain}' domain.
User Description: "{objective_text}"

Before taking action, actively reflect on the domain to determine the operational and legal constraints based purely on the data context.

You have two strict duties to ensure Context-Aware Feature Engineering:
1. RESCUE: Review the `pending_drops`. Rescue them if they are genuinely valid for this domain based on the raw data context.
2. EVICT: Review the `pending_keeps`. If a column violates the domain's strict ethical constraints or is fundamentally irrelevant context/noise based on the raw data context, you MUST drop it.
3. LOOPBACK RESOLUTION: If there are 'Previous Governance Failures' from a prior pipeline run, you MUST evict the specific features causing the failure (e.g., protected attributes or proxies) UNLESS dropping them would cause unacceptable model performance degradation for a fundamental causal driver. You must explicitly weigh the fairness vs performance tradeoff.

Raw Data (First 10 rows):
{raw_data}

Previous Governance Failures (if any):
{governance_failures}

Pending Drops (Proposed for deletion):
{pending_drops}

Pending Keeps (Proposed for retention):
{pending_keeps}

Respond ONLY with a valid JSON object in EXACTLY this structure (if there are none to rescue or evict, use an empty list []):
{{
  "domain_analysis": {{
    "ethical_and_bias_constraints": "Dynamically generate the strict legal, ethical, or fairness constraints for this specific domain.",
    "data_leakage_risks": "Identify what 'future information' looks like in this specific context."
  }},
  "rescued_features": [
    {{
      "column_name": "string",
      "chain_of_thought": "1. What does the user description state? 2. Is this actually predictive or a violation?",
      "reason": "Why this must be rescued (causal driver)."
    }}
  ],
  "evicted_features": [
    {{
      "column_name": "string",
      "chain_of_thought": "1. What does the user description state? 2. Does this violate the ethical constraints, leakage risks, or is it irrelevant noise?",
      "reason": "Why this is irrelevant to the scenario and must be evicted."
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
        err_msg = f"Target column '{target_col}' not found in the dataset. Please provide a valid target column."
        log_step_and_broadcast_sync(state, "feature_engineering", "Feature Engineering Aborted", err_msg)
        state.feature_log = FeatureLog(
            accepted=[], 
            rejected=[FeatureLogEntry(
                feature=target_col or "Unknown",
                transformation="Fatal Error",
                status="rejected",
                reason=err_msg
            )],
            final_feature_set=[]
        )
        return state

    accepted: List[FeatureLogEntry] = []
    rejected: List[FeatureLogEntry] = []
    pending_drops: Dict[str, str] = {}

    # ── Step 0: AI Semantic Feature Selection ─────────────────────────
    # Use LLM to conceptually drop IDs/names and provide reasoning for keeping others.
    try:
        # Extract 10 rows of raw data for AI context
        try:
            raw_data_str = df.head(10).to_json(orient="records", indent=2)
        except Exception:
            raw_data_str = "Error extracting raw data."

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
            schema=json.dumps(schema_info, indent=2),
            raw_data=raw_data_str
        )
        log_step_and_broadcast_sync(state, "feature_engineering", "AI Semantic Selection Started", f"Analyzing {len(schema_info)} features for semantic relevance to the objective.")
        semantic_result = get_llm_json(prompt)
        
        # Extract data dictionary and feature reasoning
        data_dict_array = semantic_result.get("step_1_data_dictionary", [])
        data_dictionary = {
            item.get("column_name"): item.get("inferred_description", "No description provided.") 
            for item in data_dict_array if isinstance(item, dict)
        }
        
        feature_reasoning = semantic_result.get("step_2_feature_selection", [])
        
        evaluated_cols = set()
        
        for item in feature_reasoning:
            if not isinstance(item, dict): continue
            c = item.get("column_name")
            action = item.get("action")
            reason = item.get("reason", "").strip()
            col_desc = data_dictionary.get(c, "No description generated.")
            imputation = item.get("imputation_strategy")
            encoding = item.get("encoding_strategy")
            cot = item.get("chain_of_thought_reasoning", "").strip()
            
            if not reason:
                reason = "Retained based on domain and objective relevance."
            
            if c and c in df.columns and c != target_col:
                evaluated_cols.add(c)
                if action == "drop":
                    full_reason = f"AI Semantic Filter: {reason}"
                    if col_desc:
                        full_reason = f"{col_desc} — {reason}"
                    if cot:
                        full_reason = f"{full_reason} [CoT: {cot}]"
                    pending_drops[c] = full_reason
                elif action == "keep":
                    if "ai_strategies" not in state.data_schema:
                        state.data_schema["ai_strategies"] = {}
                    state.data_schema["ai_strategies"][c] = {
                        "imputation_strategy": imputation,
                        "encoding_strategy": encoding,
                        "semantic_reason": reason,
                        "column_description": col_desc,
                        "chain_of_thought_reasoning": cot,
                    }
                    log_step_and_broadcast_sync(
                        state, "feature_engineering", f"AI retained '{c}'",
                        f"{col_desc} | CoT: {cot} | Imputation: {imputation} | Encoding: {encoding}"
                    )
                    
        # Drop any columns the LLM completely ignored!
        for c in list(df.columns):
            if c != target_col and c not in evaluated_cols and c not in protected_attrs:
                pending_drops[c] = "AI Semantic Filter: Omitted by AI during feature selection."

    except Exception as e:
        # If LLM fails, just proceed to math-based checks
        pass

    # ── Step 0.5: Drop Zero Variance / Constant Columns ───────────────
    for col in list(df.columns):
        if col == target_col:
            continue
        if df[col].nunique(dropna=False) <= 1:
            pending_drops[col] = "Math Check: Zero variance (constant column)."

    # ── Step 1: Drop high-missingness columns ─────────────────────────
    if state.data_health_report:
        for col, pct in state.data_health_report.missingness_flags.items():
            if col == target_col or col not in df.columns:
                continue
            if pct > HIGH_MISSINGNESS_DROP_THRESHOLD:
                pending_drops[col] = f"Math Check: {pct:.1%} missing values (threshold: {HIGH_MISSINGNESS_DROP_THRESHOLD:.0%})"

    # ── Step 2: Drop confirmed leakage columns ────────────────────────
    for col in leakage_cols:
        if col == target_col or col not in df.columns:
            continue
        pending_drops[col] = "Math Check: Flagged as potential target leakage by Data Profiling Agent."

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
            pending_drops[col] = f"Math Check: Multicollinearity: corr({col}, {max_corr_col}) = {max_corr_val:.2f} (> 0.85 threshold)."

    # ── Step 2.8: AI Post-Check Failsafe ──────────────────────────────
    rescued_cols = {}
    evicted_cols = {}
    
    pending_keeps = [c for c in df.columns if c != target_col and c not in pending_drops and c not in protected_attrs]

    if pending_drops or pending_keeps:
        log_step_and_broadcast_sync(state, "feature_engineering", "AI Post-Check Failsafe", f"Reviewing {len(pending_drops)} proposed drops and {len(pending_keeps)} proposed keeps for context-awareness.")
        
        keeps_info = {}
        for c in pending_keeps:
            keeps_info[c] = state.data_schema.get("ai_strategies", {}).get(c, {}).get("column_description", "No description provided.")

        post_prompt = _POST_CHECK_PROMPT.format(
            target_column=target_col,
            task_type=task_type.value if hasattr(task_type, 'value') else str(task_type),
            domain=state.objective.domain_tag,
            objective_text=state.objective.raw_text,
            pending_drops=json.dumps(pending_drops, indent=2),
            pending_keeps=json.dumps(keeps_info, indent=2),
            raw_data=raw_data_str,
            governance_failures=json.dumps(state.governance_audit.failure_reasons, indent=2) if state.governance_audit.failure_reasons else "None"
        )
        import time
        time.sleep(2)  # Mitigate Gemini free-tier 15RPM rate limits for rapid consecutive calls
        try:
            post_result = get_llm_json(post_prompt)
            rescues = post_result.get("rescued_features")
            evicts = post_result.get("evicted_features")
            domain_analysis = post_result.get("domain_analysis", {})
            reasoning_str = domain_analysis.get("ethical_and_bias_constraints", "") + "\n" + domain_analysis.get("data_leakage_risks", "")
            
            if not isinstance(rescues, list): rescues = []
            if not isinstance(evicts, list): evicts = []
            
            rescued_names = []
            evicted_names = []
            
            for res in rescues:
                if not isinstance(res, dict): continue
                rc = res.get("column_name")
                if rc in pending_drops:
                    rescued_cols[rc] = res.get("reason", "Rescued by AI failsafe.")
                    rescued_names.append(rc)
            for ev in evicts:
                if not isinstance(ev, dict): continue
                ec = ev.get("column_name")
                if ec in pending_keeps:
                    evicted_cols[ec] = ev.get("reason", "Evicted by AI failsafe as irrelevant context.")
                    evicted_names.append(ec)
                    
            conclusion_str = f"Rescued: {', '.join(rescued_names) if rescued_names else 'None'}. Evicted: {', '.join(evicted_names) if evicted_names else 'None'}."
            
            log_step_and_broadcast_sync(
                state, "feature_engineering", "AI Post-Check Failsafe Executed", 
                "The AI Failsafe has successfully evaluated pending drops and keeps.",
                problem=f"Evaluate {len(pending_drops)} drops and {len(pending_keeps)} keeps against domain constraints, leakage, and fairness loopbacks.",
                reasoning=reasoning_str,
                conclusion=conclusion_str
            )
        except Exception as e:
            log_step_and_broadcast_sync(state, "feature_engineering", "AI Post-Check Failed", f"LLM parsing error during failsafe: {str(e)}\nRaw Response keys: {list(post_result.keys()) if 'post_result' in locals() else 'None'}")

    # Apply final drops and record to audit log
    for col, pre_reason in pending_drops.items():
        if col not in rescued_cols:
            # Drop it
            rejected.append(FeatureLogEntry(
                feature=col,
                transformation="drop",
                status="rejected",
                reason=f"[Pre-Decision: {pre_reason}] -> [Post-Decision AI Failsafe: CONFIRMED DROP]",
                imputation_strategy=None,
            ))
            df = df.drop(columns=[col], errors="ignore")
            
    # Apply evictions
    for col, evict_reason in evicted_cols.items():
        rejected.append(FeatureLogEntry(
            feature=col,
            transformation="drop",
            status="rejected",
            reason=f"[Pre-Decision: Retained] -> [Post-Decision AI Failsafe: EVICTED - {evict_reason}]",
            imputation_strategy=None,
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

    # ── Step 4: Propose and evaluate transformations ──────────────────
    remaining_cols = [c for c in df.columns if c != target_col]
    feature_set = list(remaining_cols)
    for col in list(feature_set):
        log_step_and_broadcast_sync(state, "feature_engineering", f"Evaluating {col}", f"Testing transformations for feature {col}...")
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_categorical = not is_numeric
        unique_count = series.nunique(dropna=True)

        # Retrieve AI strategies
        ai_strat = state.data_schema.get("ai_strategies", {}).get(col, {}) or {}
        has_nulls = series.isna().any()
        imputation_strategy = (ai_strat.get("imputation_strategy") or ("mean" if is_numeric else "mode")).lower() if has_nulls else "none"

        transform_applied, transform_label, new_col_name = None, None, col
        
        rescue_msg = f" [Post-Decision AI Failsafe: RESCUED - {rescued_cols[col]}]" if col in rescued_cols else ""

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
                        reason=f"Log transform improved metric by {delta:+.4f} (skewness was {skew:.2f}).{rescue_msg}",
                        metric_delta=round(delta, 4),
                        imputation_strategy=imputation_strategy,
                    ))
                else:
                    accepted.append(FeatureLogEntry(
                        feature=col, transformation="keep_as_is", status="accepted",
                        reason=f"Kept as-is: log transform showed no improvement ({delta:+.4f}).{rescue_msg}",
                        metric_delta=round(delta, 4),
                        imputation_strategy=imputation_strategy,
                    ))
            else:
                accepted.append(FeatureLogEntry(
                    feature=col, transformation="keep_as_is", status="accepted",
                    reason=f"Numeric feature retained as-is (no significant skewness).{rescue_msg}",
                    imputation_strategy=imputation_strategy,
                ))

        elif is_categorical:
            if unique_count <= LOW_CARDINALITY_THRESHOLD:
                accepted.append(FeatureLogEntry(
                    feature=col, transformation="one_hot_encoding", status="accepted",
                    reason=f"One-hot encoding applied ({unique_count} unique values ≤ {LOW_CARDINALITY_THRESHOLD} threshold).{rescue_msg}",
                    imputation_strategy=imputation_strategy,
                ))
            else:
                accepted.append(FeatureLogEntry(
                    feature=col, transformation="frequency_encoding", status="accepted",
                    reason=f"Frequency encoding applied (high cardinality: {unique_count} unique values).{rescue_msg}",
                    imputation_strategy=imputation_strategy,
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
    encoded_count = len([e for e in accepted if e.transformation in ["one_hot", "one_hot_encoding", "target_encode", "target_encoding", "frequency_encode", "frequency_encoding"]])
    
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
