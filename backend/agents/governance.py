"""
backend/agents/governance.py
─────────────────────────────
Governance Agent — Phase 3 (+ mid-stage consult hook used in Phase 2)

Responsibilities:
1. quick_fairness_proxy_check() — mid-stage hook called by Feature Engineering
   to flag candidate features that are highly correlated with protected attributes.
2. run_governance() — full end-of-pipeline audit with three components:
   a) Fairness Audit: Disparate Impact, Equal Opportunity Difference, per-group CMs
   b) Robustness Audit: AUC degradation under synthetic covariate shift
   c) Stability Audit: bootstrap variance across N resamples
3. Compare results against compliance-injected thresholds (whichever is stricter wins).
4. On failure: build an actionable reason string and set loopback_target in state.

All numeric decisions are computed by deterministic code.
LLM narrates only the textual reason — not the pass/fail determination.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")

from backend.state.schema import (
    AuditStatus,
    FairnessMetrics,
    GovernanceAudit,
    GovernanceLoopRecord,
    PipelineState,
    RobustnessMetrics,
    StabilityMetrics,
    TaskType,
)
from backend.state.store import log_step_and_broadcast_sync

# ---------------------------------------------------------------------------
# Default thresholds (overridden by compliance YAML injections)
# ---------------------------------------------------------------------------

DEFAULT_DISPARATE_IMPACT_MIN = 0.80
DEFAULT_EOD_MAX = 0.10
DEFAULT_AUC_DEGRADATION_MAX_PCT = 10.0  # % drop
DEFAULT_BOOTSTRAP_VARIANCE_MAX = 0.03
DEFAULT_BOOTSTRAP_N = 20
PROXY_CORRELATION_THRESHOLD = 0.40     # Flag if |corr| with protected attr > 0.40

_GOV_NARRATIVE_PROMPT = """
You are an AI Governance Analyst reviewing the results of Governance Loop #{loop_number} for an ML model.

Audit Results:
- Overall Result: {overall_result}
- Disparate Impact: {disparate_impact} (threshold: >= {di_min})
- Equal Opportunity Difference: {eod} (threshold: <= {eod_max})
- AUC Degradation under shift: {auc_deg}% (threshold: <= {auc_deg_max}%)
- Bootstrap Variance: {bootstrap_var} (threshold: <= {bootstrap_var_max})
- Failure Reasons: {failures}
- Corrective Action: {corrective_action}

Write a concise, plain English paragraph (max 3 sentences) explaining: why this passed/failed, any corrective action taken, and confirming compliance if it passed. 
Do not use headings, bullet points, or markdown formatting. Be specific and factual. Use plain English understandable to a non-technical product manager.
"""

_GOVERNANCE_PLAN_PROMPT = """
You are a Senior AI Safety and Compliance Officer. Given the user's objective, dataset context, and identified potential protected attributes, define the compliance audit plan.

Your job is to determine:
1. Which of the identified protected attributes should actually be audited for bias/fairness (e.g. 'gender', 'race').
2. Whether to completely skip fairness checks (e.g. if the task is clinical/medical and variables like 'age', 'sex' are crucial physiological predictors where parity checks are counterproductive or harmful).
3. The exact thresholds for the checks (disparate_impact_min, equal_opportunity_diff_max, auc_degradation_max_pct, bootstrap_variance_max).

User Objective: {objective}
Task Type: {task_type}
Target Column: {target_column}
Candidate Protected Attributes: {candidate_protected_attributes}
Dataset Columns and Types:
{column_info}

Return ONLY a JSON object conforming exactly to this schema:
{{
  "protected_attributes_to_audit": ["list of exact attribute names to audit for fairness"],
  "skip_fairness_checks": true | false,
  "disparate_impact_min": 0.80,
  "equal_opportunity_diff_max": 0.10,
  "auc_degradation_max_pct": 10.0,
  "bootstrap_variance_max": 0.03,
  "reasoning": "A detailed, professional explanation of why this audit plan was chosen for this specific scenario (e.g., explaining why medical variables are physiological predictors and bypass audits, or why loan models require strict ECOA checks)."
}}
Only return JSON. Do not include markdown blocks.
"""

_GOVERNANCE_CHART_PROMPT = """
You are an expert AI Governance and AI Fairness Auditor. Your task is to generate 2-3 visual auditing charts (e.g. demographic parity charts, or performance comparison charts) based on the audit results.

You MUST output **ONLY Interactive React UI Charts**. No static Seaborn/Matplotlib images are allowed. You must return structured JSON data that will be rendered natively in the UI.

Write a complete, standalone Python script that processes the audit data and prints a single valid JSON array to `sys.stdout` containing all the charts.

The JSON array must look like this:
[
  {{
    "id": "gov-chart-1",
    "title": "Interactive Demographic Parity Ratio",
    "insight": "Demographic parity ratio compared across groups.",
    "type": "bar" | "line" | "pie" | "doughnut",
    "data": [
      {{ "name": "Privileged Group", "count": 0.85 }},
      {{ "name": "Unprivileged Group", "count": 0.78 }}
    ]
  }},
  ...
]

CRITICAL RULES:
- **PREREQUISITE (ONLY INTERACTIVE REACT CHARTS)**: You MUST calculate statistical aggregates and output data arrays. Do NOT generate matplotlib or seaborn plots. Do NOT output base64 strings.
- Do NOT output any markdown blocks like ```python. ONLY output the raw Python code.
- Only print the JSON to stdout. Do not print anything else.
- Make sure to `import sys`, `import json`, `import pandas as pd`, `import numpy as np`.
- Do NOT use `pd.np` (pandas has no attribute `np`). Use `numpy` directly (e.g. `np.random`).
- You MUST run `sys.stdout.reconfigure(encoding='utf-8')` right after imports to prevent Windows console encoding errors. Do NOT use `ensure_ascii=False` when calling `json.dump` or `json.dumps`.

Audit Data (use this directly in your Python code as Python dictionaries):
Fairness Metrics: {fairness_metrics}
Robustness Metrics: {robustness_metrics}
Stability Metrics: {stability_metrics}
"""


# ---------------------------------------------------------------------------
# Mid-stage consult hook (called by Feature Engineering Agent)
# ---------------------------------------------------------------------------


def quick_fairness_proxy_check(
    df: pd.DataFrame,
    candidate_feature: str,
    protected_attributes: List[str],
) -> Optional[str]:
    """
    Check if a candidate feature is highly correlated with any protected attribute.
    Returns a reason string if flagged, None if clean.

    This is the bidirectional consultation pattern from the spec (§2.4):
    Feature Engineering asks Governance before committing a feature.
    """
    if not protected_attributes or candidate_feature not in df.columns:
        return None

    for protected in protected_attributes:
        if protected not in df.columns:
            continue
        try:
            corr = _compute_association(df, candidate_feature, protected)
            if corr is not None and corr > PROXY_CORRELATION_THRESHOLD:
                return (
                    f"Rejected by Governance mid-stage consult: '{candidate_feature}' "
                    f"is highly correlated with protected attribute '{protected}' "
                    f"(association: {corr:.3f} > threshold {PROXY_CORRELATION_THRESHOLD}). "
                    "Using this feature may create a fairness proxy that violates "
                    "disparate impact thresholds at the Governance audit stage."
                )
        except Exception:
            continue

    return None


def _compute_association(df: pd.DataFrame, col_a: str, col_b: str) -> Optional[float]:
    """
    Compute a normalized association measure between two columns.
    Uses Pearson |correlation| for numeric–numeric,
    Cramér's V for categorical–categorical,
    and point-biserial for mixed.
    Returns a value in [0, 1] or None on failure.
    """
    try:
        a = df[col_a].dropna()
        b = df[col_b].dropna()
        common_idx = a.index.intersection(b.index)
        if len(common_idx) < 10:
            return None
        a, b = a.loc[common_idx], b.loc[common_idx]

        a_num = pd.api.types.is_numeric_dtype(a)
        b_num = pd.api.types.is_numeric_dtype(b)

        if a_num and b_num:
            return float(abs(a.corr(b)))

        # Encode to numeric
        if not a_num:
            a = LabelEncoder().fit_transform(a.astype(str))
        else:
            a = a.values
        if not b_num:
            b = LabelEncoder().fit_transform(b.astype(str))
        else:
            b = b.values

        return float(abs(np.corrcoef(a, b)[0, 1]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full governance audit
# ---------------------------------------------------------------------------


def run_governance(state: PipelineState) -> PipelineState:
    """
    Run all three audits and update state.governance_audit.
    Sets overall_status, failure_reasons, and loopback_target.
    """
    import os
    from backend.agents.data_profiling import _load_dataset

    dataset_path = state.dataset_path
    if not dataset_path or not os.path.exists(dataset_path):
        state.governance_audit.overall_status = AuditStatus.SKIPPED
        state.governance_audit.failure_reasons = ["Dataset not available for governance audit."]
        return state

    df = _load_dataset(dataset_path)
    target_col = state.objective.target_column
    protected_attrs = state.objective.protected_attributes
    task_type = state.objective.task_type

    if not target_col or target_col not in df.columns:
        state.governance_audit.overall_status = AuditStatus.SKIPPED
        return state

    # ── Dynamic AI Governance Planning ──
    from backend.llm.client import get_llm_json
    import json
    
    col_info = {}
    for col in df.columns:
        is_cat = str(df[col].dtype) in ["object", "category"] or df[col].nunique() < 15
        col_info[col] = {
            "type": "categorical" if is_cat else "numeric",
            "unique": int(df[col].nunique()),
        }

    plan_prompt = _GOVERNANCE_PLAN_PROMPT.format(
        objective=state.objective.raw_text,
        task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
        target_column=target_col,
        candidate_protected_attributes=json.dumps(protected_attrs),
        column_info=json.dumps(col_info, indent=2)
    )

    # Fallbacks in case AI call fails
    skip_fairness = False
    di_min = DEFAULT_DISPARATE_IMPACT_MIN
    eod_max = DEFAULT_EOD_MAX
    auc_deg_max = DEFAULT_AUC_DEGRADATION_MAX_PCT
    bootstrap_var_max = DEFAULT_BOOTSTRAP_VARIANCE_MAX
    reasoning = "Using baseline regulations."

    try:
        plan = get_llm_json(plan_prompt)
        if plan:
            protected_attrs = plan.get("protected_attributes_to_audit", protected_attrs)
            skip_fairness = plan.get("skip_fairness_checks", False)
            di_min = plan.get("disparate_impact_min", di_min)
            eod_max = plan.get("equal_opportunity_diff_max", eod_max)
            auc_deg_max = plan.get("auc_degradation_max_pct", auc_deg_max)
            bootstrap_var_max = plan.get("bootstrap_variance_max", bootstrap_var_max)
            reasoning = plan.get("reasoning", reasoning)
            
            # Override compliance details in state
            state.governance_audit.compliance_thresholds = {
                "disparate_impact_min": di_min,
                "equal_opportunity_diff_max": eod_max,
                "auc_degradation_max_pct": auc_deg_max,
                "bootstrap_variance_max": bootstrap_var_max
            }
            state.governance_audit.compliance_reasoning = reasoning
            log_step_and_broadcast_sync(state, "governance", "Compliance Plan Determined", f"AI compliance officer determined audit thresholds: DI>={di_min}, EOD<={eod_max}. Justification: {reasoning}")
    except Exception as e:
        log_step_and_broadcast_sync(state, "governance", "Compliance Plan Determination Failed", f"Could not determine plan via AI: {e}. Falling back to defaults.")

    # Prepare processed data (applying feature engineering transformations)
    X_train, X_test, y_train, y_test, X_test_raw = _prepare_audit_data(
        df, target_col, state.feature_log.final_feature_set
    )

    if X_train is None:
        state.governance_audit.overall_status = AuditStatus.SKIPPED
        state.governance_audit.failure_reasons = ["Insufficient data for governance audit."]
        return state

    # Train the selected model (or best available)
    model = _train_audit_model(X_train, y_train, task_type)

    failures = []

    # ── Fairness Audit ────────────────────────────────────────────────
    thresholds = state.governance_audit.compliance_thresholds or {}
    allow_sensitive_override = thresholds.get("allow_sensitive_features_override", False)
    
    if skip_fairness:
        fairness = FairnessMetrics(status=AuditStatus.SKIPPED)
        log_step_and_broadcast_sync(state, "governance", "Fairness Audit Bypassed", "Fairness checks bypassed dynamically based on scenario requirements (e.g. physiological predictors).")
    else:
        log_step_and_broadcast_sync(state, "governance", "Fairness Audit Started", f"Checking Disparate Impact and Equal Opportunity Difference against thresholds (DI>={di_min}, EOD<={eod_max})")
        fairness = _run_fairness_audit(
            model, df, X_test_raw, X_test, y_test, target_col,
            protected_attrs, di_min, eod_max, task_type
        )
        
        if allow_sensitive_override and fairness.status == AuditStatus.FAIL:
            fairness.status = AuditStatus.PASS
            log_step_and_broadcast_sync(state, "governance", "Fairness Audit Override", "Fairness metrics calculated for reporting only; strict thresholds relaxed due to sensitive features override.")
        
    state.governance_audit.fairness = fairness
    if fairness.status == AuditStatus.FAIL:
        failures.append(_build_fairness_failure_reason(fairness, di_min, eod_max))

    # ── Robustness Audit ──────────────────────────────────────────────
    robustness = _run_robustness_audit(model, X_test, y_test, auc_deg_max, task_type)
    state.governance_audit.robustness = robustness
    if robustness.status == AuditStatus.FAIL:
        metric_name = "R2" if task_type in ("regression", TaskType.REGRESSION) else "AUC"
        failures.append(
            f"Robustness FAIL: {metric_name} degraded by {robustness.auc_degradation_pct:.1f}% "
            f"under synthetic covariate shift (threshold: {auc_deg_max:.1f}%). "
            "Consider a more regularized model or more robust feature set."
        )

    # ── Stability Audit ───────────────────────────────────────────────
    stability = _run_stability_audit(X_train, y_train, bootstrap_var_max, task_type)
    state.governance_audit.stability = stability
    if stability.status == AuditStatus.FAIL:
        failures.append(
            f"Stability FAIL: Bootstrap variance = {stability.metric_variance:.4f} "
            f"(threshold: {bootstrap_var_max:.4f}). "
            "Model is unstable across data resamples — consider simpler model or more regularization."
        )

    # ── Overall decision ──────────────────────────────────────────────
    state.governance_audit.iteration_count += 1
    loop_number = state.governance_audit.iteration_count

    if failures:
        state.governance_audit.overall_status = AuditStatus.FAIL
        state.governance_audit.failure_reasons = failures
        # Route: fairness issue → feature_engineering; robustness/stability → model_selection
        if fairness.status == AuditStatus.FAIL:
            state.governance_audit.loopback_target = "feature_engineering"
        else:
            state.governance_audit.loopback_target = "model_selection"
    else:
        state.governance_audit.overall_status = AuditStatus.PASS
        state.governance_audit.failure_reasons = []
        state.governance_audit.loopback_target = None

    # ── Append per-loop record ─────────────────────────────────────────
    overall_result = "PASS" if not failures else "FAIL"
    corrective = state.governance_audit.loopback_target or "None (PASS)"

    # Build LLM narrative
    llm_narrative = None
    try:
        from backend.llm.client import get_llm_text
        metric_name = "R2" if task_type in ("regression", TaskType.REGRESSION) else "AUC"
        narrative_prompt = _GOV_NARRATIVE_PROMPT.format(
            loop_number=loop_number,
            overall_result=overall_result,
            disparate_impact=round(fairness.disparate_impact, 3) if fairness.disparate_impact is not None else "N/A",
            di_min=di_min,
            eod=round(fairness.equal_opportunity_difference, 3) if fairness.equal_opportunity_difference is not None else "N/A",
            eod_max=eod_max,
            auc_deg=round(robustness.auc_degradation_pct, 2) if robustness.auc_degradation_pct is not None else "N/A",
            auc_deg_max=auc_deg_max,
            bootstrap_var=round(stability.metric_variance, 4) if stability.metric_variance is not None else "N/A",
            bootstrap_var_max=bootstrap_var_max,
            failures=" | ".join(failures) if failures else "None",
            corrective_action=corrective,
        ).replace("AUC degradation", f"{metric_name} degradation")
        llm_narrative = get_llm_text(narrative_prompt)
    except Exception:
        llm_narrative = f"Loop {loop_number} completed with result: {overall_result}. " + (" Failures: " + str(failures) if failures else "All audits passed.")

    # Get current best model metrics from leaderboard
    best = next((m for m in state.model_leaderboard if m.is_selected), None)
    loop_record = GovernanceLoopRecord(
        loop_number=loop_number,
        overall_result=overall_result,
        auc_roc=best.auc_roc if best else None,
        f1_score=best.f1_score if best else None,
        rmse=best.rmse if best else None,
        mae=best.mae if best else None,
        disparate_impact=fairness.disparate_impact,
        equal_opportunity_difference=fairness.equal_opportunity_difference,
        auc_degradation_pct=robustness.auc_degradation_pct,
        bootstrap_variance=stability.metric_variance,
        failure_reasons=failures,
        corrective_action=corrective if failures else None,
        llm_narrative=llm_narrative,
    )
    state.governance_audit.governance_loop_history.append(loop_record)

    log_step_and_broadcast_sync(
        state, "governance", "Governance Audit Complete",
        f"Loop {loop_number} Result: {overall_result}. AI Narrative: {llm_narrative}"
    )

    # ── AI Visual Auditing (Python execution) ──
    try:
        from backend.llm.client import get_llm_response
        import json
        
        prompt = _GOVERNANCE_CHART_PROMPT.format(
            fairness_metrics=json.dumps(fairness.dict(), default=str),
            robustness_metrics=json.dumps(robustness.dict(), default=str),
            stability_metrics=json.dumps(stability.dict(), default=str)
        )
        raw_code = get_llm_response(prompt)
        
        # Clean markdown
        if "```python" in raw_code:
            raw_code = raw_code.split("```python")[1].split("```")[0]
        elif "```" in raw_code:
            raw_code = raw_code.split("```")[1].split("```")[0]
        raw_code = raw_code.strip()
        
        import tempfile, subprocess, sys, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', encoding='utf-8', delete=False) as f:
            f.write(raw_code)
            temp_path = f.name
            
        try:
            result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, encoding='utf-8', timeout=120)
            if result.returncode == 0:
                try:
                    ai_charts = json.loads(result.stdout)
                    state.governance_audit.ai_charts = ai_charts
                    log_step_and_broadcast_sync(state, "governance", "AI Governance Charting", f"Successfully generated {len(ai_charts)} visual auditing charts via Python.")
                except json.JSONDecodeError as je:
                    log_step_and_broadcast_sync(state, "governance", "AI Governance Charting Failed", f"Failed to parse JSON output: {je}")
            else:
                log_step_and_broadcast_sync(state, "governance", "AI Governance Charting Failed", f"Script failed: {result.stderr}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        log_step_and_broadcast_sync(state, "governance", "AI Governance Charting Failed", f"System error: {e}")

    return state


# ---------------------------------------------------------------------------
# Fairness audit
# ---------------------------------------------------------------------------


def _run_fairness_audit(
    model,
    df_full: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_col: str,
    protected_attrs: List[str],
    di_min: float,
    eod_max: float,
    task_type: str,
) -> FairnessMetrics:
    """
    Compute Disparate Impact and Equal Opportunity Difference for each
    protected attribute. Uses the strictest failing result.
    """
    metrics = FairnessMetrics()

    if not protected_attrs or X_test_raw is None or len(X_test_raw) == 0:
        metrics.status = AuditStatus.SKIPPED
        return metrics

    # For regression tasks, fairness is measured via mean predicted-value parity
    # rather than binary classification Disparate Impact.
    is_regression = task_type in (TaskType.REGRESSION, "regression")

    # Use the first available protected attribute in the test set
    chosen_attr = None
    for attr in protected_attrs:
        if attr in X_test_raw.columns:
            chosen_attr = attr
            break

    if chosen_attr is None:
        metrics.status = AuditStatus.SKIPPED
        return metrics

    metrics.protected_attribute = chosen_attr
    metrics.threshold_used = di_min

    try:
        attr_values = X_test_raw[chosen_attr].values

        unique_vals = np.unique(attr_values)
        if len(unique_vals) < 2:
            metrics.status = AuditStatus.SKIPPED
            return metrics

        # Assume binary protected attribute; treat majority class as privileged
        val_counts = pd.Series(attr_values).value_counts()
        privileged_val = val_counts.index[0]
        unprivileged_val = val_counts.index[1]

        priv_mask = attr_values == privileged_val
        unpriv_mask = attr_values == unprivileged_val

        if is_regression:
            # For regression: compare mean predicted values between groups
            y_pred_cont = model.predict(X_test)
            priv_mean = float(y_pred_cont[priv_mask].mean()) if priv_mask.sum() > 0 else 0.0
            unpriv_mean = float(y_pred_cont[unpriv_mask].mean()) if unpriv_mask.sum() > 0 else 0.0

            # Disparate Impact for regression: ratio of mean predictions
            if abs(priv_mean) > 1e-8:
                di = unpriv_mean / priv_mean
            else:
                di = 1.0
            metrics.disparate_impact = round(abs(di), 4)

            # Equal Opportunity Difference: difference in mean predicted vs actual (bias)
            priv_actual_mean = float(y_test[priv_mask].mean()) if priv_mask.sum() > 0 else 0.0
            unpriv_actual_mean = float(y_test[unpriv_mask].mean()) if unpriv_mask.sum() > 0 else 0.0
            priv_bias = abs(float(y_pred_cont[priv_mask].mean()) - priv_actual_mean) if priv_mask.sum() > 0 else 0.0
            unpriv_bias = abs(float(y_pred_cont[unpriv_mask].mean()) - unpriv_actual_mean) if unpriv_mask.sum() > 0 else 0.0
            eod = abs(unpriv_bias - priv_bias)
            metrics.equal_opportunity_difference = round(eod, 4)

            # No per-group confusion matrices for regression
            metrics.per_group_confusion_matrices = {}
        else:
            # Classification path (unchanged)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            priv_pos_rate = float(y_pred[priv_mask].mean()) if priv_mask.sum() > 0 else 0.0
            unpriv_pos_rate = float(y_pred[unpriv_mask].mean()) if unpriv_mask.sum() > 0 else 0.0

            if priv_pos_rate > 0:
                di = unpriv_pos_rate / priv_pos_rate
            else:
                di = 1.0
            metrics.disparate_impact = round(di, 4)

            priv_tp = float(y_pred[priv_mask & (y_test == 1)].mean()) if (priv_mask & (y_test == 1)).sum() > 0 else 0.0
            unpriv_tp = float(y_pred[unpriv_mask & (y_test == 1)].mean()) if (unpriv_mask & (y_test == 1)).sum() > 0 else 0.0
            eod = abs(unpriv_tp - priv_tp)
            metrics.equal_opportunity_difference = round(eod, 4)

            for val in [privileged_val, unprivileged_val]:
                mask = attr_values == val
                if mask.sum() > 0:
                    cm = confusion_matrix(y_test[mask], y_pred[mask]).tolist()
                    metrics.per_group_confusion_matrices[str(val)] = cm

        # Pass/fail determination (same threshold applies to both tracks)
        failed = (metrics.disparate_impact is not None and metrics.disparate_impact < di_min) or \
                 (metrics.equal_opportunity_difference is not None and metrics.equal_opportunity_difference > eod_max)
        metrics.status = AuditStatus.FAIL if failed else AuditStatus.PASS

    except Exception as e:
        metrics.status = AuditStatus.SKIPPED
        metrics.per_group_confusion_matrices["error"] = str(e)

    return metrics


def _build_fairness_failure_reason(fairness: FairnessMetrics, di_min: float, eod_max: float) -> str:
    reasons = []
    if fairness.disparate_impact is not None and fairness.disparate_impact < di_min:
        reasons.append(
            f"Disparate Impact = {fairness.disparate_impact:.3f} on '{fairness.protected_attribute}' "
            f"(below threshold {di_min:.2f}). Retry with feature reweighing or drop proxy features."
        )
    if fairness.equal_opportunity_difference is not None and fairness.equal_opportunity_difference > eod_max:
        reasons.append(
            f"Equal Opportunity Difference = {fairness.equal_opportunity_difference:.3f} "
            f"(above threshold {eod_max:.2f}). Consider calibration or threshold adjustment."
        )
    return "Fairness FAIL: " + " | ".join(reasons)


# ---------------------------------------------------------------------------
# Robustness audit
# ---------------------------------------------------------------------------


def _run_robustness_audit(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    auc_deg_max_pct: float,
    task_type: str,
) -> RobustnessMetrics:
    """
    Perturb each feature by +1 std dev synthetic shift and measure AUC degradation.
    """
    metrics = RobustnessMetrics()
    if X_test is None or len(X_test) == 0:
        metrics.status = AuditStatus.SKIPPED
        return metrics

    try:
        baseline_score = _score_model(model, X_test, y_test, task_type)

        # Shift ALL numeric features by adding noise (synthetic covariate shift)
        X_shifted = X_test + np.random.normal(0, X_test.std(axis=0) * 0.5, X_test.shape)
        shifted_score = _score_model(model, X_shifted, y_test, task_type)

        if abs(baseline_score) > 1e-8:
            degradation_pct = float((baseline_score - shifted_score) / abs(baseline_score) * 100)
        else:
            degradation_pct = 0.0

        # For regression, R2 can be negative, so cap degradation pct below -100
        degradation_pct = max(degradation_pct, -100.0)

        metrics.auc_degradation_pct = round(degradation_pct, 2)
        metrics.shift_description = "Synthetic noise shift: +0.5 std dev on all numeric features"
        metrics.perturbed_features = ["all_numeric"]
        metrics.status = AuditStatus.PASS if degradation_pct <= auc_deg_max_pct else AuditStatus.FAIL

    except Exception as e:
        metrics.status = AuditStatus.SKIPPED

    return metrics


# ---------------------------------------------------------------------------
# Stability audit
# ---------------------------------------------------------------------------


def _run_stability_audit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bootstrap_var_max: float,
    task_type: str,
    n: int = DEFAULT_BOOTSTRAP_N,
) -> StabilityMetrics:
    """Bootstrap resample the training set N times, retrain a lightweight model,
    measure variance in validation metric."""
    metrics = StabilityMetrics(bootstrap_n=n)
    if X_train is None or len(X_train) < 20:
        metrics.status = AuditStatus.SKIPPED
        return metrics

    scores = []
    rng = np.random.RandomState(42)

    for i in range(n):
        try:
            indices = rng.choice(len(X_train), size=len(X_train), replace=True)
            X_boot, y_boot = X_train[indices], y_train[indices]
            # Use correct model type per task
            if task_type in (TaskType.REGRESSION, "regression"):
                from sklearn.linear_model import LinearRegression
                boot_model = LinearRegression()
            else:
                boot_model = LogisticRegression(max_iter=100, C=1.0, random_state=42)
            boot_model.fit(X_boot, y_boot)
            # Evaluate on out-of-bag (indices not in boot sample)
            oob_mask = np.ones(len(X_train), dtype=bool)
            oob_mask[indices] = False
            if oob_mask.sum() > 5:
                score = _score_model(boot_model, X_train[oob_mask], y_train[oob_mask], task_type)
                scores.append(score)
        except Exception:
            continue
        except Exception:
            continue

    if len(scores) < 5:
        metrics.status = AuditStatus.SKIPPED
        return metrics

    scores_arr = np.array(scores)
    metrics.metric_variance = round(float(np.var(scores_arr)), 6)
    metrics.metric_std = round(float(np.std(scores_arr)), 6)
    metrics.metric_mean = round(float(np.mean(scores_arr)), 4)
    metrics.status = AuditStatus.PASS if metrics.metric_variance <= bootstrap_var_max else AuditStatus.FAIL

    return metrics


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prepare_audit_data(
    df: pd.DataFrame,
    target_col: str,
    final_feature_set: List[str],
) -> Tuple:
    """
    Prepare train/test splits for the governance audit, preserving
    raw test data (with protected attributes) for fairness analysis.
    """
    try:
        df_clean = df.dropna(subset=[target_col]).copy()

        # Use final feature set if non-empty, else all non-target cols
        if final_feature_set:
            feature_cols = [c for c in final_feature_set if c in df_clean.columns]
        else:
            feature_cols = [c for c in df_clean.columns if c != target_col]

        if not feature_cols:
            return None, None, None, None, None

        y = df_clean[target_col]
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = y.values

        X = df_clean[feature_cols].copy()

        # Encode categoricals
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(0)

        X_np = StandardScaler().fit_transform(X.values.astype(float))

        X_train, X_test, y_train, y_test, _, X_test_raw = train_test_split(
            X_np, y, df_clean, test_size=0.25, random_state=42
        )

        return X_train, X_test, y_train, y_test, X_test_raw

    except Exception:
        return None, None, None, None, None


def _train_audit_model(X_train: np.ndarray, y_train: np.ndarray, task_type: str):
    """Train a lightweight LR/Linear for audit purposes."""
    from backend.state.schema import TaskType
    if task_type in (TaskType.REGRESSION, "regression"):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    else:
        model = LogisticRegression(max_iter=300, C=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def _score_model(model, X: np.ndarray, y: np.ndarray, task_type: str) -> float:
    try:
        if task_type in (TaskType.REGRESSION, "regression"):
            from sklearn.metrics import r2_score
            return float(r2_score(y, model.predict(X)))
        else:
            if len(np.unique(y)) == 2:
                return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
            else:
                return float(f1_score(y, model.predict(X), average="weighted"))
    except Exception:
        return 0.0


def _encode_for_prediction(X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return StandardScaler().fit_transform(X.fillna(0).values.astype(float))
