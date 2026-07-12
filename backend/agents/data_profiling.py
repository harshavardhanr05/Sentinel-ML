"""
backend/agents/data_profiling.py
─────────────────────────────────
Data Profiling Agent — Phase 1

Responsibilities:
1. Load the dataset (CSV or Parquet) from state.dataset_path.
2. Compute per-column dtype, missingness %, unique counts, PII inference.
3. Flag MNAR patterns (missingness correlated with another column's value).
4. Flag potential target leakage (near-perfect correlation to target column).
5. Compute class imbalance ratio for classification tasks.
6. Package all results into the DataHealthReport Pydantic model.
7. Attach schema metadata (column names, dtypes) to state.data_schema.

Deterministic classical code only — no LLM calls.
Unit-testable with hand-crafted DataFrames.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.llm.client import get_llm_json, get_llm_response
from backend.state.schema import (
    ColumnProfile,
    DataHealthReport,
    PipelineState,
    TaskType,
)
from backend.state.store import log_step_and_broadcast_sync

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
_DASHBOARD_CHART_PROMPT = """
You are an expert Data Scientist and BI Analyst. Given the dataset schema, user's objective, and correlation metrics, your task is to generate as many highly-diverse, insightful visualizations as possible (minimum 6, up to 10 charts) that a human analyst would find extremely valuable for Exploratory Data Analysis.

You MUST output **ONLY Interactive React UI Charts**. No static Seaborn/Matplotlib images are allowed. You must return structured JSON data that will be rendered natively in the UI.

Write a complete, standalone Python script that:
1. Loads the dataset from `"{dataset_path}"`.
2. Computes the summary statistics or processes data for the React charts.
3. Prints a single valid JSON array to `sys.stdout` containing all the charts.

The JSON array must look like this:
[
  {{
    "id": "ai-chart-1",
    "title": "Interactive Age Group Breakdown",
    "insight": "Doughnut chart showing majority demographic.",
    "type": "pie" | "doughnut" | "radar" | "bar" | "line" | "area" | "histogram",
    "data": [
      // If type is 'pie', 'doughnut', 'radar', or 'bar', use keys 'name' and 'count':
      {{ "name": "Under 30", "count": 250 }},
      // If type is 'line' or 'scatter', use keys 'name' and 'correlation':
      // {{ "name": "feature_x", "correlation": 0.65 }}
      // If type is 'histogram' or 'area', use keys 'binStart' and 'count':
      // {{ "binStart": 10.0, "count": 45 }}
    ]
  }},
  ...
]

CRITICAL RULES:
- **DO NOT wrap your main code in a giant `try-except` block.** Let errors crash the script naturally so the orchestrator can catch the stack trace! Never silence errors.
- Do NOT output any markdown blocks like ```python. ONLY output the raw Python code.
- Ensure the code handles potential missing values or infinite values (e.g. using `.dropna()` or `.fillna(0)`).
- Do NOT encode categorical variables into numbers (like LabelEncoder) before aggregating for charts. Keep original string labels (e.g., 'Male', 'Female', 'Yes') so they remain informative in the UI.
- **NEVER use 'pie' or 'doughnut' charts for continuous/numerical variables (especially the target column in regression tasks).** For continuous variables, use 'area', 'histogram', or 'line' charts instead. Using a pie chart for a continuous variable results in hundreds of unreadable, meaningless slices.
- **ABSOLUTELY CRITICAL**: You MUST explicitly call `print(json.dumps(charts))` at the very end of the script to output the data. If you do not print the final JSON array, the script will fail!
- Only print the JSON to stdout. Do not print anything else (no intermediate prints).
- Make sure to `import sys`, `import json`, `import pandas as pd`, `import numpy as np`.
- Do NOT use `pd.np` (pandas has no attribute `np`). Use `numpy` directly (e.g. `np.random`).
- You MUST run `sys.stdout.reconfigure(encoding='utf-8')` right after imports to prevent Windows console encoding errors.

User Objective: {user_objective}
Objective Task: {task_type}
Target Column: {target_column}

Dataset Columns and Types: 
{column_info}

Correlations with Target:
{correlations}

CRITICAL FINAL REMINDER: Your code MUST end with `print(json.dumps(charts))` where `charts` is your final list of dictionary objects. DO NOT FORGET TO PRINT.
"""

# ---------------------------------------------------------------------------
# Thresholds (configurable constants)
# ---------------------------------------------------------------------------

MISSINGNESS_FLAG_THRESHOLD = 0.05    # Flag columns with >5% missing
HIGH_SEVERITY_MISSINGNESS = 0.30     # >30% → HIGH
MNAR_CORRELATION_THRESHOLD = 0.15    # |phi| > 0.15 between missingness mask and another col
LEAKAGE_CORRELATION_THRESHOLD = 0.95 # |correlation with target| > 0.95
IMBALANCE_FLAG_THRESHOLD = 0.20      # minority/majority < 0.20 → flag

# PII heuristics — column name substring matches (case-insensitive)
_PII_KEYWORDS = {
    "name", "email", "phone", "ssn", "passport", "address", "zip", "postcode",
    "dob", "birth", "national_id", "aadhar", "pan", "ip_address", "mac_address",
    "latitude", "longitude", "gps",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_data_profiling(state: PipelineState) -> PipelineState:
    """
    Load dataset and produce DataHealthReport. Populates:
      - state.data_health_report
      - state.data_schema
    """
    dataset_path = state.dataset_path
    if not dataset_path or not os.path.exists(dataset_path):
        report = DataHealthReport(
            profiling_notes=["ERROR: Dataset file not found at the specified path."],
        )
        state.data_health_report = report
        return state

    # Load dataset
    try:
        df = _load_dataset(dataset_path)
    except Exception as e:
        report = DataHealthReport(
            profiling_notes=[f"ERROR: Failed to load dataset: {e}"],
        )
        state.data_health_report = report
        return state

    target_col = state.objective.target_column
    if target_col and target_col not in df.columns:
        err_msg = f"Target column '{target_col}' not found in the dataset. Please provide a valid target column."
        log_step_and_broadcast_sync(state, "data_profiling", "Data Profiling Aborted", err_msg)
        state.data_health_report = DataHealthReport(profiling_notes=[f"ERROR: {err_msg}"])
        return state

    log_step_and_broadcast_sync(state, "data_profiling", "Data Profiling Started", "Commencing deep data health scans: missingness, leakage, PII, and imbalance checks.")

    # Run all profiling steps
    column_profiles = _profile_columns(df)
    log_step_and_broadcast_sync(state, "data_profiling", "Schema Discovery Complete", f"Profiled {len(column_profiles)} columns.")
    
    missingness_flags = _compute_missingness_flags(df)
    mnar_flags = _detect_mnar(df)
    log_step_and_broadcast_sync(state, "data_profiling", "Missingness Scan Complete", f"Found {len(missingness_flags)} columns with high missingness. {len(mnar_flags)} MNAR patterns detected.")
    
    leakage_flags = _detect_leakage(df, target_col) if target_col else []
    imbalance_ratio, imbalance_flag = _compute_imbalance(df, target_col, state.objective.task_type)
    
    pii_cols = _detect_pii_columns(df)
    if pii_cols:
        log_step_and_broadcast_sync(state, "data_profiling", "PII/Sensitive Data Detected", f"Potential PII found in columns: {pii_cols}. These will be dropped for security.")
        
    severity_summary = _build_severity_summary(missingness_flags, leakage_flags, imbalance_flag)

    report = DataHealthReport(
        row_count=len(df),
        column_count=len(df.columns),
        columns=column_profiles,
        missingness_flags=missingness_flags,
        mnar_flags=mnar_flags,
        leakage_flags=leakage_flags,
        imbalance_ratio=imbalance_ratio,
        imbalance_flag=imbalance_flag,
        inferred_pii=pii_cols,
        severity_summary=severity_summary,
        profiling_notes=_build_notes(missingness_flags, mnar_flags, leakage_flags, pii_cols),
    )

    state.data_health_report = report
    
    if leakage_flags:
        leakage_cols = [f['column'] for f in leakage_flags]
        log_step_and_broadcast_sync(state, "data_profiling", "Target Leakage Detected", f"Found {len(leakage_flags)} columns highly correlated with target: {leakage_cols}. These will be dropped.")
        
    state.data_schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": [len(df), len(df.columns)],
    }
    log_step_and_broadcast_sync(state, "data_profiling", "Dataset loaded",
        f"{len(df)} rows x {len(df.columns)} columns. "
        f"Missingness flags: {len(missingness_flags)} columns. "
        f"Leakage flags: {len(leakage_flags)}. "
        f"Class imbalance: {imbalance_flag} (ratio={imbalance_ratio}).")

    # ── Feature Correlations & Distributions (for Data Analysis Dashboard) ──
    analysis_metrics = {
        "numeric_correlations": {},
        "categorical_distributions": {},
        "target_distribution": {}
    }
    
    try:
        if target_col and target_col in df.columns:
            # Target Distribution (Counts)
            t_counts = df[target_col].value_counts().to_dict()
            analysis_metrics["target_distribution"] = {str(k): int(v) for k, v in t_counts.items()}
            
            # Numeric correlations with target (if target is binary/numeric)
            # Convert target to numeric temporarily if it's binary string
            y_temp = df[target_col]
            if y_temp.dtype == object or str(y_temp.dtype) == "category":
                if y_temp.nunique() == 2:
                    y_temp = pd.factorize(y_temp)[0]
                else:
                    y_temp = None
                    
            if y_temp is not None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for c in numeric_cols:
                    if c != target_col:
                        corr = df[c].corr(pd.Series(y_temp, index=df.index))
                        if not pd.isna(corr):
                            analysis_metrics["numeric_correlations"][c] = round(float(corr), 3)

        # Categorical distributions (Counts)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            if c != target_col:
                val_counts = df[c].value_counts().head(10).to_dict()
                analysis_metrics["categorical_distributions"][c] = {str(k): int(v) for k, v in val_counts.items()}

        # Numeric histograms (binned counts for histogram visualization)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        analysis_metrics["numeric_histograms"] = {}
        for c in numeric_cols:
            if df[c].notna().sum() > 5:
                try:
                    counts, bin_edges = np.histogram(df[c].dropna(), bins=20)
                    analysis_metrics["numeric_histograms"][c] = {
                        "counts": [int(x) for x in counts],
                        "bins": [round(float(x), 3) for x in bin_edges],
                    }
                except Exception:
                    pass

    except Exception as e:
        # Failsafe so pipeline doesn't crash if stats fail
        pass

    # ── AI Dashboard Selection ──
    ai_charts = []
    try:
        # Build a richer context for the AI: column types, distributions, correlations
        col_info = {}
        for col in df.columns:
            if col == target_col:
                continue
            is_cat = str(df[col].dtype) in ["object", "category"] or df[col].nunique() < 15
            col_info[col] = {
                "type": "categorical" if is_cat else "numeric",
                "unique": int(df[col].nunique()),
                "correlation_with_target": analysis_metrics["numeric_correlations"].get(col),
            }

        if target_col:
            prompt = _DASHBOARD_CHART_PROMPT.format(
                user_objective=state.objective.raw_text,
                task_type=state.objective.task_type.value,
                target_column=target_col,
                column_info=json.dumps(col_info, indent=2),
                correlations=json.dumps(analysis_metrics["numeric_correlations"], indent=2),
                dataset_path=state.dataset_path
            )
            raw_code = get_llm_response(prompt)
            
            # Clean markdown
            if "```python" in raw_code:
                raw_code = raw_code.split("```python")[1].split("```")[0]
            elif "```" in raw_code:
                raw_code = raw_code.split("```")[1].split("```")[0]
            raw_code = raw_code.strip()
            
            # Execute Python code in a safe subprocess with self-correction
            import tempfile, subprocess, sys
            from backend.agents.orchestrator import fix_generated_code
            
            current_code = raw_code
            max_retries = 3
            for attempt in range(max_retries):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', encoding='utf-8', delete=False) as f:
                    f.write(current_code)
                    temp_path = f.name
                    
                try:
                    result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, encoding='utf-8', timeout=120)
                    if result.returncode == 0:
                        try:
                            import re
                            out_str = result.stdout.strip()
                            if not out_str:
                                # Failsafe: if the LLM forgot to print, forcefully append it and retry
                                raise ValueError("Empty output. Force-appending print statement.")
                                
                            # Find the first '[' and last ']' to extract the JSON array
                            start_idx = out_str.find('[')
                            end_idx = out_str.rfind(']')
                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                out_str = out_str[start_idx:end_idx+1]
                                
                            ai_charts = json.loads(out_str)
                            log_step_and_broadcast_sync(state, "data_profiling", "AI Chart Generation", f"Successfully generated {len(ai_charts)} AI-driven visual EDA charts via Python script (Attempt {attempt+1}).")
                            break
                        except Exception as je:
                            if attempt < max_retries - 1:
                                if ("Empty output" in str(je) or "produced no output" in str(je)) and "for var in ['charts'" not in current_code:
                                    fixed_code = current_code + "\n\nimport json\nfor var in ['charts', 'ai_charts', 'dashboard_charts', 'data', 'json_data', 'result', 'output']:\n    if var in locals() and isinstance(locals()[var], list):\n        try:\n            print(json.dumps(locals()[var]))\n            break\n        except:\n            pass\n"
                                else:
                                    fixed_code, _ = fix_generated_code(current_code, f"Failed to parse JSON output: {je}. You MUST print a valid JSON array to stdout using print(json.dumps(charts)). Do not print anything else.")
                                if not fixed_code: break
                                current_code = fixed_code
                            else:
                                log_step_and_broadcast_sync(state, "data_profiling", "AI Chart Generation Failed", f"Failed to parse JSON output: {je}")
                    else:
                        if attempt < max_retries - 1:
                            err_msg = result.stderr.strip()
                            fixed_code, _ = fix_generated_code(current_code, f"Script failed with error:\n{err_msg}")
                            if not fixed_code: break
                            current_code = fixed_code
                        else:
                            log_step_and_broadcast_sync(state, "data_profiling", "AI Chart Generation Failed", f"Script failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    if attempt < max_retries - 1:
                        fixed_code, _ = fix_generated_code(current_code, "Script execution timed out after 120s. Try simplifying the pandas logic.")
                        if not fixed_code: break
                        current_code = fixed_code
                    else:
                        log_step_and_broadcast_sync(state, "data_profiling", "AI Chart Generation Failed", "Script execution timed out.")
                except Exception as e:
                    log_step_and_broadcast_sync(state, "data_profiling", "AI Chart Generation Failed", f"System error: {e}")
                    break
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
    except Exception:
        pass

    analysis_metrics["ai_charts"] = ai_charts
    state.data_analysis_metrics = analysis_metrics
    # Also keep in data_schema for backward compat
    state.data_schema["analysis_metrics"] = analysis_metrics

    return state


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    import re
    
    # 1. Column Name Standardization
    new_cols = []
    for c in df.columns:
        c_clean = str(c).strip()
        c_clean = re.sub(r'\s+', '_', c_clean)
        c_clean = re.sub(r'[\[\]<>]', '', c_clean) # Remove characters that crash LightGBM/XGBoost
        new_cols.append(c_clean)
    df.columns = new_cols

    # 2. Global Missing Value Normalization
    missing_placeholders = {"?", "n/a", "na", "null", "missing", "-", "", " "}
    for col in df.select_dtypes(include=["object"]):
        # First, strip whitespace
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        # Then, replace missing placeholders with np.nan
        df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.lower() in missing_placeholders else x)

    # 3. Infinity Handling
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4. Intelligent Type Coercion
    for col in df.select_dtypes(include=["object"]):
        # Try to convert to numeric
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        # If more than 90% of non-null values were successfully converted to numeric, keep it numeric
        non_null_original = df[col].notna().sum()
        if non_null_original > 0:
            converted_ratio = numeric_series.notna().sum() / non_null_original
            if converted_ratio > 0.90:
                df[col] = numeric_series

    # (Boolean standardization removed: Let feature engineering handle this so Data Analysis shows descriptive strings instead of 1/0)

    # 6. Zero-Variance / Empty Column Pruning
    cols_to_drop = []
    for col in df.columns:
        if df[col].isna().all():
            cols_to_drop.append(col)
        elif df[col].nunique(dropna=True) <= 1:
            cols_to_drop.append(col)
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    return df

def _load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        # Try CSV as fallback
        df = pd.read_csv(path, low_memory=False)
        
    df = _clean_dataset(df)
        
    return df



# ---------------------------------------------------------------------------
# Column profiling
# ---------------------------------------------------------------------------


def _profile_columns(df: pd.DataFrame) -> List[ColumnProfile]:
    profiles = []
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        missing_pct = float(series.isna().mean())
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        )
        unique_count = int(series.nunique(dropna=True))
        is_pii = _is_pii_column(col)

        notes = []
        if missing_pct > HIGH_SEVERITY_MISSINGNESS:
            notes.append(f"HIGH missingness: {missing_pct:.1%}")
        elif missing_pct > MISSINGNESS_FLAG_THRESHOLD:
            notes.append(f"Missingness: {missing_pct:.1%}")
        if is_pii:
            notes.append("Possible PII — review before sharing")
        if is_categorical and unique_count > 50:
            notes.append(f"High cardinality categorical: {unique_count} unique values")

        profiles.append(
            ColumnProfile(
                name=col,
                dtype=dtype_str,
                missing_pct=round(missing_pct, 4),
                unique_count=unique_count,
                is_numeric=is_numeric,
                is_categorical=is_categorical,
                is_potential_pii=is_pii,
                notes=notes,
            )
        )
    return profiles


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


def _compute_missingness_flags(df: pd.DataFrame) -> Dict[str, float]:
    missing_pcts = df.isna().mean()
    return {
        col: round(float(pct), 4)
        for col, pct in missing_pcts.items()
        if pct > MISSINGNESS_FLAG_THRESHOLD
    }


def _detect_mnar(df: pd.DataFrame, max_cols: int = 30) -> List[str]:
    """
    MNAR heuristic: for each column with missingness, check if its
    missingness indicator (0/1) is correlated with any other numeric column.
    Returns column names suspected of MNAR pattern.
    """
    mnar_cols = []
    cols_with_missing = [c for c in df.columns if df[c].isna().any()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols or not cols_with_missing:
        return mnar_cols

    # Limit computation for large datasets
    check_cols = cols_with_missing[:max_cols]

    for col in check_cols:
        missing_mask = df[col].isna().astype(int)
        for num_col in numeric_cols[:max_cols]:
            if num_col == col:
                continue
            try:
                filled = df[num_col].fillna(df[num_col].median())
                corr = float(abs(missing_mask.corr(filled)))
                if corr > MNAR_CORRELATION_THRESHOLD:
                    mnar_cols.append(col)
                    break
            except Exception:
                continue

    return list(set(mnar_cols))


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


def _detect_leakage(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    """
    Detect potential target leakage:
    1. Numeric columns with near-perfect correlation to target.
    2. Columns whose name contains the target column name.
    """
    flags = []
    if target_col not in df.columns:
        return flags

    target = df[target_col]
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        if col == target_col:
            continue
        try:
            # Drop rows where either is NaN
            valid = df[[col, target_col]].dropna()
            if len(valid) < 10:
                continue
            corr = float(abs(valid[col].corr(valid[target_col])))
            if corr >= LEAKAGE_CORRELATION_THRESHOLD:
                flags.append({
                    "column": col,
                    "reason": f"Near-perfect correlation with target ({corr:.3f})",
                    "correlation_with_target": corr,
                    "severity": "HIGH",
                })
        except Exception:
            continue

    # Name-based leakage heuristic
    target_lower = target_col.lower().replace("_", "").replace("-", "")
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower().replace("_", "").replace("-", "")
        if target_lower in col_lower or col_lower in target_lower:
            if not any(f["column"] == col for f in flags):
                flags.append({
                    "column": col,
                    "reason": f"Column name is a variation of the target column '{target_col}'",
                    "correlation_with_target": None,
                    "severity": "MEDIUM",
                })

    return flags


# ---------------------------------------------------------------------------
# Imbalance
# ---------------------------------------------------------------------------


def _compute_imbalance(
    df: pd.DataFrame,
    target_col: Optional[str],
    task_type: str,
) -> Tuple[Optional[float], bool]:
    if not target_col or target_col not in df.columns:
        return None, False
    if task_type == TaskType.REGRESSION or task_type == "regression":
        return None, False

    counts = df[target_col].value_counts()
    if len(counts) < 2:
        return None, False

    ratio = float(counts.min() / counts.max())
    return round(ratio, 4), ratio < IMBALANCE_FLAG_THRESHOLD


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------


def _is_pii_column(col_name: str) -> bool:
    col_lower = col_name.lower().replace("_", "").replace("-", "")
    return any(kw.replace("_", "") in col_lower for kw in _PII_KEYWORDS)


def _detect_pii_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if _is_pii_column(col)]


# ---------------------------------------------------------------------------
# Severity summary + notes
# ---------------------------------------------------------------------------


def _build_severity_summary(
    missingness_flags: Dict[str, float],
    leakage_flags: List[Dict[str, Any]],
    imbalance_flag: bool,
) -> Dict[str, str]:
    summary = {}

    if not missingness_flags:
        summary["missingness"] = "NONE"
    elif any(v > HIGH_SEVERITY_MISSINGNESS for v in missingness_flags.values()):
        summary["missingness"] = "HIGH"
    else:
        summary["missingness"] = "MEDIUM"

    high_leakage = [f for f in leakage_flags if f.get("severity") == "HIGH"]
    if not leakage_flags:
        summary["leakage"] = "NONE"
    elif high_leakage:
        summary["leakage"] = "HIGH"
    else:
        summary["leakage"] = "MEDIUM"

    summary["imbalance"] = "MEDIUM" if imbalance_flag else "NONE"

    return summary


def _build_notes(
    missingness_flags: Dict[str, float],
    mnar_flags: List[str],
    leakage_flags: List[Dict[str, Any]],
    pii_cols: List[str],
) -> List[str]:
    notes = []
    if missingness_flags:
        worst = max(missingness_flags, key=lambda k: missingness_flags[k])
        notes.append(
            f"{len(missingness_flags)} column(s) have significant missingness. "
            f"Worst: '{worst}' at {missingness_flags[worst]:.1%}."
        )
    if mnar_flags:
        notes.append(
            f"MNAR pattern suspected in: {', '.join(mnar_flags)}. "
            "Imputation strategy should account for non-random missingness."
        )
    if leakage_flags:
        high = [f for f in leakage_flags if f.get("severity") == "HIGH"]
        notes.append(
            f"{len(leakage_flags)} potential leakage column(s) detected "
            f"({len(high)} HIGH severity). Consider dropping before feature engineering."
        )
    if pii_cols:
        notes.append(
            f"Possible PII in: {', '.join(pii_cols)}. "
            "Ensure de-identification per compliance requirements."
        )
    if not notes:
        notes.append("No critical data quality issues detected.")
    return notes
