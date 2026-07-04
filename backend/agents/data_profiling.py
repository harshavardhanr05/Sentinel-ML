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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.state.schema import (
    ColumnProfile,
    DataHealthReport,
    PipelineState,
    TaskType,
)

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

    # Run all profiling steps
    column_profiles = _profile_columns(df)
    missingness_flags = _compute_missingness_flags(df)
    mnar_flags = _detect_mnar(df)
    leakage_flags = _detect_leakage(df, target_col) if target_col else []
    imbalance_ratio, imbalance_flag = _compute_imbalance(df, target_col, state.objective.task_type)
    pii_cols = _detect_pii_columns(df)
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
    state.data_schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": [len(df), len(df.columns)],
    }

    return state


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    else:
        # Try CSV as fallback
        return pd.read_csv(path, low_memory=False)


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
