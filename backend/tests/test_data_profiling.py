"""
backend/tests/test_data_profiling.py
──────────────────────────────────────
Unit tests for the Data Profiling Agent.
Uses hand-crafted DataFrames with known, planted issues.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from backend.agents.data_profiling import (
    _compute_missingness_flags,
    _compute_imbalance,
    _detect_leakage,
    _detect_mnar,
    _detect_pii_columns,
    _profile_columns,
    run_data_profiling,
)
from backend.state.schema import PipelineState, ObjectiveState, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_state_with_df(df: pd.DataFrame, target_col: str = "target", task_type: str = "classification") -> tuple:
    """Save a DataFrame to a temp CSV and return (state, filepath)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()

    state = PipelineState()
    state.dataset_path = tmp.name
    state.objective = ObjectiveState(
        raw_text="test objective",
        task_type=TaskType(task_type),
        target_column=target_col,
    )
    state.data_schema = {"columns": list(df.columns)}

    return state, tmp.name


# ---------------------------------------------------------------------------
# Missingness tests
# ---------------------------------------------------------------------------


class TestMissingness:
    def test_flags_columns_above_threshold(self):
        df = pd.DataFrame({
            "a": [1, None, None, None, None, None, None],  # ~86% missing
            "b": [1, 2, 3, 4, 5, 6, 7],                   # 0% missing
        })
        flags = _compute_missingness_flags(df)
        assert "a" in flags
        assert flags["a"] > 0.5
        assert "b" not in flags

    def test_no_flags_when_no_missingness(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        flags = _compute_missingness_flags(df)
        assert flags == {}

    def test_exact_threshold_boundary(self):
        # Exactly 5% missing should NOT be flagged (threshold is strictly >5%)
        df = pd.DataFrame({"a": [None] + [1] * 19})  # 5% missing
        flags = _compute_missingness_flags(df)
        assert "a" not in flags


# ---------------------------------------------------------------------------
# Leakage detection tests
# ---------------------------------------------------------------------------


class TestLeakageDetection:
    def test_detects_high_correlation_leakage(self):
        n = 100
        target = np.random.randint(0, 2, n)
        # Create a column that is almost perfectly correlated with target
        leak = target.copy().astype(float)
        leak[0] = 1 - leak[0]  # Flip one value → correlation ~0.98

        df = pd.DataFrame({"target": target, "leak_col": leak, "normal": np.random.randn(n)})
        flags = _detect_leakage(df, "target")
        flagged_cols = [f["column"] for f in flags]
        assert "leak_col" in flagged_cols

    def test_does_not_flag_low_correlation(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "target": np.random.randint(0, 2, 100),
            "feature": np.random.randn(100),  # Random → low correlation
        })
        flags = _detect_leakage(df, "target")
        flagged_cols = [f["column"] for f in flags]
        assert "feature" not in flagged_cols

    def test_detects_name_based_leakage(self):
        df = pd.DataFrame({
            "default_flag": [0, 1, 0, 1],
            "default_flag_encoded": [0, 1, 0, 1],  # Name variation → leakage
        })
        flags = _detect_leakage(df, "default_flag")
        flagged_cols = [f["column"] for f in flags]
        assert "default_flag_encoded" in flagged_cols


# ---------------------------------------------------------------------------
# Class imbalance tests
# ---------------------------------------------------------------------------


class TestImbalance:
    def test_flags_severe_imbalance(self):
        # 5% minority class
        target = [1] * 5 + [0] * 95
        df = pd.DataFrame({"target": target})
        ratio, flag = _compute_imbalance(df, "target", "classification")
        assert flag is True
        assert ratio < 0.20

    def test_no_flag_for_balanced(self):
        target = [0] * 50 + [1] * 50
        df = pd.DataFrame({"target": target})
        ratio, flag = _compute_imbalance(df, "target", "classification")
        assert flag is False
        assert abs(ratio - 1.0) < 0.01

    def test_regression_returns_none(self):
        df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
        ratio, flag = _compute_imbalance(df, "target", "regression")
        assert ratio is None
        assert flag is False


# ---------------------------------------------------------------------------
# PII detection tests
# ---------------------------------------------------------------------------


class TestPIIDetection:
    def test_detects_pii_columns(self):
        df = pd.DataFrame({
            "name": ["Alice"],
            "email": ["a@b.com"],
            "age": [30],
            "income": [50000],
        })
        pii = _detect_pii_columns(df)
        assert "name" in pii
        assert "email" in pii
        assert "age" not in pii
        assert "income" not in pii


# ---------------------------------------------------------------------------
# End-to-end profiling tests
# ---------------------------------------------------------------------------


class TestRunDataProfiling:
    def test_end_to_end_basic(self, tmp_path):
        df = pd.DataFrame({
            "age": [25, 30, None, 40, 50] * 20,
            "income": [50000, 60000, 70000, None, 80000] * 20,
            "gender": ["M", "F", "M", "F", "M"] * 20,
            "default_flag": [0, 1, 0, 0, 1] * 20,
        })
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        state = PipelineState()
        state.dataset_path = csv_path
        state.objective = ObjectiveState(
            raw_text="Predict default",
            task_type=TaskType.CLASSIFICATION,
            target_column="default_flag",
            protected_attributes=["gender"],
        )
        state.data_schema = {"columns": list(df.columns)}

        result = run_data_profiling(state)

        assert result.data_health_report is not None
        assert result.data_health_report.row_count == 100
        assert result.data_health_report.column_count == 4

    def test_leakage_flagged_in_report(self, tmp_path):
        n = 200
        target = np.random.randint(0, 2, n)
        leak = target.copy().astype(float)

        df = pd.DataFrame({"feature": np.random.randn(n), "target": target, "target_encoded": leak})
        csv_path = str(tmp_path / "leak.csv")
        df.to_csv(csv_path, index=False)

        state = PipelineState()
        state.dataset_path = csv_path
        state.objective = ObjectiveState(
            raw_text="Test", task_type=TaskType.CLASSIFICATION, target_column="target"
        )
        state.data_schema = {"columns": list(df.columns)}

        result = run_data_profiling(state)
        flagged = [f["column"] for f in result.data_health_report.leakage_flags]
        assert "target_encoded" in flagged

    def test_missing_file_returns_error_report(self):
        state = PipelineState()
        state.dataset_path = "/nonexistent/path.csv"
        state.objective = ObjectiveState(raw_text="Test", task_type=TaskType.CLASSIFICATION)
        state.data_schema = {}

        result = run_data_profiling(state)
        assert result.data_health_report is not None
        assert any("not found" in n or "ERROR" in n for n in result.data_health_report.profiling_notes)
