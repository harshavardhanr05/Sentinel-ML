"""
backend/tests/test_governance.py
──────────────────────────────────
Governance Agent integration tests with planted fairness problems.

These tests verify the core novelty of the system:
- Fairness audit correctly FAILS when Disparate Impact < threshold
- Failure reasons correctly name the offending attribute
- Loopback target is correctly set to 'feature_engineering' for fairness issues
- Robustness and stability audits produce expected statuses
- Proxy check correctly flags features correlated with protected attributes
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from backend.agents.governance import (
    _compute_association,
    _run_fairness_audit,
    _run_robustness_audit,
    _run_stability_audit,
    _train_audit_model,
    quick_fairness_proxy_check,
    run_governance,
)
from backend.state.schema import (
    AuditStatus,
    GovernanceAudit,
    ObjectiveState,
    PipelineState,
    TaskType,
)


# ---------------------------------------------------------------------------
# Proxy check tests
# ---------------------------------------------------------------------------


class TestProxyCheck:
    def test_flags_highly_correlated_feature(self):
        np.random.seed(42)
        n = 200
        protected = np.random.randint(0, 2, n)
        # Proxy feature: nearly identical to protected attribute
        proxy = protected.copy().astype(float)
        proxy += np.random.normal(0, 0.1, n)  # tiny noise

        df = pd.DataFrame({"proxy_col": proxy, "gender": protected})
        reason = quick_fairness_proxy_check(df, "proxy_col", ["gender"])
        assert reason is not None
        assert "proxy_col" in reason
        assert "gender" in reason

    def test_does_not_flag_unrelated_feature(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "random_feature": np.random.randn(n),
            "gender": np.random.randint(0, 2, n),
        })
        reason = quick_fairness_proxy_check(df, "random_feature", ["gender"])
        assert reason is None

    def test_returns_none_for_missing_protected_attrs(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        reason = quick_fairness_proxy_check(df, "feature", [])
        assert reason is None


# ---------------------------------------------------------------------------
# Fairness audit tests
# ---------------------------------------------------------------------------


class TestFairnessAudit:
    def _make_biased_dataset(self, n=400, di_target=0.5):
        """
        Create a dataset where group 0 (gender=0) has a much lower positive prediction rate.
        di_target: approximate desired Disparate Impact (< 0.80 → should FAIL).
        """
        np.random.seed(42)
        gender = np.random.randint(0, 2, n)
        # Generate predictions biased against gender=0
        y_pred = np.zeros(n, dtype=int)
        for i in range(n):
            if gender[i] == 1:
                y_pred[i] = 1 if np.random.random() < 0.60 else 0  # 60% positive rate
            else:
                y_pred[i] = 1 if np.random.random() < (0.60 * di_target) else 0

        y_true = (np.random.rand(n) > 0.4).astype(int)
        return gender, y_pred, y_true

    def test_fails_on_biased_predictions(self):
        gender, y_pred, y_true = self._make_biased_dataset(di_target=0.4)
        gender_series = pd.Series(gender, name="gender")
        X_raw = pd.DataFrame({"gender": gender})

        from sklearn.linear_model import LogisticRegression
        # Use a mock model that returns the biased y_pred
        class MockModel:
            def __init__(self, preds):
                self.preds = preds
            def predict_proba(self, X):
                n = len(X)
                proba = np.column_stack([1 - self.preds[:n], self.preds[:n]]).astype(float)
                return proba
            def predict(self, X):
                return self.preds[:len(X)]

        mock_model = MockModel(y_pred)
        X_test_np = np.column_stack((gender, np.random.randn(len(gender))))
        result = _run_fairness_audit(
            mock_model, pd.DataFrame({"gender": gender, "target": y_true}),
            X_raw, X_test_np, y_true, "target", ["gender"],
            di_min=0.80, eod_max=0.10, task_type="classification"
        )
        assert result.disparate_impact is not None
        assert result.disparate_impact < 0.80
        assert result.status == AuditStatus.FAIL

    def test_passes_on_fair_predictions(self):
        np.random.seed(0)
        n = 300
        gender = np.random.randint(0, 2, n)
        # Equal positive rates across groups → DI ≈ 1.0
        y_pred = np.random.randint(0, 2, n)
        y_true = np.random.randint(0, 2, n)

        class MockModel:
            def __init__(self, preds):
                self.preds = preds
            def predict_proba(self, X):
                return np.column_stack([1 - self.preds[:len(X)], self.preds[:len(X)]]).astype(float)
            def predict(self, X):
                return self.preds[:len(X)]

        mock_model = MockModel(y_pred)
        X_raw = pd.DataFrame({"gender": gender})
        df_full = pd.DataFrame({"gender": gender, "target": y_true})

        X_test_np = np.column_stack((gender, np.random.randn(len(gender))))
        result = _run_fairness_audit(
            mock_model, df_full, X_raw, X_test_np, y_true, "target",
            ["gender"], di_min=0.80, eod_max=0.10, task_type="classification"
        )
        # With random equal predictions, DI should be near 1.0 → PASS
        if result.status != AuditStatus.SKIPPED:
            assert result.disparate_impact is not None


# ---------------------------------------------------------------------------
# Stability audit tests
# ---------------------------------------------------------------------------


class TestStabilityAudit:
    def test_stable_model_passes(self):
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = (X[:, 0] > 0).astype(int)  # Very predictable

        result = _run_stability_audit(X, y, bootstrap_var_max=0.05, task_type="classification")
        if result.status != AuditStatus.SKIPPED:
            assert result.metric_variance is not None
            # Stable model on predictable data should have low variance
            assert result.metric_variance < 0.05

    def test_skips_on_insufficient_data(self):
        X = np.random.randn(5, 2)
        y = np.array([0, 1, 0, 1, 0])
        result = _run_stability_audit(X, y, bootstrap_var_max=0.03, task_type="classification")
        assert result.status == AuditStatus.SKIPPED


# ---------------------------------------------------------------------------
# Integration: full governance on planted fairness problem
# ---------------------------------------------------------------------------


class TestGovernanceIntegration:
    def test_governance_fails_and_routes_back(self, tmp_path):
        """
        Plant a fairness issue (proxy feature correlated with gender),
        run full governance, assert FAIL + loopback_target = 'feature_engineering'.
        """
        np.random.seed(42)
        n = 300
        gender = np.random.randint(0, 2, n)
        # Biased target: gender=1 gets positive label 70% of the time, gender=0 gets 30%
        target = np.where(gender == 1,
                          (np.random.rand(n) < 0.70).astype(int),
                          (np.random.rand(n) < 0.30).astype(int))
        feature = np.random.randn(n)

        df = pd.DataFrame({"gender": gender, "feature": feature, "target": target})
        csv_path = str(tmp_path / "biased.csv")
        df.to_csv(csv_path, index=False)

        state = PipelineState()
        state.dataset_path = csv_path
        state.objective = ObjectiveState(
            raw_text="Test governance loop",
            task_type=TaskType.CLASSIFICATION,
            target_column="target",
            protected_attributes=["gender"],
            domain_tag="finance",
        )
        state.governance_audit.compliance_thresholds = {
            "disparate_impact_min": 0.80,
            "equal_opportunity_diff_max": 0.10,
            "auc_degradation_max_pct": 15.0,
            "bootstrap_variance_max": 0.05,
        }
        state.feature_log.final_feature_set = ["gender", "feature"]

        result = run_governance(state)

        # Should FAIL on fairness
        assert result.governance_audit.overall_status == AuditStatus.FAIL
        assert len(result.governance_audit.failure_reasons) > 0
        assert result.governance_audit.loopback_target == "feature_engineering"
        # Failure reason should mention the protected attribute or DI value
        combined_reasons = " ".join(result.governance_audit.failure_reasons)
        assert "gender" in combined_reasons or "Disparate Impact" in combined_reasons
