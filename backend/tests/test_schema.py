"""
backend/tests/test_schema.py
──────────────────────────────
Tests for Pydantic state schema serialization and key model behaviors.
This is the first test to pass (Phase 0 acceptance criterion).
"""

import json
import uuid
from datetime import datetime

import pytest

from backend.state.schema import (
    AuditStatus,
    CalibrationPoint,
    DataHealthReport,
    DecisionCard,
    DecisionLogEntry,
    DeployRecommendation,
    ExplainabilityOutput,
    FeatureLog,
    FeatureLogEntry,
    FairnessMetrics,
    GovernanceAudit,
    ModelLeaderboardEntry,
    ObjectiveState,
    PipelineState,
    RobustnessMetrics,
    StabilityMetrics,
    StageStatus,
    TaskType,
    UserAction,
)


class TestPipelineStateBasics:
    def test_default_instantiation(self):
        """PipelineState should instantiate with sensible defaults."""
        state = PipelineState()
        assert state.run_id is not None
        assert len(state.run_id) == 36  # UUID format
        assert state.current_stage == "objective_intake"
        assert state.is_paused is False
        assert state.decisions_log == []
        assert state.final_recommendation == DeployRecommendation.PENDING

    def test_json_round_trip(self):
        """State must serialize to JSON and deserialize back losslessly."""
        state = PipelineState(
            run_id=str(uuid.uuid4()),
            objective=ObjectiveState(
                raw_text="Predict loan default for finance domain",
                task_type=TaskType.CLASSIFICATION,
                target_column="default_flag",
                protected_attributes=["gender", "age_bucket"],
                domain_tag="finance",
            ),
        )
        json_str = state.model_dump_json()
        loaded = PipelineState.model_validate_json(json_str)
        assert loaded.run_id == state.run_id
        assert loaded.objective.target_column == "default_flag"
        assert loaded.objective.protected_attributes == ["gender", "age_bucket"]

    def test_dict_round_trip(self):
        state = PipelineState()
        state_dict = state.model_dump(mode="json")
        loaded = PipelineState.model_validate(state_dict)
        assert loaded.run_id == state.run_id

    def test_full_state_round_trip(self):
        """All nested models must survive serialization."""
        state = PipelineState()
        # Populate every sub-model
        state.data_health_report = DataHealthReport(
            row_count=1000,
            column_count=15,
            missingness_flags={"income": 0.12},
            leakage_flags=[{"column": "score_leak", "severity": "HIGH"}],
            imbalance_ratio=0.15,
            imbalance_flag=True,
        )
        state.feature_log = FeatureLog(
            accepted=[
                FeatureLogEntry(feature="age", status="accepted", reason="Useful predictor"),
            ],
            rejected=[
                FeatureLogEntry(feature="zip_code", status="rejected", reason="Proxy", governance_flagged=True),
            ],
            final_feature_set=["age", "income"],
        )
        state.model_leaderboard = [
            ModelLeaderboardEntry(
                model_name="XGBoost (Optuna)",
                model_family="gradient_boosting",
                auc_roc=0.82,
                f1_score=0.74,
                is_selected=True,
                calibration_curve=[
                    CalibrationPoint(bin_mean_predicted=0.3, fraction_of_positives=0.28)
                ],
            )
        ]
        state.governance_audit = GovernanceAudit(
            fairness=FairnessMetrics(
                disparate_impact=0.65,
                equal_opportunity_difference=0.08,
                status=AuditStatus.FAIL,
                protected_attribute="gender",
            ),
            robustness=RobustnessMetrics(auc_degradation_pct=4.2, status=AuditStatus.PASS),
            stability=StabilityMetrics(metric_variance=0.015, status=AuditStatus.PASS),
            overall_status=AuditStatus.FAIL,
            failure_reasons=["Disparate Impact = 0.65 below threshold 0.80"],
            compliance_checklist=["ECOA_disparate_impact_0.80_rule"],
        )
        state.explainability = ExplainabilityOutput(
            global_shap_values={"income": 0.32, "age": 0.21},
            top_features_summary=["income", "age"],
        )
        state.decisions_log = [
            DecisionLogEntry(
                stage="data_profiling",
                proposed_action="Proceed to Feature Engineering",
                reasoning="No critical data issues",
                user_action=UserAction.APPROVE,
            )
        ]

        # Serialize → deserialize
        json_str = state.model_dump_json()
        loaded = PipelineState.model_validate_json(json_str)

        assert loaded.data_health_report.row_count == 1000
        assert loaded.feature_log.rejected[0].governance_flagged is True
        assert loaded.model_leaderboard[0].auc_roc == 0.82
        assert loaded.governance_audit.fairness.disparate_impact == 0.65
        assert loaded.decisions_log[0].user_action in (UserAction.APPROVE, "approve")


class TestPipelineStateMethods:
    def test_mark_stage(self):
        state = PipelineState()
        state.mark_stage("data_profiling", StageStatus.COMPLETE)
        assert state.stage_statuses.data_profiling == StageStatus.COMPLETE
        assert state.current_stage == "data_profiling"

    def test_append_decision_is_append_only(self):
        state = PipelineState()
        entry = DecisionLogEntry(
            stage="test", proposed_action="test action", reasoning="test reason"
        )
        state.append_decision(entry)
        assert len(state.decisions_log) == 1
        # Appending again should not overwrite
        state.append_decision(entry)
        assert len(state.decisions_log) == 2

    def test_set_and_clear_checkpoint(self):
        state = PipelineState()
        card = DecisionCard(
            stage="data_profiling",
            proposed_action="Proceed to feature engineering",
            reasoning="No issues found",
        )
        state.set_checkpoint(card)
        assert state.is_paused is True
        assert state.pending_approval is not None

        state.clear_checkpoint()
        assert state.is_paused is False
        assert state.pending_approval is None


class TestEnums:
    def test_task_type_values(self):
        assert TaskType.CLASSIFICATION == "classification"
        assert TaskType.REGRESSION == "regression"

    def test_audit_status_values(self):
        assert AuditStatus.PASS == "PASS"
        assert AuditStatus.FAIL == "FAIL"

    def test_user_action_values(self):
        assert UserAction.APPROVE == "approve"
        assert UserAction.REJECT == "reject"
        assert UserAction.COUNTER_PROPOSE == "counter_propose"
