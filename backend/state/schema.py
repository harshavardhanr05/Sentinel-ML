"""
backend/state/schema.py
───────────────────────
Pydantic v2 models for the shared PipelineState that all agents read/write.
This is the canonical inter-agent communication protocol (§2.3 of the spec).

Design rules:
- Every field that an agent writes has a matching docstring explaining ownership.
- All nested models are exported at module level for easy import.
- PipelineState is the top-level container — it is what gets persisted to SQLite
  and returned by GET /runs/{id}/state.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    LOOPED_BACK = "looped_back"
    COMPLETE = "complete"
    FAILED = "failed"


class UserAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    COUNTER_PROPOSE = "counter_propose"
    PENDING = "pending"


class AuditStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT_RUN"


class DeployRecommendation(str, Enum):
    DEPLOY = "DEPLOY"
    NO_DEPLOY = "NO_DEPLOY"
    CONDITIONAL = "CONDITIONAL"
    PENDING = "PENDING"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ObjectiveState(BaseModel):
    """Parsed from the user's natural-language objective by the Orchestrator Agent."""

    raw_text: str = Field(description="Original NL objective as typed by the user")
    task_type: TaskType = Field(default=TaskType.UNKNOWN)
    target_column: Optional[str] = None
    target_column_candidates: List[str] = Field(default_factory=list)
    optimization_priority: Optional[str] = Field(
        default=None,
        description="e.g. 'minimize false negatives', 'maximize AUC', 'balanced F1'",
    )
    protected_attributes: List[str] = Field(
        default_factory=list,
        description="Columns declared or inferred as fairness-sensitive (e.g. gender, age_bucket)",
    )
    protected_attribute_reasoning: Dict[str, str] = Field(
        default_factory=dict,
        description="{col_name: 'why it is considered a protected/sensitive attribute'}",
    )
    domain_tag: str = Field(
        default="generic",
        description="Domain for compliance YAML lookup: finance, healthcare, generic, etc.",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True when Orchestrator couldn't parse required fields — triggers clarification checkpoint",
    )
    clarification_needed: List[str] = Field(
        default_factory=list,
        description="List of fields the user needs to clarify",
    )
    feature_selection_top_k: Optional[int] = None
    feature_optimization: str = Field(
        default="none",
        description="Advanced feature optimization: 'none', 'pca', or 'tree'",
    )

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_pct: float = 0.0
    unique_count: int = 0
    is_numeric: bool = False
    is_categorical: bool = False
    is_potential_pii: bool = False
    notes: List[str] = Field(default_factory=list)


class DataHealthReport(BaseModel):
    """Produced by the Data Profiling Agent."""

    row_count: int = 0
    column_count: int = 0
    columns: List[ColumnProfile] = Field(default_factory=list)
    missingness_flags: Dict[str, float] = Field(
        default_factory=dict,
        description="col_name → missing_pct for columns above 5% threshold",
    )
    mnar_flags: List[str] = Field(
        default_factory=list,
        description="Columns where missingness is correlated with another column (MNAR heuristic)",
    )
    leakage_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{column, reason, correlation_with_target}] for suspected leakage",
    )
    imbalance_ratio: Optional[float] = Field(
        default=None,
        description="minority_class_count / majority_class_count; None for regression",
    )
    imbalance_flag: bool = False
    inferred_pii: List[str] = Field(default_factory=list)
    severity_summary: Dict[str, str] = Field(
        default_factory=dict,
        description="{'missingness': 'HIGH', 'leakage': 'MEDIUM', 'imbalance': 'LOW'}",
    )
    profiling_notes: List[str] = Field(default_factory=list)


class FeatureLogEntry(BaseModel):
    feature: str
    transformation: Optional[str] = None
    status: str = Field(description="'accepted' or 'rejected'")
    reason: str
    metric_delta: Optional[float] = Field(
        default=None,
        description="Validation metric improvement from adding this feature",
    )
    governance_flagged: bool = Field(
        default=False,
        description="True if the Governance mid-stage consult flagged this as a fairness proxy",
    )
    imputation_strategy: Optional[str] = Field(
        default=None,
        description="AI-recommended imputation: 'mean', 'median', 'zero', 'unknown'",
    )
    encoding_strategy: Optional[str] = Field(
        default=None,
        description="AI-recommended encoding: 'one_hot', 'target_encoding', 'ordinal'",
    )


class FeatureLog(BaseModel):
    """Produced by the Feature Engineering Agent."""

    accepted: List[FeatureLogEntry] = Field(default_factory=list)
    rejected: List[FeatureLogEntry] = Field(default_factory=list)
    final_feature_set: List[str] = Field(default_factory=list)


class CalibrationPoint(BaseModel):
    bin_mean_predicted: float
    fraction_of_positives: float


class ModelLeaderboardEntry(BaseModel):
    """One row in the model leaderboard. Produced by Model Selection + Cost Awareness agents."""

    model_name: str
    model_family: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    auc_roc: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None

    train_auc_roc: Optional[float] = None
    train_f1_score: Optional[float] = None
    train_rmse: Optional[float] = None
    train_mae: Optional[float] = None
    calibration_curve: List[CalibrationPoint] = Field(default_factory=list)
    cost_estimate_seconds: Optional[float] = Field(
        default=None, description="Estimated wall-clock for full tuning run"
    )
    cost_estimate_note: Optional[str] = None
    is_selected: bool = False
    explainability_summary: Optional[str] = None
    features_used: List[str] = Field(default_factory=list, description="List of feature names used to train this model")


class FairnessMetrics(BaseModel):
    disparate_impact: Optional[float] = None
    equal_opportunity_difference: Optional[float] = None
    demographic_parity_difference: Optional[float] = None
    per_group_confusion_matrices: Dict[str, Any] = Field(default_factory=dict)
    protected_attribute: Optional[str] = None
    threshold_used: Optional[float] = None
    status: AuditStatus = AuditStatus.NOT_RUN


class RobustnessMetrics(BaseModel):
    auc_degradation_pct: Optional[float] = None
    shift_description: Optional[str] = None
    perturbed_features: List[str] = Field(default_factory=list)
    status: AuditStatus = AuditStatus.NOT_RUN


class StabilityMetrics(BaseModel):
    bootstrap_n: int = 0
    metric_variance: Optional[float] = None
    metric_std: Optional[float] = None
    metric_mean: Optional[float] = None
    status: AuditStatus = AuditStatus.NOT_RUN


class GovernanceLoopRecord(BaseModel):
    """Snapshot of one governance iteration — appended each loop."""

    loop_number: int
    overall_result: str  # 'PASS' or 'FAIL'
    auc_roc: Optional[float] = None
    f1_score: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    disparate_impact: Optional[float] = None
    equal_opportunity_difference: Optional[float] = None
    auc_degradation_pct: Optional[float] = None
    bootstrap_variance: Optional[float] = None
    failure_reasons: List[str] = Field(default_factory=list)
    corrective_action: Optional[str] = None  # loopback target
    llm_narrative: Optional[str] = None  # LLM-generated explanation
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentStepEntry(BaseModel):
    """Lightweight step log — one entry per meaningful agent action (not a checkpoint)."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: str
    step_name: str
    details: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GovernanceAudit(BaseModel):
    """Produced by the Governance Agent, thresholds injected by the Compliance Agent."""

    fairness: FairnessMetrics = Field(default_factory=FairnessMetrics)
    robustness: RobustnessMetrics = Field(default_factory=RobustnessMetrics)
    stability: StabilityMetrics = Field(default_factory=StabilityMetrics)
    compliance_checklist: List[str] = Field(
        default_factory=list,
        description="Regulation codes injected by Compliance Agent (e.g. ECOA_0.80_rule)",
    )
    compliance_thresholds: Dict[str, Any] = Field(
        default_factory=dict,
        description="Injected from YAML: {disparate_impact_min: 0.80, ...}",
    )
    overall_status: AuditStatus = AuditStatus.NOT_RUN
    failure_reasons: List[str] = Field(
        default_factory=list,
        description="Actionable reasons for any FAIL; used to route loopback with context",
    )
    loopback_target: Optional[str] = Field(
        default=None,
        description="'feature_engineering' or 'model_selection' — where to loop back on failure",
    )
    iteration_count: int = Field(
        default=0,
        description="How many governance loops have been attempted on this run",
    )
    governance_loop_history: List[GovernanceLoopRecord] = Field(
        default_factory=list,
        description="Per-loop audit records with metrics and LLM narrative",
    )


class ExplainabilityOutput(BaseModel):
    """Produced by the Explainability Agent (SHAP)."""

    global_shap_values: Dict[str, float] = Field(
        default_factory=dict,
        description="feature_name → mean |SHAP value| across validation set",
    )
    top_features_summary: List[str] = Field(default_factory=list)
    local_examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{sample_index, prediction, shap_breakdown: {feature: value}}]",
    )
    shap_plot_path: Optional[str] = Field(
        default=None, description="Saved PNG path for the global SHAP summary plot"
    )


class DecisionLogEntry(BaseModel):
    """Append-only entry in the audit trail. Written by the Checkpoint protocol."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: str
    problem_context: Optional[str] = None
    action_taken: Optional[str] = None
    proposed_action: str
    reasoning: str
    alternatives_considered: List[str] = Field(default_factory=list)
    user_action: UserAction = UserAction.PENDING
    user_note: Optional[str] = None
    agent_justification: Optional[str] = Field(
        default=None,
        description="Structured pros/cons response when user counter-proposes",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    decided_at: Optional[datetime] = None


class DecisionCard(BaseModel):
    """Payload placed in pending_approval when a checkpoint fires."""

    stage: str
    problem_context: Optional[str] = None
    action_taken: Optional[str] = None
    proposed_action: str
    reasoning: str
    alternatives_considered: List[str] = Field(default_factory=list)
    cost_estimate: Optional[str] = None
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)
    requires_response: bool = True


class StageStatusMap(BaseModel):
    """Tracks per-stage status for the live DAG UI."""

    objective_intake: StageStatus = StageStatus.PENDING
    compliance: StageStatus = StageStatus.PENDING
    data_profiling: StageStatus = StageStatus.PENDING
    feature_engineering: StageStatus = StageStatus.PENDING
    model_selection: StageStatus = StageStatus.PENDING
    governance: StageStatus = StageStatus.PENDING
    explainability: StageStatus = StageStatus.PENDING
    reporting: StageStatus = StageStatus.PENDING


# ---------------------------------------------------------------------------
# Top-level PipelineState
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """
    Top-level shared state object for the entire pipeline.
    Every agent reads and writes this. Persisted to SQLite after every node.
    """

    # Identity
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # User inputs
    objective: ObjectiveState = Field(
        default_factory=lambda: ObjectiveState(raw_text="")
    )
    dataset_path: Optional[str] = None
    dataset_filename: Optional[str] = None

    # Pipeline control
    current_stage: str = "objective_intake"
    stage_statuses: StageStatusMap = Field(default_factory=StageStatusMap)
    is_paused: bool = Field(
        default=False,
        description="True when a checkpoint is awaiting human decision",
    )
    pending_approval: Optional[DecisionCard] = Field(
        default=None,
        description="The current decision card shown to the user; None when not paused",
    )
    error_message: Optional[str] = None

    # Agent outputs
    data_schema: Dict[str, Any] = Field(default_factory=dict)
    data_health_report: Optional[DataHealthReport] = None
    data_analysis_metrics: Dict[str, Any] = Field(default_factory=dict)
    feature_log: FeatureLog = Field(default_factory=FeatureLog)
    model_leaderboard: List[ModelLeaderboardEntry] = Field(default_factory=list)
    selected_model_name: Optional[str] = None
    governance_audit: GovernanceAudit = Field(default_factory=GovernanceAudit)
    explainability: ExplainabilityOutput = Field(default_factory=ExplainabilityOutput)

    # Audit trail (append-only within a run)
    decisions_log: List[DecisionLogEntry] = Field(default_factory=list)
    agent_step_log: List[AgentStepEntry] = Field(
        default_factory=list,
        description="Verbose step-by-step agent activity log (non-checkpoint)",
    )

    # Cost estimates (from Cost Awareness Agent)
    cost_estimates: Dict[str, Any] = Field(default_factory=dict)

    # SMOTE tracking
    smote_applied: bool = False
    smote_class_distributions: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    # Final output
    model_card_path: Optional[str] = None
    audit_trail_path: Optional[str] = None
    final_recommendation: DeployRecommendation = DeployRecommendation.PENDING
    final_recommendation_reasoning: Optional[str] = None

    # Lineage (stretch — pre-declared for schema completeness)
    lineage: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}

    def mark_stage(self, stage: str, status: StageStatus) -> None:
        """Update a stage's status and refresh updated_at."""
        if hasattr(self.stage_statuses, stage):
            setattr(self.stage_statuses, stage, status)
        self.current_stage = stage
        self.updated_at = datetime.utcnow()

    def append_decision(self, entry: DecisionLogEntry) -> None:
        """Append-only write to decisions_log (NFR-4: immutable audit trail)."""
        self.decisions_log.append(entry)
        self.updated_at = datetime.utcnow()

    def log_step(self, stage: str, step_name: str, details: str) -> None:
        """Append a lightweight agent step to agent_step_log."""
        self.agent_step_log.append(AgentStepEntry(
            stage=stage,
            step_name=step_name,
            details=details,
        ))
        self.updated_at = datetime.utcnow()

    def set_checkpoint(self, card: DecisionCard) -> None:
        """Pause the pipeline and set the pending decision card."""
        self.pending_approval = card
        self.is_paused = True
        self.updated_at = datetime.utcnow()

    def clear_checkpoint(self) -> None:
        """Resume the pipeline after a checkpoint is resolved."""
        self.pending_approval = None
        self.is_paused = False
        self.updated_at = datetime.utcnow()
