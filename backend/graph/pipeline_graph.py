"""
backend/graph/pipeline_graph.py
────────────────────────────────
LangGraph StateGraph definition for the Sentinel-ML pipeline.

Node order:
  orchestrator → compliance → data_profiling → [CHECKPOINT]
  → feature_engineering → [CHECKPOINT] → model_selection → [CHECKPOINT]
  → governance → (PASS → explainability → reporting)
              → (FAIL → feature_engineering | model_selection)

The Checkpoint/Human-Interface Agent is embedded here as a pattern applied
at each stage transition, not as a separate heavyweight process (per spec §2.1).

State persistence: every node calls save_state_sync() after mutating state,
so a crash mid-run can resume from the last completed checkpoint (NFR-3).
"""

from __future__ import annotations

import sys
import os

# Ensure backend package is importable when running graph directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from backend.state.schema import (
    AuditStatus,
    DecisionCard,
    DecisionLogEntry,
    PipelineState,
    StageStatus,
    UserAction,
)
from backend.state.store import load_state_sync, save_state_sync


# ---------------------------------------------------------------------------
# Helper: checkpoint node factory
# ---------------------------------------------------------------------------


def make_checkpoint_node(stage_name: str):
    """
    Returns a LangGraph node function that:
    1. Sets is_paused = True and populates pending_approval
    2. Persists state to SQLite
    3. Returns state (pipeline halts here until resume signal via API)
    """

    def checkpoint_node(state: PipelineState) -> PipelineState:
        # pending_approval is already set by the preceding agent node.
        # This node just ensures persistence + DAG status is updated.
        state.mark_stage(stage_name, StageStatus.AWAITING_APPROVAL)
        save_state_sync(state)
        return state

    checkpoint_node.__name__ = f"checkpoint_{stage_name}"
    return checkpoint_node


# ---------------------------------------------------------------------------
# Node: orchestrator
# ---------------------------------------------------------------------------


def node_orchestrator(state: PipelineState) -> PipelineState:
    """Parse NL objective → ObjectiveState. Set clarification flag if ambiguous."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "objective_intake", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.orchestrator import run_orchestrator

    state = run_orchestrator(state)
    state.mark_stage("objective_intake", StageStatus.COMPLETE)
    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Node: compliance
# ---------------------------------------------------------------------------


def node_compliance(state: PipelineState) -> PipelineState:
    """Load domain YAML → inject thresholds into governance_audit."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "compliance", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.compliance import run_compliance

    state = run_compliance(state)
    state.mark_stage("compliance", StageStatus.COMPLETE)
    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Node: data_profiling
# ---------------------------------------------------------------------------


def node_data_profiling(state: PipelineState) -> PipelineState:
    """Profile dataset → DataHealthReport. Then set checkpoint."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "data_profiling", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.data_profiling import run_data_profiling

    state = run_data_profiling(state)
    state.mark_stage("data_profiling", StageStatus.COMPLETE)

    # Create decision card for checkpoint
    if state.data_health_report:
        report = state.data_health_report
        flags = []
        if report.leakage_flags:
            flags.append(f"⚠️ {len(report.leakage_flags)} potential leakage column(s) detected")
        if report.imbalance_flag:
            flags.append(f"⚠️ Class imbalance ratio: {report.imbalance_ratio:.2f}")
        if report.mnar_flags:
            flags.append(f"ℹ️ {len(report.mnar_flags)} column(s) with MNAR pattern")

        card = DecisionCard(
            stage="data_profiling",
            problem_context=f"Data profiled. Missingness max severity: {report.severity_summary.get('missingness')}.",
            action_taken=f"Generated {len(report.columns)} column profiles and detected data health issues.",
            proposed_action=(
                f"Proceed to Feature Engineering with {report.row_count} rows × "
                f"{report.column_count} columns. "
                + (" | ".join(flags) if flags else "No critical issues found.")
            ),
            reasoning=(
                f"Data profiling is complete. The dataset has {report.row_count} rows. "
                f"Missingness: {len(report.missingness_flags)} columns above 5%. "
                f"Severity summary: {report.severity_summary}. "
                "Approving will begin feature transformation proposals."
            ),
            alternatives_considered=[
                "Drop all columns with >30% missingness before feature engineering",
                "Apply SMOTE for class imbalance before model selection",
                "Exclude flagged leakage columns immediately",
            ],
            metrics_summary={
                "rows": report.row_count,
                "columns": report.column_count,
                "leakage_flags": len(report.leakage_flags),
                "imbalance_ratio": report.imbalance_ratio,
                "severity": report.severity_summary,
            },
        )
        state.set_checkpoint(card)
        # Log the proposal
        entry = DecisionLogEntry(
            stage="data_profiling",
            problem_context=card.problem_context,
            action_taken=card.action_taken,
            proposed_action=card.proposed_action,
            reasoning=card.reasoning,
            alternatives_considered=card.alternatives_considered,
        )
        state.append_decision(entry)

    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Node: feature_engineering
# ---------------------------------------------------------------------------


def node_feature_engineering(state: PipelineState) -> PipelineState:
    """Propose + test feature transformations. Consult Governance mid-stage."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "feature_engineering", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.feature_engineering import run_feature_engineering

    state.mark_stage("feature_engineering", StageStatus.RUNNING)
    save_state_sync(state)

    state = run_feature_engineering(state)
    state.mark_stage("feature_engineering", StageStatus.COMPLETE)

    # Build checkpoint card
    accepted = len(state.feature_log.accepted)
    rejected = len(state.feature_log.rejected)
    gov_flagged = [e for e in state.feature_log.rejected if e.governance_flagged]

    card = DecisionCard(
        stage="feature_engineering",
        problem_context=f"Raw dataset needed transformations. {rejected} features were identified as suboptimal or non-compliant.",
        action_taken=f"Tested transformations and retained {accepted} features. Enforced any top-K constraints.",
        proposed_action=(
            f"Proceed to Model Selection with {accepted} accepted features "
            f"({rejected} rejected, {len(gov_flagged)} governance-flagged as fairness proxies)."
        ),
        reasoning=(
            f"Feature engineering tested {accepted + rejected} candidate transformations. "
            f"{accepted} improved or were neutral on validation metric. "
            f"{len(gov_flagged)} were rejected after Governance flagged them as likely "
            "fairness proxies. Final feature set is ready for model training."
        ),
        alternatives_considered=[
            "Use PCA instead of individual feature selection",
            "Add polynomial interaction terms for top-3 features",
            "Use target encoding instead of one-hot for high-cardinality categoricals",
        ],
        metrics_summary={
            "accepted_features": accepted,
            "rejected_features": rejected,
            "governance_flagged": len(gov_flagged),
            "final_feature_count": len(state.feature_log.final_feature_set),
        },
    )
    state.set_checkpoint(card)
    entry = DecisionLogEntry(
        stage="feature_engineering",
        problem_context=card.problem_context,
        action_taken=card.action_taken,
        proposed_action=card.proposed_action,
        reasoning=card.reasoning,
        alternatives_considered=card.alternatives_considered,
    )
    state.append_decision(entry)

    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Node: model_selection
# ---------------------------------------------------------------------------


def node_model_selection(state: PipelineState) -> PipelineState:
    """Train 3+ model families, Optuna tuning, build leaderboard."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "model_selection", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.model_selection import run_model_selection
    from backend.agents.cost_awareness import run_cost_awareness

    state.mark_stage("model_selection", StageStatus.RUNNING)
    save_state_sync(state)

    state = run_cost_awareness(state)   # Estimate costs first
    state = run_model_selection(state)  # Then train + tune
    state.mark_stage("model_selection", StageStatus.COMPLETE)

    # Build leaderboard card
    best = next((m for m in state.model_leaderboard if m.is_selected), None)
    if best:
        auc_val = best.auc_roc if best.auc_roc is not None else 0.0
        f1_val = best.f1_score if best.f1_score is not None else (best.rmse if best.rmse is not None else 0.0)
        card = DecisionCard(
            stage="model_selection",
            problem_context=f"Evaluated {len(state.model_leaderboard)} model architectures for the best trade-off between metric and cost.",
            action_taken=f"Selected {best.model_name} as the primary candidate for deployment.",
            proposed_action=(
                f"Selected model: {best.model_name} "
                f"(AUC/R2: {auc_val:.3f}, "
                f"F1/RMSE: {f1_val:.3f}). "
                f"Est. full tuning time: {best.cost_estimate_note or 'N/A'}. "
                "Proceed to Governance Audit?"
            ),
            reasoning=(
                f"{len(state.model_leaderboard)} candidate models trained. "
                f"{best.model_name} selected based on multi-objective score "
                f"(AUC/R2 + fairness proxy). "
                f"Cost-vs-performance trade-offs have been estimated for all candidates."
            ),
            alternatives_considered=[
                f"Select {m.model_name} instead (lower cost, different metric tradeoff)" 
                for m in state.model_leaderboard if m.model_name != best.model_name
            ],
            cost_estimate=best.cost_estimate_note,
            metrics_summary={
                "selected_model": best.model_name,
                "auc_roc": best.auc_roc,
                "f1_score": best.f1_score,
                "rmse": best.rmse,
                "mae": best.mae,
                "candidates_compared": len(state.model_leaderboard),
                "smote_applied": state.smote_applied,
                "smote_class_distributions": state.smote_class_distributions,
            },
        )
        state.set_checkpoint(card)
        entry = DecisionLogEntry(
            stage="model_selection",
            problem_context=card.problem_context,
            action_taken=card.action_taken,
            proposed_action=card.proposed_action,
            reasoning=card.reasoning,
            alternatives_considered=card.alternatives_considered,
        )
        state.append_decision(entry)

    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Node: governance
# ---------------------------------------------------------------------------


def node_governance(state: PipelineState) -> PipelineState:
    """Run all three audits (fairness / robustness / stability) against compliance thresholds."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "governance", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.governance import run_governance

    state.mark_stage("governance", StageStatus.RUNNING)
    save_state_sync(state)

    state = run_governance(state)
    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Node: explainability
# ---------------------------------------------------------------------------


def node_explainability(state: PipelineState) -> PipelineState:
    """SHAP global + local explanations for the selected model."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "explainability", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.explainability import run_explainability

    state.mark_stage("explainability", StageStatus.RUNNING)
    save_state_sync(state)

    state = run_explainability(state)
    state.mark_stage("explainability", StageStatus.COMPLETE)
    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Node: reporting
# ---------------------------------------------------------------------------


def node_reporting(state: PipelineState) -> PipelineState:
    """Generate Model Card + audit trail HTML."""
    db_state = load_state_sync(state.run_id)
    if db_state:
        state = db_state

    if getattr(state.stage_statuses, "reporting", None) in (StageStatus.COMPLETE, StageStatus.APPROVED):
        return state

    from backend.agents.reporting import run_reporting

    state = run_reporting(state)
    state.mark_stage("reporting", StageStatus.COMPLETE)

    # Final deployment decision card
    rec = state.final_recommendation
    card = DecisionCard(
        stage="reporting",
        problem_context="All stages complete. Final model requires deployment approval.",
        action_taken=f"Generated Model Card and final recommendation: {rec}",
        proposed_action=f"Final recommendation: {rec}. Review Model Card and confirm deployment.",
        reasoning=state.final_recommendation_reasoning or "See Model Card for full reasoning.",
        alternatives_considered=["Review and override the recommendation manually"],
        requires_response=True,
    )
    state.set_checkpoint(card)
    entry = DecisionLogEntry(
        stage="reporting",
        problem_context=card.problem_context,
        action_taken=card.action_taken,
        proposed_action=card.proposed_action,
        reasoning=card.reasoning,
    )
    state.append_decision(entry)

    save_state_sync(state)
    return state


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_after_governance(state: PipelineState) -> Literal[
    "explainability", "feature_engineering", "model_selection"
]:
    """
    If governance passed → explainability.
    If failed due to proxy feature → feature_engineering.
    If failed due to model family issue → model_selection.
    """
    audit = state.governance_audit
    if audit.overall_status == AuditStatus.PASS or audit.overall_status == "PASS":
        return "explainability"
        
    if audit.iteration_count >= 2:
        # Break the infinite loop if we've already tried mitigating twice.
        # Force progression to explainability. The failures will be documented in the Model Card.
        audit.failure_reasons.insert(0, "Max governance loopbacks reached. Proceeding to Reporting with unresolved failures.")
        return "explainability"
        
    target = audit.loopback_target or "feature_engineering"
    # Mark the loopback stage as LOOPED_BACK in the DAG
    state.mark_stage(target, StageStatus.LOOPED_BACK)
    
    # Invalidate downstream stages so they don't get skipped by fast-returns
    if target == "feature_engineering":
        state.mark_stage("model_selection", StageStatus.PENDING)
    state.mark_stage("governance", StageStatus.PENDING)
    
    state.governance_audit.iteration_count += 1
    save_state_sync(state)
    return target


def route_after_checkpoint(state: PipelineState) -> Literal["paused", "continue"]:
    """Route to paused (wait for user) or continue (user already approved)."""
    return "paused" if state.is_paused else "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_pipeline_graph() -> StateGraph:
    """
    Assemble and compile the full LangGraph StateGraph.
    Returns the compiled graph ready for invocation.
    """
    graph = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────────
    graph.add_node("orchestrator", node_orchestrator)
    graph.add_node("compliance", node_compliance)
    graph.add_node("data_profiling", node_data_profiling)
    graph.add_node("feature_engineering", node_feature_engineering)
    graph.add_node("model_selection", node_model_selection)
    graph.add_node("governance", node_governance)
    graph.add_node("explainability", node_explainability)
    graph.add_node("reporting", node_reporting)

    # ── Linear edges ─────────────────────────────────────────────────
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "compliance")
    graph.add_edge("compliance", "data_profiling")

    # After data_profiling sets is_paused=True, the graph emits state.
    # The API resumes from feature_engineering after user approves.
    graph.add_edge("data_profiling", "feature_engineering")
    graph.add_edge("feature_engineering", "model_selection")
    graph.add_edge("model_selection", "governance")

    # ── Conditional edge: governance → pass/fail loopback ─────────────
    graph.add_conditional_edges(
        "governance",
        route_after_governance,
        {
            "explainability": "explainability",
            "feature_engineering": "feature_engineering",
            "model_selection": "model_selection",
        },
    )

    graph.add_edge("explainability", "reporting")
    graph.add_edge("reporting", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph (singleton for API use)
# ---------------------------------------------------------------------------

_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_pipeline_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building pipeline graph...")
    g = build_pipeline_graph()
    print("Graph compiled successfully.")
    try:
        mermaid = g.get_graph().draw_mermaid()
        print("\nMermaid DAG:\n")
        print(mermaid)
    except Exception as e:
        print(f"Could not draw Mermaid diagram: {e}")
