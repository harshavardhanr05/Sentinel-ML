"""
backend/agents/orchestrator.py
───────────────────────────────
Orchestrator Agent — Phase 1

Responsibilities:
1. Accept the raw NL objective and the uploaded dataset path from PipelineState.
2. Use Gemini to extract: task_type, target_column, optimization_priority,
   protected_attributes, domain_tag — in strict JSON matching ObjectiveState.
3. Validate the JSON against ObjectiveState Pydantic model; retry once on failure.
4. If required fields are ambiguous, set is_ambiguous=True + clarification_needed[]
   so the Checkpoint protocol can ask the user before the pipeline continues.
5. Write parsed ObjectiveState back into PipelineState.

LLM contract: reasoning layer only — it narrates over column names from schema
metadata, never sees raw row data (NFR-6 privacy constraint).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from backend.llm.client import get_llm_json
from backend.state.schema import ObjectiveState, PipelineState, TaskType

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PARSE_OBJECTIVE_PROMPT = """
You are the Orchestrator of an ML governance pipeline. Analyze the user's objective
and the dataset's column names to produce a structured JSON configuration.

User's objective:
"{objective}"

Dataset column names:
{columns}

Produce a JSON object with EXACTLY these fields (no extras):
{{
  "task_type": "classification" or "regression" or "unknown",
  "target_column": "the most likely target column name, or null if uncertain",
  "target_column_candidates": ["list", "of", "candidate", "columns"],
  "optimization_priority": "plain-language priority from the objective, e.g. 'minimize false negatives', or null",
  "protected_attributes": ["list of exact column names that are sensitive demographic attributes"],
  "protected_attribute_reasoning": {{"col_name": "1-2 sentence explanation of why this column is a protected/sensitive attribute and what bias risk it carries"}},
  "domain_tag": "finance, healthcare, hr, generic, etc.",
  "feature_selection_top_k": integer or null (if the user specifically asks to only use the top N features),
  "feature_optimization": "none", "pca", or "tree" (default to "none", set to "pca" or "tree" if user asks for PCA or feature importance optimization),
  "is_ambiguous": true/false (true ONLY if the objective is completely unclear or target column is unidentifiable),
  "clarification_needed": ["list of questions or missing info to ask the user, if ambiguous"]
}}

Rules:
- Use ONLY column names that actually exist in the dataset for target_column and protected_attributes.
- If the objective clearly states the target column, use it. If not, infer from context.
- domain_tag: finance if credit/loan/default/risk; healthcare if patient/diagnosis/clinical;
  hr if employee/salary/hiring; retail if sales/customer/product; otherwise generic.
- For protected_attributes: If the dataset domain is sensitive (e.g., finance, loan, HR, hiring, housing, education), YOU MUST actively scan the columns and ALWAYS extract demographic variables (e.g. race, gender, sex, age, religion) as protected_attributes, even if the user didn't explicitly request them. Only skip this if the domain is physical, medical, or survival.
- protected_attribute_reasoning must contain an entry for EVERY column in protected_attributes.
  Explain concisely: what the column represents, why it is sensitive, and what discrimination risk it creates.
- is_ambiguous: true if target_column is null OR if protected_attributes can't be inferred.
- Return ONLY the JSON object, no prose.
"""

_COUNTER_PROPOSE_PROMPT = """
The user has counter-proposed a change to the orchestrator's initial parsing:
User suggestion: "{user_note}"

Original parsing:
{original}

Respond with a structured pros/cons comparison as JSON:
{{
  "agent_choice": {{...original fields...}},
  "user_suggestion_interpretation": "what the user seems to want",
  "pros_of_agent_choice": ["list"],
  "cons_of_agent_choice": ["list"],
  "pros_of_user_suggestion": ["list"],
  "cons_of_user_suggestion": ["list"],
  "recommendation": "agent_choice or user_suggestion",
  "recommendation_reasoning": "one sentence"
}}
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_orchestrator(state: PipelineState) -> PipelineState:
    """
    Parse the NL objective and populate state.objective.
    Called by the orchestrator LangGraph node.
    """
    objective_text = state.objective.raw_text
    if not objective_text.strip():
        state.objective.is_ambiguous = True
        state.objective.clarification_needed = ["Please provide a business objective."]
        return state

    # Get column names from state (populated by dataset upload)
    columns: List[str] = list(state.data_schema.get("columns", []))
    if not columns:
        state.objective.is_ambiguous = True
        state.objective.clarification_needed = [
            "Dataset was not loaded yet — cannot infer target column."
        ]
        return state

    prompt = _PARSE_OBJECTIVE_PROMPT.format(
        objective=objective_text,
        columns=json.dumps(columns),
    )

    try:
        parsed: Dict[str, Any] = get_llm_json(prompt)
        obj = _build_objective_state(objective_text, parsed, columns)
        state.objective = obj
        
        from backend.state.schema import DecisionCard
        
        # Build dynamic quick-select options
        alts = []
        candidates = parsed.get("target_column_candidates", [])
        if isinstance(candidates, list) and len(candidates) > 0:
            alts = [f"Select '{c}' as target column" for c in candidates[:4] if c != obj.target_column]
        alts.append("I will provide the target column manually")

        if obj.is_ambiguous:
            problem = "The provided objective was ambiguous or missing a target column."
            action = "Paused pipeline. Inferred likely target candidates and requesting clarification."
            proposed = "Clarify objective and target column"
            reasoning = " ".join(obj.clarification_needed) or "Could not parse objective automatically. Please choose from the inferred target columns below or type it manually."
        else:
            problem = "Please confirm the AI's understanding of the objective and target column."
            action = f"Parsed target column as '{obj.target_column}'. Pausing for user confirmation."
            proposed = f"Proceed with target column: {obj.target_column}"
            reasoning = f"Based on the objective, the AI determined '{obj.target_column}' is the best target. If this is incorrect, you can change it below."

        card = DecisionCard(
            stage="objective_intake",
            problem_context=problem,
            action_taken=action,
            proposed_action=proposed,
            reasoning=reasoning,
            alternatives_considered=alts,
            requires_response=True,
        )
        state.set_checkpoint(card)
            
    except Exception as e:
        raise RuntimeError(
            f"Could not parse objective automatically (error: {e}). "
            "Please ensure your GEMINI_API_KEY is set in the .env file."
        )

    return state


def handle_counter_propose(
    state: PipelineState, user_note: str
) -> Dict[str, Any]:
    """
    When the user counter-proposes at the objective checkpoint, return a
    structured pros/cons comparison (not silent compliance).
    """
    original = state.objective.model_dump()
    prompt = _COUNTER_PROPOSE_PROMPT.format(
        user_note=user_note,
        original=json.dumps(original, indent=2),
    )
    try:
        return get_llm_json(prompt)
    except Exception:
        return {
            "recommendation": "user_suggestion",
            "recommendation_reasoning": "Deferring to user preference as the automatic comparison could not be generated.",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_objective_state(
    raw_text: str, parsed: Dict[str, Any], available_columns: List[str]
) -> ObjectiveState:
    """
    Build an ObjectiveState from the LLM's parsed response.
    Validates that any named columns actually exist in the dataset.
    """
    # Create a case-insensitive lookup map for available columns
    col_map = {c.lower(): c for c in available_columns}

    # Normalize task type
    task_str = str(parsed.get("task_type", "unknown")).lower()
    task_type = TaskType(task_str) if task_str in TaskType._value2member_map_ else TaskType.UNKNOWN

    # Validate target column (case-insensitive)
    target_col = parsed.get("target_column")
    if target_col:
        target_col = col_map.get(str(target_col).lower())

    # Determine domain tag
    domain_tag = parsed.get("domain_tag", "generic")

    # Validate protected attributes (case-insensitive)
    raw_protected = parsed.get("protected_attributes", [])
    protected = []
    for c in raw_protected:
        actual = col_map.get(str(c).lower())
        if actual and actual not in protected:
            protected.append(actual)

    # Extract per-attribute reasoning
    raw_reasoning = parsed.get("protected_attribute_reasoning", {})
    protected_reasoning = {}
    for col, reason in raw_reasoning.items():
        actual = col_map.get(str(col).lower())
        if actual in protected:
            protected_reasoning[actual] = str(reason)

    # Fallback: Auto-extract if domain is sensitive but LLM failed to extract
    if not protected and domain_tag in ("finance", "hr", "loan", "housing", "education"):
        demographic_keywords = {"gender", "sex", "age", "race", "ethnicity", "religion", "marital_status"}
        for c in available_columns:
            if c.lower() in demographic_keywords and c not in protected:
                protected.append(c)
                protected_reasoning[c] = f"Auto-flagged '{c}' as a protected attribute due to sensitive domain."

    # Validate candidates (case-insensitive)
    raw_candidates = parsed.get("target_column_candidates", [])
    candidates = []
    for c in raw_candidates:
        actual = col_map.get(str(c).lower())
        if actual and actual not in candidates:
            candidates.append(actual)

    # Determine ambiguity
    is_ambiguous = parsed.get("is_ambiguous", False)
    clarification_needed = parsed.get("clarification_needed", [])

    if not target_col:
        is_ambiguous = True
        if "target_column" not in clarification_needed:
            clarification_needed.append(
                "Could not determine the target column. Please specify which column to predict."
            )

    return ObjectiveState(
        raw_text=raw_text,
        task_type=task_type,
        target_column=target_col,
        target_column_candidates=candidates,
        optimization_priority=parsed.get("optimization_priority"),
        protected_attributes=protected,
        protected_attribute_reasoning=protected_reasoning,
        domain_tag=parsed.get("domain_tag", "generic"),
        is_ambiguous=bool(is_ambiguous),
        clarification_needed=clarification_needed,
        feature_selection_top_k=parsed.get("feature_selection_top_k"),
        feature_optimization=parsed.get("feature_optimization", "none"),
    )
