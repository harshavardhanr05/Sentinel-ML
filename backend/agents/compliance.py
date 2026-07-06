"""
backend/agents/compliance.py
──────────────────────────────
Compliance Agent — Phase 3

Responsibilities:
1. Read state.objective.domain_tag to determine which YAML to load.
2. Load the matching compliance YAML from COMPLIANCE_CONFIG_DIR.
3. Inject regulatory thresholds into state.governance_audit.compliance_thresholds
   and compliance_checklist before the Governance stage runs.

Config-driven: adding a new domain requires ONLY a new YAML file, no code change (NFR-5).
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from backend.state.schema import PipelineState
from backend.state.store import log_step_and_broadcast_sync

_CONFIG_DIR = os.getenv("COMPLIANCE_CONFIG_DIR", "./backend/config/compliance/")

_COMPLIANCE_TEMPLATE_PROMPT = """
You are a Principal AI Compliance Officer and Regulatory Analyst.
Given a machine learning prediction objective, target column, domain, and candidate protected attributes, generate the applicable compliance rules and fairness thresholds.

Prediction Objective: "{objective}"
Domain: "{domain}"
Target Column: "{target_column}"
Candidate Protected Attributes: {protected_attributes}

Your task is to output a custom compliance profile for this domain.
Guidelines:
1. If the domain is high-risk/regulated (e.g. finance, credit, lending, insurance, human resources, hiring, housing, education):
   - Set "requires_explainability" to true.
   - Inject regulatory compliance checklist items (e.g. ECOA, Fair Housing Act, GDPR Article 22).
   - Set strict thresholds (disparate_impact_min: 0.80, equal_opportunity_diff_max: 0.10).
2. If the domain is clinical/medical/healthcare:
   - Allow sensitive features override (allow_sensitive_features_override: true) because physiological variables are important.
   - Set regulations (e.g., HIPAA de-identification, FDA AI software guidelines).
   - Require high robustness and low AUC degradation.
3. For any other domains (e.g. logistics, energy, e-commerce, weather, gaming):
   - Set regulations standard to general industry best practices (e.g., GDPR_explainability, model_safety_audit).
   - Tailor thresholds accordingly.

Return ONLY a valid JSON object conforming exactly to this schema:
{{
  "domain": "{domain}",
  "regulations": ["List of regulatory codes or checklist rules"],
  "fairness_thresholds": {{
    "disparate_impact_min": 0.80,
    "equal_opportunity_diff_max": 0.10,
    "auc_degradation_max_pct": 10.0,
    "bootstrap_variance_max": 0.03,
    "allow_sensitive_features_override": true | false
  }},
  "requires_explainability": true | false
}}
"""


def run_compliance(state: PipelineState) -> PipelineState:
    """
    Load the domain's compliance YAML. If the YAML file doesn't exist,
    dynamically generate context-aware compliance rules via LLM.
    """
    domain_tag = state.objective.domain_tag or "generic"
    config_path = os.path.join(_CONFIG_DIR, f"{domain_tag}.yaml")

    # If static YAML exists, use it; otherwise, dynamically generate it!
    if os.path.exists(config_path):
        config = _load_compliance_config(domain_tag)
        log_step_and_broadcast_sync(state, "compliance", "Compliance Rules Loaded", f"Loaded static regulatory thresholds for domain '{domain_tag}': {list(config.get('fairness_thresholds', {}).keys())}")
    else:
        log_step_and_broadcast_sync(state, "compliance", "Compliance Rules Construction", f"No static config found for domain '{domain_tag}'. Dynamically generating compliance regulations using AI...")
        from backend.llm.client import get_llm_json
        import json
        
        prompt = _COMPLIANCE_TEMPLATE_PROMPT.format(
            objective=state.objective.raw_text,
            domain=domain_tag,
            target_column=state.objective.target_column or "None",
            protected_attributes=json.dumps(state.objective.protected_attributes)
        )
        
        try:
            config = get_llm_json(prompt) or _default_config()
            log_step_and_broadcast_sync(state, "compliance", "Compliance Rules Loaded", f"AI dynamically generated compliance rules for '{domain_tag}' with regulations: {config.get('regulations', [])}")
        except Exception as e:
            config = _default_config()
            log_step_and_broadcast_sync(state, "compliance", "Compliance Rules Fallback", f"AI compliance generation failed ({e}). Falling back to generic rules.")

    # Inject into governance_audit
    state.governance_audit.compliance_checklist = config.get("regulations", [])
    state.governance_audit.compliance_thresholds = config.get("fairness_thresholds", {})
    
    return state


def _load_compliance_config(domain_tag: str) -> Dict[str, Any]:
    config_path = os.path.join(_CONFIG_DIR, f"{domain_tag}.yaml")

    if not os.path.exists(config_path):
        # Fall back to generic
        config_path = os.path.join(_CONFIG_DIR, "generic.yaml")

    if not os.path.exists(config_path):
        return _default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or _default_config()
    except Exception:
        return _default_config()


def _default_config() -> Dict[str, Any]:
    return {
        "domain": "generic",
        "regulations": ["GDPR_explainability"],
        "fairness_thresholds": {
            "disparate_impact_min": 0.80,
            "equal_opportunity_diff_max": 0.10,
            "auc_degradation_max_pct": 10.0,
            "bootstrap_variance_max": 0.03,
            "allow_sensitive_features_override": False,
        },
        "requires_explainability": True,
    }
