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

_CONFIG_DIR = os.getenv("COMPLIANCE_CONFIG_DIR", "./backend/config/compliance/")


def run_compliance(state: PipelineState) -> PipelineState:
    """
    Load the domain's compliance YAML and inject thresholds into governance_audit.
    Falls back to generic.yaml if the domain is unknown.
    """
    domain_tag = state.objective.domain_tag or "generic"
    config = _load_compliance_config(domain_tag)

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
        },
        "requires_explainability": True,
    }
