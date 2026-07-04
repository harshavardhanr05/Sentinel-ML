# Autonomous ML Pipeline Agent with Human-in-the-Loop Governance
## Technical Specification, Architecture & SRS Document
**Project type:** Solo build | **Target consumer:** Antigravity (AI coding agent) | **Version:** 1.0

---

# 1. Project Overview

## 1.1 Problem Statement
Most AutoML systems (H2O, AutoGluon, DataRobot) optimize purely for a performance metric and hand a model to a human with no visibility into *how* it got there, whether it's fair, whether it's robust, or whether a regulator would accept it. Separately, most "agentic AI" demos are single-agent chatbots wrapping an API call — they don't reason, don't check in with a human, and don't produce artifacts a real data team would recognize.

## 1.2 Project Goal
Build a **multi-agent, human-in-the-loop, regulation-aware ML pipeline system** that:
- Takes any raw tabular dataset + a plain-language business objective, for **any domain** (finance, healthcare, retail, etc.)
- Autonomously plans and executes the full ML lifecycle (profiling → feature engineering → model selection → governance → explainability → deployment)
- **Pauses at every meaningful decision point** to propose an action, explain its reasoning, and let the user approve, reject, or counter-propose — with the agent justifying trade-offs either way
- Produces the deliverables a real Data Scientist / Data Engineer / MLOps engineer would expect at each stage
- Self-governs deployment via fairness, robustness, stability, and compliance audits, with an explicit deploy/no-deploy recommendation
- Presents all of this through a polished, visual, low-friction UI (live pipeline DAG, decision cards, insight dashboards, audit trail)

## 1.3 Why This Has Novelty (headline story for viva/interviews/resume)
1. **Checkpoint-driven autonomy** — not "human in the loop" as an afterthought, but a formal protocol (Propose → Pause → Justify → Log) at every stage transition.
2. **Self-governance** — the system can refuse to deploy its own model, and the Governance Agent can send work *back* to earlier stages with a specific, actionable reason (a real feedback loop, not a linear pipeline).
3. **Regulation-aware, domain-agnostic** — a Compliance Agent dynamically injects the right regulatory checklist (ECOA/EEOC, HIPAA, GDPR, DPDP) based on the declared or inferred domain, without hardcoding per-domain logic.
4. **Bidirectional agent consultation** — agents can query each other mid-stage (e.g., Feature Engineering asks Governance "will this feature likely trip a fairness check?") rather than only auditing at the very end.

## 1.4 Headline Novelty Metric
> "Autonomous governance intervention rate" — the % of runs where the Governance Agent rejected the first candidate and triggered a redesign loop, plus the measured improvement (e.g., Disparate Impact 0.68 → 0.84) achieved without direct human edits to the model.

---

# 2. System Architecture

## 2.1 Agent Roster (Final, including all add-ons)

| # | Agent | Role |
|---|-------|------|
| 1 | **Orchestrator Agent** | Parses objective, builds task plan, owns the shared state, routes control between agents |
| 2 | **Compliance Agent** | Infers/accepts domain tag, loads relevant regulatory checklist, injects constraints into Governance |
| 3 | **Data Profiling Agent** | Schema inference, missingness, leakage, imbalance, drift-risk report |
| 4 | **Feature Engineering Agent** | Proposes/tests transformations; can consult Governance before committing a feature |
| 5 | **Model Selection & Tuning Agent** | Candidate models, Optuna search, leaderboard with explainability summary |
| 6 | **Governance Agent** | Fairness / robustness / stability audits; approves or bounces the pipeline back with a reason |
| 7 | **Explainability Agent** | Global + local explanations (SHAP), separate from Governance for clean separation of concerns |
| 8 | **Cost-Awareness Agent** | Estimates compute/time cost of each candidate approach before it runs; surfaces cost-vs-performance trade-offs |
| 9 | **Version Control / Lineage Agent** | Lightweight MLflow-style run, model, and decision versioning |
| 10 | **Synthetic Stress-Test Agent** | Generates adversarial/edge-case synthetic samples pre-deployment, beyond standard shift testing |
| 11 | **Reporting Agent** | Final model card (Google Model Card format), audit trail export |
| 12 | **Checkpoint / Human-Interface Agent** | Owns the Propose→Pause→Justify→Log protocol; formats decision cards for the UI |

*(This is 12 logical agents — for a solo MVP, several are implemented as distinct LangGraph nodes/functions rather than separate heavyweight processes; see §5 MVP scoping.)*

## 2.2 High-Level Flow

```
User Objective (NL) + Dataset
        │
        ▼
 ┌───────────────┐
 │ Orchestrator  │──────────────┐
 └───────┬───────┘              │
         ▼                      ▼
 ┌───────────────┐      ┌───────────────┐
 │  Compliance    │      │  Checkpoint /  │
 │  Agent         │◄────►│  Human UI Agent│
 └───────┬───────┘      └───────┬───────┘
         ▼                      │ (approve/reject/counter at every stage)
 ┌───────────────┐              │
 │ Data Profiling │◄─────────────┘
 └───────┬───────┘
         ▼
 ┌───────────────┐   consult   ┌───────────────┐
 │ Feature Eng.   │────────────►│  Governance    │
 └───────┬───────┘◄────────────│  Agent         │
         ▼                      └───────┬───────┘
 ┌───────────────┐                      │ pass/fail + reason
 │ Model Select/  │──────────────────────┘
 │ Tuning + Cost  │
 │ Awareness      │
 └───────┬───────┘
         ▼
 ┌───────────────┐   FAIL → loop back to Feature Eng. or Model Selection
 │ Governance     │───────────────────────────────────────────┐
 │ Audit (Fair/   │                                           │
 │ Robust/Stable) │  PASS                                     │
 └───────┬───────┘                                           │
         ▼                                                    │
 ┌───────────────┐      ┌───────────────┐                     │
 │ Explainability │      │ Synthetic      │                    │
 │ Agent          │      │ Stress-Test    │                    │
 └───────┬───────┘      └───────┬───────┘                     │
         └──────────┬───────────┘                             │
                     ▼                                        │
             ┌───────────────┐                                │
             │ Version Control│                                │
             │ / Lineage Agent│                                │
             └───────┬───────┘                                │
                     ▼                                        │
             ┌───────────────┐                                │
             │ Reporting Agent│──── Model Card + Audit Trail   │
             └───────┬───────┘                                │
                     ▼                                        │
             Deploy / No-Deploy Recommendation ◄───────────────┘
```

## 2.3 Shared State (Inter-Agent Communication Protocol)
All agents read/write a single structured state object — this is what makes them "well communicable with each other" rather than passing loose text.

```json
{
  "run_id": "uuid",
  "objective": {
    "raw_text": "predict loan default, minimize false negatives, must be fair across gender/age",
    "task_type": "classification",
    "target_column": "default_flag",
    "protected_attributes": ["gender", "age_bucket"],
    "domain_tag": "finance"
  },
  "current_stage": "governance_audit",
  "data_schema": { "columns": [...], "dtypes": {...}, "inferred_pii": [...] },
  "data_health_report": { "missingness": {...}, "leakage_flags": [...], "imbalance_ratio": 0.12 },
  "feature_log": { "accepted": [...], "rejected": [{"feature": "zip_code", "reason": "proxy for race, flagged by Governance consult"}] },
  "model_leaderboard": [ {"model": "XGBoost", "auc": 0.81, "cost_estimate_sec": 42}, ... ],
  "governance_audit": {
    "fairness": {"disparate_impact": 0.65, "threshold": 0.80, "status": "FAIL"},
    "robustness": {"auc_degradation_pct": 4.2, "status": "PASS"},
    "stability": {"bootstrap_variance": 0.015, "status": "PASS"},
    "compliance_checklist": ["ECOA_0.80_rule", "GDPR_explainability"]
  },
  "decisions_log": [
    {"stage": "feature_engineering", "proposed": "...", "reasoning": "...", "user_action": "approved", "timestamp": "..."}
  ],
  "pending_approval": {"stage": "model_selection", "card": {...}},
  "explainability": { "global_shap": {...}, "local_examples": [...] },
  "cost_estimates": {...},
  "lineage": { "run_history": [...], "model_versions": [...] },
  "final_recommendation": null
}
```
LangGraph's native state-graph handles this as a typed state object passed between nodes — no separate message bus needed for the MVP.

## 2.4 The Checkpoint Protocol (core UX/agentic pattern)
At every stage transition:
1. **Propose** — agent states: what it's about to do, why (tied back to the objective), and what alternatives it considered.
2. **Pause** — pipeline halts; UI shows a decision card with Approve / Reject / Suggest Alternative / Ask Why.
3. **Justify** — if the user suggests an alternative, the agent must respond with a structured pros/cons comparison against its own choice, not silently comply.
4. **Log** — every proposal, user action, and final choice is appended to `decisions_log` for the audit trail.

Checkpoints occur at minimum: data cleaning strategy, feature selection, model family choice, hyperparameter search space, fairness threshold selection, and final deployment decision.

---

# 3. Technology Stack & Environment

## 3.1 Core Stack — Every Tool Below Is Free / Open-Source

| Layer | Technology | Cost | Reason |
|---|---|---|---|
| Agent orchestration | **LangGraph** | Free, open-source (MIT) | Native cyclic graphs — required for governance feedback loops; state-graph model fits the shared-state design directly |
| LLM (reasoning/orchestration only) | **Google Gemini API — free tier**, via `google-genai` SDK (model: `gemini-2.5-flash` or `gemini-flash-lite`) — with **Ollama** (Llama 3.1 8B / Qwen2.5) as a fully-offline free fallback | Free — Google AI Studio's free tier requires no credit card and no billing account, with generous daily request limits on Flash-tier models. Note: this is **not** unlocked by a Google AI Pro subscription — Pro is a separate consumer app plan and doesn't include API credits; the free tier exists independently of any subscription | Reasoning/orchestration layer only, never the predictor. Gemini's free tier is hosted (no local GPU needed) and more capable than a small local model, so it's the better default; Ollama remains useful if you want the system to run fully offline or hit the daily free-tier request cap during heavy testing |
| Classical ML | **scikit-learn**, **XGBoost**, **LightGBM (optional)** | Free, open-source | Actual model fitting/tuning |
| Hyperparameter search | **Optuna** | Free, open-source | Bayesian tuning, multi-objective support |
| Explainability | **SHAP** | Free, open-source | Global + local explanations |
| Fairness metrics | **Fairlearn** or **AIF360** | Free, open-source (Microsoft / IBM) | Disparate Impact, Equal Opportunity Difference, per-group confusion matrices |
| Data profiling | **pandas**, **ydata-profiling** (optional) | Free, open-source | Missingness, distributions |
| Backend API | **FastAPI** | Free, open-source | Serves the LangGraph pipeline, exposes REST + WebSocket endpoints |
| Real-time updates | **WebSockets** (FastAPI native) | Free | Push pipeline state changes to UI live (DAG progress, decision cards) |
| Frontend | **React** + **Tailwind** + **React Flow** | Free, open-source | Matches existing React/Flask-style skillset |
| Charts/dashboards | **Recharts** or **Plotly.js** | Free, open-source | Fairness metric cards, drift-risk gauges, leaderboard visuals |
| Experiment tracking | **MLflow** (local/lightweight mode) | Free, open-source | Version Control / Lineage Agent backend |
| Database | **SQLite** (MVP) → **PostgreSQL** (stretch) | Free, open-source | Persist runs, decisions_log, model versions |
| Config for compliance rules | **YAML** files per domain tag | Free (plain text format) | Pluggable, no-code-change extensibility |
| Containerization | **Docker** (Docker Desktop free tier for individual/personal use) + `docker-compose` | Free for individual use | Reproducible local environment |
| Datasets | **UCI Adult Income**, **COMPAS** (ProPublica), synthetic data via `sklearn.datasets`/`Faker` | Free, publicly licensed | Fairness-auditing benchmarks with existing literature |

> **No paid API key is required anywhere in this stack.** For the LLM reasoning layer, use Google's **free Gemini API tier** (`gemini-2.5-flash`) via a free Google AI Studio API key — no billing account needed — with **Ollama** as a fully-offline fallback if you'd rather not depend on any external service, or if you hit the free tier's daily request cap during heavy testing. Isolate whichever you pick behind one adapter function (`backend/llm/client.py`) so switching between Gemini-free, Ollama, or (later, optionally) a paid hosted API is a one-line config change, not a rewrite.

**Getting a free Gemini API key:** go to https://aistudio.google.com, sign in with any Google account (your existing one is fine), click "Get API key," and create one — this is separate from your AI Pro subscription and doesn't touch its billing. Free-tier limits (checked periodically by Google) are generous enough for iterative development and demoing; if you exceed them mid-testing, the pipeline should fail gracefully and let you switch the `.env` `LLM_PROVIDER` to `ollama` without any code change.

**Hardware note (Ollama fallback only):** Llama 3.1 8B / Mistral 7B / Qwen2.5 7B run comfortably on a machine with 16GB RAM (CPU-only, a bit slower) or any GPU with ≥8GB VRAM. If your machine is more limited, use a smaller model (e.g., **Qwen2.5 3B** or **Llama 3.2 3B**) via Ollama.

## 3.2 Environment Setup Requirements

**Python:** 3.11+
**Node:** 20+ (for React frontend)
**Gemini API key (free):** from https://aistudio.google.com — no billing required
**Ollama (optional, offline fallback):** latest version from https://ollama.com

```bash
# 1. Get a free Gemini API key
#    Visit https://aistudio.google.com -> "Get API key" -> copy it into .env below

# 2. (Optional) Install Ollama as an offline fallback
#    macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
#    Windows: download installer from ollama.com
#    ollama pull llama3.1:8b   # or: ollama pull qwen2.5:3b for lighter hardware
#    ollama serve

# 3. Backend Python environment
python -m venv venv
source venv/bin/activate
pip install langgraph langchain-google-genai langchain-ollama fastapi uvicorn[standard] \
    scikit-learn xgboost optuna shap fairlearn pandas numpy \
    mlflow pyyaml python-multipart websockets pydantic

# 4. Frontend
npx create-react-app frontend --template typescript
cd frontend && npm install reactflow recharts axios tailwindcss
```

**Environment variables (`.env`):**
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=<your free key from aistudio.google.com>
GEMINI_MODEL=gemini-2.5-flash
# fallback config, only used if LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DATABASE_URL=sqlite:///pipeline_runs.db
COMPLIANCE_CONFIG_DIR=./config/compliance/
```



**Directory skeleton:**
```
ml-governance-agent/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── compliance.py
│   │   ├── data_profiling.py
│   │   ├── feature_engineering.py
│   │   ├── model_selection.py
│   │   ├── governance.py
│   │   ├── explainability.py
│   │   ├── cost_awareness.py
│   │   ├── lineage.py
│   │   ├── stress_test.py
│   │   └── reporting.py
│   ├── graph/
│   │   └── pipeline_graph.py        # LangGraph StateGraph definition
│   ├── state/
│   │   └── schema.py                # Pydantic models for shared state
│   ├── config/
│   │   └── compliance/
│   │       ├── finance.yaml
│   │       ├── healthcare.yaml
│   │       └── generic.yaml
│   ├── api/
│   │   ├── main.py                  # FastAPI app
│   │   └── ws.py                    # WebSocket handlers
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PipelineDAG.tsx
│   │   │   ├── DecisionCard.tsx
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── InsightDashboard.tsx
│   │   │   └── AuditTrailViewer.tsx
│   │   └── App.tsx
├── docker-compose.yml
└── README.md
```

---

# 4. Software Requirements Specification (SRS)

## 4.1 Purpose
This SRS defines the functional and non-functional requirements for the Autonomous ML Pipeline Agent with Human-in-the-Loop Governance, to be implemented solo and fed into an AI coding agent (Antigravity) for scaffolding and iterative build-out.

## 4.2 Scope
A locally-runnable (Dockerized) web application: user uploads a dataset + states an objective in natural language; the system runs a multi-agent pipeline with mandatory human checkpoints, producing a governed, explainable, deployment-recommended model along with a full audit trail — for **any tabular dataset in any domain**.

## 4.3 Functional Requirements

### FR-1: Objective Intake
- FR-1.1 System shall accept a natural-language business objective and a dataset (CSV/Parquet upload).
- FR-1.2 System shall parse the objective into: task type (classification/regression), target column candidate(s), optimization priority (e.g., minimize false negatives), and candidate protected attributes.
- FR-1.3 If the objective is ambiguous, system shall ask clarifying questions before proceeding (not guess silently).
- FR-1.4 User shall be able to confirm/edit the inferred target column and protected attributes before the pipeline starts.

### FR-2: Domain & Compliance Detection
- FR-2.1 System shall infer or accept a user-declared domain tag (finance, healthcare, generic, etc.).
- FR-2.2 System shall load the corresponding compliance checklist from a YAML config (ECOA/EEOC for finance, HIPAA-style de-identification checks for healthcare, GDPR explainability requirement if EU context flagged, DPDP Act flags for Indian data context).
- FR-2.3 Compliance requirements shall be injected as explicit thresholds/constraints into the Governance Agent's audit stage.

### FR-3: Data Profiling
- FR-3.1 System shall auto-detect column types, missingness patterns (including MNAR flags), class imbalance ratio, and potential target leakage.
- FR-3.2 System shall produce a structured Data Health Report (not just visualizations) with explicit flags and severity levels.
- FR-3.3 This report shall be presented at a checkpoint before feature engineering begins.

### FR-4: Feature Engineering
- FR-4.1 System shall propose feature transformations (encoding, interactions, temporal features where applicable) and test each via validation-metric delta.
- FR-4.2 System shall be able to consult the Governance Agent mid-stage on whether a candidate feature is a likely fairness proxy, before committing it.
- FR-4.3 System shall log every accepted and rejected feature with a reason.
- FR-4.4 Every feature-set change shall pass through the Checkpoint Protocol.

### FR-5: Model Selection & Tuning
- FR-5.1 System shall train and compare at least 3 candidate model families (e.g., Logistic Regression, tree ensemble, optionally a small NN).
- FR-5.2 System shall use Optuna for hyperparameter search with a multi-objective score (not accuracy alone).
- FR-5.3 System shall produce a leaderboard with per-model metric comparison — AUC-ROC, F1, and a **calibration curve** per candidate — and a short explainability summary.
- FR-5.4 The Cost-Awareness Agent shall estimate compute/time cost for each candidate before it is run, and present cost-vs-performance trade-offs at the checkpoint.

### FR-6: Governance Audit
- FR-6.1 System shall run a Fairness Audit (Disparate Impact, Equal Opportunity Difference, per-group confusion matrices) against configurable thresholds.
- FR-6.2 System shall run a Robustness Audit (performance degradation under synthetic covariate shift).
- FR-6.3 System shall run a Stability Audit (variance across bootstrapped resamples).
- FR-6.4 If any audit fails, system shall route control back to Feature Engineering or Model Selection with a specific, actionable reason (not a generic failure message).
- FR-6.5 All audit results shall be checked against the Compliance Agent's injected regulatory thresholds, not just internal defaults.

### FR-7: Explainability
- FR-7.1 System shall generate global feature-importance explanations (SHAP summary) for the selected model.
- FR-7.2 System shall generate local explanations for representative/edge-case predictions.
- FR-7.3 Explainability shall be presented as a distinct deliverable from the Governance audit.

### FR-8: Synthetic Stress-Testing
- FR-8.1 Before final deployment approval, system shall generate adversarial/edge-case synthetic samples and evaluate model behavior on them.
- FR-8.2 Results shall be added to the Governance audit as a supplementary signal, flagged separately from the standard robustness check.

### FR-9: Version Control / Lineage
- FR-9.1 System shall log every run, model version, and decision set (MLflow-backed).
- FR-9.2 System shall allow the user to view/compare prior runs.

### FR-10: Human-in-the-Loop Checkpoints
- FR-10.1 At every checkpoint, system shall present: proposed action, reasoning tied to the objective, and alternatives considered.
- FR-10.2 User shall be able to Approve, Reject, or Suggest an Alternative at each checkpoint.
- FR-10.3 If the user suggests an alternative, system shall respond with a structured pros/cons comparison before proceeding, never silently comply.
- FR-10.4 Every checkpoint decision (proposal, user action, final choice, timestamp) shall be appended to an immutable decisions log.

### FR-11: Reporting
- FR-11.1 System shall generate a final Model Card (Google Model Card format) summarizing objective, chosen model, performance, fairness results, compliance status, and an explicit deploy / no-deploy recommendation with reasoning.
- FR-11.2 System shall export the full audit trail (all decisions + overrides) in a human-readable format (PDF/HTML).

### FR-12: UI/UX
- FR-12.1 System shall render a live pipeline DAG showing stage status: completed (green), pending approval (pulsing yellow), rejected/looped-back (red).
- FR-12.2 System shall provide a chat-style reasoning panel alongside the DAG.
- FR-12.3 System shall render inline decision cards at each checkpoint with Approve/Reject/Discuss actions.
- FR-12.4 System shall render an Insight Dashboard (SHAP plots, fairness metric cards, drift-risk gauges, leaderboard).
- FR-12.5 System shall render an Audit Trail Viewer as a timeline of every decision and who made it (agent vs. user).
- FR-12.6 System shall work across any uploaded tabular dataset without requiring code changes for a new domain (config-driven compliance rules only).

## 4.4 Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Usability | Every agent proposal must be readable by a non-ML-expert stakeholder — plain language, no unexplained jargon in decision cards |
| NFR-2 | Performance | Data profiling on a dataset up to ~100K rows / 100 columns shall complete in under 60 seconds on a standard laptop (no GPU required for MVP) |
| NFR-3 | Reliability | Pipeline state must be persisted after every stage so a crash/restart resumes from the last checkpoint, not from scratch |
| NFR-4 | Auditability | Decisions log must be append-only / immutable within a run — no silent overwrites |
| NFR-5 | Extensibility | Adding a new domain's compliance rules must require only a new YAML file, no code change |
| NFR-6 | Security | Uploaded datasets shall not leave the local/self-hosted environment; if using the Gemini free tier, only schema-level metadata and small samples needed for reasoning are sent, never full raw data — or run entirely offline via the Ollama fallback for zero external data exposure |
| NFR-7 | Transparency | Every LLM-driven decision must be traceable to which agent produced it and what data it saw |
| NFR-8 | Portability | Entire system shall run via `docker-compose up` with no manual dependency installation |
| NFR-9 | Maintainability | Each agent implemented as an isolated module/function with a defined input/output schema (Pydantic), independently testable |
| NFR-10 | Scalability (design-only for MVP) | Architecture shall be documented to support swapping SQLite → PostgreSQL and adding worker queues (Celery/Redis) without redesigning the agent graph |

## 4.5 Constraints
- Solo developer — MVP scope must be achievable without a team (see §5).
- No production-grade auth/multi-tenancy required for MVP (single local user).
- LLM used only for reasoning/orchestration/NL parsing — never for the actual model fitting (must be explicitly defensible in viva: "why not just use sklearn directly for this part").

## 4.6 Assumptions
- Dataset is tabular (CSV/Parquet); non-tabular support is out of scope.
- User has a free Gemini API key from Google AI Studio (aistudio.google.com), unrelated to any Google AI Pro subscription; no paid API key of any kind is required for the LLM layer. Running fully offline via Ollama is supported as a drop-in alternative through the same adapter.
- "Any domain" means the *pipeline logic* is domain-agnostic; regulatory coverage is limited to the domains configured in `config/compliance/` (finance, healthcare, generic/EU, generic/India) — adding more is a config task, not a functional gap.

---

# 5. MVP Scope vs. Stretch Scope (solo-build reality check)

## 5.1 MVP (build this first — demoable in ~4–5 weeks part-time)
- Agents: Orchestrator, Compliance, Data Profiling, Feature Engineering, Model Selection + Cost estimate, Governance (fairness + robustness + stability), Reporting, Checkpoint/Human-Interface.
- UI: DAG view + decision cards + basic dashboard (fairness/leaderboard) + audit trail list.
- 2 domains configured (e.g., finance + generic) to prove the config-driven compliance claim.
- One full working feedback loop demonstrated end-to-end (Governance rejects → Feature Eng. retries → passes).

## 5.2 Stretch (post-MVP, documented as "designed for extensibility")
- Explainability Agent as a fully separate SHAP-driven module with local + global views.
- Synthetic Stress-Test Agent.
- Version Control / Lineage Agent with full MLflow UI integration.
- Bidirectional mid-stage agent consultation (Feature Eng. ↔ Governance) beyond end-of-pipeline audits only.
- PostgreSQL + multi-user auth.
- Additional compliance domains (healthcare/HIPAA, EU/GDPR, India/DPDP) fully fleshed out.

---

# 6. In-Depth Step-by-Step Implementation Plan (solo, fully granular)

This section is written so each numbered step is a concrete, checkable unit of work — hand this whole section to Antigravity phase-by-phase and it has enough detail to scaffold each piece without guessing.

## Phase 0 — Environment & Skeleton (2–3 days)

1. Install Docker Desktop (free) and confirm `docker --version` works.
2. Get a free Gemini API key from https://aistudio.google.com (no billing account needed) and save it to `.env` as `GEMINI_API_KEY`. Optionally, also install Ollama per §3.2 as an offline fallback and confirm `ollama serve` responds locally.
3. Create the project root folder and the full directory skeleton from §3.2 (`backend/agents`, `backend/graph`, `backend/state`, `backend/config/compliance`, `backend/api`, `backend/tests`, `frontend/src/components`).
4. Initialize a git repo; add a `.gitignore` (venv, node_modules, `*.db`, `.env`).
5. Set up the Python virtual environment and install the backend dependency list from §3.2.
6. Scaffold the React app (`create-react-app` with TypeScript template) and install frontend dependencies from §3.2.
7. Write `backend/llm/client.py`: a single adapter function `get_llm_response(prompt)` that reads `LLM_PROVIDER` from `.env` and routes to either the Gemini free-tier API or the local Ollama endpoint — every agent calls this one function, never the SDKs directly, so switching providers later never touches agent code.
8. Write `backend/state/schema.py`: define the full shared-state object from §2.3 as nested Pydantic models (`ObjectiveState`, `DataHealthReport`, `FeatureLog`, `ModelLeaderboardEntry`, `GovernanceAudit`, `DecisionLogEntry`, `PipelineState` as the top-level container).
9. Write one `pytest` test that instantiates `PipelineState` with dummy data and asserts it serializes/deserializes to JSON correctly — this is your first passing test and confirms the schema is usable.
10. Write a minimal `backend/graph/pipeline_graph.py`: a LangGraph `StateGraph` with a single placeholder node ("hello_world") that calls `get_llm_response()` with a test prompt and returns its response into the state. Run it once end-to-end from a `if __name__ == "__main__"` block to confirm LangGraph ↔ LLM adapter wiring works before building anything else on top of it.
11. Commit. This phase is done when: Docker works, the LLM adapter successfully gets a response from Gemini (or Ollama), the Pydantic state schema has a passing test, and a trivial LangGraph node can call it and get a response back.

## Phase 1 — Core Pipeline Skeleton + Data Profiling (Week 1)

1. In `pipeline_graph.py`, define all MVP node placeholders (`orchestrator`, `compliance`, `data_profiling`, `feature_engineering`, `model_selection`, `governance`, `reporting`, `checkpoint`) as empty pass-through functions first, and wire the edges exactly as in the flow diagram in §2.2 — including the conditional edge from `governance` back to `feature_engineering`/`model_selection` on failure. Confirm the graph compiles and can be visualized (LangGraph has a built-in `.get_graph().draw_mermaid()` or similar) before writing any real logic.
2. Implement `backend/agents/orchestrator.py`:
   - Accepts the raw NL objective + uploaded file path.
   - Prompts the local LLM to extract: task type, target column candidate(s), optimization priority, candidate protected attributes, domain tag — instruct the model to respond in strict JSON matching your Pydantic schema.
   - Parse and validate the JSON response against the Pydantic model; if parsing fails, retry once with a stricter prompt before falling back to asking the user directly.
   - If required fields are ambiguous/missing, set a flag so the Checkpoint Agent asks the user a clarifying question instead of guessing.
3. Implement `backend/agents/data_profiling.py`:
   - Load the dataset with pandas.
   - Compute: per-column dtype, missingness % per column, a simple MNAR heuristic (e.g., missingness correlated with another column's value), class balance ratio on the target, and a leakage heuristic (columns with near-perfect correlation to the target, or columns whose name matches the target column's domain, e.g., a column literally named `outcome_encoded`).
   - Package results into the `DataHealthReport` Pydantic model.
   - Write 2–3 unit tests using small hand-crafted DataFrames with known issues (e.g., a column that's 60% missing, a leaked column) and assert the report flags them correctly.
4. Build `backend/api/main.py`: a FastAPI app with one POST endpoint `/runs` that accepts a file upload + objective text, creates a `run_id`, kicks off the LangGraph pipeline asynchronously (background task), and returns the `run_id` immediately.
5. Add a GET endpoint `/runs/{run_id}/state` that returns the current `PipelineState` as JSON — this is what the frontend will poll or subscribe to.
6. Manually test via `curl`/Postman: upload a small CSV, confirm a run_id comes back, confirm `/runs/{run_id}/state` shows the objective correctly parsed and the data health report populated after a few seconds.
7. This phase is done when: you can hit one API endpoint with a real CSV + objective and get back a correctly-parsed objective and a correct data health report, with tests passing.

## Phase 2 — Feature Engineering + Model Selection + Cost Awareness (Week 2)

1. Implement `backend/agents/feature_engineering.py`:
   - Given the data health report, propose a shortlist of transformations (e.g., one-hot/target encoding for categoricals, log-transform for skewed numerics, simple interaction terms for the top-2 most important raw features).
   - For each candidate transformation: fit a quick baseline model (e.g., Logistic Regression) with and without it on a held-out validation split, compute the metric delta, and keep it only if it helps or is neutral.
   - Log every accepted/rejected feature with a plain-language reason string (this feeds the UI decision card later).
2. Implement the Governance-consult hook: before finalizing the feature set, call a lightweight function in `governance.py` (`quick_fairness_proxy_check(feature_name, protected_attributes, correlation)`) that flags any feature highly correlated with a protected attribute — this is the "mid-stage consultation" pattern from §2.4, not the full end-of-pipeline audit yet.
3. Implement `backend/agents/model_selection.py`:
   - Define at least 3 candidate model families (Logistic Regression, Random Forest or XGBoost, and one more of your choice).
   - Wire Optuna for hyperparameter search on the top 1–2 candidates, using a multi-objective score (e.g., weighted combination of AUC and a simple fairness proxy, or just AUC first for MVP simplicity with fairness deferred to the full Governance stage).
   - Compute AUC-ROC, F1, and calibration curve data (predicted probability bins vs. observed frequency) for each candidate; assemble into the leaderboard structure.
4. Implement `backend/agents/cost_awareness.py`:
   - Before running a candidate's full tuning search, time a single quick fit on a small data sample, extrapolate to estimate full-run wall-clock time, and attach that estimate to each leaderboard entry so the checkpoint decision card can show a cost-vs-performance trade-off.
5. Implement the Checkpoint Protocol for these two stages: when `feature_engineering` and `model_selection` finish, populate `pending_approval` in the shared state with a structured "decision card" payload (proposed action, reasoning, alternatives considered, cost estimate) instead of auto-advancing the graph.
6. Add a FastAPI endpoint `POST /runs/{run_id}/decision` that accepts `{action: "approve"|"reject"|"counter_propose", note?: string}`; on `approve` the LangGraph run resumes past the checkpoint; on `counter_propose`, route the note back into the relevant agent, which must respond with a structured pros/cons comparison before re-proposing.
7. Write tests: one that forces a rejection at the feature-engineering checkpoint and confirms the pipeline actually pauses and waits (does not silently continue), and one that confirms a resumed run picks up from the correct stage.
8. This phase is done when: you can run the pipeline up to the model leaderboard, see a paused checkpoint via the API, submit an approve/reject/counter-propose decision, and see the pipeline correctly resume or loop.

## Phase 3 — Governance + Compliance (Week 3 — the core novelty phase)

1. Create `backend/config/compliance/finance.yaml` and `generic.yaml` with structured fields: `regulations: [...]`, `fairness_thresholds: {disparate_impact_min: 0.80, equal_opportunity_diff_max: 0.10}`, `requires_explainability: true/false`.
2. Implement `backend/agents/compliance.py`: given the domain tag from the orchestrator's parsed objective, load the matching YAML (default to `generic.yaml` if no match), and inject its thresholds into the state's `governance_audit.compliance_checklist` field before the Governance stage runs.
3. Implement `backend/agents/governance.py` with three distinct audit functions:
   - **Fairness audit** using Fairlearn/AIF360: compute Disparate Impact and Equal Opportunity Difference across each declared protected attribute; compute per-group confusion matrices.
   - **Robustness audit**: perturb 2–3 key numeric features by a synthetic shift (e.g., shift distribution by 1 std dev, or resample from a shifted distribution) and measure AUC degradation.
   - **Stability audit**: bootstrap-resample the training data N times (e.g., N=20), retrain a lightweight version of the winning model each time, and measure variance in the validation metric.
4. Compare each audit's result against both the internal default threshold and the compliance-injected threshold (whichever is stricter wins) — this is what makes compliance genuinely enforce something rather than being decorative.
5. Implement the conditional LangGraph edge: if any audit fails, construct a specific, actionable reason string (e.g., "Disparate Impact = 0.65 on `gender`, below the 0.80 threshold required for this finance-domain run — retry with reweighing or drop feature `zip_code`, which is acting as a proxy") and route back to `feature_engineering` (for a proxy-feature issue) or `model_selection` (for a pure performance/robustness issue).
6. Deliberately construct 2–3 small test datasets where you inject a known fairness problem (e.g., a proxy feature correlated with a protected attribute) and write integration tests asserting: the audit correctly fails, the reason string correctly names the offending feature, and the loop-back actually re-triggers feature engineering.
7. Run the full loop manually end-to-end at least once and record the before/after Disparate Impact numbers — this becomes your real (non-illustrative) headline metric for §1.4 and the resume bullet in §7.
8. This phase is done when: you have at least one recorded, reproducible run where Governance rejected a candidate, the pipeline looped back, and the retried candidate passed — with real numbers, not placeholders.

## Phase 4 — Reporting + Frontend Core (Week 4)

1. Implement `backend/agents/reporting.py`: assemble a Model Card (Google Model Card format — sections: model details, intended use, factors/protected attributes, metrics, ethical considerations, fairness results, deploy/no-deploy recommendation with reasoning) as a Markdown/HTML template populated from the final `PipelineState`.
2. Add an audit-trail exporter that renders `decisions_log` as a readable HTML/PDF timeline.
3. Add FastAPI endpoints: `GET /runs/{run_id}/model-card` and `GET /runs/{run_id}/audit-trail`.
4. Frontend — `PipelineDAG.tsx`: use React Flow to render the stage graph from §2.2; poll (or WebSocket-subscribe to) `/runs/{run_id}/state` and color nodes by status (completed = green, pending approval = pulsing yellow, looped-back = red).
5. Frontend — `DecisionCard.tsx`: render the `pending_approval` payload (proposed action, reasoning, alternatives, cost estimate) with Approve / Reject / Counter-Propose buttons wired to the `/runs/{run_id}/decision` endpoint.
6. Frontend — `ChatPanel.tsx`: a simple scrolling log of agent reasoning text pulled from `decisions_log`, shown alongside the DAG.
7. Frontend — `InsightDashboard.tsx`: Recharts bar chart for the model leaderboard, fairness metric cards (Disparate Impact, Equal Opportunity Difference) with pass/fail coloring, and a simple drift/robustness gauge.
8. Frontend — `AuditTrailViewer.tsx`: a chronological list rendering every `decisions_log` entry with timestamp, stage, proposal, and user action.
9. Wire a WebSocket channel (`backend/api/ws.py`) so the DAG and chat panel update live instead of requiring manual refresh/polling.
10. This phase is done when: you can start a run from the UI, watch the DAG update live, interact with a decision card, and see the model card + audit trail render at the end.

## Phase 5 — Multi-Dataset Validation, Metrics, Polish (Week 5)

1. Download UCI Adult Income and COMPAS (both free, publicly licensed) and run the full pipeline end-to-end on each, plus one synthetic dataset for a third domain (e.g., a synthetic healthcare-flavored dataset via `sklearn.datasets.make_classification` with renamed columns). Prefer Adult Income/COMPAS over LendingClub for your primary demo runs — both are standard, well-documented fairness-auditing benchmarks with existing literature to cite in viva, and using fresh datasets here keeps this project reading as a distinct build rather than a re-skin of your earlier Credit Risk Monitor work.
2. For each run, record: number of self-correction loops, time-to-convergence, whether the run succeeded without any human edit to the model itself, and the before/after fairness metric when a loop-back occurred.
3. Compute your final "autonomous governance intervention rate" across all runs (see §1.4) using real numbers.
4. Fill in the resume bullet template in §7 with your real X/Y/N values.
5. Write the README: setup instructions matching §3.2 exactly, a short "how to run a demo in 5 minutes" section, and a link to/embedded copy of the architecture diagram from §2.2.
6. Do a full manual UX pass: for every checkpoint, read the decision card as if you were a non-technical stakeholder and confirm it makes sense without reading any code.
7. This phase is done when: you have reproducible results across 3 datasets/domains, a filled-in resume bullet, and a README that lets a stranger clone-and-run the project.

## Phase 6 — Stretch (only if time remains)

1. Implement `backend/agents/explainability.py` as its own SHAP-driven node (global summary plot + local force-plot for 2–3 representative predictions), and add its own dashboard panel.
2. Implement `backend/agents/stress_test.py`: generate adversarial/edge-case synthetic samples (e.g., extreme values, rare combinations of protected-attribute groups) and evaluate the final model's behavior on them as a supplementary Governance signal.
3. Wire MLflow properly in `backend/agents/lineage.py`: log every run, parameter set, and model artifact; add a simple "compare runs" view in the frontend.
4. Extend bidirectional consultation: let Model Selection ask Governance a quick "would this model family likely fail robustness?" question before committing to full tuning, not just Feature Engineering asking about proxies.
5. Add 1–2 more domain YAMLs (healthcare/HIPAA, EU/GDPR) fully fleshed out with real threshold values researched from public documentation.

## 6.1 Testing Strategy
- Unit tests per agent (input/output schema conformance) using `pytest`.
- Integration test: full pipeline run on a small synthetic dataset with a known, deliberately-planted fairness issue — assert the Governance loop-back fires.
- Manual UX pass: confirm every checkpoint decision card is understandable without reading code.

## 6.2 Risk Notes
- **LLM non-determinism** in the reasoning/proposal text: keep all numeric decisions (metrics, thresholds, pass/fail) computed by deterministic classical-ML code; the LLM only narrates/reasons over already-computed numbers, never invents them. This is important both for correctness and for your viva defense.
- **Free-tier LLM constraints**: Gemini's free tier has daily request limits, and a local Ollama model can be less reliable at strict JSON formatting than a larger hosted model — mitigate both by keeping prompts short and structured, validating every LLM JSON response against the Pydantic schema, and retrying once with a stricter "respond with ONLY valid JSON, no prose" instruction before falling back to a rules-based default. If you hit Gemini's daily cap mid-testing, switch `.env`'s `LLM_PROVIDER` to `ollama` with zero code changes.
- **Scope creep**: the 12-agent full vision is real but large — the MVP/stretch split in §5 exists specifically so you always have a demoable, defensible product regardless of how much time you actually get.

---

# 7. Resume Bullet Draft (fill in X/Y/N once built)

> Built a multi-agent, human-in-the-loop autonomous ML pipeline (LangGraph) that self-governs model deployment via fairness (Disparate Impact), robustness (synthetic covariate shift), and stability (bootstrap variance) audits — autonomously triggering redesign loops that improved fairness DI from X→Y without direct human edits to the model, across N datasets/domains, with every decision gated through an auditable approve/reject/counter-propose checkpoint.

Keep the numbers honest to your actual test runs (§6, Phase 5) rather than reusing the illustrative 0.68→0.84 example from the design discussion.

# 8. Deliverable Checklist (for resume/report use)
- [ ] Architecture diagram (this document, §2.2, can be redrawn as a polished diagram)
- [ ] Working end-to-end demo with at least one visible governance loop-back
- [ ] Model Cards generated for 2–3 datasets across different domains
- [ ] Quantified novelty metric: governance intervention rate + fairness improvement (X→Y)
- [ ] Audit trail export sample
- [ ] README with setup instructions matching §3.2

---

*End of document — ready for ingestion into Antigravity for scaffolding.*
