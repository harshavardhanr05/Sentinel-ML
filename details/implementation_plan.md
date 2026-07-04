# Sentinel-ML: Autonomous ML Pipeline Agent with Human-in-the-Loop Governance
## Implementation Plan (Revised)

**Project root:** `c:\Users\harsh\Code\Sentinel-ML\`
**LLM:** Gemini free tier (`gemini-2.5-flash`) — no Ollama needed
**OS:** Windows (PowerShell)

---

## Decisions Resolved from Feedback

| # | Issue | Decision |
|---|-------|----------|
| 1 | Missing persistence layer | ✅ Add `backend/state/store.py` — saves `PipelineState` to SQLite after every node; loaded on `GET /runs/{id}/state` |
| 2 | Explainability consistency | ✅ **Pull `explainability.py` into MVP** (cheap to add, meaningful deliverable). `stress_test.py` and `lineage.py` remain stretch. |
| 3 | 3 compliance YAMLs scope risk | ✅ Build all 3 (finance, healthcare, generic) — but healthcare.yaml can be a thin stub in Week 3; flesh out only if time permits |
| 4 | Windows environment | ✅ No Ollama. LLM adapter = Gemini only. All shell commands use PowerShell syntax. |
| 5 | Checkpoint Agent not a file | ✅ Intentional — embedded in `backend/graph/pipeline_graph.py` (checkpoint nodes) + `backend/api/main.py` (decision endpoint). Not a separate module. |

---

## Architecture Summary

**12 logical agents → 10 backend modules + 2 embedded in graph/API:**

| Agent | File | Phase |
|-------|------|-------|
| Orchestrator | `agents/orchestrator.py` | 1 |
| Compliance | `agents/compliance.py` | 3 |
| Data Profiling | `agents/data_profiling.py` | 1 |
| Feature Engineering | `agents/feature_engineering.py` | 2 |
| Model Selection + Cost | `agents/model_selection.py` + `agents/cost_awareness.py` | 2 |
| Governance | `agents/governance.py` | 3 |
| Explainability | `agents/explainability.py` | **MVP (Phase 4)** |
| Reporting | `agents/reporting.py` | 4 |
| Checkpoint / Human-Interface | Embedded in `graph/pipeline_graph.py` + `api/main.py` | 1–4 |
| Version Control / Lineage | `agents/lineage.py` | **Stretch (Phase 6)** |
| Synthetic Stress-Test | `agents/stress_test.py` | **Stretch (Phase 6)** |

---

## Full File List (MVP)

### Backend
```
backend/
├── llm/
│   └── client.py                  # Gemini adapter (LLM_PROVIDER=gemini hardcoded default)
├── state/
│   ├── schema.py                  # All Pydantic models
│   └── store.py                   # [NEW] SQLite read/write for PipelineState (NFR-3)
├── graph/
│   └── pipeline_graph.py          # LangGraph StateGraph + all node wiring + Checkpoint protocol
├── agents/
│   ├── orchestrator.py
│   ├── compliance.py
│   ├── data_profiling.py
│   ├── feature_engineering.py
│   ├── model_selection.py
│   ├── cost_awareness.py
│   ├── governance.py
│   ├── explainability.py          # MVP (SHAP global + local)
│   └── reporting.py
├── config/
│   └── compliance/
│       ├── finance.yaml           # Full thresholds
│       ├── healthcare.yaml        # Thin stub in Week 3, flesh out in Phase 6
│       └── generic.yaml           # Full thresholds
├── api/
│   ├── main.py                    # FastAPI app + decision endpoint (Checkpoint protocol)
│   └── ws.py                      # WebSocket live state push
└── tests/
    ├── test_schema.py
    ├── test_data_profiling.py
    ├── test_governance.py
    └── test_pipeline_integration.py
```

### `backend/state/store.py` — Persistence Layer (NFR-3)
```python
# Responsibilities:
# - save_state(run_id, state: PipelineState) → upserts to pipeline_runs.db (SQLite)
# - load_state(run_id) → PipelineState | None
# - list_runs() → list[{run_id, stage, created_at}]
# Called by every LangGraph node after it mutates state, and by GET /runs/{id}/state
```

### Frontend
```
frontend/src/
├── App.tsx
├── api/
│   └── client.ts                  # axios wrapper + WebSocket hook
└── components/
    ├── PipelineDAG.tsx             # ReactFlow DAG, live status colors
    ├── DecisionCard.tsx            # Checkpoint UI: Approve/Reject/Counter-Propose
    ├── ChatPanel.tsx               # Scrolling agent reasoning log
    ├── InsightDashboard.tsx        # Leaderboard + fairness cards + gauge
    ├── AuditTrailViewer.tsx        # Decision timeline
    └── ExplainabilityPanel.tsx     # SHAP plots (global + local)
```

### Infrastructure
```
.env.example
.gitignore
docker-compose.yml
requirements.txt
README.md
```

---

## Phase Plan

### Phase 0 — Environment & Skeleton
1. Create full directory skeleton
2. Init git repo + `.gitignore`
3. `requirements.txt` (no Ollama packages)
4. `backend/llm/client.py` — Gemini only, reads `GEMINI_API_KEY` from `.env`
5. `backend/state/schema.py` — full Pydantic state
6. `backend/state/store.py` — SQLite persistence
7. `backend/graph/pipeline_graph.py` — hello-world node, LLM wiring test
8. `docker-compose.yml` scaffold
9. React app scaffold (Vite + TypeScript + Tailwind + ReactFlow)
10. First passing test: schema serialization round-trip

### Phase 1 — Core Pipeline + Data Profiling
1. All LangGraph nodes as pass-through placeholders, edges wired
2. `agents/orchestrator.py` — NL → ObjectiveState via Gemini
3. `agents/data_profiling.py` — missingness, leakage, imbalance, MNAR
4. FastAPI: `POST /runs`, `GET /runs/{id}/state` (loads from SQLite)
5. Unit tests: data_profiling with hand-crafted DFs

### Phase 2 — Feature Engineering + Model Selection + Cost
1. `agents/feature_engineering.py` — transforms + metric-delta validation
2. Governance consult hook: `quick_fairness_proxy_check()` in governance.py
3. `agents/model_selection.py` — LR, RF/XGB, 3rd family; Optuna; leaderboard
4. `agents/cost_awareness.py` — wall-clock estimator
5. Checkpoint protocol: `pending_approval` in state, persisted to SQLite
6. `POST /runs/{id}/decision` endpoint (Approve/Reject/Counter-Propose)

### Phase 3 — Governance + Compliance
1. `config/compliance/finance.yaml` + `generic.yaml` (full); `healthcare.yaml` (stub)
2. `agents/compliance.py` — load YAML → inject thresholds
3. `agents/governance.py` — Fairness (DI, EOD, per-group CM) + Robustness + Stability
4. Conditional LangGraph edge: fail → loop back with actionable reason string
5. Integration tests with planted fairness issues

### Phase 4 — Reporting + Explainability + Frontend
1. `agents/explainability.py` — SHAP global summary + local force plots
2. `agents/reporting.py` — Google Model Card + audit trail HTML
3. FastAPI: `/model-card`, `/audit-trail`, `/explainability`
4. WebSocket channel (`ws.py`)
5. All 5 React components + `ExplainabilityPanel.tsx`

### Phase 5 — Validation + Polish
1. Run on UCI Adult Income + COMPAS
2. Record governance intervention rate + before/after DI
3. README (Windows setup, 5-min demo instructions)
4. UX polish pass

### Phase 6 — Stretch
1. `agents/stress_test.py` — adversarial samples
2. `agents/lineage.py` — MLflow integration
3. Healthcare/HIPAA YAML fully fleshed out
4. Bidirectional Model Selection ↔ Governance consultation

---

## Verification Plan

### Automated
```powershell
pytest backend/tests/ -v
```
- Schema round-trip
- Data profiling flags (missingness, leakage, imbalance)
- Governance loop-back fires on planted fairness issue

### Manual
- `docker-compose up` → upload Adult Income CSV → watch DAG live → interact with checkpoint cards → read Model Card + audit trail
- Record real DI before/after numbers for resume bullet

---

> [!NOTE]
> **Windows setup note:** All shell commands use PowerShell. Python venv activation uses `.\venv\Scripts\Activate.ps1`. No `curl | sh` Ollama install ever appears. React uses Vite (`npm create vite`) not CRA.
