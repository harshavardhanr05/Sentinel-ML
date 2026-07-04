# 🛡️ Sentinel-ML

**Autonomous ML Pipeline Agent with Human-in-the-Loop Governance**

A multi-agent, regulation-aware ML governance system built with LangGraph + FastAPI + React. Takes any tabular dataset + a plain-language business objective, runs the full ML lifecycle autonomously, and **pauses at every meaningful decision point** for human approval — with every decision permanently logged in an auditable trail.

---

## What Makes This Different

| Feature | What it means |
|---------|--------------|
| **Checkpoint-driven autonomy** | Not "human in the loop" as an afterthought — formal Propose → Pause → Justify → Log protocol at every stage |
| **Self-governance** | The system can refuse to deploy its own model and send work *back* to earlier stages with a specific reason |
| **Regulation-aware** | Compliance Agent dynamically injects ECOA/GDPR/HIPAA thresholds based on declared domain — just add a YAML |
| **Bidirectional consultation** | Feature Engineering Agent asks Governance "will this feature trip a fairness check?" before committing |

---

## Architecture

```
User Objective + Dataset
        │
        ▼
 Orchestrator → Compliance → Data Profiling → [CHECKPOINT]
        → Feature Engineering → [CHECKPOINT]
        → Model Selection + Cost Estimate → [CHECKPOINT]
        → Governance Audit (Fairness + Robustness + Stability)
              FAIL → loops back to Feature Eng. or Model Selection
              PASS → Explainability → Reporting → DEPLOY / NO-DEPLOY
```

**12 logical agents:** Orchestrator, Compliance, Data Profiling, Feature Engineering, Model Selection, Cost Awareness, Governance, Explainability, Reporting, Checkpoint/Human-Interface (embedded in graph/API), Version Control/Lineage (stretch), Stress Test (stretch).

---

## Tech Stack (all free / open-source)

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph |
| LLM (reasoning only) | Gemini 2.5 Flash (free tier) |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Hyperparameter search | Optuna |
| Explainability | SHAP |
| Fairness | Fairlearn |
| Backend API | FastAPI + WebSockets |
| Frontend | React + Vite + Tailwind + ReactFlow + Recharts |
| Persistence | SQLite (every node saves state → crash-safe) |
| Experiment tracking | MLflow |

---

## Quick Start (Windows)

### Prerequisites
- Python 3.11+
- Node.js 20+
- Free Gemini API key from https://aistudio.google.com (no billing required)

### 1. Clone & setup

```powershell
git clone <repo>
cd Sentinel-ML

# Copy env template
Copy-Item .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Python venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run backend

```powershell
.\venv\Scripts\Activate.ps1
uvicorn backend.api.main:app --reload --port 8000
```

### 3. Run frontend

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### 4. (Optional) Docker

```powershell
# Copy .env.example → .env and fill in GEMINI_API_KEY
docker-compose up
```

---

## 5-Minute Demo

1. Open http://localhost:5173
2. Type objective: `"Predict loan default. Minimize false negatives. Must be fair across gender. Domain: finance."`
3. Upload `data/samples/adult_income.csv` (download from UCI ML Repository)
4. Click **Start Pipeline**
5. Watch the DAG update live — pipeline will pause at each checkpoint
6. Review each **Decision Card** and click Approve/Reject/Suggest Alternative
7. Watch the Governance Agent run fairness/robustness/stability audits
8. If DI < 0.80: observe the automatic loopback to Feature Engineering
9. View the final Model Card and Audit Trail

---

## Running Tests

```powershell
.\venv\Scripts\Activate.ps1
pytest backend/tests/ -v
```

Key tests:
- `test_schema.py` — state serialization round-trips
- `test_data_profiling.py` — missingness, leakage, imbalance detection
- `test_governance.py` — fairness audit with planted bias, loopback routing

---

## Compliance Domains

| Domain | Regulations | File |
|--------|-------------|------|
| `finance` | ECOA 4/5ths rule, EEOC, GDPR, FCRA | `backend/config/compliance/finance.yaml` |
| `generic` | GDPR, EU AI Act | `backend/config/compliance/generic.yaml` |
| `healthcare` | HIPAA, DPDP (stub) | `backend/config/compliance/healthcare.yaml` |

To add a new domain: create `backend/config/compliance/<domain>.yaml`. No code change needed.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/runs` | Start pipeline (upload CSV + objective) |
| `GET` | `/runs/{id}/state` | Get current PipelineState |
| `POST` | `/runs/{id}/decision` | Submit checkpoint decision |
| `POST` | `/runs/{id}/objective` | Update parsed objective (after ambiguity) |
| `GET` | `/runs/{id}/model-card` | Download Model Card markdown |
| `GET` | `/runs/{id}/audit-trail` | Download Audit Trail HTML |
| `WS` | `/ws/{id}` | Live state updates |

---

## Project Structure

```
Sentinel-ML/
├── backend/
│   ├── agents/          # 10 agent modules (one file per agent)
│   ├── api/             # FastAPI app + WebSocket manager
│   ├── config/          # Compliance YAML configs
│   ├── graph/           # LangGraph StateGraph definition
│   ├── llm/             # Gemini adapter
│   ├── state/           # Pydantic schema + SQLite store
│   └── tests/           # pytest unit + integration tests
├── frontend/
│   └── src/
│       ├── api/         # Axios client + WebSocket hook
│       └── components/  # PipelineDAG, DecisionCard, InsightDashboard, etc.
├── data/samples/        # Sample datasets
├── .env.example         # Environment template
├── requirements.txt
└── docker-compose.yml
```

---

## Resume Bullet (fill in X→Y once you have real numbers)

> Built a multi-agent, human-in-the-loop autonomous ML pipeline (LangGraph) that self-governs model deployment via fairness (Disparate Impact), robustness (synthetic covariate shift), and stability (bootstrap variance) audits — autonomously triggering redesign loops that improved DI from **X→Y** without direct human edits to the model, across **N** datasets/domains, with every decision gated through an auditable approve/reject/counter-propose checkpoint.

---

## License

MIT
