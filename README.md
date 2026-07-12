# 🛡️ Sentinel-ML — AI-Governed ML Automation Platform

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangGraph-Multi--Agent-FF4F00?logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
</p>

> **A production-grade, human-in-the-loop machine learning pipeline orchestration system powered by LangGraph, Agentic AI, and modern DevOps tooling.**

Sentinel-ML automates the entire ML lifecycle — from raw data upload to deployment decision — while enforcing fairness, compliance, and full audit traceability at every step. Every agent action pauses for your approval before proceeding, giving you complete control over automated ML decisions.

---

## 🗂️ Table of Contents

1. [Key Features](#-key-features)
2. [Tech Stack & Architecture](#-tech-stack--architecture)
3. [System Architecture](#-system-architecture)
4. [Pipeline Walkthrough](#-pipeline-walkthrough)
5. [Case Study 1 — Student Depression Classification](#-case-study-1--student-depression-classification)
6. [Case Study 2 — House Price Regression](#-case-study-2--house-price-regression)
7. [UI Components Reference](#-ui-components-reference)
8. [Compliance & Governance Engine](#-compliance--governance-engine)
9. [Supported Datasets (Samples)](#-supported-datasets-samples)
10. [Project Structure](#-project-structure)
11. [Setup & Running Locally](#-setup--running-locally)
12. [Environment Variables](#-environment-variables)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Human-in-the-Loop** | Every agent decision (target selection, feature engineering, model choice, governance) pauses and awaits explicit user Approve / Reject / Suggest Alt. |
| **Multi-Agent Orchestration** | 8 specialized agents (Orchestrator, Compliance, Data Profiling, Feature Engineering, Model Selection, Governance, Explainability, Reporting) coordinated via a LangGraph state machine |
| **15+ ML Algorithms** | Random Forest, XGBoost, LightGBM, Gradient Boosting, Extra Trees, Voting Ensemble, Stacking Ensemble, Ridge, ElasticNet, KNN, SVM, MLP, AdaBoost, Decision Tree, Huber Regressor |
| **Automated Fairness Auditing** | Disparate Impact, Equal Opportunity Difference, AUC Degradation % — enforced via `fairlearn` against configurable thresholds |
| **SHAP Explainability** | Global feature importance bar charts + local correct/misclassified example breakdowns |
| **Governance Audit Trail** | Full immutable log of every agent action, decision checkpoint, and feature selection step |
| **Compliance Profiles** | Pre-built YAML profiles for Finance, Healthcare, and Generic domains (GDPR, EU AI Act, HIPAA) |
| **AI Code Assistant** | In-pipeline chat interface to request custom feature engineering or model code from the configured LLM |
| **MLflow Integration** | Experiment tracking with run IDs, metrics, and artifact logging for every trained model |
| **Rich Data Analytics Dashboard** | Auto-generated EDA charts — histograms, bar charts, area charts, radar, correlation heatmaps, scatter plots — tailored by task type |
| **Real-time Pipeline DAG** | Live visual graph showing current node, completed stages, and agent status |
| **Model Card & Report Export** | Auto-generated model card with metrics, fairness scores, feature importance, and deployment recommendation |
| **Docker Support** | One-command deployment via `docker-compose.yml` |

---

## 🧰 Tech Stack & Architecture

### Backend

| Layer | Technology | Version |
|---|---|---|
| **Agent Orchestration** | LangGraph | >= 0.2.0 |
| **LLM Integration** | Model-Agnostic via LangChain (Gemini, Groq, etc.) | >= 0.1.0 |
| **LLM Framework** | LangChain Core | >= 0.3.0 |
| **Alt LLM** | OpenAI SDK | >= 1.0.0 |
| **API Framework** | FastAPI | >= 0.111.0 |
| **ASGI Server** | Uvicorn (with standard extras) | >= 0.30.0 |
| **Real-time Comms** | WebSockets | >= 12.0 |
| **Data Validation** | Pydantic v2 + pydantic-settings | >= 2.7.0 |
| **ORM / Persistence** | SQLAlchemy + aiosqlite (SQLite) | >= 2.0.0 |
| **Experiment Tracking** | MLflow | >= 2.13.0 |
| **Configuration** | PyYAML + python-dotenv | >= 6.0 / 1.0 |
| **PDF Reporting** | Jinja2 + WeasyPrint | >= 3.1 / 62.0 |

### Compatible LLM Models

Sentinel-ML is model-agnostic and interfaces with a wide variety of LLM providers. Current natively supported platforms include:
- **Google Gemini**: `gemini-2.5-flash`, `gemini-1.5-pro`, etc.
- **Groq**: Fast inference for models like `llama-3.3-70b-versatile`
- **Cerebras**: High-speed inference for models like `gpt-oss-120b`
- **xAI / Grok**: `grok-4.3`
- **Ollama**: Local models like `qwen2.5:1.5b`, `llama3.2:1b`
- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

Set your active provider via `LLM_PROVIDER` and configure the respective API keys and models in `.env`.

### ML / Data Science

| Library | Purpose | Version |
|---|---|---|
| **scikit-learn** | Core ML algorithms (RF, Ridge, SVM, MLP, etc.) | >= 1.4.0 |
| **XGBoost** | Gradient boosting (XGB) | >= 2.0.0 |
| **LightGBM** | Fast gradient boosting | >= 4.3.0 |
| **Optuna** | Hyperparameter optimization | >= 3.6.0 |
| **SHAP** | Model explainability / feature attribution | >= 0.45.0 |
| **fairlearn** | Fairness metrics & bias detection | >= 0.10.0 |
| **pandas** | Data manipulation | >= 2.2.0 |
| **numpy** | Numerical computing | >= 1.26.0 |
| **imbalanced-learn** | SMOTE / class imbalance handling | >= 0.12.0 |

### Frontend

| Technology | Purpose |
|---|---|
| **React 18 + TypeScript** | UI framework |
| **Vite** | Build tool & dev server |
| **Recharts** | Data visualization (charts, radar, area, bar) |
| **Lucide React** | Icon library |
| **Vanilla CSS** | Custom dark-theme design system |

---

## 🏛️ System Architecture

![System Architecture](assets/screenshots/architecture.png)

*High-level architecture of the Sentinel-ML platform, showing the React frontend, FastAPI backend, LangGraph state machine, and ML engine services.*

### Agent Pipeline Flow

![Agent Pipeline Flow](assets/screenshots/pipeline_flow.png)

*The step-by-step flowchart of the multi-agent orchestration, highlighting the checkpoints where human approval is required.*

---

## 📸 Platform Overview

![Sentinel-ML Landing Page](assets/screenshots/01_home_screen_1783764354249.png)
*The Sentinel-ML dashboard: Start a new pipeline with a business objective and a CSV/Parquet dataset.*

---

## 📋 Pipeline Walkthrough

### Step 1 — Start a New Pipeline Run

Upload any CSV or Parquet file and describe your goal in plain English.

![New Run Form](assets/screenshots/01_home_screen_1783764354249.png)

*The intake form supports free-text business objectives like "Predict student depression. Must be fair across age and gender."*

---

### Step 2 — Objective Intake Decision

The AI analyzes your objective and proposes a target column. You review and approve before any processing begins.

![Objective Intake Decision Card](assets/screenshots/decision_card_objective_intake_1783830764376.png)

*The Decision Card shows the proposed action, the agent reasoning, and alternatives considered. You can Approve, Reject, or Suggest an alternative.*

---

### Step 3 — Live Pipeline DAG

A real-time directed acyclic graph shows which agent is running, which are complete, and the entire pipeline topology.

![Pipeline DAG](assets/screenshots/03_pipeline_dag_1783764564796.png)

*Live DAG visualization with node status badges. Blue = current, Green = completed, Grey = pending.*

---

### Step 4 — Data Profiling Decision

The Data Profiling agent generates column statistics, detects nulls, cardinality, skewness, and flags potential data leakage columns.

![Data Profiling Decision Card](assets/screenshots/04_decision_card_data_profiling_1783764859925.png)

*Decision card for data profiling: shows dataset shape, missingness severity, and leakage warnings.*

---

### Step 5 — Data Analysis Dashboard

A rich, multi-chart analytics dashboard auto-generated from your dataset. Charts are chosen dynamically based on the task type (classification vs. regression) and feature types.

**Classification EDA (Student Depression Dataset)**

![Data Analysis Dashboard](assets/screenshots/data_analysis_charts_1783830851768.png)

**Regression EDA (House Price Dataset)**

![EDA Dashboard Regression](assets/screenshots/eda_charts_1783831569348.png)

*Charts include: target distribution, categorical frequency bars, numerical histograms (area charts for continuous targets), feature correlation bars, radar chart, and scatter plots.*

---

### Step 6 — Feature Engineering Decision

The Feature Engineering agent encodes categoricals, scales numerics, handles missing values, detects fairness proxy variables, and selects the optimal feature subset.

![Feature Engineering Decision Card](assets/screenshots/07_decision_card_feature_engineering_1783764999430.png)

*Shows accepted features, rejected features, and governance-flagged fairness proxies before proceeding to model training.*

---

### Step 7 — AI Code Assistant (Chat Panel)

At any stage you can open the AI chat panel and request custom code — for example, asking the agent to write a custom feature transformation or explain a preprocessing choice.

**Classification Pipeline — AI Code Request**

![AI Code Request Classification](assets/screenshots/loan_06_ai_code_request_1783766937764.png)

**Regression Pipeline — AI Code Request**

![AI Code Request Regression](assets/screenshots/reg_06_ai_code_request_1783765828959.png)

---

### Step 8 — Model Selection Decision

The agent trains **15+ ML algorithms** simultaneously, evaluates them on validation data, and presents a ranked leaderboard. It selects the best model based on a fairness-weighted multi-objective score.

![Model Selection Decision Card](assets/screenshots/08_decision_card_modeling_1783765041232.png)

*Decision card showing the selected model, evaluation metrics (AUC/R2, F1/RMSE), and fairness score.*

---

### Step 9 — Model Leaderboard

The Leaderboard tab shows all trained models ranked by primary metric, with train/validation comparison to detect overfitting.

**Classification Leaderboard**

![Model Leaderboard Classification](assets/screenshots/09_leaderboard_dashboard_1783765065206.png)

**Regression Leaderboard**

![Model Leaderboard Regression](assets/screenshots/leaderboard_regression_1783831688320.png)

---

### Step 10 — Governance & Fairness Audit

The Governance agent runs three fairness metrics against configurable thresholds. If any metric fails, it triggers automatic model redesign loops (up to 3 attempts) before recommending NO_DEPLOY.

![Governance Decision Card](assets/screenshots/10_decision_card_governance_1783765276607.png)

![Governance Audit Detail](assets/screenshots/governance_audit_1783831083770.png)

---

### Step 11 — SHAP Explainability

The Explainability panel provides global feature importance (SHAP values) and local examples (correct positives, correct negatives, misclassified samples) with AI-generated plain-English narration.

**Classification — SHAP**

![SHAP Explainability Classification](assets/screenshots/explainability_shap_1783831216238.png)

**Regression — SHAP**

![SHAP Explainability Regression](assets/screenshots/reg_10_explainability_1783766056409.png)

---

### Step 12 — Governance Audit Trail

The Audit Trail tab provides a complete immutable log: every decision checkpoint, feature selection step, and agent activity entry with timestamps.

**Classification Audit Trail**

![Audit Trail Classification](assets/screenshots/12_audit_trail_dashboard_1783765386289.png)

**Regression Audit Trail**

![Audit Trail Regression](assets/screenshots/audit_trail_regression_1783831801258.png)

---

### Step 13 — Agent Reasoning Log

The Reasoning tab shows the sequential chain-of-thought log: which agent ran, what it decided, and why — with the user approve/reject at each checkpoint.

![Agent Reasoning Log](assets/screenshots/03_data_profiling_agent_chat_1783765507922.png)

---

### Step 14 — Reporting & Model Card

The Reporting agent generates a deployment recommendation with a full model card.

![Reporting Decision Card](assets/screenshots/reporting_nodeploy_1783831198586.png)

![Pipeline Completed](assets/screenshots/pipeline_completed_1783831331832.png)

---

## 🎓 Case Study 1 — Student Depression Classification

**Dataset**: student_dataset_10000_rows.csv (10,000 rows x 18 features)  
**Objective**: Predict student depression risk. Must be fair across age and gender. Domain: healthcare.  
**Task Type**: Binary Classification  
**Compliance Profile**: healthcare.yaml (HIPAA + EU AI Act)

### Results

| Metric | Value |
|---|---|
| **Selected Model** | XGBoost |
| **Validation AUC** | 0.997 |
| **F1 Score (Val)** | 0.997 |
| **Top Features** | sleep_hours, anxiety_level, stress_level, daily_social_media_hours, physical_activity |
| **Fairness Proxies Rejected** | 2 features flagged and removed |
| **Governance Status** | Fairness audit required 3 redesign loops; final: NO_DEPLOY pending review |
| **Agent Activity Entries** | 77 events logged |

### Key Insights from SHAP

- **sleep_hours** is the dominant predictor — insufficient sleep is the strongest signal for depression risk
- **anxiety_level** and **stress_level** compound the effect
- **physical_activity** is protective — students who exercise show lower depression risk
- **daily_social_media_hours** increases vulnerability via comparison-driven stress

---

## 🏠 Case Study 2 — House Price Regression

**Dataset**: house_price_regression_dataset.csv (1,000 rows x 8 features)  
**Objective**: Predict house prices based on features like square footage, bedrooms, and neighborhood quality.  
**Task Type**: Continuous Regression  
**Compliance Profile**: generic.yaml

### Results

| Metric | Value |
|---|---|
| **Selected Model** | Linear Regression |
| **Validation R2** | 0.9984 |
| **Validation RMSE** | 10,196.58 |
| **Train R2** | 0.9986 |
| **Train RMSE** | 9,623.40 |
| **Top Features** | Square_Footage, Neighborhood_Quality, Num_Bedrooms, Year_Built, Lot_Size |
| **Runners-up** | Ridge (R2 0.9984), Huber Regressor (R2 0.9984), Gradient Boosting (R2 0.9965) |
| **Total Models Evaluated** | 17 |
| **Pipeline Status** | COMPLETED |

---

## 🖥️ UI Components Reference

| Component | File | Description |
|---|---|---|
| **PipelineDAG** | PipelineDAG.tsx | Real-time directed graph of pipeline stages with node status |
| **DecisionCard** | DecisionCard.tsx | Approve/Reject/Suggest Alt. modal for agent checkpoints |
| **DataAnalysisDashboard** | DataAnalysisDashboard.tsx | Auto-generated EDA charts (adapts to classification vs. regression) |
| **InsightDashboard** | InsightDashboard.tsx | Model leaderboard, governance metrics, cost/performance |
| **ExplainabilityPanel** | ExplainabilityPanel.tsx | SHAP global importance + local example breakdowns |
| **AuditTrailViewer** | AuditTrailViewer.tsx | Immutable audit log with Decision Checkpoints, Feature Selection, Agent Activity |
| **ChatPanel** | ChatPanel.tsx | AI code assistant — live LLM chat with code generation |

---

## 🔒 Compliance & Governance Engine

![Governance Engine Infographic](assets/screenshots/governance_engine.png)

*The Sentinel-ML Governance Engine enforces domain-specific compliance rules and runs fairness audits before deployment.*

Sentinel-ML ships with three domain-specific compliance profiles in backend/config/compliance/:

### healthcare.yaml
- **Regulations**: HIPAA, EU AI Act (High Risk), GDPR
- **Protected attributes**: age, gender, race, ethnicity
- **Fairness thresholds**: Disparate Impact >= 0.80, Equal Opp. Diff <= 0.10, AUC Degradation <= 10%
- **Banned features**: SSN, diagnosis codes used as raw features

### finance.yaml
- **Regulations**: EU AI Act, GDPR Right to Explanation
- **Protected attributes**: age, gender, marital_status, zip_code
- **Fairness thresholds**: Disparate Impact >= 0.80, Equal Opp. Diff <= 0.10
- **Additional**: Credit score proxy detection

### generic.yaml
- **Regulations**: GDPR Data Minimization, GDPR Right to Explanation
- **General-purpose** fairness checks with relaxed thresholds

### Governance Audit Metrics

| Metric | Threshold | Interpretation |
|---|---|---|
| Disparate Impact | >= 0.80 | Higher = fairer |
| Equal Opp. Difference | <= 0.10 | Lower = fairer |
| AUC Degradation % | <= 10% | Lower = fairer |

If any metric fails, the pipeline:
1. Triggers an automatic model redesign loop (removes proxy features, re-trains)
2. Retries up to **3 times**
3. If still failing, recommends **NO_DEPLOY** and halts with full audit evidence

---

## 📦 Supported Datasets (Samples)

The data/samples/ directory ships with 13 example datasets:

| Dataset | Type | Rows | Use Case |
|---|---|---|---|
| student_dataset_10000_rows.csv | Classification | 10,000 | Mental health / depression prediction |
| house_price_regression_dataset.csv | Regression | 1,000 | Real estate price prediction |
| diabetes_prediction_dataset.csv | Classification | ~100K | Healthcare risk prediction |
| Loan_Default.csv | Classification | ~150K | Finance: loan default risk |
| adult_income.csv | Classification | ~48K | Income prediction (fairness benchmark) |
| titanic.csv | Classification | 891 | Classic survival prediction |
| winequality-red.csv | Regression/Clf | 1,599 | Wine quality scoring |
| Car_Price_Prediction.csv | Regression | ~300 | Automotive pricing |
| kidney_disease.csv | Classification | 400 | Medical diagnosis |
| Teen_Mental_Health_Dataset.csv | Classification | ~1K | Adolescent health screening |
| customer_shopping_behavior.csv | Regression/Clf | ~100K | Retail analytics |
| ai-impact-jobs-layoff-risk-dataset.csv | Classification | ~100K | HR / workforce risk |
| gta_v_worldwide_sales_player_analytics_2013_2026.csv | Regression | ~350K | Gaming analytics |

---

## 🗂️ Project Structure

```
Sentinel-ML/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py        # Entry point agent, task type detection
│   │   ├── compliance.py          # Domain profile loader, constraint enforcement
│   │   ├── data_profiling.py      # Column stats, null detection, leakage flags
│   │   ├── feature_engineering.py # Encoding, scaling, SMOTE, proxy detection
│   │   ├── model_selection.py     # 15+ algorithms, cross-validation, ranking
│   │   ├── governance.py          # Fairness metrics, redesign loops
│   │   ├── explainability.py      # SHAP values, local examples
│   │   ├── reporting.py           # Model card, deployment recommendation
│   │   └── cost_awareness.py      # Cost/performance trade-off estimation
│   ├── api/
│   │   └── main.py                # FastAPI endpoints + WebSocket handler
│   ├── config/
│   │   └── compliance/
│   │       ├── finance.yaml
│   │       ├── healthcare.yaml
│   │       └── generic.yaml
│   ├── graph/                     # LangGraph graph definition
│   ├── llm/                       # Model-agnostic LLM wrappers
│   ├── state/
│   │   └── schema.py              # PipelineState Pydantic model
│   └── export/                    # PDF/HTML report generation
│
├── frontend/
│   └── src/
│       ├── App.tsx                # Main app, routing, WebSocket client
│       ├── api/client.ts          # REST + WS API client
│       └── components/
│           ├── PipelineDAG.tsx
│           ├── DecisionCard.tsx
│           ├── DataAnalysisDashboard.tsx
│           ├── InsightDashboard.tsx
│           ├── ExplainabilityPanel.tsx
│           ├── AuditTrailViewer.tsx
│           └── ChatPanel.tsx
│
├── data/
│   ├── samples/                   # 13 example datasets
│   └── uploads/                   # User-uploaded files (runtime)
│
├── artifacts/                     # MLflow artifacts storage
├── docker-compose.yml
├── Dockerfile.backend
├── requirements.txt
└── run.bat                        # Windows quick-start script
```
---

## 🚀 Setup & Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- API key for your chosen LLM provider

### 1. Clone and Set Up Backend

`ash
git clone <repo-url>
cd Sentinel-ML

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Set Up Frontend

`ash
cd frontend
npm install
```
### 3. Configure Environment

`ash
cp .env.example .env
# Edit .env with your API keys (see below)
```
### 4. Run the Platform

**Option A — Windows Quick Start**
`at
run.bat
```
**Option B — Manual**
`ash
# Terminal 1 — Backend
uvicorn backend.api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```
**Option C — Docker**
`bash
docker-compose up --build
```
### 5. Open the App

Navigate to http://localhost:5173 and click **New Run** to start a pipeline.

---

## 🔑 Environment Variables

```env
# ── LLM (at least one required) ───────────────────
LLM_PROVIDER=cerebras # Options: cerebras, groq, grok, gemini, ollama, openai

GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile

CEREBRAS_API_KEY=your_cerebras_key
CEREBRAS_MODEL=gpt-oss-120b

GROK_API_KEY=your_grok_key
GROK_MODEL=grok-4.3

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b

# Backend settings
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
DATABASE_URL=sqlite+aiosqlite:///./pipeline_runs.db

# MLflow (optional)
MLFLOW_TRACKING_URI=./mlruns

# CORS
FRONTEND_URL=http://localhost:5173
```
---

## 📊 Technical Specifications Summary

| Spec | Value |
|---|---|
| **ML Algorithms Supported** | 17 (including ensembles) |
| **Max Governance Redesign Loops** | 3 per pipeline run |
| **State Persistence** | SQLite (async, via aiosqlite + SQLAlchemy) |
| **Real-time Updates** | WebSocket push from backend |
| **Supported File Formats** | CSV, Parquet |
| **Task Types** | Binary Classification, Multi-class Classification, Regression |
| **Fairness Metrics** | Disparate Impact, Equal Opportunity Difference, AUC Degradation % |
| **Explainability** | SHAP (global + local) |
| **Compliance Domains** | Healthcare (HIPAA), Finance (EU AI Act), Generic (GDPR) |
| **Experiment Tracking** | MLflow |
| **Deployment** | Docker + docker-compose, or local dev mode |
| **Frontend** | React 18 + TypeScript + Vite |
| **API** | REST + WebSocket (FastAPI) |
| **Python** | >= 3.10 |
| **Node.js** | >= 18 |

---

*Sentinel-ML — Bringing governance, fairness, and transparency to automated machine learning.*

*Built with LangGraph · Agentic AI · FastAPI · React · SHAP · fairlearn*

