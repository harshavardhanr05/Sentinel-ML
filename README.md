<div align="center">

<img src="https://img.shields.io/badge/Sentinel--ML-ML%20Automation%20Platform-6366f1?style=for-the-badge&logo=shield&logoColor=white" alt="Sentinel-ML"/>

# Sentinel-ML

### An Agentic, Human-in-the-Loop ML Automation Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=white)](https://react.dev)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Pipeline-FF6B35?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*From raw CSV to a governed, explainable, production-ready ML model — with a human checkpointing every decision.*

</div>

---

## Table of Contents

- [What is Sentinel-ML?](#-what-is-sentinel-ml)
- [Key Features](#-key-features)
- [Architecture & Workflow](#-architecture--workflow)
- [Detailed Agent Activity & Feature Logic](#-detailed-agent-activity--feature-logic)
- [Governance & Reporting Mechanism](#-governance--reporting-mechanism)
- [UI Walkthrough — Classification (Loan Default prediction)](#-ui-walkthrough--classification-loan-default-prediction)
- [UI Walkthrough — Regression (Car Price prediction)](#-ui-walkthrough--regression-car-price-prediction)
- [Supported LLM Providers](#-supported-llm-providers)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)

---

## 🔍 What is Sentinel-ML?

**Sentinel-ML** is a full-stack, agentic machine learning platform built on the philosophy that **AI should assist humans, not act as a black box**. While most AutoML systems ingest data and spit out models with no explanation of *why* choices were made, Sentinel-ML structure-binds every machine learning step into an explicit, human-auditable checkpoint.

You upload a dataset and state your objective in plain English. The AI agents profile the data, generate mathematical feature transformations, train and tune model suites, evaluate compliance, and construct global/local explainability profiles. However, the pipeline pauses at every major milestone. As the human operator, you review the agent justifications and either **Approve**, **Reject**, or **Suggest Alternatives** (which prompts the AI to write and execute custom code in real-time).

---

## ✨ Key Features

- **8 Multi-Agent Collaboration**: Specialized agents cooperate to handle orchestration, data profiling, engineering, modeling, cost estimation, compliance, explainability, and reporting.
- **Human-in-the-Loop Checkpoints**: Decision cards pause execution at key stages, requiring explicit human sign-off or instructions.
- **Real-Time Reasoning Stream**: A WebSocket-driven live console shows step-by-step thoughts, warnings, and code execution.
- **Dynamic AI Code Generation**: Natural language suggestions (e.g., *"Apply log transformation to the loan_amount column"*) are compiled into Python code by the LLM and executed in a sandboxed subprocess with automatic self-correction.
- **Fairness & Governance Auditing**: Automatic checks for disparate impact and equal opportunity difference across protected attributes (e.g., age, gender).
- **Interactive Visualizations**: Dynamically calculated correlation heatmaps, feature distribution histograms, and model leaderboard benchmarks.
- **Explainability & SHAP Plots**: Mean global feature impact plots paired with plain-English narratives and sample-specific prediction explanations.
- **Immutable Audit Trail**: Exportable compliance log of all decisions, LLM justifications, and user inputs.

---

## 🏗️ Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │Pipeline │ │Decision  │ │Data Analysis │ │Model          │  │
│  │DAG      │ │Cards     │ │Dashboard     │ │Leaderboard    │  │
│  └─────────┘ └──────────┘ └──────────────┘ └───────────────┘  │
│  ┌─────────┐ ┌──────────┐ ┌──────────────────────────────────┐ │
│  │SHAP     │ │Reasoning │ │Audit Trail Viewer                │ │
│  │Explain  │ │Chat Log  │ │                                  │ │
│  └─────────┘ └──────────┘ └──────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WebSocket + REST API
┌──────────────────────────▼──────────────────────────────────────┐
│                    Backend (FastAPI + LangGraph)                 │
│                                                                 │
│  ┌───────────┐   ┌──────────────────────────────────────────┐  │
│  │ REST API  │   │         LangGraph Pipeline Graph         │  │
│  │ /runs     │   │                                          │  │
│  │ /decision │   │  Orchestrator → Compliance →             │  │
│  │ /ws/{id}  │   │  DataProfiling → FeatureEng →            │  │
│  │ /export   │   │  ModelSelection → Governance ⟲           │  │
│  └───────────┘   │  Explainability → Reporting              │  │
│                  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The pipeline is modeled as a compiled state graph using **LangGraph**. Stages pause by shifting the state's `is_paused` flag to `True`, which yields control back to the API. When a human operator sends an approval or alternative request, the state is updated, `is_paused` is set to `False`, and execution resumes in a background worker.

---

## 📋 Detailed Agent Activity & Feature Logic

### 1. Objective Intake & Compliance Context
The **Orchestrator Agent** parses the plain-English target description. It identifies the target column, the ML task type (Binary Classification, Multiclass Classification, or Regression), and checks for protected attributes (e.g., `Gender`, `age`, `Race`). The **Compliance Agent** immediately cross-references these attributes with configuration YAML rules (e.g., standard finance or healthcare templates) to load specific fairness thresholds.

### 2. Feature Selection & Transformation Heuristics
The **Feature Engineering Agent** is tasked with preparing features for the models. It performs:
- **Imputation Heuristics**: Assigns mode/constant imputation for categoricals and mean/median for numerical columns.
- **Categorical Encoding**: Maps columns to `OneHotEncoder` or `OrdinalEncoder`/`TargetEncoder` depending on cardinality (e.g., high-cardinality strings like `Model` are target-encoded, while binary fields like `Gender` are one-hot encoded).
- **Outlier Pruning**: Optionally fits an `IsolationForest` on the training partition to drop corrupt samples.
- **Feature Pruning**: Drops columns that have high missingness (e.g., >30%) or those flagged as leakage risks (e.g., features having >98% correlation with the target).
- **Polynomial expansion**: Restricts combinatorial explosion by running a `SelectKBest` test to pick the top 20 features, generating degree-2 interactions only on those, keeping features bounded.

### 3. Model Training & Leaders
The **Model Selection Agent** fits multiple algorithms simultaneously:
- **Classification**: Logistic Regression (baseline), Random Forest (ensemble), Extra Trees, XGBoost, and LightGBM.
- **Regression**: Linear Regression, Ridge, Lasso, Random Forest Regressor, XGBoost Regressor, and LightGBM Regressor.

A cost-awareness module fits each family on a small sample of the data (`n=500`) first, timing it, and projects full training costs. Models are ranked based on a multi-objective score combining raw accuracy (e.g. Validation AUC or R²) and a fairness proxy to select the best candidate.

---

## ⚖️ Governance & Reporting Mechanism

```
                       ┌─────────────────────────┐
                       │   Model Selection Fit   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │    Governance Audit     │
                       └────────────┬────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
         [ Fairness Fails ]                  [ Audits Pass ]
                  │                                   │
                  ▼                                   ▼
┌────────────────────────────────────┐       ┌─────────────────┐
│ Identify Fairness-Proxy Features   │       │   Reporting:    │
│ (Correlation with protected attrib)│       │   Deploy Ready  │
│                                    │       └─────────────────┘
│ Drop Proxies & Loop Back           │
│ to Feature Engineering & Retrain  │
└────────────────────────────────────┘
```

The **Governance Agent** runs detailed validation checks on the selected model:
1. **Fairness Audit**: Measures **Disparate Impact** (ratio of selection rate for unprivileged vs privileged groups) and **Equal Opportunity Difference** (difference in True Positive Rates).
2. **Robustness Audit**: Runs bootstrapped resamples of the dataset to measure variance in validation metrics.
3. **Stability Audit**: Measures drift in feature importances.

**The Auto-Remediation Loop**: If a fairness attribute fails the threshold (e.g., Disparate Impact is below `0.80` or above `1.25`), the agent searches for features that are highly correlated with the protected attribute. It flags these as **Fairness Proxies**, marks them as rejected, and loops the graph back to the Feature Engineering node to drop them, retrain the models, and re-audit.

### The Reporting Agent & Model Card Output
Once the audits pass (or complete), the **Reporting Agent** builds:
- An **HTML Model Card** detailing the dataset dimensions, training parameters, leaderboard results, SHAP importances, and bias metrics.
- A **Deployment Recommendation**:
  - `DEPLOY`: Model passes all performance and fairness audits.
  - `NO_DEPLOY`: Model has severe fairness/bias issues or poor robustness.
  - `WARNING`: Small issues detected, require human sign-off.
- An **Audit Trail**: A detailed ledger recording every state transition, the user's action (`APPROVE`/`REJECT`/`COUNTER_PROPOSE`), the timestamp, and the exact code executed.

---

## 📸 UI Walkthrough — Classification (Loan Default prediction)

> **Dataset**: `Loan_Default.csv` (10,000 sampled rows, 34 columns)
> **Objective**: Predict loan default (`Status`), using `Gender` and `age` as protected attributes for fairness auditing.

### 1. Home Screen — Launch Run
Navigate to `http://localhost:5173`. Enter the plain text objective and select `Loan_Default.csv`.

![Home Screen](./assets/screenshots/classification/loan_01_run_started.png)

---

### 2. Live Pipeline DAG View
The **Pipeline** tab renders the LangGraph flow, showing stages updating as they process.

![Pipeline DAG](./assets/screenshots/classification/loan_02_pipeline_dag.png)

---

### 3. Data Profiling & Flagged Imbalances
The profiling agent flags dataset dimensions, class imbalance, and potential data leakage.

![Data Profiling Decision Card](./assets/screenshots/classification/loan_03_data_profiling_decision.png)

---

### 4. Interactive Data Analysis Dashboard
The **Data Analysis** tab shows distributions, a correlation heatmap, and target metrics.

![Data Analysis Dashboard](./assets/screenshots/classification/loan_04_data_analysis.png)

---

### 5. Feature Engineering Proposals
Proposes preprocessing strategies, column imputers, scaling, and encodings.

![Feature Engineering Proposals](./assets/screenshots/classification/loan_05_feature_engineering_decision.png)

---

### 6. AI Code Request (Suggesting Alternatives)
Clicking **Suggest Alt.** and typing *"Apply log transformation to the loan_amount column to reduce skewness"* triggers the LLM to write and execute a custom script.

![AI Code Execution](./assets/screenshots/classification/loan_06_ai_code_request.png)

---

### 7. Model Selection & Comparison
Compiles performance metrics (AUC-ROC, F1, Precision, Recall) for candidate models.

![Model Selection](./assets/screenshots/classification/loan_07_model_selection_decision.png)

---

### 8. Interactive Model Leaderboard
The **Leaderboard** tab visualizes candidate comparisons across multiple performance goals.

![Leaderboard](./assets/screenshots/classification/loan_08_leaderboard.png)

---

### 9. Governance Audit Results
Performs bias audits, disparate impact assessments, and outputs recommendations.

![Governance Decision Card](./assets/screenshots/classification/loan_09_governance_decision.png)

---

### 10. SHAP Explainability & LLM Interpretation
Global SHAP importances paired with plain-English narratives explaining model behavior.

![SHAP Explainability](./assets/screenshots/classification/loan_10_explainability.png)

---

### 11. Immutable Audit Trail
The **Audit Trail** tab records all operator actions, timestamps, and justification entries.

![Audit Trail](./assets/screenshots/classification/loan_11_audit_trail.png)

---

### 12. WebSocket-Streamed Agent Reasoning
The **Reasoning** tab shows the continuous log of agent step executions.

![Reasoning Logs](./assets/screenshots/classification/loan_12_reasoning_chat.png)

---

### 13. Completed Run State
Final model card has been generated and the run finishes.

![Completed Run](./assets/screenshots/classification/loan_13_completed.png)

---


## 📸 UI Walkthrough — Regression (Car Price prediction)

> **Dataset**: `Car_Price_Prediction.csv` (1,001 rows, 8 columns)
> **Objective**: Predict the selling `Price` of used cars (regression) based on attributes like Mileage, Make, Year, and Engine Size.

### 1. Initialization
Input the regression objective and upload `Car_Price_Prediction.csv`.

![Regression Run Start](./assets/screenshots/regression/reg_01_run_started.png)

---

### 2. Regression Pipeline DAG
The orchestrator automatically sets the task type to `regression`, altering the models trained downstream.

![Regression DAG](./assets/screenshots/regression/reg_02_pipeline_dag.png)

---

### 3. Data Profiling & Targets
Identifies numeric distributions and correlation metrics relative to `Price`.

![Data Profiling Card](./assets/screenshots/regression/reg_03_data_profiling_decision.png)

---

### 4. Regression Analysis Charts
Charts showing relationships between attributes and selling prices.

![Regression Data Analysis](./assets/screenshots/regression/reg_04_data_analysis.png)

---

### 5. Preprocessing & Encoding Configuration
Sets up categorical encoders (target encoding on high-cardinality columns) and scaling.

![Feature Engineering Config](./assets/screenshots/regression/reg_05_feature_engineering_decision.png)

---

### 6. Sandbox Code Execution (Log Mileage)
Applies a log-transformation script to `Mileage` to handle skewed numeric distributions.

![Log Transformation Execution](./assets/screenshots/regression/reg_06_ai_code_request.png)

---

### 7. Regression Model Selection
Evaluates models (XGBoost Regressor, LightGBM Regressor, Random Forest, Lasso) on RMSE, MAE, and R².

![Regression Models](./assets/screenshots/regression/reg_07_model_selection_decision.png)

---

### 8. Regression Leaderboard
Ranks regression algorithms according to error and training cost.

![Leaderboard Analysis](./assets/screenshots/regression/reg_08_leaderboard.png)

---

### 9. Regression Governance Checks
Audits model stability and errors across resampled validations.

![Governance Audit Card](./assets/screenshots/regression/reg_09_governance_decision.png)

---

### 10. SHAP Attribution & Interpretation
SHAP values show which attributes (e.g. Engine Size, Year) shift predictions up or down.

![SHAP Explanations](./assets/screenshots/regression/reg_10_explainability.png)

---

### 11. Audit Ledger
Logs all model retrains and choices made during development.

![Audit Ledger](./assets/screenshots/regression/reg_11_audit_trail.png)

---

### 12. Chat Reasoning Console
Detailed reasoning stream logs for the regression run.

![Reasoning Logs](./assets/screenshots/regression/reg_12_reasoning_chat.png)

---

### 13. Production Ready State
Final exportable artifacts (pipelines, transformers, and model parameters) are packaged.

![Completed Model Card](./assets/screenshots/regression/reg_13_completed.png)

---

## 🔌 Supported LLM Providers

Set `LLM_PROVIDER` in your `.env` to swap backends:

| Provider | Config Value | Requirement |
|----------|--------------|-------------|
| **Gemini** (default) | `LLM_PROVIDER=gemini` | `GEMINI_API_KEY` (aistudio.google.com) |
| **Cerebras** | `LLM_PROVIDER=cerebras` | `CEREBRAS_API_KEY` |
| **Groq** | `LLM_PROVIDER=groq` | `GROQ_API_KEY` |
| **Grok** (xAI) | `LLM_PROVIDER=grok` | `GROK_API_KEY` |
| **Ollama** (Local) | `LLM_PROVIDER=ollama` | Local Ollama instance running on port 11434 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Gemini API key (or other provider credentials)

### 1. Setup Backend

```bash
git clone https://github.com/yourusername/sentinel-ml.git
cd sentinel-ml
python -m venv venv

# Activate
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set Env Variables
Create a `.env` file in the root directory:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
DATABASE_URL=sqlite:///pipeline_runs.db
```

### 3. Run APIs
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** to run pipelines.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, LangGraph, scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn, SQLite, MLflow
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Recharts, Lucide Icons, WebSockets

---

<div align="center">

Built with ❤️ using FastAPI, LangGraph, and React

*Sentinel-ML — Where every ML decision has a paper trail.*

</div>
