"""
backend/agents/explainability.py
──────────────────────────────────
Explainability Agent — Phase 4 (MVP)

Responsibilities:
1. Generate global SHAP summary: feature importance across all validation samples.
2. Generate local SHAP explanations for 3 representative predictions:
   - A correctly classified positive
   - A correctly classified negative
   - A misclassified example (most interesting for stakeholders)
3. Save a global SHAP bar-chart plot as PNG.
4. Package results into state.explainability (ExplainabilityOutput).

Kept separate from Governance for clean separation of concerns (per spec §2.1).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backend.state.schema import ExplainabilityOutput, PipelineState

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

_OUTPUT_DIR = os.getenv("EXPLAINABILITY_OUTPUT_DIR", "./artifacts/explainability/")


def run_explainability(state: PipelineState) -> PipelineState:
    """
    Compute SHAP explanations for the selected model. Updates state.explainability.
    """
    if not _HAS_SHAP:
        state.explainability = ExplainabilityOutput(
            top_features_summary=["SHAP not available — install shap package"],
        )
        return state

    import os as _os
    from backend.agents.data_profiling import _load_dataset

    dataset_path = state.dataset_path
    if not dataset_path or not _os.path.exists(dataset_path):
        return state

    df = _load_dataset(dataset_path)
    target_col = state.objective.target_column
    final_features = state.feature_log.final_feature_set

    if not target_col or target_col not in df.columns:
        return state

    X, y, feature_names = _prepare_data(df, target_col, final_features)
    if X is None:
        return state

    model = _get_model(state, X, y)
    if model is None:
        return state

    # ── Global SHAP ───────────────────────────────────────────────────
    try:
        sample_size = min(200, len(X))
        X_sample = X[:sample_size]

        explainer = _build_explainer(model, X_sample)
        shap_values = explainer(X_sample)

        # Handle multi-output SHAP (binary classification → use positive class)
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]  # positive class
        else:
            vals = np.array(shap_values)

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        global_shap_dict = {
            (feature_names[i] if i < len(feature_names) else f"feature_{i}"): round(float(mean_abs_shap[i]), 4)
            for i in range(len(mean_abs_shap))
        }

        # Sort by importance
        global_shap_sorted = dict(sorted(global_shap_dict.items(), key=lambda x: x[1], reverse=True))
        top_5 = list(global_shap_sorted.keys())[:5]

        # ── Local examples ─────────────────────────────────────────────
        local_examples = _build_local_examples(model, X_sample, y[:sample_size], vals, feature_names)

        # ── Save plot ─────────────────────────────────────────────────
        plot_path = _save_shap_plot(global_shap_sorted, state.run_id)

        state.explainability = ExplainabilityOutput(
            global_shap_values=global_shap_sorted,
            top_features_summary=top_5,
            local_examples=local_examples,
            shap_plot_path=plot_path,
        )

    except Exception as e:
        state.explainability = ExplainabilityOutput(
            top_features_summary=[f"SHAP computation failed: {e}"],
        )

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_data(df, target_col, final_features):
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    try:
        df_clean = df.dropna(subset=[target_col]).copy()
        feature_cols = [c for c in final_features if c in df_clean.columns] if final_features else \
                       [c for c in df_clean.columns if c != target_col]
        if not feature_cols:
            return None, None, None

        y = df_clean[target_col]
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = y.values

        X = df_clean[feature_cols].copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(0)
        X_np = StandardScaler().fit_transform(X.values.astype(float))

        return X_np, y, feature_cols
    except Exception:
        return None, None, None


def _get_model(state, X, y):
    """Retrain the selected model for SHAP analysis."""
    try:
        selected = state.selected_model_name or ""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)

        if "XGBoost" in selected:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=100, verbosity=0, random_state=42,
                                       eval_metric="logloss", use_label_encoder=False)
        elif "Random Forest" in selected or "LightGBM" in selected:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=500, random_state=42)

        model.fit(X_train, y_train)
        return model
    except Exception:
        return None


def _build_explainer(model, X_sample):
    """Build appropriate SHAP explainer based on model type."""
    try:
        # Try TreeExplainer first (faster for tree-based models)
        return shap.TreeExplainer(model)
    except Exception:
        try:
            return shap.LinearExplainer(model, X_sample)
        except Exception:
            return shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 50))


def _build_local_examples(model, X, y, shap_vals, feature_names):
    """Build local SHAP breakdowns for 3 representative examples."""
    examples = []
    try:
        y_pred = model.predict(X)
        correct_pos = np.where((y_pred == 1) & (y == 1))[0]
        correct_neg = np.where((y_pred == 0) & (y == 0))[0]
        misclassified = np.where(y_pred != y)[0]

        for label, candidates in [
            ("correct_positive", correct_pos),
            ("correct_negative", correct_neg),
            ("misclassified", misclassified),
        ]:
            if len(candidates) == 0:
                continue
            idx = candidates[0]
            shap_breakdown = {
                (feature_names[i] if i < len(feature_names) else f"feature_{i}"): round(float(shap_vals[idx, i]), 4)
                for i in range(min(10, shap_vals.shape[1]))
            }
            examples.append({
                "type": label,
                "sample_index": int(idx),
                "actual_label": int(y[idx]),
                "predicted_label": int(y_pred[idx]),
                "shap_breakdown": shap_breakdown,
            })
    except Exception:
        pass
    return examples


def _save_shap_plot(global_shap: Dict[str, float], run_id: str) -> Optional[str]:
    """Save a horizontal bar chart of global SHAP values."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(_OUTPUT_DIR, f"{run_id}_shap_global.png")

        features = list(global_shap.keys())[:15]
        values = [global_shap[f] for f in features]
        colors = ["#6366f1" if v > 0 else "#ef4444" for v in values]

        fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))
        bars = ax.barh(range(len(features)), values, color=colors, edgecolor="none")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception:
        return None
