"""
backend/api/main.py
────────────────────
FastAPI application for Sentinel-ML.

Endpoints:
  POST   /runs                          — Upload dataset + objective, start pipeline
  GET    /runs                          — List all runs
  GET    /runs/{run_id}/state           — Get current PipelineState (loaded from SQLite)
  POST   /runs/{run_id}/decision        — Submit checkpoint decision (approve/reject/counter)
  POST   /runs/{run_id}/objective       — Update objective fields (target column, protected attrs)
  GET    /runs/{run_id}/model-card      — Get Model Card markdown
  GET    /runs/{run_id}/audit-trail     — Get Audit Trail HTML
  DELETE /runs/{run_id}                 — Delete a run
  WS     /ws/{run_id}                   — WebSocket for live state updates

Checkpoint protocol is embedded here (POST /runs/{run_id}/decision).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from backend.state.schema import (
    DecisionLogEntry,
    PipelineState,
    StageStatus,
    UserAction,
)
from backend.state.store import delete_run, init_db, list_runs, load_state, save_state

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads/")
os.makedirs(_UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Sentinel-ML API",
    description="Multi-agent Human-in-the-Loop ML Governance Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
from backend.api.ws import ConnectionManager
ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    action: str  # "approve" | "reject" | "counter_propose"
    note: Optional[str] = None


class ObjectiveUpdateRequest(BaseModel):
    target_column: Optional[str] = None
    protected_attributes: Optional[List[str]] = None
    domain_tag: Optional[str] = None
    task_type: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    current_stage: str
    is_paused: bool
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# POST /runs — start a new pipeline run
# ---------------------------------------------------------------------------


@app.post("/runs", status_code=202)
async def create_run(
    background_tasks: BackgroundTasks,
    objective: str = Form(..., description="Plain-language business objective"),
    file: UploadFile = File(..., description="Dataset CSV or Parquet"),
):
    """
    Start a new pipeline run. Returns run_id immediately.
    Pipeline executes in the background.
    """
    run_id = str(uuid.uuid4())

    # Save uploaded file
    ext = os.path.splitext(file.filename)[1] if file.filename else ".csv"
    upload_path = os.path.join(_UPLOAD_DIR, f"{run_id}{ext}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize state
    state = PipelineState(
        run_id=run_id,
        dataset_path=upload_path,
        dataset_filename=file.filename,
    )
    state.objective.raw_text = objective

    # Quick schema extraction before pipeline starts
    try:
        import pandas as pd
        import re
        if ext in (".parquet", ".pq"):
            df = pd.read_parquet(upload_path)
        else:
            df = pd.read_csv(upload_path, low_memory=False)
            
        new_cols = []
        for c in df.columns:
            c_clean = str(c).strip()
            c_clean = re.sub(r'\s+', '_', c_clean)
            c_clean = re.sub(r'[\[\]<>]', '', c_clean)
            new_cols.append(c_clean)
        df.columns = new_cols
        
        # Save the cleaned dataset back to the upload path so LLM scripts load the clean version
        if ext in (".parquet", ".pq"):
            df.to_parquet(upload_path, index=False)
        else:
            df.to_csv(upload_path, index=False)
        
        state.data_schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
    except Exception:
        pass

    await save_state(state)

    # Run pipeline in background
    background_tasks.add_task(_run_pipeline_background, run_id, upload_path, ws_manager)

    return {"run_id": run_id, "status": "started"}


async def _run_pipeline_background(run_id: str, dataset_path: str, manager):
    """Execute the LangGraph pipeline and broadcast state updates via WebSocket."""
    try:
        from backend.graph.pipeline_graph import get_compiled_graph

        state = await load_state(run_id)
        if not state:
            return

        graph = get_compiled_graph()

        # Stream execution — LangGraph yields after each node
        async for chunk in graph.astream(state):
            updated_state = await load_state(run_id)
            if updated_state:
                await manager.broadcast(run_id, updated_state.model_dump(mode="json"))

            # Check if pipeline paused at a checkpoint
            if updated_state and updated_state.is_paused:
                # Do NOT wait here! Just exit the runner.
                # When the user submits a decision, submit_decision will spawn a new runner.
                break

    except Exception as e:
        state = await load_state(run_id)
        if state:
            state.error_message = str(e)
            await save_state(state)
            await manager.broadcast(run_id, state.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# GET /runs — list all runs
# ---------------------------------------------------------------------------


@app.get("/runs", response_model=List[RunSummary])
async def get_runs():
    return await list_runs()


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/state — get current state
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/state")
async def get_run_state(run_id: str):
    state = await load_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return state.model_dump(mode="json")


@app.post("/runs/{run_id}/broadcast")
async def broadcast_state(run_id: str):
    """Trigger an immediate WebSocket broadcast of the current state."""
    state = await load_state(run_id)
    if state:
        await ws_manager.broadcast(run_id, state.model_dump(mode="json"))
    return {"status": "broadcasted"}


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/decision — submit a checkpoint decision
# ---------------------------------------------------------------------------


@app.post("/runs/{run_id}/decision")
async def submit_decision(run_id: str, request: DecisionRequest, background_tasks: BackgroundTasks):
    """
    Submit a checkpoint decision.
    On approve: clears pending_approval, sets is_paused=False → pipeline resumes.
    On reject: clears is_paused but sets stage to REJECTED → pipeline stops at that stage.
    On counter_propose: records the note, keeps is_paused=True until agent justifies.
    """
    state = await load_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if not state.is_paused:
        raise HTTPException(status_code=400, detail="Run is not currently awaiting a decision")

    action_str = request.action.lower()
    try:
        action = UserAction(action_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action_str}")

    original_step_count = len(state.agent_step_log)

    # Update the last pending decision log entry
    pending_card = state.pending_approval
    agent_justification = None
    execution_failed_fatally = False
    
    if action == UserAction.APPROVE:
        if pending_card and pending_card.stage == "objective_intake" and getattr(state.objective, "is_ambiguous", False):
            raise HTTPException(
                status_code=400, 
                detail="Cannot approve an ambiguous objective. Please click 'Suggest Alt.' and provide the exact name of the target column."
            )

    if action == UserAction.COUNTER_PROPOSE and request.note:
        if pending_card and pending_card.stage == "objective_intake":
            user_text = request.note.strip()
            possible_cols = state.data_schema.get("columns", [])
            matched_col = next((c for c in possible_cols if c.strip().lower() == user_text.lower()), None)
            
            if matched_col:
                # Bypass LLM and forcefully accept the user's exact column match
                state.objective.target_column = matched_col
                state.objective.is_ambiguous = False
                state.objective.clarification_needed = []
                state.clear_checkpoint()
                agent_justification = f"Target column explicitly set to '{matched_col}'."
            else:
                # Let the orchestrator try to parse it
                state.objective.raw_text += f"\nUser clarification: {request.note}"
                from backend.agents.orchestrator import run_orchestrator
                
                state.clear_checkpoint() # Clear temporarily to see if it triggers again
                state = run_orchestrator(state)
                
                if state.is_paused:
                    # Still ambiguous! Update state and return early, keeping it paused
                    await save_state(state)
                    await ws_manager.broadcast(run_id, state.model_dump(mode="json"))
                    return {
                        "run_id": run_id,
                        "action_recorded": "counter_propose (still ambiguous)",
                        "agent_justification": "Still ambiguous. Please provide more details or type the exact column name.",
                        "pipeline_resumed": False,
                    }
                
                agent_justification = "Objective updated successfully based on your clarification."
        else:
            # Get agent's pros/cons justification and potential generated code
            agent_justification, generated_code, ai_summary, ai_technique = _get_agent_justification(state, request.note)
            
            note_lower = request.note.lower() if request.note else ""
            system_notes = []
            
            if "smotetomek" in note_lower or "smote-tomek" in note_lower:
                state.smote_applied = True
                state.objective.oversampler_type = "smotetomek"
                system_notes.append("SMOTETomek hybrid balancing has been queued for the Model Selection phase.")
                generated_code = None
            elif "smoteenn" in note_lower or "smote-enn" in note_lower:
                state.smote_applied = True
                state.objective.oversampler_type = "smoteenn"
                system_notes.append("SMOTEENN hybrid balancing has been queued for the Model Selection phase.")
                generated_code = None
            elif "smote" in note_lower:
                state.smote_applied = True
                state.objective.oversampler_type = "smote"
                system_notes.append("SMOTE class balancing has been queued for the Model Selection phase.")
                generated_code = None  # Bypass AI execution for native features
                
            if "isolation forest" in note_lower or "outlier" in note_lower:
                state.objective.outlier_removal = "isolation_forest"
                system_notes.append("Isolation Forest outlier removal queued for training data.")
                generated_code = None
                
            if "quantile" in note_lower:
                state.objective.numeric_scaler = "quantile"
                system_notes.append("Numeric scaling set to QuantileTransformer (uniform).")
                generated_code = None
            elif "power" in note_lower or "yeo-johnson" in note_lower:
                state.objective.numeric_scaler = "power"
                system_notes.append("Numeric scaling set to PowerTransformer (Yeo-Johnson).")
                generated_code = None
            elif "robust scaler" in note_lower or "robust scale" in note_lower:
                state.objective.numeric_scaler = "robust"
                system_notes.append("Numeric scaling set to RobustScaler.")
                generated_code = None
            elif "minmax scaler" in note_lower or "minmax scale" in note_lower or "min-max" in note_lower:
                state.objective.numeric_scaler = "minmax"
                system_notes.append("Numeric scaling set to MinMaxScaler.")
                generated_code = None
            elif "standard scaler" in note_lower or "standard scale" in note_lower:
                state.objective.numeric_scaler = "standard"
                system_notes.append("Numeric scaling set to StandardScaler.")
                generated_code = None
                
            if pending_card and pending_card.stage == "feature_engineering":
                if "pca" in note_lower:
                    state.objective.feature_optimization = "pca"
                    system_notes.append("Feature optimization set to PCA dimensionality reduction.")
                    generated_code = None
                elif "polynomial" in note_lower:
                    state.objective.feature_optimization = "polynomial"
                    system_notes.append("Feature optimization set to Polynomial Features.")
                    generated_code = None
                elif "tree" in note_lower:
                    state.objective.feature_optimization = "tree"
                    system_notes.append("Feature optimization set to Tree-based pruning.")
                    generated_code = None
                    
            if pending_card and pending_card.stage == "model_selection":
                if "select " in note_lower and " instead" in note_lower:
                    start_idx = note_lower.find("select ") + 7
                    end_idx = note_lower.find(" instead")
                    model_target = note_lower[start_idx:end_idx].strip().replace("'", "").replace('"', "")
                    
                    found = False
                    for model_entry in state.model_leaderboard:
                        if model_entry.model_name.lower() == model_target:
                            model_entry.is_selected = True
                            state.selected_model_name = model_entry.model_name
                            found = True
                        else:
                            model_entry.is_selected = False
                            
                    if found:
                        system_notes.append(f"Model selection explicitly overridden to: {state.selected_model_name}.")
                        generated_code = None
            log_stage = pending_card.stage if pending_card else state.current_stage
            
            async def _instant_log(step_name: str, details: str, **kwargs):
                state.log_step(log_stage, step_name, details, **kwargs)
                await save_state(state)
                await ws_manager.broadcast(run_id, state.model_dump(mode="json"))

            for note in system_notes:
                await _instant_log("Native Feature Queued", note)
                
            # Execute generated code if present
            if generated_code and generated_code.strip() != "null":
                import sys
                import subprocess
                import os
                import tempfile
                from backend.agents.orchestrator import fix_generated_code
                await _instant_log("AI Code Execution Initiated", "Attempting to run dynamically generated Python script for alternative suggestion.", is_ai_code_request=True, generated_code=generated_code, ai_summary=ai_summary, ai_technique=ai_technique)
                
                max_retries = 5
                current_code = generated_code
                timeout_count = 0
                
                for attempt in range(max_retries):
                    try:
                        # Create a robust wrapper script
                        script_content = f'''import sys
import pandas as pd
import numpy as np

{current_code}

if __name__ == "__main__":
    try:
        df = pd.read_csv(sys.argv[1])
        if "apply_transformation" in locals():
            df_transformed = apply_transformation(df)
            df_transformed.to_csv(sys.argv[2], index=False)
        else:
            print("Failed to execute custom code: function 'apply_transformation' not found in generated code.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
'''
                        # Write to temp file
                        fd, temp_script_path = tempfile.mkstemp(suffix=".py")
                        os.close(fd)
                        with open(temp_script_path, "w", encoding="utf-8") as f:
                            f.write(script_content)
                            
                        new_path = state.dataset_path.replace(".csv", "_custom_transformed.csv")
                        
                        # Calculate dynamic timeout based on dataset size (min 5 minutes, +10s per MB)
                        try:
                            file_size_mb = os.path.getsize(state.dataset_path) / (1024 * 1024)
                            dynamic_timeout = int(max(300, 150 + (file_size_mb * 10)))
                        except Exception:
                            dynamic_timeout = 300
                            
                        # Execute in subprocess with dynamic timeout
                        result = subprocess.run(
                            [sys.executable, temp_script_path, state.dataset_path, new_path],
                            capture_output=True,
                            text=True,
                            timeout=dynamic_timeout
                        )
                        
                        os.remove(temp_script_path)
                        
                        if result.returncode == 0:
                            # Success!
                            import pandas as pd
                            df_transformed = pd.read_csv(new_path)
                            state.dataset_path = new_path
                            if state.feature_log and state.feature_log.final_feature_set:
                                target = state.objective.target_column
                                state.feature_log.final_feature_set = [c for c in df_transformed.columns if c != target]
                                
                            msg = f"Successfully generated and executed custom Python code to apply the requested transformation (Attempt {attempt + 1})."
                            system_notes.append(msg)
                            await _instant_log("Custom Transformation Succeeded", msg, is_ai_code_request=True, ai_summary=ai_summary, ai_technique=ai_technique)
                            break  # Success, exit the retry loop
                        else:
                            # Subprocess failed
                            err_msg = result.stderr.strip()
                            if "ModuleNotFoundError" in err_msg or "ImportError" in err_msg:
                                # Try auto-healing dependencies first
                                lines = err_msg.splitlines()
                                missing_module = None
                                for line in reversed(lines):
                                    if "No module named" in line:
                                        missing_module = line.split("'")[1]
                                        break
                                if missing_module and attempt < max_retries - 1:
                                    install_msg = f"Auto-installing missing dependency '{missing_module}'..."
                                    system_notes.append(install_msg)
                                    await _instant_log("Auto-Healing Triggered", install_msg)
                                    try:
                                        subprocess.check_call([sys.executable, "-m", "pip", "install", missing_module], timeout=120)
                                        success_msg = f"Successfully installed '{missing_module}'. Retrying execution..."
                                        system_notes.append(success_msg)
                                        await _instant_log("Dependency Installed", success_msg)
                                        continue  # Retry execution with same code
                                    except Exception:
                                        err_msg += f"\nFailed to install {missing_module}."
                            
                            err_msg = f"Attempted to execute custom data transformation, but encountered an error:\n{err_msg}"
                            system_notes.append(f"Attempt {attempt + 1} failed.")
                            await _instant_log(f"Execution Failed (Attempt {attempt + 1})", err_msg, is_ai_code_request=True, code_error=err_msg)
                            timeout_count = 0  # reset consecutive timeouts
                            
                            if attempt < max_retries - 1:
                                await _instant_log("AI Self-Correction", "Requesting AI to fix the code based on the error.")
                                fixed_code, fix_method = fix_generated_code(current_code, err_msg, context_msg=request.note or "")
                                if fixed_code:
                                    await _instant_log("AI Self-Correction Applied", "Generated fix for the error.", is_ai_code_request=True, fixed_code=fixed_code, fix_method=fix_method)
                                    current_code = fixed_code
                                else:
                                    execution_failed_fatally = True
                                    break
                            else:
                                execution_failed_fatally = True
                                
                    except subprocess.TimeoutExpired:
                        err_msg = "Code execution timed out after 150 seconds (possible endless loop)."
                        system_notes.append(f"Attempt {attempt + 1} failed: {err_msg}")
                        await _instant_log(f"Execution Timeout (Attempt {attempt + 1})", err_msg, is_ai_code_request=True, code_error=err_msg)
                        try:
                            os.remove(temp_script_path)
                        except:
                            pass
                            
                        timeout_count += 1
                        if timeout_count >= 3:
                            await _instant_log("Execution Aborted", "Consecutive timeout limit reached. Aborting AI code generation.")
                            execution_failed_fatally = True
                            break
                            
                        if attempt < max_retries - 1:
                            await _instant_log("AI Self-Correction", "Requesting AI to fix the code based on the timeout error with full context.")
                            fixed_code, fix_method = fix_generated_code(current_code, err_msg, context_msg=request.note or "")
                            if fixed_code:
                                await _instant_log("AI Self-Correction Applied", "Generated fix for the timeout.", is_ai_code_request=True, fixed_code=fixed_code, fix_method=fix_method)
                                current_code = fixed_code
                            else:
                                execution_failed_fatally = True
                                break
                        else:
                            execution_failed_fatally = True
                    except Exception as e:
                        err_msg = f"Attempted to execute custom data transformation, but encountered a system error: {e}"
                        system_notes.append(err_msg)
                        await _instant_log("Execution Failed", err_msg)
                        execution_failed_fatally = True
                        break
                        
            if execution_failed_fatally:
                fail_msg = "Dynamic code execution failed after maximum retries. Remaining at the current decision card."
                system_notes.append(fail_msg)
                await _instant_log("Execution Aborted", fail_msg)
                agent_justification = "Failed to execute your suggestion. " + fail_msg
            
            if system_notes:
                agent_justification += "\n\nSystem Notes:\n- " + "\n- ".join(system_notes)

    new_ai_logs = [
        step.model_dump(mode="json") 
        for step in state.agent_step_log[original_step_count:] 
        if step.is_ai_code_request
    ]

    # Update the last decision log entry with the user's choice
    if state.decisions_log:
        last_entry = state.decisions_log[-1]
        if last_entry.user_action in (UserAction.PENDING, UserAction.COUNTER_PROPOSE):
            last_entry.user_action = action
            last_entry.user_note = request.note
            last_entry.agent_justification = agent_justification
            
            if last_entry.ai_execution_logs:
                last_entry.ai_execution_logs.extend(new_ai_logs)
            else:
                last_entry.ai_execution_logs = new_ai_logs
                
            last_entry.decided_at = datetime.utcnow()
            
            if state.pending_approval:
                state.pending_approval.ai_execution_logs = last_entry.ai_execution_logs

    if not execution_failed_fatally:
        if action == UserAction.APPROVE:
            state.clear_checkpoint()
            if pending_card:
                state.mark_stage(pending_card.stage, StageStatus.APPROVED)
                if pending_card.stage == "reporting":
                    state.current_stage = "completed"
        elif action == UserAction.REJECT:
            state.is_paused = False  # Unblock but mark rejected
            if pending_card:
                state.mark_stage(pending_card.stage, StageStatus.REJECTED)
            state.pending_approval = None

    await save_state(state)

    # Broadcast updated state via WebSocket
    await ws_manager.broadcast(run_id, state.model_dump(mode="json"))

    # Spawn fresh background runner if pipeline is unpaused
    if not state.is_paused and not execution_failed_fatally:
        background_tasks.add_task(_run_pipeline_background, run_id, state.dataset_path, ws_manager)

    return {
        "run_id": run_id,
        "action_recorded": action_str,
        "agent_justification": agent_justification,
        "ai_execution_logs": new_ai_logs,
        "pipeline_resumed": not state.is_paused,
    }


from typing import Tuple

def _get_agent_justification(state: PipelineState, user_note: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Ask the Orchestrator agent for a structured pros/cons and potential code on the counter-proposal."""
    try:
        from backend.agents.orchestrator import handle_counter_propose
        result = handle_counter_propose(state, user_note)
        
        flaws = result.get('flaws_and_drawbacks_of_user_suggestion', [])
        benefits = result.get('benefits_of_user_suggestion', [])
        
        flaws_text = "\n".join([f"  • {f}" for f in flaws]) if flaws else "  • None identified."
        benefits_text = "\n".join([f"  • {b}" for b in benefits]) if benefits else "  • None identified."
        
        justification_str = (
            f"User Suggestion Interpretation: {result.get('user_suggestion_interpretation', 'N/A')}\n\n"
            f"⚠️ Flaws/Risks of your suggestion:\n{flaws_text}\n\n"
            f"✅ Benefits of your suggestion:\n{benefits_text}\n\n"
            f"AI Conclusion: {result.get('recommendation_reasoning', 'Executing as requested.')}"
        )
        
        code_str = result.get('generated_code', None)
        ai_summary = result.get('ai_summary', None)
        ai_technique = result.get('ai_technique', None)
        return justification_str, code_str, ai_summary, ai_technique
    except Exception as e:
        return f"Agent justification could not be generated: {e}", None, None, None


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/objective — update objective fields after ambiguity check
# ---------------------------------------------------------------------------


@app.post("/runs/{run_id}/objective")
async def update_objective(run_id: str, request: ObjectiveUpdateRequest):
    state = await load_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if request.target_column is not None:
        state.objective.target_column = request.target_column
    if request.protected_attributes is not None:
        state.objective.protected_attributes = request.protected_attributes
    if request.domain_tag is not None:
        state.objective.domain_tag = request.domain_tag
    if request.task_type is not None:
        from backend.state.schema import TaskType
        try:
            state.objective.task_type = TaskType(request.task_type)
        except ValueError:
            pass

    state.objective.is_ambiguous = False
    state.objective.clarification_needed = []
    state.clear_checkpoint()

    await save_state(state)
    return {"run_id": run_id, "objective_updated": True}


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/model-card
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/model-card")
async def get_model_card(run_id: str):
    state = await load_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    if state.model_card_path and os.path.exists(state.model_card_path):
        with open(state.model_card_path, "r", encoding="utf-8") as f:
            card_md = f.read()
    else:
        # Generate on-the-fly
        from backend.agents.reporting import _render_model_card
        card_md = _render_model_card(state)
        
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Card - {run_id}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.0/github-markdown-dark.min.css">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body {{
                box-sizing: border-box;
                min-width: 200px;
                max-width: 980px;
                margin: 0 auto;
                padding: 45px;
                background-color: #0d1117;
            }}
            .markdown-body {{
                box-sizing: border-box;
                min-width: 200px;
                max-width: 980px;
                margin: 0 auto;
                padding: 45px;
            }}
        </style>
    </head>
    <body class="markdown-body">
        <div id="content"></div>
        <script>
            const markdownContent = {json.dumps(card_md)};
            document.getElementById('content').innerHTML = marked.parse(markdownContent);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/audit-trail
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/audit-trail")
async def get_audit_trail(run_id: str):
    state = await load_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    if state.audit_trail_path and os.path.exists(state.audit_trail_path):
        return FileResponse(state.audit_trail_path, media_type="text/html")
    from backend.agents.reporting import _render_audit_trail
    html = _render_audit_trail(state)
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/export
# ---------------------------------------------------------------------------

def _instantiate_unfitted_model(state: PipelineState):
    """Instantiate an unfitted model based on the state."""
    selected = state.selected_model_name or ""
    task_type = getattr(state.objective.task_type, "value", str(state.objective.task_type)).lower()
    is_regression = task_type == "regression"
    
    if "XGBoost" in selected:
        import xgboost as xgb
        if is_regression:
            return xgb.XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
        else:
            return xgb.XGBClassifier(n_estimators=100, verbosity=0, random_state=42, eval_metric="logloss")
    elif "Random Forest" in selected or "LightGBM" in selected:
        if is_regression:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        if is_regression:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=500, random_state=42)


@app.get("/runs/{run_id}/export")
async def export_model(run_id: str):
    import joblib
    from backend.agents.data_profiling import _load_dataset
    from backend.agents.model_selection import _prepare_data
    from backend.agents.explainability import _get_model

    state = await load_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
        
    if not state.selected_model_name:
        raise HTTPException(status_code=400, detail="No model selected yet")
        
    dataset_path = state.dataset_path
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Original dataset not found")

    df = _load_dataset(dataset_path)
    target_col = state.objective.target_column
    final_features = state.feature_log.final_feature_set

    if not target_col or target_col not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found in dataset")

    # Reconstruct the preprocessing logic for the Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder

    # Build LabelEncoder if target is categorical
    y_raw = df[target_col]
    label_encoder = None
    if y_raw.dtype == object:
        label_encoder = LabelEncoder()
        y_train_full = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y_train_full = y_raw.values

    # Determine features
    df_clean = df.dropna(subset=[target_col]).copy()
    feature_cols = [c for c in final_features if c in df_clean.columns] if final_features else \
                   [c for c in df_clean.columns if c != target_col]
    X_train_full = df_clean[feature_cols].copy()

    # Apply outlier removal on training data BEFORE fitting pipeline
    if getattr(state.objective, "outlier_removal", "none") == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.05, random_state=42)
        yhat = iso.fit_predict(X_train_full)
        mask = yhat != -1
        X_train_full = X_train_full[mask]
        y_train_full = y_train_full[mask]

    # Build Transformers
    ai_strats = state.data_schema.get("ai_strategies", {})
    transformers = []
    
    for col in feature_cols:
        strat = ai_strats.get(col, {}) or {}
        imp_strat = (strat.get("imputation_strategy") or "mean").lower()
        enc_strat = (strat.get("encoding_strategy") or "one_hot").lower()
        
        is_cat = str(X_train_full[col].dtype) in ["object", "category"]
        
        # Match imputer exactly like model_selection.py
        if imp_strat == "zero":
            imputer = SimpleImputer(strategy="constant", fill_value="Unknown") if is_cat else SimpleImputer(strategy="constant", fill_value=0)
        elif imp_strat == "unknown":
            imputer = SimpleImputer(strategy="constant", fill_value="Unknown") if is_cat else SimpleImputer(strategy="mean")
        elif imp_strat == "median":
            imputer = SimpleImputer(strategy="most_frequent") if is_cat else SimpleImputer(strategy="median")
        elif imp_strat == "mode":
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            imputer = SimpleImputer(strategy="most_frequent") if is_cat else SimpleImputer(strategy="mean")
        
        steps = [("imputer", imputer)]
        
        if is_cat:
            if enc_strat in ["ordinal", "target_encoding"]:
                steps.append(("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            else:
                steps.append(("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        else:
            scaler_type = getattr(state.objective, "numeric_scaler", "standard")
            if scaler_type == "robust":
                from sklearn.preprocessing import RobustScaler
                steps.append(("scaler", RobustScaler()))
            elif scaler_type == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                steps.append(("scaler", MinMaxScaler()))
            elif scaler_type == "quantile":
                from sklearn.preprocessing import QuantileTransformer
                steps.append(("scaler", QuantileTransformer(output_distribution="uniform", random_state=42)))
            elif scaler_type == "power":
                from sklearn.preprocessing import PowerTransformer
                steps.append(("scaler", PowerTransformer(method="yeo-johnson")))
            else:
                steps.append(("scaler", StandardScaler()))
            
        transformers.append((col, Pipeline(steps), [col]))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)
    
    pipeline_steps = [("preprocessor", preprocessor)]

    # Add optional features
    if state.objective.feature_optimization == "polynomial":
        from sklearn.preprocessing import PolynomialFeatures
        pipeline_steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    elif state.objective.feature_optimization == "pca":
        from sklearn.decomposition import PCA
        pipeline_steps.append(("pca", PCA(n_components=0.95, random_state=42)))

    # Apply SMOTE only to training data (NOT to the pipeline since it's for inference)
    if state.objective.task_type != "regression" and getattr(state.objective, "oversampler_type", "none") != "none":
        # Pre-transform training data to apply SMOTE
        # Fit on the FULL source schema so ColumnTransformer can accept the raw dataset during inference!
        X_source_full = df_clean.drop(columns=[target_col]).copy()
        X_tmp = preprocessor.fit_transform(X_source_full)
        if state.objective.feature_optimization == "polynomial":
            X_tmp = pipeline_steps[-1][1].fit_transform(X_tmp)
        elif state.objective.feature_optimization == "pca":
            X_tmp = pipeline_steps[-1][1].fit_transform(X_tmp)
            
        oversampler_type = getattr(state.objective, "oversampler_type", "smote")
        if oversampler_type == "smotetomek":
            from imblearn.combine import SMOTETomek
            oversampler = SMOTETomek(random_state=42)
        elif oversampler_type == "smoteenn":
            from imblearn.combine import SMOTEENN
            oversampler = SMOTEENN(random_state=42)
        else:
            from imblearn.over_sampling import SMOTE
            oversampler = SMOTE(random_state=42)
            
        try:
            X_tmp, y_train_full = oversampler.fit_resample(X_tmp, y_train_full)
            # Re-initialize preprocessor/poly/pca since we already fitted them manually on pre-smote data
            # Wait, for the final Pipeline to be ready for inference, it must be fit on X_train_full.
            # But we just did SMOTE on the *transformed* data!
            # Sklearn Pipelines don't support SMOTE. 
            # We must fit the model manually on the SMOTE transformed data!
            model = _get_model(state, X_tmp, y_train_full)
            if model is None: raise Exception("Model fit failed")
            pipeline_steps[-1] = (pipeline_steps[-1][0], pipeline_steps[-1][1]) if len(pipeline_steps) > 1 else pipeline_steps[0]
            pipeline_steps.append(("model", model))
            final_pipeline = Pipeline(pipeline_steps)
            
        except Exception:
            # Fallback
            model = _instantiate_unfitted_model(state)
            pipeline_steps.append(("model", model))
            final_pipeline = Pipeline(pipeline_steps)
            final_pipeline.fit(X_source_full, y_train_full)
    else:
        model = _instantiate_unfitted_model(state)
        pipeline_steps.append(("model", model))
        final_pipeline = Pipeline(pipeline_steps)
        final_pipeline.fit(X_source_full, y_train_full)
    # Backwards compatibility for older scikit-learn versions loading LogisticRegression
    actual_model = final_pipeline.named_steps.get("model")
    if actual_model is not None and type(actual_model).__name__ == "LogisticRegression":
        if not hasattr(actual_model, "multi_class"):
            actual_model.multi_class = "auto"

    # Save to disk as a dict to include label encoder if needed
    export_obj = {
        "pipeline": final_pipeline,
        "target_encoder": label_encoder,
        "target_column": target_col,
        "features": feature_cols
    }
    
    model_path = os.path.join(_UPLOAD_DIR, f"{run_id}_model.joblib")
    joblib.dump(export_obj, model_path)

    return FileResponse(
        path=model_path,
        filename=f"sentinel_model.joblib",
        media_type="application/octet-stream"
    )


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/explainability
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/explainability")
async def get_explainability(run_id: str):
    state = await load_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    return state.explainability.model_dump()


# ---------------------------------------------------------------------------
# DELETE /runs/{run_id}
# ---------------------------------------------------------------------------


@app.delete("/runs/{run_id}")
async def remove_run(run_id: str):
    deleted = await delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# WebSocket /ws/{run_id}
# ---------------------------------------------------------------------------


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await ws_manager.connect(websocket, run_id)
    try:
        # Send current state immediately on connect
        state = await load_state(run_id)
        if state:
            await websocket.send_json(state.model_dump(mode="json"))
        while True:
            # Keep connection alive; server pushes updates on state changes
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, run_id)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "sentinel-ml-api"}
