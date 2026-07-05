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
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
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
        if ext in (".parquet", ".pq"):
            df = pd.read_parquet(upload_path)
        else:
            df = pd.read_csv(upload_path, nrows=5)
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
                # Wait for user decision via /runs/{run_id}/decision
                await _wait_for_decision(run_id)

    except Exception as e:
        state = await load_state(run_id)
        if state:
            state.error_message = str(e)
            await save_state(state)
            await manager.broadcast(run_id, state.model_dump(mode="json"))


async def _wait_for_decision(run_id: str, poll_interval: float = 1.0, max_wait: int = 3600):
    """Poll SQLite until is_paused becomes False (user submitted a decision)."""
    waited = 0
    while waited < max_wait:
        await asyncio.sleep(poll_interval)
        state = await load_state(run_id)
        if state and not state.is_paused:
            return
        waited += poll_interval


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
async def submit_decision(run_id: str, request: DecisionRequest):
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

    # Update the last pending decision log entry
    pending_card = state.pending_approval
    agent_justification = None
    
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
            # Get agent's pros/cons justification before proceeding for other stages
            agent_justification = _get_agent_justification(state, request.note)
            if request.note and "smote" in request.note.lower():
                state.smote_applied = True
                agent_justification += "\n\nSystem Note: SMOTE class balancing has been queued for the Model Selection phase."

    # Update the last decision log entry with the user's choice
    if state.decisions_log:
        last_entry = state.decisions_log[-1]
        if last_entry.user_action == UserAction.PENDING:
            last_entry.user_action = action
            last_entry.user_note = request.note
            last_entry.agent_justification = agent_justification
            last_entry.decided_at = datetime.utcnow()

    if action in (UserAction.APPROVE, UserAction.COUNTER_PROPOSE):
        state.clear_checkpoint()
        if pending_card:
            state.mark_stage(pending_card.stage, StageStatus.APPROVED)
    elif action == UserAction.REJECT:
        state.is_paused = False  # Unblock but mark rejected
        if pending_card:
            state.mark_stage(pending_card.stage, StageStatus.REJECTED)
        state.pending_approval = None

    await save_state(state)

    # Broadcast updated state via WebSocket
    await ws_manager.broadcast(run_id, state.model_dump(mode="json"))

    return {
        "run_id": run_id,
        "action_recorded": action_str,
        "agent_justification": agent_justification,
        "pipeline_resumed": not state.is_paused,
    }


def _get_agent_justification(state: PipelineState, user_note: str) -> str:
    """Ask the Orchestrator agent for a structured pros/cons on the counter-proposal."""
    try:
        from backend.agents.orchestrator import handle_counter_propose
        result = handle_counter_propose(state, user_note)
        return (
            f"Recommendation: {result.get('recommendation', 'N/A')}. "
            f"Reasoning: {result.get('recommendation_reasoning', '')}. "
            f"Pros of agent choice: {'; '.join(result.get('pros_of_agent_choice', [])[:2])}."
        )
    except Exception:
        return "Agent justification could not be generated."


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
        return FileResponse(state.model_card_path, media_type="text/markdown")
    # Generate on-the-fly
    from backend.agents.reporting import _render_model_card
    card_md = _render_model_card(state)
    return PlainTextResponse(card_md, media_type="text/markdown")


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
