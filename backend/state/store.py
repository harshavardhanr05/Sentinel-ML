"""
backend/state/store.py
──────────────────────
SQLite persistence for PipelineState (NFR-3: resume from last checkpoint on crash/restart).

Every LangGraph node calls save_state() after it mutates state.
GET /runs/{id}/state calls load_state() to serve the current state.

Schema: a single table `pipeline_runs` with columns:
  - run_id       TEXT PRIMARY KEY
  - state_json   TEXT  (full PipelineState serialized as JSON)
  - current_stage TEXT
  - is_paused    INTEGER (0/1)
  - created_at   TEXT
  - updated_at   TEXT

Uses SQLAlchemy Core (not ORM) with aiosqlite for async support.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from backend.state.schema import PipelineState

# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

_DB_URL = os.getenv("DATABASE_URL", "sqlite:///pipeline_runs.db")

# Convert sqlite:/// → sqlite+aiosqlite:/// for async support
_ASYNC_DB_URL = _DB_URL.replace("sqlite:///", "sqlite+aiosqlite:///", 1)

_async_engine: Optional[AsyncEngine] = None
_metadata = MetaData()

pipeline_runs = Table(
    "pipeline_runs",
    _metadata,
    Column("run_id", String(36), primary_key=True),
    Column("state_json", Text, nullable=False),
    Column("current_stage", String(64), nullable=False, default="objective_intake"),
    Column("is_paused", Integer, nullable=False, default=0),
    Column("created_at", String(32), nullable=False),
    Column("updated_at", String(32), nullable=False),
)


def get_async_engine() -> AsyncEngine:
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(_ASYNC_DB_URL, echo=False)
    return _async_engine

_sync_engine = None

def get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
    return _sync_engine


async def init_db() -> None:
    """Create the pipeline_runs table if it doesn't exist. Called on FastAPI startup."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(_metadata.create_all)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


async def save_state(state: PipelineState) -> None:
    """
    Upsert (insert or replace) the full PipelineState for a run.
    Called after every LangGraph node completes.
    """
    engine = get_async_engine()
    state_dict = state.model_dump(mode="json")
    state_json = json.dumps(state_dict, default=str)
    now = datetime.utcnow().isoformat()

    async with engine.begin() as conn:
        # Try update first
        result = await conn.execute(
            update(pipeline_runs)
            .where(pipeline_runs.c.run_id == state.run_id)
            .values(
                state_json=state_json,
                current_stage=state.current_stage,
                is_paused=int(state.is_paused),
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            # Row doesn't exist yet — insert
            await conn.execute(
                insert(pipeline_runs).values(
                    run_id=state.run_id,
                    state_json=state_json,
                    current_stage=state.current_stage,
                    is_paused=int(state.is_paused),
                    created_at=state.created_at.isoformat() if state.created_at else now,
                    updated_at=now,
                )
            )


async def load_state(run_id: str) -> Optional[PipelineState]:
    """
    Load the latest persisted PipelineState for a run.
    Returns None if the run_id doesn't exist.
    """
    engine = get_async_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            select(pipeline_runs.c.state_json).where(
                pipeline_runs.c.run_id == run_id
            )
        )
        row = result.fetchone()
        if row is None:
            return None
        state_dict = json.loads(row.state_json)
        return PipelineState.model_validate(state_dict)


async def list_runs() -> List[dict]:
    """
    Return a summary list of all runs (for the run-history UI).
    Returns [{run_id, current_stage, is_paused, created_at, updated_at}].
    """
    engine = get_async_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            select(
                pipeline_runs.c.run_id,
                pipeline_runs.c.current_stage,
                pipeline_runs.c.is_paused,
                pipeline_runs.c.created_at,
                pipeline_runs.c.updated_at,
            ).order_by(pipeline_runs.c.created_at.desc())
        )
        rows = result.fetchall()
        return [
            {
                "run_id": r.run_id,
                "current_stage": r.current_stage,
                "is_paused": bool(r.is_paused),
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]


async def delete_run(run_id: str) -> bool:
    """Delete a run and its state from the DB. Returns True if deleted."""
    from sqlalchemy import delete as sa_delete

    engine = get_async_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            sa_delete(pipeline_runs).where(pipeline_runs.c.run_id == run_id)
        )
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# Sync helpers (for use in tests and LangGraph nodes that aren't async)
# ---------------------------------------------------------------------------


def save_state_sync(state: PipelineState) -> None:
    """Synchronous version of save_state for use inside LangGraph nodes."""
    engine = get_sync_engine()
    state_dict = state.model_dump(mode="json")
    state_json = json.dumps(state_dict, default=str)
    now = datetime.utcnow().isoformat()

    with engine.begin() as conn:
        result = conn.execute(
            update(pipeline_runs)
            .where(pipeline_runs.c.run_id == state.run_id)
            .values(
                state_json=state_json,
                current_stage=state.current_stage,
                is_paused=int(state.is_paused),
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            conn.execute(
                insert(pipeline_runs).values(
                    run_id=state.run_id,
                    state_json=state_json,
                    current_stage=state.current_stage,
                    is_paused=int(state.is_paused),
                    created_at=state.created_at.isoformat() if state.created_at else now,
                    updated_at=now,
                )
            )


def load_state_sync(run_id: str) -> Optional[PipelineState]:
    """Synchronous version of load_state."""
    engine = get_sync_engine()
    with engine.connect() as conn:
        result = conn.execute(
            select(pipeline_runs.c.state_json).where(
                pipeline_runs.c.run_id == run_id
            )
        )
        row = result.fetchone()
        if row is None:
            return None
        state_dict = json.loads(row.state_json)
        return PipelineState.model_validate(state_dict)


def log_step_and_broadcast_sync(state: PipelineState, stage: str, step_name: str, details: str, **kwargs) -> None:
    """Helper to log a step, save state to DB, and trigger immediate WS broadcast."""
    state.log_step(stage, step_name, details, **kwargs)
    save_state_sync(state)
    import requests
    try:
        requests.post(f"http://127.0.0.1:8000/runs/{state.run_id}/broadcast", timeout=0.5)
    except Exception:
        pass
