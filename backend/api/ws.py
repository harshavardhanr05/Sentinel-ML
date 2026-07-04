"""
backend/api/ws.py
──────────────────
WebSocket connection manager for Sentinel-ML.

Manages per-run WebSocket connections and broadcasts state updates
to all connected clients watching a given run_id.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, List

from fastapi import WebSocket


class ConnectionManager:
    """Manages active WebSocket connections per run_id."""

    def __init__(self):
        # run_id → list of active WebSocket connections
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        self._connections[run_id].append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: str) -> None:
        if run_id in self._connections:
            try:
                self._connections[run_id].remove(websocket)
            except ValueError:
                pass

    async def broadcast(self, run_id: str, data: Dict[str, Any]) -> None:
        """Send a JSON message to all clients watching this run."""
        if run_id not in self._connections:
            return

        dead = []
        for ws in self._connections[run_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws, run_id)

    async def send_to_run(self, run_id: str, message: str) -> None:
        """Send a plain text message to all clients watching this run."""
        if run_id not in self._connections:
            return
        for ws in self._connections[run_id]:
            try:
                await ws.send_text(message)
            except Exception:
                pass

    def get_connection_count(self, run_id: str) -> int:
        return len(self._connections.get(run_id, []))
