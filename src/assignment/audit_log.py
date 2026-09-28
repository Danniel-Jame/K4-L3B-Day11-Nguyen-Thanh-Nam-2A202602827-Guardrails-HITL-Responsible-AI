"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(
        self, *, user_id: str, text: str, request_id: str | None = None
    ) -> str:
        """Store an input event and return its correlation id."""
        correlation_id = request_id or str(uuid.uuid4())
        started_at = time.perf_counter()
        self._open[correlation_id] = started_at
        self.logs.append(
            {
                "request_id": correlation_id,
                "user_id": user_id,
                "event": "input",
                "input": text,
                "timestamp": utc_now_iso(),
            }
        )
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store the decision and response, including elapsed time when correlated."""
        correlation_id = request_id or user_id
        started_at = self._open.pop(correlation_id, None)
        entry = {
            "request_id": request_id,
            "user_id": user_id,
            "event": "output",
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "timestamp": utc_now_iso(),
        }
        if started_at is not None:
            entry["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        self.logs.append(entry)

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
