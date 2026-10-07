"""그래프 실행 시간을 로컬에 기록합니다. 외부 LangSmith 기록과 같은 trace_id를 씁니다."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler


class ExecutionTrace(BaseCallbackHandler):
    def __init__(self, path: Path, trace_id: str):
        self.path = path
        self.trace_id = trace_id
        self.lock = threading.Lock()
        self.nodes = {}

    def _write(self, event: dict):
        record = {"trace_id": self.trace_id, "ts": datetime.now(timezone.utc).isoformat(), **event}
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        self.nodes[str(run_id)] = name
        task = inputs.get("current_task", {}) if isinstance(inputs, dict) else {}
        self._write({"event": "start", "run_id": str(run_id), "parent_run_id": str(parent_run_id) if parent_run_id else None,
                     "node": name, "task_id": task.get("task_id")})

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        details = {}
        if isinstance(outputs, dict):
            for key in ("plan_reason", "retry_targets", "status", "quality_evaluation"):
                if key in outputs:
                    details[key] = outputs[key]
            if "plan" in outputs:
                details["task_ids"] = [task["task_id"] for task in outputs["plan"]]
            if "decision_log" in outputs:
                details["decisions"] = outputs["decision_log"]
        self._write({"event": "end", "run_id": str(run_id), "node": self.nodes.get(str(run_id)), **details})

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._write({"event": "error", "run_id": str(run_id), "node": self.nodes.get(str(run_id)),
                     "error": type(error).__name__})
