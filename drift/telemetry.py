"""
Data model for drift-detection evaluation.

Lightweight dataclasses (stdlib only) that capture per-step telemetry
and per-run labels. Bridges to run_one_task.py JSONL output format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class StepRecord:
    """One step in an agent trajectory with all signals needed for drift detection."""

    step: int
    code: str
    tests_passed: bool
    n_passed: int = 0
    n_failed: int = 0
    n_errors: int = 0

    # Signals — populated by metrics module or loaded from simulated_metrics
    entropy: float | None = None
    confidence_delta: float | None = None
    logprob_variance: float | None = None
    patch_complexity: float | None = None

    # Trajectory-aware signals
    test_stagnation: float = 0.0
    patch_oscillation: float = 0.0
    edit_target_concentration: float = 0.0

    # Detector output (filled after detection pass)
    severity: str | None = None  # "normal" | "warning" | "critical"

    # Source of entropy data: "simulated" or "real"
    entropy_source: str = "simulated"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunRecord:
    """Complete trajectory with run-level labels."""

    run_id: str
    task_id: str
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False
    t_fail: int | None = None  # first unrecovered failure step (failure runs only)
    t_alarm: int | None = None  # first step where detector fires warning+ (filled post-detection)

    @property
    def lead_time(self) -> int | None:
        """Steps of advance warning: t_fail - t_alarm. Only meaningful for failure runs."""
        if self.t_fail is None or self.t_alarm is None:
            return None
        return self.t_fail - self.t_alarm

    @property
    def is_false_alarm(self) -> bool:
        """Alarm fired but run succeeded."""
        return self.success and self.t_alarm is not None

    def to_dict(self) -> dict:
        d = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "success": self.success,
            "t_fail": self.t_fail,
            "t_alarm": self.t_alarm,
            "lead_time": self.lead_time,
            "is_false_alarm": self.is_false_alarm,
            "total_steps": len(self.steps),
        }
        return d

    def to_jsonl(self) -> str:
        """Serialize to JSONL: one line per step + one summary line."""
        lines = []
        for s in self.steps:
            lines.append(json.dumps(s.to_dict()))
        lines.append(json.dumps({"_run_summary": True, **self.to_dict()}))
        return "\n".join(lines)


def load_run_log(path: str | Path) -> RunRecord:
    """
    Load a RunRecord from a JSONL file produced by run_one_task.py.

    Bridges the run_one_task output format to our drift data model.
    """
    path = Path(path)
    steps = []
    summary = {}

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("_run_summary"):
            summary = d
        else:
            sm = d.get("simulated_metrics", {})
            steps.append(StepRecord(
                step=d["step"],
                code=d.get("code_applied", ""),
                tests_passed=d["tests_passed"],
                n_passed=d.get("n_passed", 0),
                n_failed=d.get("n_failed", 0),
                n_errors=d.get("n_errors", 0),
                entropy=sm.get("entropy"),
                confidence_delta=sm.get("confidence_delta"),
                logprob_variance=sm.get("logprob_variance"),
            ))

    return RunRecord(
        run_id=summary.get("run_id", "unknown"),
        task_id=summary.get("task_id", "unknown"),
        steps=steps,
        success=summary.get("success", False),
        t_fail=summary.get("t_fail"),
    )


def save_run_log(run: RunRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.to_jsonl() + "\n")
