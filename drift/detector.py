"""
Online drift detector with rolling-window baseline + threshold classification.

Emits per-step severity: normal / warning / critical.
Supports two modes for A/B comparison (GAP C):
  - entropy-only:  uses entropy + confidence_delta signals
  - composite:     adds patch_complexity proxy

Algorithm adapted from entropy_calculator.DriftCorrelator.detect_entropy_spikes:
rolling window mean/std with spike detection, refactored for streaming use.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from statistics import mean, stdev

from drift.metrics import StepMetrics


class Severity(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DetectorConfig:
    """Thresholds for drift severity classification."""

    # Entropy thresholds (absolute)
    entropy_warn: float = 2.0
    entropy_crit: float = 3.5

    # Confidence delta thresholds (inverted: smaller = worse)
    conf_delta_warn: float = 1.2
    conf_delta_crit: float = 0.6

    # Patch-complexity thresholds
    pcs_warn: float = 0.35
    pcs_crit: float = 0.65

    # Rolling-window spike detection (relative to baseline)
    window_size: int = 3
    spike_sigma: float = 1.5

    # Whether to include patch_complexity in severity vote
    use_patch_complexity: bool = False

    # Trajectory-aware signal thresholds
    tss_warn: float = 0.3
    tss_crit: float = 0.6
    pos_warn: float = 0.4
    pos_crit: float = 0.7
    etc_warn: float = 0.5
    etc_crit: float = 0.8

    # Per-signal enable flags (for ablation)
    use_entropy: bool = True
    use_confidence_delta: bool = True
    use_test_stagnation: bool = False
    use_patch_oscillation: bool = False
    use_edit_target_concentration: bool = False

    # How many "critical" votes needed for CRITICAL severity
    # 2+ for multi-signal confirmation (reduces false alarms)
    critical_vote_threshold: int = 2
    warning_vote_threshold: int = 1


class DriftDetector:
    """
    Streaming drift detector. Call observe() once per agent step.

    Maintains a rolling window of entropy values for adaptive spike detection,
    plus absolute threshold checks on all signals. Emits severity per step.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._entropy_window: deque[float] = deque(maxlen=self.config.window_size)
        self._history: list[tuple[int, Severity]] = []
        self._step_count = 0
        self._alarm_step: int | None = None

    @property
    def alarm_step(self) -> int | None:
        """First step where severity >= WARNING. None if no alarm yet."""
        return self._alarm_step

    @property
    def history(self) -> list[tuple[int, Severity]]:
        return list(self._history)

    def observe(self, step_idx: int, metrics: StepMetrics) -> Severity:
        """
        Process one step and return its severity.

        Voting scheme (adapted from entropy_calculator.classify_severity):
        - Each signal contributes a vote: 0 (ok), 1 (warn), or 2 (crit)
        - Sum determines severity via thresholds
        """
        crit_votes = 0
        warn_votes = 0

        # --- Entropy: absolute threshold + spike relative to baseline ---
        if self.config.use_entropy:
            if metrics.entropy >= self.config.entropy_crit:
                crit_votes += 1
            elif metrics.entropy >= self.config.entropy_warn:
                warn_votes += 1

            if self._is_spike(metrics.entropy):
                warn_votes += 1

        # --- Confidence delta (inverted: smaller is worse) ---
        if self.config.use_confidence_delta:
            if metrics.confidence_delta > 0:
                if metrics.confidence_delta <= self.config.conf_delta_crit:
                    crit_votes += 1
                elif metrics.confidence_delta <= self.config.conf_delta_warn:
                    warn_votes += 1

        # --- Patch complexity (GAP C signal) ---
        if self.config.use_patch_complexity:
            if metrics.patch_complexity >= self.config.pcs_crit:
                crit_votes += 1
            elif metrics.patch_complexity >= self.config.pcs_warn:
                warn_votes += 1

        # --- Test stagnation ---
        if self.config.use_test_stagnation:
            if metrics.test_stagnation >= self.config.tss_crit:
                crit_votes += 1
            elif metrics.test_stagnation >= self.config.tss_warn:
                warn_votes += 1

        # --- Patch oscillation ---
        if self.config.use_patch_oscillation:
            if metrics.patch_oscillation >= self.config.pos_crit:
                crit_votes += 1
            elif metrics.patch_oscillation >= self.config.pos_warn:
                warn_votes += 1

        # --- Edit target concentration ---
        if self.config.use_edit_target_concentration:
            if metrics.edit_target_concentration >= self.config.etc_crit:
                crit_votes += 1
            elif metrics.edit_target_concentration >= self.config.etc_warn:
                warn_votes += 1

        # --- Aggregate ---
        if crit_votes >= self.config.critical_vote_threshold:
            severity = Severity.CRITICAL
        elif crit_votes >= 1 or warn_votes >= self.config.warning_vote_threshold:
            severity = Severity.WARNING
        else:
            severity = Severity.NORMAL

        self._entropy_window.append(metrics.entropy)
        self._history.append((step_idx, severity))
        self._step_count += 1

        if severity != Severity.NORMAL and self._alarm_step is None:
            self._alarm_step = step_idx

        return severity

    def _is_spike(self, entropy: float) -> bool:
        """
        Detect if current entropy is a spike relative to rolling baseline.
        Adapted from DriftCorrelator.detect_entropy_spikes (lines 247-268).
        """
        if len(self._entropy_window) < self.config.window_size:
            return False
        w = list(self._entropy_window)
        w_mean = mean(w)
        w_std = stdev(w) if len(w) > 1 else 0.0
        threshold = w_mean + self.config.spike_sigma * max(w_std, 0.1)
        return entropy > threshold

    def reset(self) -> None:
        """Clear state for a new trajectory."""
        self._entropy_window.clear()
        self._history.clear()
        self._step_count = 0
        self._alarm_step = None


def run_detector_on_run(run, config: DetectorConfig | None = None) -> None:
    """
    Run drift detection on a RunRecord in-place.

    Populates each step's .severity and the run's .t_alarm.
    Requires steps to already have metrics populated.

    Args:
        run: a drift.telemetry.RunRecord
        config: DetectorConfig (default: entropy-only)
    """
    det = DriftDetector(config)
    original_code = ""
    if run.steps:
        original_code = run.steps[0].code

    for step in run.steps:
        m = StepMetrics(
            entropy=step.entropy or 0.0,
            confidence_delta=step.confidence_delta or 0.0,
            logprob_variance=step.logprob_variance or 0.0,
            patch_complexity=step.patch_complexity or 0.0,
            test_stagnation=step.test_stagnation or 0.0,
            patch_oscillation=step.patch_oscillation or 0.0,
            edit_target_concentration=step.edit_target_concentration or 0.0,
        )
        sev = det.observe(step.step, m)
        step.severity = sev.value

    run.t_alarm = det.alarm_step
