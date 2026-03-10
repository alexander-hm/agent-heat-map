"""
Drift detection module for agent trajectory analysis.

Self-contained package (stdlib only). Provides:
  - telemetry: data model for step/run records, JSONL I/O
  - metrics: entropy, confidence delta, patch-complexity proxy, trajectory-aware signals
  - detector: online rolling-window drift detector with severity classification
  - ablations: named detector configurations for ablation experiments
"""

from drift.telemetry import StepRecord, RunRecord
from drift.metrics import (
    StepMetrics,
    compute_patch_complexity,
    compute_entropy_from_logprobs,
    compute_confidence_delta_from_logprobs,
    compute_test_stagnation,
    compute_patch_oscillation,
    compute_edit_target_concentration,
)
from drift.detector import DriftDetector, DetectorConfig, Severity
from drift.ablations import ABLATION_REGISTRY
