"""
Named detector configurations for ablation experiments.

Each configuration enables a specific subset of signals. All share the same
threshold values so that comparisons are fair — only the signal inclusion differs.
"""

from __future__ import annotations

from .detector import DetectorConfig


def _make_config(**enabled_signals: bool) -> DetectorConfig:
    """Create a DetectorConfig with all signals off, then enable the specified ones."""
    base = DetectorConfig(
        use_entropy=False,
        use_confidence_delta=False,
        use_patch_complexity=False,
        use_test_stagnation=False,
        use_patch_oscillation=False,
        use_edit_target_concentration=False,
        critical_vote_threshold=1,
        warning_vote_threshold=1,
    )
    for signal, value in enabled_signals.items():
        setattr(base, signal, value)
    return base


# ── Individual signals ──────────────────────────────────────────────────

ENTROPY_ONLY = _make_config(use_entropy=True, use_confidence_delta=True)
ENTROPY_ONLY.critical_vote_threshold = 2

PCS_ONLY = _make_config(use_patch_complexity=True)

TSS_ONLY = _make_config(use_test_stagnation=True)

POS_ONLY = _make_config(use_patch_oscillation=True)

ETC_ONLY = _make_config(use_edit_target_concentration=True)


# ── Paper's original configurations ────────────────────────────────────

ORIGINAL_ENTROPY = DetectorConfig(
    use_entropy=True,
    use_confidence_delta=True,
    use_patch_complexity=False,
    use_test_stagnation=False,
    use_patch_oscillation=False,
    use_edit_target_concentration=False,
)

ORIGINAL_COMPOSITE = DetectorConfig(
    use_entropy=True,
    use_confidence_delta=True,
    use_patch_complexity=True,
    use_test_stagnation=False,
    use_patch_oscillation=False,
    use_edit_target_concentration=False,
)


# ── New composite configurations ───────────────────────────────────────

TRAJECTORY_AWARE = _make_config(
    use_test_stagnation=True,
    use_patch_oscillation=True,
    use_edit_target_concentration=True,
)

PCS_PLUS_TRAJECTORY = _make_config(
    use_patch_complexity=True,
    use_test_stagnation=True,
    use_patch_oscillation=True,
    use_edit_target_concentration=True,
)

FULL_COMPOSITE = DetectorConfig(
    use_entropy=True,
    use_confidence_delta=True,
    use_patch_complexity=True,
    use_test_stagnation=True,
    use_patch_oscillation=True,
    use_edit_target_concentration=True,
)

NO_ENTROPY_COMPOSITE = _make_config(
    use_patch_complexity=True,
    use_test_stagnation=True,
    use_patch_oscillation=True,
    use_edit_target_concentration=True,
)


# ── Ablation registry (used by eval pipeline) ──────────────────────────
# Each entry: (short_name, human_label, config)
ABLATION_REGISTRY: list[tuple[str, str, DetectorConfig]] = [
    ("entropy",     "Entropy+ConfΔ only",              ENTROPY_ONLY),
    ("pcs",         "PCS only",                        PCS_ONLY),
    ("tss",         "Test Stagnation only",            TSS_ONLY),
    ("pos",         "Patch Oscillation only",          POS_ONLY),
    ("etc",         "Edit Target Conc. only",          ETC_ONLY),
    ("orig_eo",     "Original Entropy-Only",           ORIGINAL_ENTROPY),
    ("orig_comp",   "Original Composite",              ORIGINAL_COMPOSITE),
    ("traj_aware",  "Trajectory-Aware (TSS+POS+ETC)",  TRAJECTORY_AWARE),
    ("pcs_traj",    "PCS + Trajectory-Aware",          PCS_PLUS_TRAJECTORY),
    ("no_entropy",  "All except Entropy",              NO_ENTROPY_COMPOSITE),
    ("full",        "Full Composite (all signals)",    FULL_COMPOSITE),
]
