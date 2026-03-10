"""Tests for drift.detector — online drift detection + severity classification."""

import pytest
from drift.detector import DriftDetector, DetectorConfig, Severity, run_detector_on_run
from drift.metrics import StepMetrics
from drift.telemetry import StepRecord, RunRecord


class TestDriftDetector:
    def test_all_normal(self):
        """Low-entropy, high-confidence steps → all NORMAL, no alarm."""
        det = DriftDetector()
        for i in range(5):
            sev = det.observe(i, StepMetrics(entropy=0.5, confidence_delta=3.0))
            assert sev == Severity.NORMAL
        assert det.alarm_step is None

    def test_high_entropy_triggers_warning(self):
        """Entropy above warn threshold → WARNING."""
        det = DriftDetector(DetectorConfig(entropy_warn=2.0, entropy_crit=3.5))
        sev = det.observe(0, StepMetrics(entropy=2.5, confidence_delta=3.0))
        assert sev == Severity.WARNING
        assert det.alarm_step == 0

    def test_critical_requires_multiple_signals(self):
        """CRITICAL needs >= 2 critical votes (multi-signal confirmation)."""
        det = DriftDetector(DetectorConfig(
            entropy_crit=3.5,
            conf_delta_crit=0.6,
            critical_vote_threshold=2,
        ))
        # Only entropy is critical → WARNING (1 crit vote)
        sev = det.observe(0, StepMetrics(entropy=4.0, confidence_delta=2.0))
        assert sev == Severity.WARNING

        det.reset()
        # Both entropy AND confidence_delta critical → CRITICAL (2 votes)
        sev = det.observe(0, StepMetrics(entropy=4.0, confidence_delta=0.3))
        assert sev == Severity.CRITICAL

    def test_spike_detection(self):
        """A sudden entropy jump above rolling baseline triggers warning."""
        det = DriftDetector(DetectorConfig(
            window_size=3,
            spike_sigma=1.5,
            entropy_warn=5.0,   # set high so absolute threshold doesn't trigger
            entropy_crit=10.0,
        ))
        # Build baseline: steady low entropy
        for i in range(3):
            det.observe(i, StepMetrics(entropy=1.0, confidence_delta=3.0))
        # Spike
        sev = det.observe(3, StepMetrics(entropy=3.0, confidence_delta=3.0))
        assert sev == Severity.WARNING
        assert det.alarm_step == 3

    def test_alarm_step_is_first_warning(self):
        """alarm_step records the first non-NORMAL step."""
        det = DriftDetector(DetectorConfig(entropy_warn=2.0))
        det.observe(0, StepMetrics(entropy=0.5))
        det.observe(1, StepMetrics(entropy=0.5))
        det.observe(2, StepMetrics(entropy=2.5))  # first warning
        det.observe(3, StepMetrics(entropy=3.0))  # also warning
        assert det.alarm_step == 2

    def test_history_tracking(self):
        det = DriftDetector()
        det.observe(0, StepMetrics(entropy=0.5))
        det.observe(1, StepMetrics(entropy=0.5))
        assert len(det.history) == 2
        assert det.history[0] == (0, Severity.NORMAL)

    def test_reset_clears_state(self):
        det = DriftDetector(DetectorConfig(entropy_warn=2.0))
        det.observe(0, StepMetrics(entropy=3.0))
        assert det.alarm_step == 0
        det.reset()
        assert det.alarm_step is None
        assert len(det.history) == 0


class TestCompositeDetector:
    """Tests for patch_complexity integration (GAP C)."""

    def test_pcs_disabled_by_default(self):
        """Without use_patch_complexity, high PCS doesn't affect severity."""
        det = DriftDetector(DetectorConfig(use_patch_complexity=False))
        sev = det.observe(0, StepMetrics(entropy=0.5, patch_complexity=0.9))
        assert sev == Severity.NORMAL

    def test_pcs_enabled_triggers_warning(self):
        """With use_patch_complexity, high PCS contributes a warning vote."""
        det = DriftDetector(DetectorConfig(
            use_patch_complexity=True,
            pcs_warn=0.35,
            pcs_crit=0.65,
        ))
        sev = det.observe(0, StepMetrics(entropy=0.5, patch_complexity=0.5))
        assert sev == Severity.WARNING

    def test_pcs_critical_with_entropy(self):
        """PCS critical + entropy critical → CRITICAL (2 votes)."""
        det = DriftDetector(DetectorConfig(
            use_patch_complexity=True,
            entropy_crit=3.5,
            pcs_crit=0.65,
            critical_vote_threshold=2,
        ))
        sev = det.observe(0, StepMetrics(entropy=4.0, patch_complexity=0.8))
        assert sev == Severity.CRITICAL

    def test_confident_wrong_scenario(self):
        """
        GAP C core test: low entropy (model is confident) but high PCS
        (large/churny patch). Entropy-only misses it; composite catches it.
        """
        entropy_only = DriftDetector(DetectorConfig(use_patch_complexity=False))
        composite = DriftDetector(DetectorConfig(use_patch_complexity=True, pcs_warn=0.35))

        m = StepMetrics(entropy=0.8, confidence_delta=3.0, patch_complexity=0.6)

        sev_eo = entropy_only.observe(0, m)
        sev_comp = composite.observe(0, m)

        assert sev_eo == Severity.NORMAL, "Entropy-only should miss confident-wrong"
        assert sev_comp == Severity.WARNING, "Composite should catch via PCS"


class TestRunDetectorOnRun:
    def test_populates_severity_and_alarm(self):
        """run_detector_on_run fills step.severity and run.t_alarm."""
        steps = [
            StepRecord(step=0, code="a", tests_passed=True, entropy=0.5, confidence_delta=3.0),
            StepRecord(step=1, code="b", tests_passed=True, entropy=0.5, confidence_delta=3.0),
            StepRecord(step=2, code="c", tests_passed=False, entropy=3.0, confidence_delta=0.8),
        ]
        run = RunRecord(run_id="test-1", task_id="t", steps=steps, success=False, t_fail=2)

        run_detector_on_run(run, DetectorConfig(entropy_warn=2.0))

        assert steps[0].severity == "normal"
        assert steps[1].severity == "normal"
        assert steps[2].severity in ("warning", "critical")
        assert run.t_alarm == 2

    def test_lead_time_computation(self):
        """Lead time = t_fail - t_alarm."""
        steps = [
            StepRecord(step=0, code="", tests_passed=True, entropy=0.5),
            StepRecord(step=1, code="", tests_passed=True, entropy=2.5),  # alarm here
            StepRecord(step=2, code="", tests_passed=True, entropy=3.0),
            StepRecord(step=3, code="", tests_passed=False, entropy=3.5),  # fail here
        ]
        run = RunRecord(run_id="lt", task_id="t", steps=steps, success=False, t_fail=3)

        run_detector_on_run(run, DetectorConfig(entropy_warn=2.0))

        assert run.t_alarm == 1
        assert run.lead_time == 2  # 3 - 1


class TestTrajectoryAwareSignals:
    """Tests for TSS, POS, ETC signal voting in the detector."""

    def test_tss_critical_triggers_vote(self):
        """High test stagnation with use_test_stagnation=True → CRITICAL."""
        det = DriftDetector(DetectorConfig(
            use_entropy=False,
            use_confidence_delta=False,
            use_test_stagnation=True,
            tss_crit=0.6,
            critical_vote_threshold=1,
        ))
        sev = det.observe(0, StepMetrics(test_stagnation=0.8))
        assert sev == Severity.CRITICAL

    def test_tss_warning_triggers_vote(self):
        det = DriftDetector(DetectorConfig(
            use_entropy=False,
            use_confidence_delta=False,
            use_test_stagnation=True,
            tss_warn=0.3,
            tss_crit=0.6,
        ))
        sev = det.observe(0, StepMetrics(test_stagnation=0.4))
        assert sev == Severity.WARNING

    def test_pos_critical_triggers_vote(self):
        det = DriftDetector(DetectorConfig(
            use_entropy=False,
            use_confidence_delta=False,
            use_patch_oscillation=True,
            pos_crit=0.7,
            critical_vote_threshold=1,
        ))
        sev = det.observe(0, StepMetrics(patch_oscillation=0.9))
        assert sev == Severity.CRITICAL

    def test_etc_critical_triggers_vote(self):
        det = DriftDetector(DetectorConfig(
            use_entropy=False,
            use_confidence_delta=False,
            use_edit_target_concentration=True,
            etc_crit=0.8,
            critical_vote_threshold=1,
        ))
        sev = det.observe(0, StepMetrics(edit_target_concentration=0.9))
        assert sev == Severity.CRITICAL

    def test_disabled_signals_no_effect(self):
        """With all new signals disabled, high values have no effect (backward compat)."""
        det = DriftDetector(DetectorConfig(
            use_test_stagnation=False,
            use_patch_oscillation=False,
            use_edit_target_concentration=False,
        ))
        sev = det.observe(0, StepMetrics(
            entropy=0.5,
            confidence_delta=3.0,
            test_stagnation=1.0,
            patch_oscillation=1.0,
            edit_target_concentration=1.0,
        ))
        assert sev == Severity.NORMAL

    def test_full_composite_all_signals(self):
        """Full composite with all signals enabled — multiple critical votes → CRITICAL."""
        det = DriftDetector(DetectorConfig(
            use_entropy=True,
            use_confidence_delta=True,
            use_patch_complexity=True,
            use_test_stagnation=True,
            use_patch_oscillation=True,
            use_edit_target_concentration=True,
            critical_vote_threshold=2,
        ))
        sev = det.observe(0, StepMetrics(
            entropy=4.0,
            confidence_delta=0.3,
            patch_complexity=0.8,
            test_stagnation=0.9,
            patch_oscillation=0.9,
            edit_target_concentration=0.9,
        ))
        assert sev == Severity.CRITICAL

    def test_entropy_guard(self):
        """With use_entropy=False, high entropy does NOT trigger a vote."""
        det = DriftDetector(DetectorConfig(
            use_entropy=False,
            use_confidence_delta=False,
        ))
        sev = det.observe(0, StepMetrics(entropy=10.0, confidence_delta=0.01))
        assert sev == Severity.NORMAL
