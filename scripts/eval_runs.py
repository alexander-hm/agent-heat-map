#!/usr/bin/env python3
"""
Batch evaluation: generate trajectories, run drift detectors, produce CSV + ROC figure.

Generates 5 trajectory templates × 8 tasks = 40 runs, then compares
entropy-only vs. composite (entropy + patch-complexity) drift detectors.
Demonstrates GAP C: confident-wrong cases missed by entropy but caught by composite.

Modes:
    scripted (default):  run all template trajectories with simulated metrics
    validate:            run live agent on a task subset and log real entropy
    both:                run validation first, then scripted batch

Usage:
    python scripts/eval_runs.py
    python scripts/eval_runs.py --output-dir outputs/exp01 --seed 42
    python scripts/eval_runs.py --mode validate --native-client
    python scripts/eval_runs.py --mode both --native-client
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from toy_tasks.registry import TASKS, get_task, list_task_ids
from drift.telemetry import StepRecord, RunRecord
from drift.metrics import (
    compute_patch_complexity, StepMetrics,
    compute_test_stagnation, compute_patch_oscillation,
    compute_edit_target_concentration,
)
from drift.detector import DriftDetector, DetectorConfig, Severity, run_detector_on_run

# Import run infrastructure via importlib (scripts/ is not a package)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_one_task", str(REPO_ROOT / "scripts" / "run_one_task.py")
)
_rot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rot)
_run_trajectory = _rot.run_trajectory
_run_live_agent = _rot.run_live_agent
_label_run = _rot.label_run


# ---------------------------------------------------------------------------
# Trajectory templates
# ---------------------------------------------------------------------------

def _add_noise(val: float, rng: random.Random, sigma: float = 0.12) -> float:
    return max(0.0, val + rng.gauss(0, sigma))


def _template_quick_correct(task: dict, rng: random.Random) -> list[dict]:
    """1-step: apply correct solution immediately."""
    return [
        {"code": task["correct_code"],
         "simulated_metrics": {"entropy": _add_noise(0.5, rng),
                               "confidence_delta": _add_noise(3.0, rng)}},
    ]


def _template_struggle_correct(task: dict, rng: random.Random) -> list[dict]:
    """4 steps: buggy → wrong → buggy → correct."""
    buggy = (REPO_ROOT / task["source_file"]).read_text()
    return [
        {"code": buggy,
         "simulated_metrics": {"entropy": _add_noise(1.5, rng),
                               "confidence_delta": _add_noise(2.0, rng)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(2.2, rng),
                               "confidence_delta": _add_noise(1.5, rng)}},
        {"code": buggy,
         "simulated_metrics": {"entropy": _add_noise(2.5, rng),
                               "confidence_delta": _add_noise(1.2, rng)}},
        {"code": task["correct_code"],
         "simulated_metrics": {"entropy": _add_noise(0.8, rng),
                               "confidence_delta": _add_noise(2.8, rng)}},
    ]


def _template_high_entropy_fail(task: dict, rng: random.Random) -> list[dict]:
    """5 steps: entropy climbs, never solves. Entropy-only should catch this."""
    buggy = (REPO_ROOT / task["source_file"]).read_text()
    return [
        {"code": buggy,
         "simulated_metrics": {"entropy": _add_noise(1.5, rng),
                               "confidence_delta": _add_noise(2.0, rng)}},
        {"code": buggy,
         "simulated_metrics": {"entropy": _add_noise(2.5, rng),
                               "confidence_delta": _add_noise(1.5, rng)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(3.0, rng),
                               "confidence_delta": _add_noise(1.0, rng)}},
        {"code": buggy,
         "simulated_metrics": {"entropy": _add_noise(3.5, rng),
                               "confidence_delta": _add_noise(0.8, rng)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(4.0, rng),
                               "confidence_delta": _add_noise(0.5, rng)}},
    ]


def _template_confident_wrong(task: dict, rng: random.Random) -> list[dict]:
    """
    5 steps: model is confident (low entropy) but applies wrong fix repeatedly.
    KEY FOR GAP C: entropy-only should miss this; composite should catch via PCS.
    """
    buggy = (REPO_ROOT / task["source_file"]).read_text()
    return [
        {"code": buggy,
         "simulated_metrics": {"entropy": _add_noise(1.0, rng, sigma=0.08),
                               "confidence_delta": _add_noise(2.5, rng, sigma=0.08)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(0.6, rng, sigma=0.08),
                               "confidence_delta": _add_noise(3.2, rng, sigma=0.08)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(0.5, rng, sigma=0.08),
                               "confidence_delta": _add_noise(3.5, rng, sigma=0.08)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(0.5, rng, sigma=0.08),
                               "confidence_delta": _add_noise(3.5, rng, sigma=0.08)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(0.4, rng, sigma=0.08),
                               "confidence_delta": _add_noise(3.6, rng, sigma=0.08)}},
    ]


def _template_regression(task: dict, rng: random.Random) -> list[dict]:
    """5 steps: solves, then regresses to wrong code."""
    buggy = (REPO_ROOT / task["source_file"]).read_text()
    return [
        {"code": buggy,
         "simulated_metrics": {"entropy": _add_noise(1.0, rng),
                               "confidence_delta": _add_noise(2.5, rng)}},
        {"code": task["correct_code"],
         "simulated_metrics": {"entropy": _add_noise(0.8, rng),
                               "confidence_delta": _add_noise(3.0, rng)}},
        {"code": task["correct_code"],
         "simulated_metrics": {"entropy": _add_noise(0.7, rng),
                               "confidence_delta": _add_noise(3.0, rng)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(1.2, rng),
                               "confidence_delta": _add_noise(2.2, rng)}},
        {"code": task["wrong_code"],
         "simulated_metrics": {"entropy": _add_noise(1.5, rng),
                               "confidence_delta": _add_noise(1.8, rng)}},
    ]


TEMPLATES = {
    "quick_correct": _template_quick_correct,
    "struggle_correct": _template_struggle_correct,
    "high_entropy_fail": _template_high_entropy_fail,
    "confident_wrong": _template_confident_wrong,
    "regression": _template_regression,
}


# ---------------------------------------------------------------------------
# Run processing
# ---------------------------------------------------------------------------

def build_run_record(
    task_id: str,
    template: str,
    log_entries: list[dict],
    run_label: dict,
    original_code: str,
) -> RunRecord:
    """Convert raw log entries → RunRecord with all signals computed for each step."""
    steps = []
    prev_codes: list[str] = []
    test_history: list[tuple[int, int]] = []
    code_history: list[str] = []

    for entry in log_entries:
        sm = entry.get("simulated_metrics", {})
        code = entry.get("code_applied", "")

        pcs = compute_patch_complexity(
            current_code=code,
            original_code=original_code,
            prev_codes=prev_codes if prev_codes else None,
        )

        ent = entry.get("entropy") if entry.get("entropy") is not None else sm.get("entropy", 0.0)
        cd = entry.get("confidence_delta") if entry.get("confidence_delta") is not None else sm.get("confidence_delta", 0.0)
        esrc = entry.get("entropy_source", "simulated")

        test_history.append((entry.get("n_passed", 0), entry.get("n_failed", 0)))
        code_history.append(code)

        tss = compute_test_stagnation(test_history)
        pos = compute_patch_oscillation(code_history)
        etc_val = compute_edit_target_concentration(code_history, original_code)

        steps.append(StepRecord(
            step=entry["step"],
            code=code,
            tests_passed=entry["tests_passed"],
            n_passed=entry.get("n_passed", 0),
            n_failed=entry.get("n_failed", 0),
            n_errors=entry.get("n_errors", 0),
            entropy=ent,
            confidence_delta=cd,
            logprob_variance=sm.get("logprob_variance", 0.0),
            patch_complexity=pcs,
            test_stagnation=tss,
            patch_oscillation=pos,
            edit_target_concentration=etc_val,
            entropy_source=esrc,
        ))
        prev_codes.append(code)

    return RunRecord(
        run_id=run_label["run_id"],
        task_id=task_id,
        steps=steps,
        success=run_label["success"],
        t_fail=run_label["t_fail"],
    )


# ---------------------------------------------------------------------------
# ROC computation (no sklearn dependency)
# ---------------------------------------------------------------------------

def compute_roc(scores: list[float], labels: list[bool]) -> tuple[list[tuple[float, float]], float]:
    """
    Compute ROC curve and AUC from continuous scores and binary labels.

    Args:
        scores: higher score = more likely failure
        labels: True = failure run, False = success run
    Returns:
        (list of (fpr, tpr) points, AUC)
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [(0, 0), (1, 1)], 0.5

    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    points = [(0.0, 0.0)]

    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        points.append((fp / n_neg, tp / n_pos))

    auc = sum(
        (points[i][0] - points[i - 1][0]) * (points[i][1] + points[i - 1][1]) / 2
        for i in range(1, len(points))
    )
    return points, auc


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "run_id", "task_id", "template", "success", "t_fail", "total_steps",
    "max_entropy", "max_pcs",
    "eo_t_alarm", "eo_lead_time", "eo_detected",
    "comp_t_alarm", "comp_lead_time", "comp_detected",
    "is_confident_wrong", "rescued_by_composite",
]


def make_row(template: str, run: RunRecord, eo: RunRecord, comp: RunRecord) -> dict:
    max_ent = max((s.entropy or 0 for s in run.steps), default=0)
    max_pcs = max((s.patch_complexity or 0 for s in run.steps), default=0)

    eo_detected = eo.t_alarm is not None
    comp_detected = comp.t_alarm is not None

    is_confident_wrong = (not run.success) and (not eo_detected)
    rescued = is_confident_wrong and comp_detected

    return {
        "run_id": run.run_id,
        "task_id": run.task_id,
        "template": template,
        "success": run.success,
        "t_fail": run.t_fail,
        "total_steps": len(run.steps),
        "max_entropy": round(max_ent, 3),
        "max_pcs": round(max_pcs, 3),
        "eo_t_alarm": eo.t_alarm,
        "eo_lead_time": eo.lead_time,
        "eo_detected": eo_detected,
        "comp_t_alarm": comp.t_alarm,
        "comp_lead_time": comp.lead_time,
        "comp_detected": comp_detected,
        "is_confident_wrong": is_confident_wrong,
        "rescued_by_composite": rescued,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(eo_roc, eo_auc, comp_roc, comp_auc, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        eo_fpr = [p[0] for p in eo_roc]
        eo_tpr = [p[1] for p in eo_roc]
        comp_fpr = [p[0] for p in comp_roc]
        comp_tpr = [p[1] for p in comp_roc]

        ax.plot(eo_fpr, eo_tpr, "b-o", markersize=3, label=f"Entropy-only (AUC={eo_auc:.2f})")
        ax.plot(comp_fpr, comp_tpr, "r-s", markersize=3, label=f"Composite (AUC={comp_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Detection Rate)")
        ax.set_title("ROC: Entropy-Only vs Composite Drift Detector")
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  (figure generation failed: {e})", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Validation mode — run live agent on a subset, log real entropy, compare
# ---------------------------------------------------------------------------

VALIDATION_TASKS = [
    "binary_search", "fibonacci", "flatten_list", "merge_sorted",
    "is_palindrome", "gcd", "caesar_cipher", "matrix_transpose",
]

VALIDATION_CSV_FIELDS = [
    "task_id", "step", "entropy", "confidence_delta", "entropy_source",
    "patch_complexity", "test_stagnation", "patch_oscillation",
    "edit_target_concentration", "tests_passed", "n_passed", "n_failed",
]


def run_validation(
    output_dir: Path,
    model: str | None = None,
    max_steps: int = 6,
    use_native_client: bool = False,
) -> None:
    """Run live agent on VALIDATION_TASKS, log real entropy, print comparison."""
    from drift.metrics import compute_entropy_from_logprobs, compute_confidence_delta_from_logprobs

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "validation_steps.jsonl"
    csv_path = output_dir / "validation_summary.csv"

    all_lines: list[str] = []
    csv_rows: list[dict] = []

    for task_id in VALIDATION_TASKS:
        task = get_task(task_id)
        original_code = (REPO_ROOT / task["source_file"]).read_text()

        print(f"\n[validate] Running live agent on {task_id}...", file=sys.stderr)
        log_entries = _run_live_agent(
            task_id=task_id,
            model=model,
            max_steps=max_steps,
            use_native_client=use_native_client,
        )
        run_label = _label_run(log_entries)
        run = build_run_record(task_id, "live", log_entries, run_label, original_code)

        for step in run.steps:
            d = step.to_dict()
            d["task_id"] = task_id
            all_lines.append(json.dumps(d))

            csv_rows.append({
                "task_id": task_id,
                "step": step.step,
                "entropy": round(step.entropy or 0.0, 4),
                "confidence_delta": round(step.confidence_delta or 0.0, 4),
                "entropy_source": step.entropy_source,
                "patch_complexity": round(step.patch_complexity or 0.0, 4),
                "test_stagnation": round(step.test_stagnation or 0.0, 4),
                "patch_oscillation": round(step.patch_oscillation or 0.0, 4),
                "edit_target_concentration": round(step.edit_target_concentration or 0.0, 4),
                "tests_passed": step.tests_passed,
                "n_passed": step.n_passed,
                "n_failed": step.n_failed,
            })

        status = "SUCCESS" if run.success else f"FAIL(t={run.t_fail})"
        print(f"  [{task_id}] {status} — {len(run.steps)} step(s)", file=sys.stderr)

    jsonl_path.write_text("\n".join(all_lines) + "\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*72}", file=sys.stderr)
    print("VALIDATION RESULTS", file=sys.stderr)
    print(f"{'='*72}", file=sys.stderr)
    print(f"{'task_id':<20s} {'step':>4s} {'entropy':>8s} {'conf_Δ':>8s} "
          f"{'source':>10s} {'pcs':>6s} {'tss':>6s} {'pos':>6s} {'etc':>6s} {'pass':>5s}",
          file=sys.stderr)
    print(f"{'-'*20} {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}",
          file=sys.stderr)
    for r in csv_rows:
        print(f"{r['task_id']:<20s} {r['step']:>4d} {r['entropy']:>8.4f} "
              f"{r['confidence_delta']:>8.4f} {r['entropy_source']:>10s} "
              f"{r['patch_complexity']:>6.4f} {r['test_stagnation']:>6.4f} "
              f"{r['patch_oscillation']:>6.4f} {r['edit_target_concentration']:>6.4f} "
              f"{'Y' if r['tests_passed'] else 'N':>5s}",
              file=sys.stderr)
    print(f"{'='*72}", file=sys.stderr)
    print(f"Outputs:", file=sys.stderr)
    print(f"  JSONL: {jsonl_path}", file=sys.stderr)
    print(f"  CSV:   {csv_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Ablation mode — evaluate all detector configs on the same trajectories
# ---------------------------------------------------------------------------

def run_ablation(
    output_dir: Path,
    rng: random.Random | None = None,
    use_native_client: bool = False,
) -> None:
    """
    Run all detector configurations from ABLATION_REGISTRY on the same set of
    trajectories and produce a comparison table.

    When use_native_client is True, generates live agent trajectories on
    VALIDATION_TASKS instead of the 40 scripted runs.
    """
    from drift.ablations import ABLATION_REGISTRY
    from collections import defaultdict

    output_dir.mkdir(parents=True, exist_ok=True)

    all_run_records: list[tuple[str, RunRecord, dict]] = []

    if use_native_client:
        print("[ablation] Running LIVE agent trajectories (native SDK)...", file=sys.stderr)
        for task_id in VALIDATION_TASKS:
            task = get_task(task_id)
            original_code = (REPO_ROOT / task["source_file"]).read_text()

            print(f"  [{task_id}] running agent...", file=sys.stderr)
            log_entries = _run_live_agent(
                task_id=task_id,
                use_native_client=True,
            )
            if not log_entries:
                print(f"  [{task_id}] no entries returned, skipping", file=sys.stderr)
                continue

            run_label = _label_run(log_entries)
            run = build_run_record(task_id, "live", log_entries, run_label, original_code)
            all_run_records.append(("live", run, run_label))

            status = "SUCCESS" if run.success else f"FAIL(t={run.t_fail})"
            print(f"  [{task_id}] {status} — {len(run.steps)} step(s)", file=sys.stderr)

        print(f"[ablation] {len(all_run_records)} live trajectories collected.", file=sys.stderr)

    else:
        assert rng is not None, "rng is required for scripted ablation"
        task_ids = list_task_ids()
        total = len(task_ids) * len(TEMPLATES)
        print(f"[ablation] Generating {total} scripted trajectories...", file=sys.stderr)

        for ti, task_id in enumerate(task_ids):
            task = get_task(task_id)
            original_code = (REPO_ROOT / task["source_file"]).read_text()

            for tname, tfunc in TEMPLATES.items():
                run_id = f"{task_id}-{tname}"
                steps = tfunc(task, rng)

                log_entries = _run_trajectory(task_id, steps, run_id=run_id)
                run_label = _label_run(log_entries)

                run = build_run_record(task_id, tname, log_entries, run_label, original_code)
                all_run_records.append((tname, run, run_label))

                idx = ti * len(TEMPLATES) + list(TEMPLATES).index(tname) + 1
                status = "OK" if run.success else f"FAIL(t={run.t_fail})"
                print(f"  [{idx:2d}/{total}] {run_id:40s} {status}", file=sys.stderr)

    print(f"\n[ablation] Evaluating {len(ABLATION_REGISTRY)} detector configs...", file=sys.stderr)

    results_rows: list[dict] = []

    for short_name, label, config in ABLATION_REGISTRY:
        for tname, run, run_label in all_run_records:
            run_copy = copy.deepcopy(run)
            run_detector_on_run(run_copy, config)

            detected = run_copy.t_alarm is not None
            lead_time = None
            if not run_copy.success and detected and run_copy.t_fail is not None:
                lead_time = run_copy.t_fail - run_copy.t_alarm

            results_rows.append({
                "detector": short_name,
                "detector_label": label,
                "task_id": run_copy.task_id,
                "run_id": run_copy.run_id,
                "template": tname,
                "success": run_copy.success,
                "detected": detected,
                "t_alarm": run_copy.t_alarm,
                "t_fail": run_copy.t_fail,
                "lead_time": lead_time,
                "n_steps": len(run_copy.steps),
            })

    ablation_csv = output_dir / "ablation_results.csv"
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(ablation_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n{'='*90}", file=sys.stderr)
    print("ABLATION RESULTS", file=sys.stderr)
    print(f"{'='*90}", file=sys.stderr)
    print(f"{'Detector':<35s} {'Runs':>5s} {'Det':>5s} {'Miss':>5s} {'FP':>4s} "
          f"{'DR':>7s} {'FAR':>7s} {'Lead':>6s}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)

    by_detector: dict[str, list[dict]] = defaultdict(list)
    for row in results_rows:
        by_detector[row["detector"]].append(row)

    for short_name, label, _ in ABLATION_REGISTRY:
        rows = by_detector[short_name]
        failures = [r for r in rows if not r["success"]]
        successes = [r for r in rows if r["success"]]
        detected_failures = sum(1 for r in failures if r["detected"])
        missed_failures = len(failures) - detected_failures
        false_positives = sum(1 for r in successes if r["detected"])
        dr = detected_failures / max(len(failures), 1)
        far = false_positives / max(len(successes), 1)
        lead_times = [r["lead_time"] for r in failures if r["lead_time"] is not None]
        avg_lead = f"{sum(lead_times)/len(lead_times):.1f}" if lead_times else "-"

        print(f"{label:<35s} {len(rows):>5d} {detected_failures:>5d} {missed_failures:>5d} "
              f"{false_positives:>4d} {dr:>6.1%} {far:>6.1%} {avg_lead:>6s}", file=sys.stderr)

    print(f"{'='*90}", file=sys.stderr)
    print(f"\nCSV: {ablation_csv}", file=sys.stderr)

    ablation_jsonl = output_dir / "ablation_steps.jsonl"
    lines: list[str] = []
    for short_name, label, config in ABLATION_REGISTRY:
        for tname, run, _ in all_run_records:
            run_copy = copy.deepcopy(run)
            run_detector_on_run(run_copy, config)
            for step in run_copy.steps:
                d = step.to_dict()
                d["run_id"] = run_copy.run_id
                d["task_id"] = run_copy.task_id
                d["template"] = tname
                d["detector"] = short_name
                d["detector_label"] = label
                lines.append(json.dumps(d))
    ablation_jsonl.write_text("\n".join(lines) + "\n")
    print(f"JSONL: {ablation_jsonl}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Scripted batch (original evaluation pipeline)
# ---------------------------------------------------------------------------

def _run_scripted_batch(output_dir: Path, rng: random.Random) -> None:
    """Run all scripted template trajectories and produce CSV + ROC figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    eo_config = DetectorConfig(use_patch_complexity=False)
    comp_config = DetectorConfig(use_patch_complexity=True)

    rows = []
    all_jsonl_lines = []

    task_ids = list_task_ids()
    total = len(task_ids) * len(TEMPLATES)
    print(f"Running {total} trajectories ({len(task_ids)} tasks × {len(TEMPLATES)} templates)...",
          file=sys.stderr)

    for ti, task_id in enumerate(task_ids):
        task = get_task(task_id)
        original_code = (REPO_ROOT / task["source_file"]).read_text()

        for tname, tfunc in TEMPLATES.items():
            run_id = f"{task_id}-{tname}"
            steps = tfunc(task, rng)

            log_entries = _run_trajectory(task_id, steps, run_id=run_id)
            run_label = _label_run(log_entries)

            run = build_run_record(task_id, tname, log_entries, run_label, original_code)

            run_eo = copy.deepcopy(run)
            run_detector_on_run(run_eo, eo_config)

            run_comp = copy.deepcopy(run)
            run_detector_on_run(run_comp, comp_config)

            row = make_row(tname, run, run_eo, run_comp)
            rows.append(row)

            for step in run.steps:
                d = step.to_dict()
                d["run_id"] = run_id
                d["task_id"] = task_id
                d["template"] = tname
                d["eo_severity"] = run_eo.steps[step.step].severity
                d["comp_severity"] = run_comp.steps[step.step].severity
                all_jsonl_lines.append(json.dumps(d))

            status = "OK" if run.success else f"FAIL(t={run.t_fail})"
            idx = ti * len(TEMPLATES) + list(TEMPLATES).index(tname) + 1
            print(f"  [{idx:2d}/{total}] {run_id:40s} {status:12s} "
                  f"eo={'Y' if run_eo.t_alarm is not None else '-'}  "
                  f"comp={'Y' if run_comp.t_alarm is not None else '-'}",
                  file=sys.stderr)

    jsonl_path = output_dir / "all_steps.jsonl"
    jsonl_path.write_text("\n".join(all_jsonl_lines) + "\n")

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_success = sum(1 for r in rows if r["success"])
    n_fail = n_total - n_success

    eo_detected_fails = sum(1 for r in rows if not r["success"] and r["eo_detected"])
    comp_detected_fails = sum(1 for r in rows if not r["success"] and r["comp_detected"])
    eo_false_alarms = sum(1 for r in rows if r["success"] and r["eo_detected"])
    comp_false_alarms = sum(1 for r in rows if r["success"] and r["comp_detected"])

    dr_eo = eo_detected_fails / max(n_fail, 1)
    dr_comp = comp_detected_fails / max(n_fail, 1)
    far_eo = eo_false_alarms / max(n_success, 1)
    far_comp = comp_false_alarms / max(n_success, 1)

    eo_lead_times = [r["eo_lead_time"] for r in rows if r["eo_lead_time"] is not None]
    comp_lead_times = [r["comp_lead_time"] for r in rows if r["comp_lead_time"] is not None]

    n_confident_wrong = sum(1 for r in rows if r["is_confident_wrong"])
    n_rescued = sum(1 for r in rows if r["rescued_by_composite"])

    labels = [not r["success"] for r in rows]
    eo_scores = [r["max_entropy"] for r in rows]
    comp_scores = [max(r["max_entropy"], r["max_pcs"] * 5.0) for r in rows]

    eo_roc, eo_auc = compute_roc(eo_scores, labels)
    comp_roc, comp_auc = compute_roc(comp_scores, labels)

    fig_path = output_dir / "roc_comparison.png"
    fig_ok = generate_figure(eo_roc, eo_auc, comp_roc, comp_auc, fig_path)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total runs:          {n_total}  ({n_success} success, {n_fail} fail)")
    print(f"")
    print(f"{'Metric':<28s} {'Entropy-Only':>14s} {'Composite':>14s}")
    print(f"{'-'*28} {'-'*14} {'-'*14}")
    print(f"{'Detection rate (recall)':<28s} {dr_eo:>13.1%} {dr_comp:>13.1%}")
    print(f"{'False alarm rate':<28s} {far_eo:>13.1%} {far_comp:>13.1%}")
    print(f"{'ROC AUC':<28s} {eo_auc:>14.3f} {comp_auc:>14.3f}")
    med_eo = sorted(eo_lead_times)[len(eo_lead_times)//2] if eo_lead_times else None
    med_comp = sorted(comp_lead_times)[len(comp_lead_times)//2] if comp_lead_times else None
    eo_lt_str = f"{med_eo}" if med_eo is not None else "n/a"
    comp_lt_str = f"{med_comp}" if med_comp is not None else "n/a"
    print(f"{'Median lead time (steps)':<28s} {eo_lt_str:>14s} {comp_lt_str:>14s}")
    print(f"")
    print(f"GAP C Analysis:")
    print(f"  Confident-wrong failures (entropy missed): {n_confident_wrong}")
    print(f"  Rescued by composite:                      {n_rescued}")
    print(f"  Rescue rate:                               "
          f"{n_rescued/max(n_confident_wrong,1):.0%}" if n_confident_wrong else "n/a")
    print(f"")
    print(f"Outputs:")
    print(f"  CSV:    {csv_path}")
    print(f"  JSONL:  {jsonl_path}")
    if fig_ok:
        print(f"  Figure: {fig_path}")
    print("=" * 65)

    stats = {
        "n_total": n_total, "n_success": n_success, "n_fail": n_fail,
        "dr_eo": round(dr_eo, 4), "dr_comp": round(dr_comp, 4),
        "far_eo": round(far_eo, 4), "far_comp": round(far_comp, 4),
        "auc_eo": round(eo_auc, 4), "auc_comp": round(comp_auc, 4),
        "median_lead_time_eo": med_eo, "median_lead_time_comp": med_comp,
        "n_confident_wrong": n_confident_wrong, "n_rescued": n_rescued,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of drift detectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode", choices=["scripted", "validate", "both", "ablate"], default="scripted",
        help="Evaluation mode: scripted, validate, both, or ablate (default: scripted)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model for live/validate modes")
    parser.add_argument("--max-steps", type=int, default=6, help="Max agent steps for live modes")
    parser.add_argument(
        "--native-client", action="store_true",
        help="Use native Tinker SDK for real logprobs (requires tinker + transformers)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    native = getattr(args, "native_client", False)

    if args.mode == "validate":
        run_validation(
            output_dir=output_dir,
            model=args.model,
            max_steps=args.max_steps,
            use_native_client=native,
        )
    elif args.mode == "both":
        run_validation(
            output_dir=output_dir,
            model=args.model,
            max_steps=args.max_steps,
            use_native_client=native,
        )
        rng = random.Random(args.seed)
        _run_scripted_batch(output_dir, rng)
    elif args.mode == "ablate":
        rng = random.Random(args.seed)
        run_ablation(output_dir, rng, use_native_client=native)
    else:
        rng = random.Random(args.seed)
        _run_scripted_batch(output_dir, rng)


if __name__ == "__main__":
    main()
