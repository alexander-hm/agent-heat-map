#!/usr/bin/env python3
"""
Generate publication-ready figures for the Agent Drift Heatmap paper appendix.
Version 2: Improved heatmaps, live-data-aware, readable at column width.

Usage:
  python scripts/gen_figures_v2.py [--input-dir outputs] [--output-dir outputs/figures]
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_PASS = "#2e7d32"
C_FAIL = "#c62828"
C_ENTROPY = "#5c6bc0"
C_PCS = "#ff7043"
C_TSS = "#26a69a"
C_POS = "#ab47bc"
C_ETC = "#ffa726"


def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def group_by_task(entries):
    """Group JSONL entries by task_id, deduplicate by step number."""
    groups = defaultdict(dict)
    for e in entries:
        task = e.get("task_id", "unknown")
        step = int(e.get("step", 0))
        # Keep the first (or most complete) entry per task+step
        if step not in groups[task]:
            groups[task][step] = e
    # Convert to sorted lists
    result = {}
    for task, step_dict in groups.items():
        result[task] = [step_dict[s] for s in sorted(step_dict.keys())]
    return result


def get_signal(entry, key, default=0.0):
    v = entry.get(key, default)
    if v is None:
        return default
    return float(v)


# ── Figure 1: Ablation Bar Chart (improved) ─────────────────────────────────

def fig_ablation_bar(csv_path, output_path):
    rows = load_csv(csv_path)
    by_det = defaultdict(list)
    for r in rows:
        by_det[r["detector_label"]].append(r)

    show_order = [
        ("Entropy+ConfΔ only", "Entropy+CΔ"),
        ("Original Entropy-Only", "Orig. Entropy"),
        ("PCS only", "PCS"),
        ("Test Stagnation only", "TSS"),
        ("Patch Oscillation only", "POS"),
        ("Edit Target Conc. only", "ETC"),
        ("Original Composite", "Orig. Comp."),
        ("Trajectory-Aware (TSS+POS+ETC)", "Traj.-Aware"),
        ("Full Composite (all signals)", "Full Comp."),
    ]

    labels, dr_vals, far_vals = [], [], []
    for full_name, short in show_order:
        if full_name not in by_det:
            continue
        group = by_det[full_name]
        failures = [r for r in group if r["success"].lower() in ("false", "0", "no")]
        successes = [r for r in group if r["success"].lower() in ("true", "1", "yes")]
        det_f = sum(1 for r in failures if r["detected"].lower() in ("true", "1", "yes"))
        fp = sum(1 for r in successes if r["detected"].lower() in ("true", "1", "yes"))
        dr = det_f / len(failures) * 100 if failures else 0
        far = fp / len(successes) * 100 if successes else 0
        labels.append(short)
        dr_vals.append(dr)
        far_vals.append(far)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars_dr = ax.bar(x - width/2, dr_vals, width, label="Detection Rate",
                     color=C_PASS, edgecolor="white", linewidth=0.5)
    bars_far = ax.bar(x + width/2, far_vals, width, label="False Alarm Rate",
                      color=C_FAIL, edgecolor="white", linewidth=0.5, alpha=0.75)

    ax.set_ylabel("Rate (%)")
    ax.set_title("Ablation: Detection Rate vs. False Alarm Rate (Live, n=8)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 118)
    ax.axhline(y=100, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    for bar in bars_dr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    for bar in bars_far:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=6.5, color=C_FAIL)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 2: Entropy Validation (all tasks) ────────────────────────────────

def fig_entropy_validation(jsonl_path, output_path):
    entries = load_jsonl(jsonl_path)
    by_task = group_by_task(entries)

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    cmap = plt.cm.Set2

    task_list = sorted(by_task.keys())
    offset = 0
    tick_positions, tick_labels = [], []

    for i, task_id in enumerate(task_list):
        steps = by_task[task_id]
        color = cmap(i / max(len(task_list) - 1, 1))

        for s in steps:
            ent = get_signal(s, "entropy")
            passed = s.get("tests_passed", False)
            if isinstance(passed, str):
                passed = passed.lower() in ("true", "1")
            marker = "o" if passed else "X"
            edge = C_PASS if passed else C_FAIL
            ax.scatter(offset, ent, c=[color], marker=marker, s=50,
                       edgecolors=edge, linewidth=1.5, zorder=3)
            offset += 1

        mid = offset - len(steps) / 2
        tick_positions.append(mid)
        short_name = task_id.replace("_", "\n")
        tick_labels.append(short_name)
        ax.axvline(x=offset - 0.5, color="gray", linestyle=":", linewidth=0.3, alpha=0.4)
        offset += 0.5

    ax.set_ylabel("Mean Entropy (nats)")
    ax.set_title("Real Token Entropy per Step (Qwen3-235B via Tinker)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_ylim(-0.005, max(0.1, ax.get_ylim()[1] * 1.1))

    # Reference band showing how narrow the range is
    ax.axhspan(0, 0.07, alpha=0.06, color=C_ENTROPY)
    ax.scatter([], [], marker="o", c="gray", edgecolors=C_PASS, label="Pass", s=30)
    ax.scatter([], [], marker="X", c="gray", edgecolors=C_FAIL, label="Fail", s=30)
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figures 3 & 4: Compact Signal Heatmaps ──────────────────────────────────

def fig_heatmap(jsonl_path, output_path, target_task=None, pick_failure=True):
    entries = load_jsonl(jsonl_path)
    by_task = group_by_task(entries)

    # Pick trajectory
    chosen_task, chosen_steps = None, None
    if target_task and target_task in by_task:
        chosen_task = target_task
        chosen_steps = by_task[target_task]
    else:
        # Find best candidate: for failure, pick longest; for success, pick one with >1 step
        best_len = 0
        for task_id, steps in by_task.items():
            last_passed = steps[-1].get("tests_passed", False)
            if isinstance(last_passed, str):
                last_passed = last_passed.lower() in ("true", "1")
            is_failure = not last_passed
            if is_failure == pick_failure:
                if pick_failure:
                    # Longest failure
                    if len(steps) > best_len:
                        best_len = len(steps)
                        chosen_task = task_id
                        chosen_steps = steps
                else:
                    # Success with most steps (more interesting than 1-step)
                    if len(steps) > best_len:
                        best_len = len(steps)
                        chosen_task = task_id
                        chosen_steps = steps

    if not chosen_steps:
        print(f"  Skipped heatmap: no {'failure' if pick_failure else 'success'} found")
        return

    n_steps = len(chosen_steps)
    signal_names = ["Entropy", "Conf. $\\Delta$", "PCS", "TSS", "POS", "ETC"]
    signal_keys = ["entropy", "confidence_delta", "patch_complexity",
                   "test_stagnation", "patch_oscillation", "edit_target_concentration"]
    n_signals = len(signal_names)

    # Build data matrix
    data = np.zeros((n_signals, n_steps))
    test_pass = []
    for j, step in enumerate(chosen_steps):
        for i, key in enumerate(signal_keys):
            data[i, j] = get_signal(step, key)
        passed = step.get("tests_passed", False)
        if isinstance(passed, str):
            passed = passed.lower() in ("true", "1")
        test_pass.append(passed)

    # Normalize each row independently for color mapping
    data_norm = np.zeros_like(data)
    for i in range(n_signals):
        row = data[i]
        rmin, rmax = row.min(), row.max()
        if rmax > rmin:
            data_norm[i] = (row - rmin) / (rmax - rmin)
        elif rmax == 0:
            data_norm[i] = 0.0
        else:
            data_norm[i] = 0.5

    # Size figure to content
    cell_w = 0.85
    cell_h = 0.5
    fig_w = max(4, n_steps * cell_w + 2.5)
    fig_h = n_signals * cell_h + 1.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "drift", ["#e8f5e9", "#fff9c4", "#ffcc80", "#ef5350"], N=256
    )
    im = ax.imshow(data_norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Annotate cells
    for i in range(n_signals):
        for j in range(n_steps):
            val = data[i, j]
            if signal_keys[i] == "confidence_delta":
                text = f"{val:.1f}"
            elif val == 0:
                text = "0"
            elif val >= 1:
                text = f"{val:.2f}"
            else:
                text = f"{val:.3f}"
            color = "black" if data_norm[i, j] < 0.65 else "white"
            fontsize = 8 if n_steps <= 8 else 6.5
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=color)

    # X labels with pass/fail coloring
    ax.set_xticks(range(n_steps))
    xlabels = [f"Step {j}" for j in range(n_steps)]
    ax.set_xticklabels(xlabels, fontsize=7.5)
    for j, p in enumerate(test_pass):
        ax.get_xticklabels()[j].set_color(C_PASS if p else C_FAIL)
        ax.get_xticklabels()[j].set_fontweight("bold")

    # Add pass/fail indicator bar below
    for j, p in enumerate(test_pass):
        color = C_PASS if p else C_FAIL
        rect = mpatches.FancyBboxPatch(
            (j - 0.4, n_signals - 0.5 + 0.15), 0.8, 0.25,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor="none", alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(j, n_signals - 0.5 + 0.27, "PASS" if p else "FAIL",
                ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")

    ax.set_yticks(range(n_signals))
    ax.set_yticklabels(signal_names, fontsize=8.5)

    outcome = "SUCCESS" if test_pass[-1] else "FAILURE"
    ax.set_title(f"{chosen_task.replace('_', ' ').title()} — {outcome} ({n_steps} steps)",
                 fontsize=11, fontweight="bold")

    # Minimal colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=15)
    cbar.set_label("Normalized", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_ylim(n_signals - 0.3, -0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 5: Signal Traces ─────────────────────────────────────────────────

def fig_signal_traces(jsonl_path, output_path, target_task=None):
    entries = load_jsonl(jsonl_path)
    by_task = group_by_task(entries)

    # Pick longest failure
    chosen_task, chosen_steps = target_task, None
    if target_task and target_task in by_task:
        chosen_steps = by_task[target_task]
    else:
        max_len = 0
        for task_id, steps in by_task.items():
            last_p = steps[-1].get("tests_passed", False)
            if isinstance(last_p, str):
                last_p = last_p.lower() in ("true", "1")
            if not last_p and len(steps) > max_len:
                max_len = len(steps)
                chosen_task = task_id
                chosen_steps = steps

    if not chosen_steps or len(chosen_steps) < 2:
        print("  Skipped signal traces: no suitable failure")
        return

    step_nums = list(range(len(chosen_steps)))

    signals = [
        ("Entropy", "entropy", C_ENTROPY, "o", "-"),
        ("PCS", "patch_complexity", C_PCS, "s", "-"),
        ("TSS", "test_stagnation", C_TSS, "^", "-"),
        ("POS", "patch_oscillation", C_POS, "D", "-"),
        ("ETC", "edit_target_concentration", C_ETC, "v", "-"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.2), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax = axes[0]
    for label, key, color, marker, ls in signals:
        vals = [get_signal(s, key) for s in chosen_steps]
        ax.plot(step_nums, vals, ls, color=color, marker=marker, label=label,
                markersize=6, linewidth=1.8, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_ylabel("Signal Value")
    ax.set_title(f"Signal Evolution: {chosen_task.replace('_', ' ')} (FAILURE)",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(-0.05, 1.1)

    # Bottom: test results
    ax2 = axes[1]
    for j, s in enumerate(chosen_steps):
        passed = s.get("tests_passed", False)
        if isinstance(passed, str):
            passed = passed.lower() in ("true", "1")
        n_p = int(s.get("n_passed", 0))
        n_f = int(s.get("n_failed", 0))
        ax2.bar(j, n_p, color=C_PASS, alpha=0.75, width=0.5)
        ax2.bar(j, -n_f, color=C_FAIL, alpha=0.75, width=0.5)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Tests")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xticks(step_nums)
    ax2.set_xticklabels([str(i) for i in step_nums])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 6: Dual Ablation (Scripted vs Live) ──────────────────────────────

def fig_dual_ablation(scripted_csv, live_csv, output_path):
    def rates(csv_path):
        rows = load_csv(csv_path)
        by_det = defaultdict(list)
        for r in rows:
            by_det[r["detector_label"]].append(r)
        result = {}
        for label, group in by_det.items():
            failures = [r for r in group if r["success"].lower() in ("false", "0", "no")]
            successes = [r for r in group if r["success"].lower() in ("true", "1", "yes")]
            det_f = sum(1 for r in failures if r["detected"].lower() in ("true", "1", "yes"))
            fp = sum(1 for r in successes if r["detected"].lower() in ("true", "1", "yes"))
            dr = det_f / len(failures) * 100 if failures else 0
            result[label] = dr
        return result

    show = [
        ("Entropy+ConfΔ only", "Entropy+CΔ"),
        ("PCS only", "PCS"),
        ("Test Stagnation only", "TSS"),
        ("Patch Oscillation only", "POS"),
        ("Edit Target Conc. only", "ETC"),
        ("Original Composite", "Orig. Comp."),
        ("Trajectory-Aware (TSS+POS+ETC)", "Traj.-Aware"),
        ("Full Composite (all signals)", "Full Comp."),
    ]

    s_rates = rates(scripted_csv) if Path(scripted_csv).exists() else {}
    l_rates = rates(live_csv) if Path(live_csv).exists() else {}

    labels, s_dr, l_dr = [], [], []
    for full, short in show:
        if full in s_rates or full in l_rates:
            labels.append(short)
            s_dr.append(s_rates.get(full, 0))
            l_dr.append(l_rates.get(full, 0))

    if not labels:
        print("  Skipped dual ablation: no data")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x - width/2, s_dr, width, label="Scripted (n=40)",
           color="#5c6bc0", edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, l_dr, width, label="Live (n=8)",
           color="#ef5350", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate: Scripted vs. Live Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 118)
    ax.axhline(y=100, color="gray", linestyle=":", linewidth=0.5)

    for i, (sv, lv) in enumerate(zip(s_dr, l_dr)):
        if sv > 0:
            ax.text(i - width/2, sv + 2, f"{sv:.0f}", ha="center", fontsize=6.5)
        if lv > 0:
            ax.text(i + width/2, lv + 2, f"{lv:.0f}", ha="center", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating appendix figures (v2)...\n")

    # Prefer live ablation data; fall back to validation; fall back to scripted
    live_jsonl = inp / "ablation_steps.jsonl"
    val_jsonl = inp / "validation_steps.jsonl"
    data_jsonl = live_jsonl if live_jsonl.exists() else val_jsonl

    # 1. Ablation bar chart
    ablation_csv = inp / "ablation_results.csv"
    if ablation_csv.exists():
        fig_ablation_bar(ablation_csv, out / "fig_ablation_bar.pdf")

    # 2. Entropy validation
    if data_jsonl.exists():
        fig_entropy_validation(data_jsonl, out / "fig_entropy_validation.pdf")

    # 3. Failure heatmap — pick a long failure
    if data_jsonl.exists():
        fig_heatmap(data_jsonl, out / "fig_heatmap_failure.pdf", pick_failure=True)

    # 4. Success heatmap — pick a success with >1 step if possible
    if data_jsonl.exists():
        fig_heatmap(data_jsonl, out / "fig_heatmap_success.pdf", pick_failure=False)

    # 5. Signal traces for failure
    if data_jsonl.exists():
        fig_signal_traces(data_jsonl, out / "fig_signal_traces.pdf")

    # 6. Dual ablation (if scripted CSV also exists)
    scripted_csv = inp / "ablation_results_scripted.csv"
    if scripted_csv.exists() and ablation_csv.exists():
        fig_dual_ablation(str(scripted_csv), str(ablation_csv),
                          out / "fig_dual_ablation.pdf")

    print(f"\nDone. Figures in: {out}/")


if __name__ == "__main__":
    main()