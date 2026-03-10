#!/usr/bin/env python3
"""
Generate drift heatmap visualizations from evaluation logs.

Produces:
  - A self-contained HTML report with per-run heatmap grids (default)
  - Terminal ASCII heatmaps for quick inspection (--terminal)

Usage:
    python scripts/gen_heatmap.py                           # HTML report
    python scripts/gen_heatmap.py --terminal                 # all runs, terminal
    python scripts/gen_heatmap.py --terminal --run-id X      # single run, terminal
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_steps(jsonl_path: Path) -> dict[str, list[dict]]:
    """Load all_steps.jsonl into {run_id: [step_dicts]} grouped by run."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        runs[d["run_id"]].append(d)
    for steps in runs.values():
        steps.sort(key=lambda s: s["step"])
    return dict(runs)


# ---------------------------------------------------------------------------
# Terminal heatmap
# ---------------------------------------------------------------------------

SEV_SYMBOLS = {"normal": "░░", "warning": "▓▓", "critical": "██", None: "··"}
SEV_ANSI = {"normal": "\033[92m", "warning": "\033[93m", "critical": "\033[91m", None: "\033[90m"}
RESET = "\033[0m"
BOLD = "\033[1m"


def terminal_heatmap(run_id: str, steps: list[dict]) -> str:
    """Render a single run as a terminal heatmap."""
    lines = []
    lines.append(f"{BOLD}{'─' * 70}")
    task = steps[0].get("task_id", "?")
    template = steps[0].get("template", "?")
    success = steps[-1].get("tests_passed", False)
    status = f"\033[92mSUCCESS{RESET}" if success else f"\033[91mFAIL{RESET}"
    lines.append(f"{BOLD}{run_id}{RESET}  ({task} / {template})  {status}")
    lines.append("")

    # Header
    step_nums = [s["step"] for s in steps]
    hdr = f"  {'Signal':<16s}"
    for s in step_nums:
        hdr += f" {s:^5d}"
    lines.append(hdr)
    lines.append(f"  {'─' * 16} " + " ".join("─────" for _ in step_nums))

    # Entropy row
    row = f"  {'Entropy':<16s}"
    for s in steps:
        e = s.get("entropy", 0) or 0
        row += f" {e:5.2f}"
    lines.append(row)

    # PCS row
    row = f"  {'PatchComplexity':<16s}"
    for s in steps:
        p = s.get("patch_complexity", 0) or 0
        row += f" {p:5.3f}"
    lines.append(row)

    # Conf delta row
    row = f"  {'ConfDelta':<16s}"
    for s in steps:
        c = s.get("confidence_delta", 0) or 0
        row += f" {c:5.2f}"
    lines.append(row)

    # TSS row
    row = f"  {'TestStagnation':<16s}"
    for s in steps:
        v = s.get("test_stagnation", 0) or 0
        row += f" {v:5.3f}"
    lines.append(row)

    # POS row
    row = f"  {'PatchOscillation':<16s}"
    for s in steps:
        v = s.get("patch_oscillation", 0) or 0
        row += f" {v:5.3f}"
    lines.append(row)

    # ETC row
    row = f"  {'EditTargetConc':<16s}"
    for s in steps:
        v = s.get("edit_target_concentration", 0) or 0
        row += f" {v:5.3f}"
    lines.append(row)

    # Entropy-only severity
    row = f"  {'EO Severity':<16s}"
    for s in steps:
        sev = s.get("eo_severity")
        sym = SEV_SYMBOLS.get(sev, "??")
        color = SEV_ANSI.get(sev, "")
        row += f" {color}{sym:^5s}{RESET}"
    lines.append(row)

    # Composite severity
    row = f"  {'Comp Severity':<16s}"
    for s in steps:
        sev = s.get("comp_severity")
        sym = SEV_SYMBOLS.get(sev, "??")
        color = SEV_ANSI.get(sev, "")
        row += f" {color}{sym:^5s}{RESET}"
    lines.append(row)

    # Tests row
    row = f"  {'Tests':<16s}"
    for s in steps:
        passed = s.get("tests_passed", False)
        sym = "\033[92m  ✓  \033[0m" if passed else "\033[91m  ✗  \033[0m"
        row += f" {sym}"
    lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agent Drift Heatmap Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
         background: #1a1a2e; color: #e0e0e0; padding: 24px; }
  h1 { color: #e94560; margin-bottom: 8px; font-size: 22px; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 13px; }
  .summary { background: #16213e; padding: 16px; border-radius: 8px;
             margin-bottom: 24px; display: inline-block; }
  .summary td { padding: 3px 14px 3px 0; }
  .summary .label { color: #888; }
  .summary .val { font-weight: bold; }
  .run { background: #16213e; border-radius: 8px; padding: 16px;
         margin-bottom: 16px; border-left: 4px solid #333; }
  .run.success { border-left-color: #4CAF50; }
  .run.fail { border-left-color: #f44336; }
  .run-header { display: flex; justify-content: space-between;
                align-items: center; margin-bottom: 10px; }
  .run-id { font-weight: bold; font-size: 14px; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  .badge-ok { background: #4CAF50; color: #fff; }
  .badge-fail { background: #f44336; color: #fff; }
  .badge-rescued { background: #FF9800; color: #fff; margin-left: 6px; }
  .grid { display: grid; gap: 2px; margin-top: 8px; }
  .grid-row { display: contents; }
  .row-label { background: #0f3460; padding: 4px 8px; font-size: 11px;
               color: #aaa; display: flex; align-items: center;
               min-width: 110px; border-radius: 3px; }
  .cell { padding: 4px 6px; text-align: center; font-size: 11px;
          border-radius: 3px; min-width: 52px; }
  .cell-hdr { background: #0f3460; color: #aaa; font-weight: bold; }
  .sev-normal  { background: #1b5e20; color: #a5d6a7; }
  .sev-warning { background: #e65100; color: #ffcc80; }
  .sev-critical { background: #b71c1c; color: #ef9a9a; }
  .sev-none { background: #333; color: #666; }
  .test-pass { background: #1b5e20; color: #a5d6a7; }
  .test-fail { background: #b71c1c; color: #ef9a9a; }
  .metric-cell { background: #1a1a2e; color: #ccc; border: 1px solid #333; }
  .legend { margin-top: 24px; padding: 12px; background: #16213e;
            border-radius: 8px; font-size: 12px; }
  .legend span { margin-right: 16px; }
  .leg-box { display: inline-block; width: 14px; height: 14px;
             vertical-align: middle; margin-right: 4px; border-radius: 2px; }
</style>
</head>
<body>
<h1>Agent Drift Heatmap Report</h1>
<p class="subtitle">Generated from outputs/all_steps.jsonl &mdash; x-axis: steps, colors: drift severity</p>

{summary_html}

<div class="legend">
  <strong>Legend:</strong>
  <span><span class="leg-box" style="background:#1b5e20"></span> Normal</span>
  <span><span class="leg-box" style="background:#e65100"></span> Warning</span>
  <span><span class="leg-box" style="background:#b71c1c"></span> Critical</span>
  <span style="margin-left:20px">EO = Entropy-Only &nbsp; Comp = Composite (entropy + patch-complexity)</span>
</div>

{runs_html}

</body>
</html>
"""


def _sev_class(sev: str | None) -> str:
    return f"sev-{sev}" if sev in ("normal", "warning", "critical") else "sev-none"


def html_run_block(run_id: str, steps: list[dict]) -> str:
    task = steps[0].get("task_id", "?")
    template = steps[0].get("template", "?")
    success = steps[-1].get("tests_passed", False)
    cls = "success" if success else "fail"
    badge = '<span class="badge badge-ok">SUCCESS</span>' if success else '<span class="badge badge-fail">FAIL</span>'

    # Check if this is a rescued confident-wrong case
    all_eo_normal = all(s.get("eo_severity") == "normal" for s in steps)
    any_comp_warn = any(s.get("comp_severity") in ("warning", "critical") for s in steps)
    rescued = (not success) and all_eo_normal and any_comp_warn
    if rescued:
        badge += ' <span class="badge badge-rescued">RESCUED BY COMPOSITE</span>'

    n_steps = len(steps)
    cols = n_steps + 1  # +1 for row label column

    rows_html = []

    # Header row
    cells = ['<div class="row-label"></div>']
    for s in steps:
        cells.append(f'<div class="cell cell-hdr">Step {s["step"]}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # Entropy row
    cells = ['<div class="row-label">Entropy</div>']
    for s in steps:
        e = s.get("entropy", 0) or 0
        cells.append(f'<div class="cell metric-cell">{e:.2f}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # PCS row
    cells = ['<div class="row-label">Patch Complexity</div>']
    for s in steps:
        p = s.get("patch_complexity", 0) or 0
        cells.append(f'<div class="cell metric-cell">{p:.3f}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # TSS row
    cells = ['<div class="row-label">Test Stagnation</div>']
    for s in steps:
        v = s.get("test_stagnation", 0) or 0
        cells.append(f'<div class="cell metric-cell">{v:.3f}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # POS row
    cells = ['<div class="row-label">Patch Oscillation</div>']
    for s in steps:
        v = s.get("patch_oscillation", 0) or 0
        cells.append(f'<div class="cell metric-cell">{v:.3f}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # ETC row
    cells = ['<div class="row-label">Edit Target Conc.</div>']
    for s in steps:
        v = s.get("edit_target_concentration", 0) or 0
        cells.append(f'<div class="cell metric-cell">{v:.3f}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # EO severity row
    cells = ['<div class="row-label">EO Severity</div>']
    for s in steps:
        sev = s.get("eo_severity")
        label = (sev or "–").upper()[:4]
        cells.append(f'<div class="cell {_sev_class(sev)}">{label}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # Composite severity row
    cells = ['<div class="row-label">Comp Severity</div>']
    for s in steps:
        sev = s.get("comp_severity")
        label = (sev or "–").upper()[:4]
        cells.append(f'<div class="cell {_sev_class(sev)}">{label}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    # Tests row
    cells = ['<div class="row-label">Tests</div>']
    for s in steps:
        passed = s.get("tests_passed", False)
        cls_t = "test-pass" if passed else "test-fail"
        sym = "✓" if passed else "✗"
        np = s.get("n_passed", 0)
        nf = s.get("n_failed", 0)
        cells.append(f'<div class="cell {cls_t}">{sym} {np}/{np+nf}</div>')
    rows_html.append(f'<div class="grid-row">{"".join(cells)}</div>')

    grid_style = f"grid-template-columns: 110px repeat({n_steps}, 1fr);"

    return f"""
<div class="run {cls}">
  <div class="run-header">
    <span class="run-id">{run_id} <span style="color:#888;font-weight:normal">({task} / {template})</span></span>
    {badge}
  </div>
  <div class="grid" style="{grid_style}">
    {"".join(rows_html)}
  </div>
</div>
"""


def generate_html_report(runs: dict[str, list[dict]], output_path: Path, stats_path: Path | None = None):
    """Generate a self-contained HTML heatmap report."""

    # Summary section
    stats = {}
    if stats_path and stats_path.exists():
        stats = json.loads(stats_path.read_text())

    summary_rows = ""
    if stats:
        summary_rows = f"""
<table>
<tr><td class="label">Total runs</td><td class="val">{stats.get('n_total', '?')}</td>
    <td class="label" style="padding-left:24px">Success</td><td class="val">{stats.get('n_success', '?')}</td>
    <td class="label" style="padding-left:24px">Fail</td><td class="val">{stats.get('n_fail', '?')}</td></tr>
<tr><td class="label">EO Detection Rate</td><td class="val">{stats.get('dr_eo', 0):.1%}</td>
    <td class="label" style="padding-left:24px">Comp Detection Rate</td><td class="val">{stats.get('dr_comp', 0):.1%}</td></tr>
<tr><td class="label">EO AUC</td><td class="val">{stats.get('auc_eo', 0):.3f}</td>
    <td class="label" style="padding-left:24px">Comp AUC</td><td class="val">{stats.get('auc_comp', 0):.3f}</td></tr>
<tr><td class="label">Confident-wrong rescued</td><td class="val">{stats.get('n_rescued', 0)} / {stats.get('n_confident_wrong', 0)}</td></tr>
</table>
"""
    summary_html = f'<div class="summary">{summary_rows}</div>' if summary_rows else ""

    # Per-run blocks (failures first, then successes)
    ordered = sorted(runs.items(), key=lambda kv: (kv[1][-1].get("tests_passed", False), kv[0]))
    runs_html = "\n".join(html_run_block(rid, steps) for rid, steps in ordered)

    html = HTML_TEMPLATE.replace("{summary_html}", summary_html).replace("{runs_html}", runs_html)
    output_path.write_text(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate drift heatmap visualizations")
    parser.add_argument("--input", default="outputs/all_steps.jsonl", help="Path to all_steps.jsonl")
    parser.add_argument("--output", default="outputs/heatmap_report.html", help="HTML output path")
    parser.add_argument("--terminal", action="store_true", help="Print terminal heatmap instead of HTML")
    parser.add_argument("--run-id", type=str, default=None, help="Show only this run (terminal mode)")
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    if not jsonl_path.is_absolute():
        jsonl_path = REPO_ROOT / jsonl_path

    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found. Run eval_runs.py first.", file=sys.stderr)
        sys.exit(1)

    runs = load_steps(jsonl_path)

    if args.terminal:
        if args.run_id:
            if args.run_id not in runs:
                print(f"Error: run_id '{args.run_id}' not found. Available: {list(runs.keys())[:5]}...",
                      file=sys.stderr)
                sys.exit(1)
            print(terminal_heatmap(args.run_id, runs[args.run_id]))
        else:
            for rid, steps in sorted(runs.items()):
                print(terminal_heatmap(rid, steps))
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        stats_path = output_path.parent / "stats.json"
        generate_html_report(runs, output_path, stats_path)
        print(f"HTML report: {output_path}", file=sys.stderr)
        print(f"Contains {len(runs)} run heatmaps", file=sys.stderr)


if __name__ == "__main__":
    main()
