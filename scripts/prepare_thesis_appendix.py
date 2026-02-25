#!/usr/bin/env python3
"""
Prepare thesis appendix material from PVP results.

Reads a PVP run directory (e.g. pvp_run3) and writes:
  - Appendix A: Full per-scenario metric tables (CSV + Markdown), Phase 0 MAE_q table
  - Appendix B: Manifest of trajectory/plot artifacts (for supplementary plots)
  - Appendix C: Phase 4 reproducibility raw data summary + hardware env template
  - Appendix D: HIL timing and tolerance band tables (SC-6a, SC-6b placeholder)
  - Phase 5 table: HIL metrics (R12, R13) and SC-6a Pass/Fail per metric (CSV + Markdown)
  - Appendix E: PVP logging checklist and JSON file naming conventions
  - Appendix F: Motor parameters and SNN training hyperparameters (reference tables)

Usage:
  poetry run python embark-evaluation/scripts/prepare_thesis_appendix.py
  poetry run python embark-evaluation/scripts/prepare_thesis_appendix.py --run pvp_run3
  poetry run python embark-evaluation/scripts/prepare_thesis_appendix.py --output docs/thesis_appendix

Default output: embark-evaluation/appendix (so it appears in the repo tree when opening the project).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

# Repo paths
SCRIPT_DIR = Path(__file__).resolve().parent
EMBARK_EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EMBARK_EVAL_DIR.parent
for _p in [str(REPO_ROOT), str(EMBARK_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_BASE = EMBARK_EVAL_DIR / "pvp" / "results"

# Metrics to include in Appendix A full table (Pipeline / PVP names)
APPENDIX_A_METRICS = ["mae_i_q", "itae_i_q", "settling_time_i_q", "overshoot"]
APPENDIX_A_METRIC_LABELS = {"mae_i_q": "MAE i_q (A)", "itae_i_q": "ITAE i_q (A·s²)", "settling_time_i_q": "Settling time (s)", "overshoot": "Overshoot (%)"}


def _load_json(path: Path) -> Any:
    """Load JSON, tolerating non-standard 'Infinity' and 'NaN' in files."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r":\s*Infinity\b", ": null", text)
    text = re.sub(r":\s*NaN\b", ": null", text)
    return json.loads(text)


def _safe_float(val: Any) -> float | str:
    """Convert to float for table; use '—' for non-finite or missing."""
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        if val != val or abs(val) == float("inf"):
            return "—"
        return val
    return "—"


def _format_cell(val: Any) -> str:
    """Format a table cell for CSV/Markdown."""
    v = _safe_float(val)
    if v == "—":
        return "—"
    if isinstance(v, float):
        if v >= 1e-2 or (v != 0 and v < 1e-6):
            return f"{v:.4g}"
        return f"{v:.6f}"
    return str(v)


def build_appendix_a(results_dir: Path, out_dir: Path) -> None:
    """Appendix A: Full per-scenario metric table (PI + 3 SNNs × 6 scenarios) + Phase 0 MAE_q."""
    phase3_dir = results_dir / "phase3_discriminative"
    phase0_dir = results_dir / "phase0_ground_truth"
    if not phase3_dir.exists():
        print(f"  [skip] Appendix A: {phase3_dir} not found")
        return

    summaries_path = phase3_dir / "phase3_summaries.json"
    if not summaries_path.exists():
        print(f"  [skip] Appendix A: {summaries_path} not found")
        return

    try:
        summaries = _load_json(summaries_path)
    except Exception as e:
        print(f"  [skip] Appendix A: could not load summaries: {e}")
        return

    # Build full table: rows = (agent, scenario), columns = MAE, ITAE, settling, overshoot
    rows: list[dict[str, Any]] = []
    scenario_list = sorted(
        dict.fromkeys(
            sr.get("name") or sr.get("scenario_name")
            for m in summaries.values()
            for sr in (m.get("scenarios") or m.get("scenario_results") or [])
            if (sr.get("name") or sr.get("scenario_name"))
        )
    )

    seen = set()
    for model_name, summary in summaries.items():
        scenarios_data = summary.get("scenarios") or summary.get("scenario_results") or []
        for sr in scenarios_data:
            name = sr.get("name") or sr.get("scenario_name")
            if not name or (model_name, name) in seen:
                continue
            seen.add((model_name, name))
            metrics = sr.get("metrics") or {}
            row = {
                "Agent": model_name,
                "Scenario": name,
                **{APPENDIX_A_METRIC_LABELS.get(k, k): _safe_float(metrics.get(k)) for k in APPENDIX_A_METRICS},
            }
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: one row per (agent, scenario) with columns Agent, Scenario, MAE, ITAE, Settling, Overshoot
    csv_path = out_dir / "appendix_A_full_metrics_table.csv"
    if rows:
        fieldnames = ["Agent", "Scenario"] + [APPENDIX_A_METRIC_LABELS[k] for k in APPENDIX_A_METRICS]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                out = {}
                for k in fieldnames:
                    v = r.get(k)
                    out[k] = _format_cell(v) if k not in ("Agent", "Scenario") else (v if v is not None else "—")
                w.writerow(out)
        print(f"  Wrote {csv_path}")

    # Markdown: same table
    md_path = out_dir / "appendix_A_full_metrics_table.md"
    md_lines = [
        "# Appendix A — Full Per-Scenario Metric Tables",
        "",
        "Complete MAE, ITAE, settling time, and overshoot for PI + 3 SNNs across all scenarios.",
        "",
        "## Full metrics (4 agents × 6 scenarios)",
        "",
    ]
    if rows:
        headers = ["Agent", "Scenario"] + [APPENDIX_A_METRIC_LABELS[k] for k in APPENDIX_A_METRICS]
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            cells = []
            for h in headers:
                v = r.get(h)
                cells.append(_format_cell(v) if h not in ("Agent", "Scenario") else (str(v) if v is not None else "—"))
            md_lines.append("| " + " | ".join(cells) + " |")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Wrote {md_path}")

    # Phase 0 raw MAE_q
    phase0_rankings_path = phase0_dir / "phase0_rankings.json"
    phase0_md_path = out_dir / "appendix_A_phase0_MAE_q.md"
    if phase0_rankings_path.exists():
        try:
            rankings = _load_json(phase0_rankings_path)
            scens = sorted(next(iter(rankings.values())).keys()) if rankings else []
            md0 = [
                "# Appendix A — Phase 0 Raw MAE_q (Ground Truth)",
                "",
                "MAE_q [A] per model per scenario from wrapper-free Phase 0 calibration (R0a–R0c).",
                "",
                "| Model | " + " | ".join(scens) + " |",
                "| --- | " + " | ".join("---" for _ in scens) + " |",
            ]
            for model, scen_mae in rankings.items():
                md0.append("| " + model + " | " + " | ".join(_format_cell(scen_mae.get(s)) for s in scens) + " |")
            phase0_md_path.write_text("\n".join(md0), encoding="utf-8")
            print(f"  Wrote {phase0_md_path}")
        except Exception as e:
            print(f"  [skip] Phase 0 MAE_q table: {e}")
    else:
        phase0_md_path.write_text("# Appendix A — Phase 0 MAE_q\n\n(No phase0_rankings.json found for this run.)\n", encoding="utf-8")
        print(f"  Wrote {phase0_md_path} (no data)")


def build_appendix_b(results_dir: Path, out_dir: Path) -> None:
    """Appendix B: Manifest of trajectory files and plot instructions for supplementary plots."""
    phase3_dir = results_dir / "phase3_discriminative"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Appendix B — Supplementary Trajectory Plots",
        "",
        "The chapter body keeps **S2** (step_mid_speed_1500rpm_2A) as the representative step response.",
        "This appendix lists artifacts for step response overlays and relative-error (Plot E1) for all scenarios.",
        "",
        "## Trajectory data (Phase 3)",
        "",
    ]
    if phase3_dir.exists():
        traj_files = sorted(phase3_dir.glob("trajectory_*.json"))
        if traj_files:
            lines.append("| File | Model | Scenario |")
            lines.append("| --- | --- | --- |")
            # Known scenario suffixes (longest first so we match correctly)
            scenario_suffixes = [
                "step_low_speed_500rpm_2A", "step_mid_speed_1500rpm_2A", "step_high_speed_2500rpm_2A",
                "multi_step_bidirectional_1500rpm", "four_quadrant_transition_1500rpm", "field_weakening_2500rpm",
            ]
            for tf in traj_files:
                stem = tf.stem.replace("trajectory_", "")
                model, scenario = stem, ""
                for suf in scenario_suffixes:
                    if stem.endswith("_" + suf):
                        model = stem[: -len(suf) - 1]
                        scenario = suf
                        break
                lines.append(f"| {tf.name} | {model} | {scenario} |")
        else:
            lines.append("(No trajectory_*.json files in phase3_discriminative.)")
    else:
        lines.append("(phase3_discriminative/ not found.)")
    lines.extend([
        "",
        "## Generating supplementary figures",
        "",
        "- **Step response overlays (S1, S3–S6):** Use the same plotting logic as Plot 3.1, selecting one scenario per figure.",
        "- **Relative error (Plot E1):** Already includes all scenarios in one figure; see `plots/utils/plot_phase3.py` → `plot_relative_error_vs_pi()`.",
        "",
    ])
    (out_dir / "appendix_B_trajectory_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_dir / 'appendix_B_trajectory_manifest.md'}")


def build_appendix_c(results_dir: Path, out_dir: Path) -> None:
    """Appendix C: Phase 4 raw data summary + full hardware environment spec."""
    phase4_dir = results_dir / "phase4_reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Appendix C — Phase 4 Reproducibility Raw Data",
        "",
        "Per-run metric values for R6, R7, R8 and full hardware/software environment for reproducibility.",
        "",
    ]
    if phase4_dir.exists():
        report_path = phase4_dir / "phase4_report.txt"
        if report_path.exists():
            lines.append("## Phase 4 report (excerpt)")
            lines.append("")
            lines.append("```")
            lines.extend(report_path.read_text(encoding="utf-8").splitlines()[:30])
            lines.append("...")
            lines.append("```")
            lines.append("")
        lines.append("## Raw per-run files")
        lines.append("")
        for f in ["R6_best_incremental_snn.json", "R7_best_incremental_snn.json", "R8_best_incremental_snn.json"]:
            p = phase4_dir / f
            lines.append(f"- `{f}` " + ("(present)" if p.exists() else "(missing)"))
        lines.append("")
    else:
        lines.append("(phase4_reproducibility/ not found.)")
        lines.append("")

    # Full hardware environment template (current machine when script runs)
    lines.extend([
        "## Hardware and software environment (template)",
        "",
        "Fill or verify when capturing for a specific PVP run:",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| CPU | {platform.processor() or '(unknown)'} |",
        f"| OS | {platform.system()} {platform.release()} |",
        f"| Kernel (if Linux) | (run `uname -r`) |",
        f"| Python | {platform.python_version()} |",
    ])
    try:
        import numpy as np
        lines.append(f"| NumPy | {np.__version__} |")
    except Exception:
        lines.append("| NumPy | (not available) |")
    try:
        import torch
        lines.append(f"| PyTorch | {torch.__version__} |")
    except Exception:
        lines.append("| PyTorch | (not available) |")
    lines.extend([
        "| Deterministic setting | `torch.use_deterministic_algorithms(True, warn_only=True)`; `torch.backends.cudnn.deterministic = True`; `torch.backends.cudnn.benchmark = False` |",
        "",
        "See `embark-evaluation/plots/utils/common.py` → `setup_deterministic(seed)`.",
        "",
    ])
    (out_dir / "appendix_C_phase4_reproducibility.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_dir / 'appendix_C_phase4_reproducibility.md'}")


# SC-6a tolerance bands (same as pvp/phase5_hil.py)
_PHASE5_TOLERANCE = {
    "mae_i_q": ("relative", 0.01),
    "mae_i_d": ("relative", 0.01),
    "itae_i_q": ("relative", 0.01),
    "itae_i_d": ("relative", 0.01),
    "settling_time_i_q": ("absolute", 0.0001),  # 0.1 ms
    "overshoot": ("absolute", 0.5),  # 0.5 pp
}


def _phase5_within_tolerance(v12: Any, v13: Any, metric_key: str) -> str:
    """Return PASS/FAIL for SC-6a; — if either value missing or non-finite."""
    spec = _PHASE5_TOLERANCE.get(metric_key)
    if spec is None:
        return "—"
    v12_f = v13_f = None
    if v12 is not None and isinstance(v12, (int, float)) and abs(v12) != float("inf") and v12 == v12:
        v12_f = float(v12)
    if v13 is not None and isinstance(v13, (int, float)) and abs(v13) != float("inf") and v13 == v13:
        v13_f = float(v13)
    if v12_f is None or v13_f is None:
        return "—"
    tol_type, tol_val = spec
    if tol_type == "relative":
        ref = max(abs(v12_f), 1e-20)
        dev = abs(v13_f - v12_f) / ref
        return "PASS" if dev <= tol_val else "FAIL"
    else:
        dev = abs(v13_f - v12_f)
        return "PASS" if dev <= tol_val else "FAIL"


def build_phase5_table(results_dir: Path, out_dir: Path) -> None:
    """Phase 5 HIL metrics table: R12, R13 per scenario/metric and SC-6a Pass/Fail (CSV + MD)."""
    phase5_dir = results_dir / "phase5_hil"
    results_path = phase5_dir / "phase5_results.json" if phase5_dir.exists() else None
    if not results_path or not results_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "appendix_phase5_metrics_table.md").write_text(
            "# Phase 5 — HIL Metrics Table\n\n(No phase5_hil/ or phase5_results.json found for this run.)\n",
            encoding="utf-8",
        )
        print(f"  Wrote {out_dir / 'appendix_phase5_metrics_table.md'} (no data)")
        return

    try:
        data = _load_json(results_path)
    except Exception as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "appendix_phase5_metrics_table.md").write_text(
            f"# Phase 5 — HIL Metrics Table\n\n(Could not load phase5_results.json: {e})\n",
            encoding="utf-8",
        )
        print(f"  Wrote {out_dir / 'appendix_phase5_metrics_table.md'} (error)")
        return

    r12 = data.get("R12", {})
    r13 = data.get("R13", {})
    phase5_metrics = ["mae_i_q", "mae_i_d", "itae_i_q", "itae_i_d", "settling_time_i_q", "overshoot"]
    phase5_metric_labels = {
        "mae_i_q": "MAE i_q (A)", "mae_i_d": "MAE i_d (A)",
        "itae_i_q": "ITAE i_q (A·s²)", "itae_i_d": "ITAE i_d (A·s²)",
        "settling_time_i_q": "Settling time (s)", "overshoot": "Overshoot (%)",
    }

    rows: list[dict[str, Any]] = []
    for sr in r12.get("scenarios") or []:
        name = sr.get("name") or sr.get("scenario_name") or "—"
        m12 = sr.get("metrics") or {}
        r13_match = None
        for s in r13.get("scenarios") or []:
            if (s.get("name") or s.get("scenario_name")) == name:
                r13_match = s.get("metrics") or {}
                break
        for mk in phase5_metrics:
            v12 = m12.get(mk)
            v13 = r13_match.get(mk) if r13_match else None
            pass_fail = _phase5_within_tolerance(v12, v13, mk)
            rows.append({
                "Scenario": name,
                "Metric": phase5_metric_labels.get(mk, mk),
                "R12": _format_cell(v12),
                "R13": _format_cell(v13),
                "SC-6a": pass_fail,
            })

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "appendix_phase5_metrics_table.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Scenario", "Metric", "R12", "R13", "SC-6a"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in ["Scenario", "Metric", "R12", "R13", "SC-6a"]})
    print(f"  Wrote {csv_path}")

    # Markdown
    md_lines = [
        "# Phase 5 — HIL Metrics Table (SC-6a)",
        "",
        "R12 and R13 metric values per scenario; SC-6a = within tolerance (R12 vs R13).",
        "",
        "| Scenario | Metric | R12 | R13 | SC-6a |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md_lines.append(f"| {r['Scenario']} | {r['Metric']} | {r['R12']} | {r['R13']} | {r['SC-6a']} |")
    (out_dir / "appendix_phase5_metrics_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Wrote {out_dir / 'appendix_phase5_metrics_table.md'}")


def build_appendix_d(results_dir: Path, out_dir: Path) -> None:
    """Appendix D: HIL timing and tolerance band tables (SC-6a; SC-6b if R11 available)."""
    phase5_dir = results_dir / "phase5_hil"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Appendix D — HIL Timing and Hardware Logs",
        "",
        "Raw clock counts (when available from `model.statistics.inference_clk`) and per-metric tolerance band tables.",
        "",
    ]
    if phase5_dir.exists():
        report_path = phase5_dir / "phase5_report.txt"
        if report_path.exists():
            lines.append("## Phase 5 report (SC-6a, SC-6c)")
            lines.append("")
            lines.append("```")
            lines.extend(report_path.read_text(encoding="utf-8").splitlines())
            lines.append("```")
            lines.append("")
        results_path = phase5_dir / "phase5_results.json"
        if results_path.exists():
            try:
                data = _load_json(results_path)
                r12 = data.get("R12", {})
                r13 = data.get("R13", {})
                lines.append("## Tolerance band check table (SC-6a: R12 vs R13)")
                lines.append("")
                lines.append("| Scenario | Metric | R12 | R13 | Within band? |")
                lines.append("| --- | --- | --- | --- | --- |")
                for sr in r12.get("scenarios") or []:
                    name = sr.get("name") or sr.get("scenario_name") or "—"
                    metrics = sr.get("metrics") or {}
                    for mk in ["mae_i_q", "itae_i_q", "settling_time_i_q", "overshoot"]:
                        v12 = metrics.get(mk)
                        r13_scens = (r13.get("scenarios") or [])
                        v13 = None
                        for s in r13_scens:
                            if (s.get("name") or s.get("scenario_name")) == name:
                                v13 = (s.get("metrics") or {}).get(mk)
                                break
                        v12s = _format_cell(v12)
                        v13s = _format_cell(v13)
                        lines.append(f"| {name} | {mk} | {v12s} | {v13s} | (see report) |")
                lines.append("")
            except Exception as e:
                lines.append(f"(Could not parse phase5_results.json: {e})")
                lines.append("")
        lines.append("**SC-6b (R12 vs R11):** R11 is produced by Phase 5-Q (Akida software simulation). Compare R12 metrics to R11 manually or via a separate script; add table here when available.")
    else:
        lines.append("(phase5_hil/ not found — run Phase 5 with HIL to generate data.)")
    lines.append("")
    lines.append("**Network-attached Akida:** If the lab uses a remote board over TCP, describe host, port, and transfer metric in this section.")
    (out_dir / "appendix_D_HIL_timing_and_tolerance.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_dir / 'appendix_D_HIL_timing_and_tolerance.md'}")


def build_appendix_e(out_dir: Path) -> None:
    """Appendix E: Logging checklist, JSON naming, version-control manifest template."""
    out_dir.mkdir(parents=True, exist_ok=True)
    content = """# Appendix E — Logging and File Structure

## PVP logging checklist

| Run(s) | Logged content |
| --- | --- |
| R1, R2 | Full trajectory (t, i_q_ref, i_q, u_q) |
| R3, R4, R5 | Full trajectory |
| R12 | Full trajectory |
| R0a–R0c, R6–R11, R13, R14 (opt) | Metrics only (MAE, ITAE, settling time, overshoot, SyOps, sparsity) |

After every run and every scenario: GEM environment, controller state, SNN hidden/membrane state, and metric accumulators are fully reset.

## JSON archive file naming conventions

| Pattern | Meaning |
| --- | --- |
| `phase0_ground_truth/phase0_rankings.json` | MAE_q per model per scenario (Phase 0) |
| `phase0_ground_truth/phase0_report.txt` | Human-readable Phase 0 report |
| `phase3_discriminative/trajectory_<model>_<scenario>.json` | Trajectory (t, i_q_ref, i_q, u_q) for one agent and one scenario |
| `phase3_discriminative/R2_PI_baseline.json`, `R3_*.json`, … | Full metrics per scenario for that run |
| `phase3_discriminative/phase3_summaries.json` | All agents’ scenario results combined |
| `phase3_discriminative/phase3_mae_table.json` | MAE i_q only, per model per scenario |
| `phase4_reproducibility/R6_best_incremental_snn.json`, R7, R8 | Per-repeat metrics (R6–R8) |
| `phase4_reproducibility/phase4_sigma_table.json` | Per-metric σ across repeats |
| `phase4_reproducibility/phase4_all_metrics.json` | All repeats’ raw metrics |
| `phase5_hil/R12_akida_hil.json`, `R13_akida_hil.json` | HIL run results |
| `phase5_hil/phase5_results.json` | R12 and R13 combined for plots |

## Version-control manifest (template)

Map each PVP run ID to exact artifacts for Open Science reproducibility:

| Run ID | Model weights | Motor/benchmark config | Script / commit |
| --- | --- | --- | --- |
| R0a–R0c | `models_for_evaluation/{best_incremental_snn,intermediate_scheduled_sampling,poor_no_tanh}/model.pt` | embark benchmark STANDARD_SCENARIOS | `pvp/phase0_ground_truth.py` |
| R1, R2 | — (PI) | same | `pvp/phase1_correctness.py` |
| R3–R5 | same as Phase 0 | same | `pvp/phase3_discriminative.py` |
| R6–R8 | `best_incremental_snn/model.pt` | same | `pvp/phase4_reproducibility.py` |
| R9–R11 | (Keras / Akida sim) | (Phase 5-Q) | — |
| R12–R13 | Akida .fbz from Phase 5-Q | same | `pvp/phase5_hil.py` |

Fill in commit hash or tag and config file paths when archiving a run.
"""
    (out_dir / "appendix_E_logging_and_file_structure.md").write_text(content, encoding="utf-8")
    print(f"  Wrote {out_dir / 'appendix_E_logging_and_file_structure.md'}")


def build_appendix_f(out_dir: Path) -> None:
    """Appendix F: Motor parameters and SNN training configuration (reference tables)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Motor table from README / GEM config
    motor_table = """# Appendix F — Motor Parameters and Training Configuration

## Motor electrical parameters (PMSM)

Used for PI gain derivation and benchmark simulation (gym-electric-motor / embark).

| Parameter | Value | Unit |
| --- | --- | --- |
| Pole pairs (p) | 3 | — |
| Stator resistance (R_s) | 0.543 | Ω |
| d-axis inductance (L_d) | 1.13 | mH |
| q-axis inductance (L_q) | 1.42 | mH |
| PM flux linkage (Ψ_PM) | 16.9 | mWb |
| Maximum current (I_max) | 10.8 | A |
| DC-link voltage (V_DC) | 48 | V |
| Maximum speed (n_max) | 3000 | RPM |

Source: README.md and embark/benchmark config; same values in `notebooks/train_snn_v12.py` (V12Config).

## SNN training hyperparameters (v12 — best_incremental_snn)

Reference from `notebooks/train_snn_v12.py` (V12Config). Other probe models (v9, v10) may differ; see their training notebooks.

| Parameter | Value |
| --- | --- |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size (Phase 1) | 256 |
| Phase 1 epochs | 5 (imitation) |
| Phase 2 epochs | 15 (closed-loop BPTT) |
| Rollout length (Phase 2) | 10 → 100 (curriculum) |
| Phase 2 batch size | 64 |
| Phase 2 steps per epoch | 300 |
| Phase 2 LR factor | 0.1 |
| Gradient clip (Phase 1) | 2.0 |
| Phase 2 gradient clip | 0.5 |
| BPTT | Closed-loop through differentiable PMSM |
| Delta u max (output) | 0.2 |
| Hidden sizes | [128, 96, 64] |
| Betas (membrane) | [0.96, 0.90, 0.82] |
| Rate steps | 48 |
| Input size | 13 (refs, prev voltage, EMAs; no derivatives) |
| Output size | 2 (delta u_d, delta u_q) |

Sampling stride and windowing: see training data generation and dataset in `evaluation/pytorch_snn/utils/dataset.py` and training scripts.
"""
    (out_dir / "appendix_F_motor_and_training_config.md").write_text(motor_table, encoding="utf-8")
    print(f"  Wrote {out_dir / 'appendix_F_motor_and_training_config.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare thesis appendix material from PVP results")
    parser.add_argument("--run", type=str, default="pvp_run3", help="PVP run name under pvp/results/")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: embark-evaluation/appendix)")
    args = parser.parse_args()

    results_dir = RESULTS_BASE / args.run
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        return 1

    out_dir = Path(args.output).resolve() if args.output else (EMBARK_EVAL_DIR / "appendix")
    print(f"Results: {results_dir}")
    print(f"Output:  {out_dir}")
    print("")

    build_appendix_a(results_dir, out_dir)
    build_appendix_b(results_dir, out_dir)
    build_appendix_c(results_dir, out_dir)
    build_appendix_d(results_dir, out_dir)
    build_phase5_table(results_dir, out_dir)
    build_appendix_e(out_dir)
    build_appendix_f(out_dir)

    print("")
    print("Done. Copy the generated files from the output directory into your thesis appendix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
