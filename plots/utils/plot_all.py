"""
PVP Plot Orchestrator — Generate all plots from a completed PVP run.

Reads results from each phase subdirectory and generates publication-ready
plots into ``embark-evaluation/plots/<run_name>/``, organized by phase.

Usage:
    poetry run python embark-evaluation/plots/utils/plot_all.py --results-dir embark-evaluation/pvp/results/<run_name>
    poetry run python embark-evaluation/plots/utils/plot_all.py --results-dir embark-evaluation/pvp/results/pvp_run1 --skip 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path setup — scripts live under embark-evaluation/plots/utils/
_this_file = Path(__file__).resolve()
_plots_utils_dir = _this_file.parent  # embark-evaluation/plots/utils/
_plots_dir = _plots_utils_dir.parent  # embark-evaluation/plots/
_embark_eval_dir = _plots_dir.parent  # embark-evaluation/
_repo_root = _embark_eval_dir.parent  # repo root

for _p in [str(_repo_root), str(_embark_eval_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def generate_all_plots(results_base: Path, skip: set[int] | None = None) -> None:
    """Generate plots for all PVP phases from a completed run.

    Parameters
    ----------
    results_base : Path
        The run-level results directory, e.g., ``results/pvp_run1/``.
        Phase subdirectories are expected at:
          results_base/phase0_ground_truth/
          results_base/phase1_correctness/
          ...
    skip : set of int, optional
        Phase numbers to skip (e.g., {5} to skip HIL plots).
    """
    if skip is None:
        skip = set()

    # Output under embark-evaluation/plots/<run_name>/ (run name = results_base dir name)
    run_name = results_base.name
    plots_root = _plots_dir / run_name
    print("=" * 70)
    print("  PVP Plot Generator")
    print(f"  Results: {results_base}")
    print(f"  Output:  {plots_root}")
    if skip:
        print(f"  Skipping phases: {sorted(skip)}")
    print("=" * 70)

    # Phase 1
    if 1 not in skip:
        phase1_dir = results_base / "phase1_correctness"
        if phase1_dir.exists():
            from plots.utils.plot_phase1 import generate_phase1_plots
            generate_phase1_plots(phase1_dir, plots_root / "phase1")
        else:
            print(f"  [skip] {phase1_dir.name}/ not found")

    # Phase 2
    if 2 not in skip:
        phase2_dir = results_base / "phase2_metric_validation"
        if phase2_dir.exists():
            from plots.utils.plot_phase2 import generate_phase2_plots
            generate_phase2_plots(phase2_dir, plots_root / "phase2")
        else:
            print(f"  [skip] {phase2_dir.name}/ not found")

    # Phase 3
    if 3 not in skip:
        phase3_dir = results_base / "phase3_discriminative"
        if phase3_dir.exists():
            from plots.utils.plot_phase3 import generate_phase3_plots
            generate_phase3_plots(phase3_dir, plots_root / "phase3")
        else:
            print(f"  [skip] {phase3_dir.name}/ not found")

    # Phase 4
    if 4 not in skip:
        phase4_dir = results_base / "phase4_reproducibility"
        if phase4_dir.exists():
            from plots.utils.plot_phase4 import generate_phase4_plots
            generate_phase4_plots(phase4_dir, plots_root / "phase4")
        else:
            print(f"  [skip] {phase4_dir.name}/ not found")

    # Phase 5
    if 5 not in skip:
        phase5_dir = results_base / "phase5_hil"
        if phase5_dir.exists():
            from plots.utils.plot_phase5 import generate_phase5_plots
            generate_phase5_plots(phase5_dir, plots_root / "phase5")
        else:
            print(f"  [skip] {phase5_dir.name}/ not found — run Phase 5 with --host first")

    # Phase 6
    if 6 not in skip:
        phase6_dir = results_base / "phase6_overhead"
        if phase6_dir.exists():
            from plots.utils.plot_phase6 import generate_phase6_plots
            generate_phase6_plots(phase6_dir, plots_root / "phase6")
        else:
            print(f"  [skip] {phase6_dir.name}/ not found")

    print("\n" + "=" * 70)
    print(f"  All plots saved to: {plots_root}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PVP Plot Generator — Generate all plots from a completed PVP run."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the run-level results directory (e.g., results/pvp_run1).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        help="Phase numbers to skip (e.g., --skip 5 to skip HIL plots).",
    )
    args = parser.parse_args()

    results_base = Path(args.results_dir).resolve()
    if not results_base.exists():
        print(f"Error: Results directory not found: {results_base}")
        return 1

    generate_all_plots(results_base, skip=set(args.skip))
    return 0


if __name__ == "__main__":
    sys.exit(main())
