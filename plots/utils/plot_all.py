"""
PVP Plot Orchestrator — Generate all plots from a completed PVP run.

Reads results from each phase subdirectory and generates publication-ready
plots into ``embark-evaluation/plots/phase1/``, ``phase2/``, … ``phase6/``
directly under the plots directory.

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


def generate_all_plots(
    results_base: Path,
    skip: set[int] | None = None,
    phase5_dir_override: Path | None = None,
    phase6_dir_override: Path | None = None,
) -> None:
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
    phase5_dir_override : Path, optional
        If set, use this directory for Phase 5 data (phase5_hil); plots still go to plots/phase5/.
    phase6_dir_override : Path, optional
        If set, use this directory for Phase 6 data (phase6_overhead); plots still go to plots/phase6/.
    """
    if skip is None:
        skip = set()

    # Phase folders directly under embark-evaluation/plots/ (no run-name subfolder)
    plots_root = _plots_dir
    print("=" * 70)
    print("  PVP Plot Generator")
    print(f"  Results: {results_base}")
    print(f"  Output:  {plots_root} (phase1/ … phase6/)")
    if skip:
        print(f"  Skipping phases: {sorted(skip)}")
    if phase5_dir_override:
        print(f"  Phase 5 data: {phase5_dir_override}")
    if phase6_dir_override:
        print(f"  Phase 6 data: {phase6_dir_override}")
    print("=" * 70)

    # Phase 0
    if 0 not in skip:
        phase0_dir = results_base / "phase0_ground_truth"
        if phase0_dir.exists():
            from plots.utils.plot_phase0 import generate_phase0_plots
            generate_phase0_plots(phase0_dir, plots_root / "phase0")
        else:
            print(f"  [skip] {phase0_dir.name}/ not found")

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
        phase5_dir = phase5_dir_override if phase5_dir_override else (results_base / "phase5_hil")
        if phase5_dir.exists():
            from plots.utils.plot_phase5 import generate_phase5_plots
            generate_phase5_plots(phase5_dir, plots_root / "phase5")
        else:
            print(f"  [skip] phase5_hil/ not found — run Phase 5 with --host first or pass --phase5-dir")

    # Phase 6
    if 6 not in skip:
        phase6_dir = phase6_dir_override if phase6_dir_override else (results_base / "phase6_overhead")
        if phase6_dir.exists():
            from plots.utils.plot_phase6 import generate_phase6_plots
            generate_phase6_plots(phase6_dir, plots_root / "phase6")
        else:
            print(f"  [skip] phase6_overhead/ not found — run Phase 6 or pass --phase6-dir")

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
    parser.add_argument(
        "--only",
        type=int,
        nargs="*",
        default=[],
        help="If set, generate only these phase numbers (e.g., --only 5 for Phase 5 only).",
    )
    parser.add_argument(
        "--phase5-dir",
        type=str,
        default=None,
        help="Use this directory for Phase 5 data; plots still go to output/phase5/.",
    )
    parser.add_argument(
        "--phase6-dir",
        type=str,
        default=None,
        help="Use this directory for Phase 6 data; plots still go to output/phase6/.",
    )
    args = parser.parse_args()

    results_base = Path(args.results_dir).resolve()
    if not results_base.exists():
        print(f"Error: Results directory not found: {results_base}")
        return 1

    phase5_override = Path(args.phase5_dir).resolve() if args.phase5_dir else None
    phase6_override = Path(args.phase6_dir).resolve() if args.phase6_dir else None
    if phase5_override and not phase5_override.exists():
        print(f"Error: Phase 5 directory not found: {phase5_override}")
        return 1
    if phase6_override and not phase6_override.exists():
        print(f"Error: Phase 6 directory not found: {phase6_override}")
        return 1

    only_phases = set(args.only) if args.only else None
    if only_phases:
        skip = set(range(0, 7)) - only_phases
    else:
        skip = set(args.skip)

    generate_all_plots(
        results_base,
        skip=skip,
        phase5_dir_override=phase5_override,
        phase6_dir_override=phase6_override,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
