"""
Plot focused comparison: PI Baseline vs Incremental SNN model (formerly v12) from Run 1.

Creates normalized histogram plots (0-1 range, percentages) for control and neuromorphic metrics.
All metrics are normalized to 0-1 range based on hardware constraints and maximum observed values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_benchmark_json(json_path: Path) -> dict[str, Any]:
    """Load a benchmark JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scenario_metrics(
    benchmark_data: dict[str, Any], metric_names: list[str]
) -> dict[str, dict[str, float]]:
    """
    Extract metrics for all scenarios.
    
    Returns: {scenario_name: {metric_name: value}}
    """
    results = {}
    for scenario in benchmark_data.get("scenarios", []):
        scenario_name = scenario["name"]
        results[scenario_name] = {}
        for metric_name in metric_names:
            value = scenario["metrics"].get(metric_name)
            if value is not None and not np.isinf(value):
                results[scenario_name][metric_name] = value
            else:
                results[scenario_name][metric_name] = np.nan
    return results


def compute_normalization_factors(
    all_data: list[dict[str, Any]], metric_names: list[str]
) -> dict[str, float]:
    """
    Compute normalization factors (max values) for each metric across all data.
    Uses hardware constraints where applicable.
    """
    factors = {}
    
    # Hardware constraints
    hardware_constraints = {
        "rms_i_q": 10.8,  # Max current [A]
        "mae_i_q": 10.8,  # Max current [A]
        "max_error_i_q": 10.8,  # Max current [A]
        "overshoot": 500.0,  # Max overshoot [%] (reasonable upper bound)
        "total_spikes": 1e6,  # Max spikes (hardware limit estimate)
        "spikes_per_step": 10000,  # Max spikes per step (hardware limit estimate)
        "total_syops": 1e9,  # Max SyOps (hardware limit estimate)
        "syops_per_step": 500000,  # Max SyOps per step (from docs: typical max for rate-SNN)
        "mean_sparsity": 1.0,  # Already 0-1 range
    }
    
    for metric in metric_names:
        if metric in hardware_constraints:
            factors[metric] = hardware_constraints[metric]
        else:
            # Find max value across all data
            max_val = 0.0
            for data in all_data:
                metrics = extract_scenario_metrics(data, [metric])
                for scenario_metrics in metrics.values():
                    val = scenario_metrics.get(metric, 0)
                    if not np.isnan(val) and not np.isinf(val):
                        max_val = max(max_val, abs(val))
            factors[metric] = max(max_val, 1e-6)  # Avoid division by zero
    
    return factors


def normalize_value(value: float, factor: float) -> float:
    """Normalize a value to 0-1 range."""
    if np.isnan(value) or np.isinf(value):
        return np.nan
    return np.clip(abs(value) / factor, 0.0, 1.0)


def plot_pi_vs_incremental_snn_control_histogram(
    pi_data: dict[str, Any],
    v12_data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot normalized histogram comparison: PI vs Incremental SNN for control metrics."""
    control_metrics = ["rms_i_q", "mae_i_q", "max_error_i_q", "overshoot"]
    metric_labels = ["RMS i_q [%]", "MAE i_q [%]", "Max Error i_q [%]", "Overshoot [%]"]

    # Compute normalization factors
    norm_factors = compute_normalization_factors([pi_data, v12_data], control_metrics)

    pi_metrics = extract_scenario_metrics(pi_data, control_metrics)
    v12_metrics = extract_scenario_metrics(v12_data, control_metrics)

    # Get common scenarios
    scenarios = sorted(set(pi_metrics.keys()) & set(v12_metrics.keys()))

    if not scenarios:
        print("Warning: No common scenarios found")
        return

    n_metrics = len(control_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))

    if n_metrics == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

    for idx, (metric, label) in enumerate(zip(control_metrics, metric_labels)):
        ax = axes[idx]

        pi_values = [pi_metrics[s].get(metric, np.nan) for s in scenarios]
        v12_values = [v12_metrics[s].get(metric, np.nan) for s in scenarios]

        # Normalize values
        norm_factor = norm_factors[metric]
        pi_normalized = [normalize_value(v, norm_factor) for v in pi_values]
        v12_normalized = [normalize_value(v, norm_factor) for v in v12_values]

        # Filter out NaN values
        valid_pi = [v for v in pi_normalized if not np.isnan(v)]
        valid_v12 = [v for v in v12_normalized if not np.isnan(v)]

        if not valid_pi and not valid_v12:
            ax.text(
                0.5,
                0.5,
                f"No data\nfor {label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(label, fontweight="bold")
            continue

        # Create histogram
        ax.hist(
            valid_pi,
            bins=bins,
            alpha=0.7,
            label="PI Baseline",
            color="black",
            edgecolor="white",
            linewidth=1.5,
        )
        ax.hist(
            valid_v12,
            bins=bins,
            alpha=0.6,
            label="Incremental SNN",
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Normalized Value [0-1]", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "PI Baseline vs Incremental SNN Model - Control Metrics Comparison (Normalized)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "pi_vs_incremental_snn_control_histogram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_incremental_snn_neuromorphic_histogram(
    v12_data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot normalized histogram for neuromorphic metrics for Incremental SNN (PI doesn't have these)."""
    neuro_metrics = [
        "total_spikes",
        "spikes_per_step",
        "total_syops",
        "syops_per_step",
        "mean_sparsity",
    ]
    neuro_labels = [
        "Total Spikes [%]",
        "Spikes per Step [%]",
        "Total SyOps [%]",
        "SyOps per Step [%]",
        "Mean Sparsity [%]",
    ]

    # Compute normalization factors
    norm_factors = compute_normalization_factors([v12_data], neuro_metrics)

    v12_metrics = extract_scenario_metrics(v12_data, neuro_metrics)

    # Check if there's any non-zero data
    has_data = False
    for scenario_metrics in v12_metrics.values():
        for value in scenario_metrics.values():
            if value is not None and value != 0 and not np.isnan(value):
                has_data = True
                break
        if has_data:
            break

    if not has_data:
        print("Note: No neuromorphic metrics data found (all zeros)")
        return

    scenarios = sorted(v12_metrics.keys())
    n_metrics = len(neuro_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))

    if n_metrics == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

    for idx, (metric, label) in enumerate(zip(neuro_metrics, neuro_labels)):
        ax = axes[idx]

        v12_values = [v12_metrics[s].get(metric, np.nan) for s in scenarios]
        
        # Normalize values
        norm_factor = norm_factors[metric]
        v12_normalized = [normalize_value(v, norm_factor) for v in v12_values]
        
        valid_v12 = [v for v in v12_normalized if not np.isnan(v) and v > 0]

        if not valid_v12:
            ax.text(
                0.5,
                0.5,
                f"No data\nfor {label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(label, fontweight="bold")
            continue

        # Create histogram
        ax.hist(
            valid_v12,
            bins=bins,
            alpha=0.6,
            label="Incremental SNN",
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Normalized Value [0-1]", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Incremental SNN Model - Neuromorphic Metrics (Normalized)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "incremental_snn_neuromorphic_histogram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot PI Baseline vs Incremental SNN model comparison"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "benchmarking-results" / "run 1",
        help="Directory containing Run 1 results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "models_for_evaluation" / "plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    print("Loading benchmark results...")
    print(f"  Results directory: {args.results_dir}")

    # Load PI baseline
    pi_path = args.results_dir / "PI-baseline.json"
    if not pi_path.exists():
        print(f"Error: {pi_path} not found!")
        return 1

    # Load Incremental SNN (v12) model
    v12_path = args.results_dir / "v12_incremental_best_model.json"
    if not v12_path.exists():
        print(f"Error: {v12_path} not found!")
        return 1

    pi_data = load_benchmark_json(pi_path)
    v12_data = load_benchmark_json(v12_path)

    print(f"Loaded PI baseline and Incremental SNN model")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating normalized histogram plots...")

    # 1. Control metrics comparison
    print("  Creating control metrics histogram comparison...")
    plot_pi_vs_incremental_snn_control_histogram(pi_data, v12_data, args.output_dir)

    # 2. Neuromorphic metrics (Incremental SNN only)
    print("  Creating neuromorphic metrics histogram...")
    plot_incremental_snn_neuromorphic_histogram(v12_data, args.output_dir)

    print(f"\nDone! All plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
