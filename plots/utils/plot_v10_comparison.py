"""
Plot focused comparison for Rate-based SNN model (formerly v10): Run 1 vs Run 2, and Run 1 baseline vs all models.

This script creates normalized histogram plots (0-1 range, percentages):
1. Run 1 vs Run 2 comparison for Rate-based SNN (control and neuromorphic metrics)
2. Run 1: Baseline (PI) vs all SNN models comparison

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


def plot_rate_snn_run_comparison_histogram(
    run1_data: dict[str, Any],
    run2_data: dict[str, Any],
    output_dir: Path,
    metric_category: str,
) -> None:
    """
    Plot histogram comparison between Run 1 and Run 2 for Rate-based SNN model.
    
    Args:
        metric_category: "control" or "neuromorphic"
    """
    if metric_category == "control":
        metrics = ["rms_i_q", "mae_i_q", "max_error_i_q", "overshoot"]
        metric_labels = ["RMS i_q [%]", "MAE i_q [%]", "Max Error i_q [%]", "Overshoot [%]"]
        title_suffix = "Control Metrics"
    else:  # neuromorphic
        metrics = [
            "total_spikes",
            "spikes_per_step",
            "total_syops",
            "syops_per_step",
            "mean_sparsity",
        ]
        metric_labels = [
            "Total Spikes [%]",
            "Spikes per Step [%]",
            "Total SyOps [%]",
            "SyOps per Step [%]",
            "Mean Sparsity [%]",
        ]
        title_suffix = "Neuromorphic Metrics"

    # Compute normalization factors
    norm_factors = compute_normalization_factors([run1_data, run2_data], metrics)

    run1_metrics = extract_scenario_metrics(run1_data, metrics)
    run2_metrics = extract_scenario_metrics(run2_data, metrics)

    # Get common scenarios
    scenarios = sorted(set(run1_metrics.keys()) & set(run2_metrics.keys()))

    if not scenarios:
        print(f"Warning: No common scenarios found for {metric_category} metrics")
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        run1_values = [run1_metrics[s].get(metric, np.nan) for s in scenarios]
        run2_values = [run2_metrics[s].get(metric, np.nan) for s in scenarios]

        # Normalize values
        norm_factor = norm_factors[metric]
        run1_normalized = [normalize_value(v, norm_factor) for v in run1_values]
        run2_normalized = [normalize_value(v, norm_factor) for v in run2_values]

        # Filter out NaN values
        valid_run1 = [v for v in run1_normalized if not np.isnan(v)]
        valid_run2 = [v for v in run2_normalized if not np.isnan(v)]

        if not valid_run1 and not valid_run2:
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
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        ax.hist(
            valid_run1,
            bins=bins,
            alpha=0.6,
            label="Run 1",
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.hist(
            valid_run2,
            bins=bins,
            alpha=0.6,
            label="Run 2",
            color="coral",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Normalized Value [0-1]", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])

    plt.suptitle(
        f"Rate-based SNN Model: Run 1 vs Run 2 Comparison - {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / f"rate_based_snn_run1_vs_run2_{metric_category}_histogram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_run1_baseline_vs_models_histogram(
    baseline_data: dict[str, Any],
    models_data: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot histogram comparison of PI baseline vs all SNN models for Run 1."""
    # Control metrics
    control_metrics = ["rms_i_q", "mae_i_q", "max_error_i_q", "overshoot"]
    control_labels = ["RMS i_q [%]", "MAE i_q [%]", "Max Error i_q [%]", "Overshoot [%]"]

    # Neuromorphic metrics (if available)
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

    # Extract baseline metrics
    baseline_control = extract_scenario_metrics(baseline_data, control_metrics)
    baseline_neuro = extract_scenario_metrics(baseline_data, neuro_metrics)

    # Extract model metrics
    models_control = {}
    models_neuro = {}
    for model_name, model_data in models_data.items():
        models_control[model_name] = extract_scenario_metrics(model_data, control_metrics)
        models_neuro[model_name] = extract_scenario_metrics(model_data, neuro_metrics)

    # Compute normalization factors
    all_data = [baseline_data] + list(models_data.values())
    control_norm_factors = compute_normalization_factors(all_data, control_metrics)
    neuro_norm_factors = compute_normalization_factors(all_data, neuro_metrics)

    # Get common scenarios
    all_scenarios = set(baseline_control.keys())
    for model_metrics in models_control.values():
        all_scenarios.update(model_metrics.keys())
    scenarios = sorted(all_scenarios)

    # Plot control metrics
    n_metrics = len(control_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))

    if n_metrics == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

    for idx, (metric, label) in enumerate(zip(control_metrics, control_labels)):
        ax = axes[idx]

        # Collect normalized values
        baseline_values = [
            normalize_value(baseline_control.get(s, {}).get(metric, np.nan), control_norm_factors[metric])
            for s in scenarios
        ]
        baseline_valid = [v for v in baseline_values if not np.isnan(v)]

        # Plot baseline
        if baseline_valid:
            ax.hist(
                baseline_valid,
                bins=bins,
                alpha=0.7,
                label="PI Baseline",
                color="black",
                edgecolor="white",
                linewidth=1.5,
            )

        # Plot models
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
        for model_idx, (model_name, model_metrics) in enumerate(models_control.items()):
            model_values = [
                normalize_value(model_metrics.get(s, {}).get(metric, np.nan), control_norm_factors[metric])
                for s in scenarios
            ]
            model_valid = [v for v in model_values if not np.isnan(v)]
            
            if model_valid:
                # Format model name for display
                display_name = model_name.replace("_", " ").title()
                if "v10" in display_name.lower():
                    display_name = "Rate-based SNN"
                elif "v12" in display_name.lower():
                    display_name = "Incremental SNN"
                
                ax.hist(
                    model_valid,
                    bins=bins,
                    alpha=0.5,
                    label=display_name,
                    color=colors[model_idx],
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.set_xlabel("Normalized Value [0-1]", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Run 1: PI Baseline vs SNN Models - Control Metrics (Normalized)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "run1_baseline_vs_models_control_histogram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Plot neuromorphic metrics (only for SNN models, baseline won't have these)
    has_neuro_data = False
    for model_metrics in models_neuro.values():
        for scenario_metrics in model_metrics.values():
            for value in scenario_metrics.values():
                if value is not None and value != 0 and not np.isnan(value):
                    has_neuro_data = True
                    break
            if has_neuro_data:
                break
        if has_neuro_data:
            break

    if has_neuro_data:
        n_metrics = len(neuro_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))

        if n_metrics == 1:
            axes = [axes]

        for idx, (metric, label) in enumerate(zip(neuro_metrics, neuro_labels)):
            ax = axes[idx]

            colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

            for model_idx, (model_name, model_metrics) in enumerate(models_neuro.items()):
                model_values = [
                    normalize_value(model_metrics.get(s, {}).get(metric, np.nan), neuro_norm_factors[metric])
                    for s in scenarios
                ]
                model_valid = [v for v in model_values if not np.isnan(v) and v > 0]

                if model_valid:
                    # Format model name for display
                    display_name = model_name.replace("_", " ").title()
                    if "v10" in display_name.lower():
                        display_name = "Rate-based SNN"
                    elif "v12" in display_name.lower():
                        display_name = "Incremental SNN"
                    
                    ax.hist(
                        model_valid,
                        bins=bins,
                        alpha=0.6,
                        label=display_name,
                        color=colors[model_idx],
                        edgecolor="black",
                        linewidth=0.5,
                    )

            ax.set_xlabel("Normalized Value [0-1]", fontweight="bold")
            ax.set_ylabel("Frequency", fontweight="bold")
            ax.set_title(label, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Run 1: SNN Models - Neuromorphic Metrics (Normalized)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = output_dir / "run1_baseline_vs_models_neuromorphic_histogram.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        print("Note: No neuromorphic metrics data found (all zeros)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Rate-based SNN comparison: Run 1 vs Run 2, and Run 1 baseline vs models"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "benchmarking-results",
        help="Base directory containing run directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "models_for_evaluation" / "plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    run1_dir = args.results_dir / "run 1"
    run2_dir = args.results_dir / "run2"

    print("Loading benchmark results...")
    print(f"  Run 1 directory: {run1_dir}")
    print(f"  Run 2 directory: {run2_dir}")

    # Load Rate-based SNN (v10) for both runs
    v10_run1_path = run1_dir / "v10_v10_scheduled_sampling.json"
    v10_run2_path = run2_dir / "v10_v10_scheduled_sampling.json"

    if not v10_run1_path.exists():
        print(f"Error: {v10_run1_path} not found!")
        return 1
    if not v10_run2_path.exists():
        print(f"Error: {v10_run2_path} not found!")
        return 1

    v10_run1_data = load_benchmark_json(v10_run1_path)
    v10_run2_data = load_benchmark_json(v10_run2_path)

    print(f"\nLoaded Rate-based SNN model from both runs")

    # Load Run 1 baseline and all models
    baseline_path = run1_dir / "PI-baseline.json"
    if baseline_path.exists():
        baseline_data = load_benchmark_json(baseline_path)
        print(f"Loaded PI baseline")
    else:
        print(f"Warning: {baseline_path} not found, skipping baseline comparison")
        baseline_data = None

    # Load all SNN models from Run 1 (for baseline comparison)
    models_data = {}
    for json_file in run1_dir.glob("*.json"):
        if json_file.name == "PI-baseline.json":
            continue
        try:
            data = load_benchmark_json(json_file)
            model_name = data.get("controller_name", json_file.stem)
            models_data[model_name] = data
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating normalized histogram plots...")

    # 1. Rate-based SNN Run 1 vs Run 2 - Control metrics
    print("  Creating Rate-based SNN Run 1 vs Run 2 control metrics histogram...")
    plot_rate_snn_run_comparison_histogram(v10_run1_data, v10_run2_data, args.output_dir, "control")

    # 2. Rate-based SNN Run 1 vs Run 2 - Neuromorphic metrics
    print("  Creating Rate-based SNN Run 1 vs Run 2 neuromorphic metrics histogram...")
    plot_rate_snn_run_comparison_histogram(v10_run1_data, v10_run2_data, args.output_dir, "neuromorphic")

    # 3. Run 1: Baseline vs all models
    if baseline_data and models_data:
        print("  Creating Run 1 baseline vs models histogram comparison...")
        plot_run1_baseline_vs_models_histogram(baseline_data, models_data, args.output_dir)
    elif baseline_data:
        print("  Warning: No SNN models found in Run 1, skipping baseline comparison")
    elif models_data:
        print("  Warning: No baseline found, skipping baseline comparison")

    print(f"\nDone! All plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
