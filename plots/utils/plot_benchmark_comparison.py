"""
Plot benchmark comparison between PyTorch SNN models and Akida hardware models.

This script creates normalized histogram plots (0-1 range, percentages) comparing:
- PyTorch SNN models: embark-evaluation/models_for_evaluation/benchmarking-results/
- Akida hardware: embark-evaluation/benchmarking-results/akida/

All metrics are normalized to 0-1 range based on hardware constraints and maximum observed values.

Usage:
    # Default: Compare all results from both directories
    python embark-evaluation/plots/utils/plot_benchmark_comparison.py

    # Compare specific run
    python embark-evaluation/plots/utils/plot_benchmark_comparison.py --run "run 1"

Output:
    Generates normalized histogram plots in the output directory.
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


def format_model_name(name: str) -> str:
    """Format model name for display (replace underscores, capitalize, use descriptive names)."""
    # Replace common patterns
    name = name.replace("_", " ").replace("tdhil", "TD-HIL")
    # Replace version numbers with descriptive names (before splitting)
    if "v10" in name.lower() or "scheduled sampling" in name.lower():
        name = "Rate-based SNN"
    elif "v12" in name.lower() or "incremental" in name.lower():
        name = "Incremental SNN"
    elif "v9" in name.lower():
        name = "V9 SNN"
    # Capitalize first letter of each word
    words = name.split()
    formatted = " ".join(word.capitalize() for word in words)
    return formatted


def collect_results(
    pytorch_dir: Path, akida_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Collect all benchmark results from both directories.
    
    Returns:
        (pytorch_results, akida_results) where each is {model_name: benchmark_data}
    """
    pytorch_results = {}
    akida_results = {}

    # Collect PyTorch SNN results
    if pytorch_dir.exists():
        for json_file in pytorch_dir.rglob("*.json"):
            if json_file.name == "PI-baseline.json":
                continue  # Skip PI baseline for now
            try:
                data = load_benchmark_json(json_file)
                model_name = data.get("controller_name", json_file.stem)
                # Use run directory name if available for disambiguation
                run_name = json_file.parent.name if json_file.parent.name.startswith("run") else ""
                if run_name:
                    model_name = f"{model_name} ({run_name})"
                # Format for display
                display_name = format_model_name(model_name)
                pytorch_results[display_name] = data
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    # Collect Akida hardware results
    if akida_dir.exists():
        for json_file in akida_dir.rglob("*.json"):
            try:
                data = load_benchmark_json(json_file)
                model_name = data.get("controller_name", json_file.stem)
                display_name = format_model_name(model_name)
                akida_results[display_name] = data
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    return pytorch_results, akida_results


def extract_scenario_metrics(
    benchmark_data: dict[str, Any], metric_name: str
) -> dict[str, float]:
    """Extract a specific metric for all scenarios."""
    metrics = {}
    for scenario in benchmark_data.get("scenarios", []):
        scenario_name = scenario["name"]
        metric_value = scenario["metrics"].get(metric_name)
        if metric_value is not None and not np.isinf(metric_value):
            metrics[scenario_name] = metric_value
    return metrics


def compute_normalization_factors(
    all_results: dict[str, dict[str, Any]], metric_names: list[str]
) -> dict[str, float]:
    """
    Compute normalization factors (max values) for each metric across all results.
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
            # Find max value across all results
            max_val = 0.0
            for benchmark_data in all_results.values():
                scenario_metrics = extract_scenario_metrics(benchmark_data, metric)
                for value in scenario_metrics.values():
                    if not np.isnan(value) and not np.isinf(value):
                        max_val = max(max_val, abs(value))
            factors[metric] = max(max_val, 1e-6)  # Avoid division by zero
    
    return factors


def normalize_value(value: float, factor: float) -> float:
    """Normalize a value to 0-1 range."""
    if np.isnan(value) or np.isinf(value):
        return np.nan
    return np.clip(abs(value) / factor, 0.0, 1.0)


def create_metric_comparison_histogram(
    all_results: dict[str, dict[str, Any]],
    metric_name: str,
    metric_label: str,
    output_path: Path,
) -> None:
    """
    Create a histogram comparing a metric across all models and scenarios.
    
    Args:
        all_results: {model_name: benchmark_data}
        metric_name: Name of metric in JSON (e.g., "rms_i_q")
        metric_label: Display label (e.g., "RMS i_q [%]")
        output_path: Where to save the plot
    """
    # Collect all normalized values
    all_normalized_values = []
    model_labels = []
    
    # Compute normalization factor
    norm_factors = compute_normalization_factors(all_results, [metric_name])
    norm_factor = norm_factors[metric_name]
    
    for model_name, benchmark_data in all_results.items():
        scenario_metrics = extract_scenario_metrics(benchmark_data, metric_name)
        normalized_values = [
            normalize_value(value, norm_factor) for value in scenario_metrics.values()
        ]
        valid_values = [v for v in normalized_values if not np.isnan(v)]
        if valid_values:
            all_normalized_values.extend(valid_values)
            # Add model label for each value (for grouping if needed)
            model_labels.extend([model_name] * len(valid_values))

    if not all_normalized_values:
        print(f"Warning: No data found for metric {metric_name}")
        return

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    
    # Group by model type (PyTorch vs Akida) if possible
    pytorch_values = []
    akida_values = []
    other_values = []
    
    for model_name, benchmark_data in all_results.items():
        scenario_metrics = extract_scenario_metrics(benchmark_data, metric_name)
        normalized_values = [
            normalize_value(value, norm_factor) for value in scenario_metrics.values()
        ]
        valid_values = [v for v in normalized_values if not np.isnan(v)]
        
        if "akida" in model_name.lower() or "td-hil" in model_name.lower():
            akida_values.extend(valid_values)
        elif "pytorch" in model_name.lower() or any(x in model_name.lower() for x in ["rate", "incremental", "snn"]):
            pytorch_values.extend(valid_values)
        else:
            other_values.extend(valid_values)
    
    # Plot histograms
    if pytorch_values:
        ax.hist(
            pytorch_values,
            bins=bins,
            alpha=0.6,
            label="PyTorch SNN",
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )
    if akida_values:
        ax.hist(
            akida_values,
            bins=bins,
            alpha=0.6,
            label="Akida Hardware",
            color="coral",
            edgecolor="black",
            linewidth=0.5,
        )
    if other_values:
        ax.hist(
            other_values,
            bins=bins,
            alpha=0.6,
            label="Other",
            color="gray",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Normalized Value [0-1]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Benchmark Comparison: {metric_label}\n(PyTorch SNN vs Akida Hardware - Normalized)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot benchmark comparison between PyTorch SNN and Akida hardware models"
    )
    parser.add_argument(
        "--pytorch-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "benchmarking-results",
        help="Directory containing PyTorch SNN benchmark results",
    )
    parser.add_argument(
        "--akida-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "benchmarking-results" / "akida",
        help="Directory containing Akida hardware benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "models_for_evaluation" / "plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Specific run directory to use (e.g., 'run 1', 'run2')",
    )

    args = parser.parse_args()

    # Adjust paths if specific run is requested
    if args.run:
        args.pytorch_dir = args.pytorch_dir / args.run
        if not args.pytorch_dir.exists():
            print(f"Warning: PyTorch directory {args.pytorch_dir} does not exist")

    print("Collecting benchmark results...")
    print(f"  PyTorch SNN directory: {args.pytorch_dir}")
    print(f"  Akida hardware directory: {args.akida_dir}")

    pytorch_results, akida_results = collect_results(args.pytorch_dir, args.akida_dir)

    print(f"\nFound {len(pytorch_results)} PyTorch SNN model(s):")
    for name in pytorch_results:
        print(f"  - {name}")

    print(f"\nFound {len(akida_results)} Akida hardware model(s):")
    for name in akida_results:
        print(f"  - {name}")

    if not pytorch_results and not akida_results:
        print("Error: No benchmark results found!")
        return 1

    # Combine all results
    all_results = {**pytorch_results, **akida_results}

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating normalized histogram plots...")

    # Metric comparison histograms
    metrics_to_plot = [
        ("rms_i_q", "RMS i_q [%]"),
        ("mae_i_q", "MAE i_q [%]"),
        ("max_error_i_q", "Max Error i_q [%]"),
    ]

    for metric_key, metric_label in metrics_to_plot:
        output_path = args.output_dir / f"comparison_{metric_key}_histogram.png"
        create_metric_comparison_histogram(
            all_results, metric_key, metric_label, output_path
        )

    print(f"\nDone! All plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
