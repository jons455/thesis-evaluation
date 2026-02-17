"""Quick check: load SNN, run wrapper forward once, assert last_info has neuromorphic keys."""
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

import torch
from run_models_benchmark import LocalSNNControllerWrapper
from evaluation.analysis.evaluate_rate_snn import load_rate_model

def main():
    checkpoint = repo / "evaluation" / "trained_models" / "v9" / "v9_no_tanh.pt"
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return 1
    model, _meta = load_rate_model(checkpoint, device="cpu")
    wrapped = LocalSNNControllerWrapper(model=model)
    obs = torch.randn(1, 12)
    _ = wrapped.forward(obs)
    info = wrapped.last_info
    required = ["total_spikes", "syops", "sparsity"]
    missing = [k for k in required if k not in info]
    if missing:
        print(f"FAIL: last_info missing keys: {missing}")
        print("last_info:", info)
        return 1
    if info["total_spikes"] == 0 and info["syops"] == 0:
        print("FAIL: total_spikes and syops are still 0")
        print("last_info:", info)
        return 1
    print("OK: neuromorphic metrics present and non-zero")
    print("last_info:", info)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
