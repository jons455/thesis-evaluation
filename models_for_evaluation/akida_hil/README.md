# Akida model for PVP Phase 5 and TD-HIL

This folder holds the **Keras-trained, quantized Akida model** used for hardware-in-the-loop evaluation (PVP Phase 5, R12–R13) and for `run_models_benchmark_tdhil.py`.

**Training conducted in:** Google Colab, via `notebooks/train_learned_snn_and_keras_colab_v2.ipynb` (v3/akida run). Not trained from `evaluation/akida/utils/train.py`.

## Training source (notebook)

The model was trained from a **Colab notebook**, not from `evaluation/akida/utils/train.py`:

- **`notebooks/train_learned_snn_and_keras_colab_v2.ipynb`** — dual-framework (PyTorch SNN + Keras ANN).  
  Keras part: 5 inputs [i_d, i_q, e_d, e_q, n], Dense(128)→BN→ReLU, Dense(64)→BN→ReLU, Dense(32)→BN→ReLU, Dense(2) linear.  
  Output in Colab: `/content/output_models/keras_model/` (then typically exported to Drive or repo as **`evaluation/trained_models/v3/akida/`**).

- **`notebooks/train_learned_snn_and_keras_colab.ipynb`** (v1) — same Keras architecture; output → **`trained_models/v1/akida/`**.

So "run 3" / "v3" corresponds to the **v2 Colab notebook** (`train_learned_snn_and_keras_colab_v2.ipynb`), whose Keras output is documented as going to **v3/akida**. The `.fbz` and `.keras` in this folder were produced from that Keras model (quantize + cnn2snn), then copied here for Phase 5 and deployment.

## Files in this folder

| File | Purpose |
|------|---------|
| **akida_model.fbz** | Akida-compiled model; copy to the board for Phase 5 / TD-HIL. |
| **best_model.keras** | Keras model (float or post-training); source for re-export. |
| **calibration_data.npy** | Calibration samples for quantization. |
| **final_model.json** | Config (input_size=5, hidden_sizes=[128,64,32], 8-bit). |
| **normalization_params.npz** | i_max, u_max, n_max, error_gain, etc. |

## Deploying to the board

See ** [AKIDA_BOARD_DEPLOY.md](../../AKIDA_BOARD_DEPLOY.md)**. Copy `akida_model.fbz` to the board and start the inference server with `--model-path models/akida_model.fbz --input-shape "1,1,1,5"`.

## Rebuilding .fbz from the notebook or script

- From the same notebook run: train Keras → save `best_model.keras` / `final_model.keras` → then either export to .fbz in Colab (if akida/cnn2snn/quantizeml are installed) or copy the .keras to this repo and run `evaluation/akida/core/deploy.py` (or `evaluation/akida/utils/train.py` with `--export_akida` after loading the checkpoint).
