# Akida model for PVP Phase 5 and TD-HIL

This folder is the **canonical location** for the Akida model used in:
- **PVP Phase 5** (HIL deployment feasibility)
- **TD-HIL benchmark** (`embark-evaluation/run_models_benchmark_tdhil.py`)

## Files to place here

| File | Purpose |
|------|---------|
| **akida_model.fbz** | Required. Akida-compiled model for the board. Copy to the board when running Phase 5 / TD-HIL. |
| **best_model.keras** or **final_model.keras** | Optional. Keras source for reference or re-export. |

If you already built the model elsewhere (e.g. with `evaluation/akida` or a notebook), copy the files here:

- **From training output** (typical path):  
  `evaluation/trained_models/v4/learned_linear/akida/`  
  → copy `akida_model.fbz` and, if present, `best_model.keras` or `final_model.keras` into this folder.

- **Building .fbz from Keras**:  
  Use `evaluation/akida/utils/train.py` with `--export_akida` or the export steps in `evaluation/akida/README.md`. Then copy the resulting `.fbz` (and optionally the `.keras`) here.

## Deploying to the board and running Phase 5

See **[AKIDA_BOARD_DEPLOY.md](../../AKIDA_BOARD_DEPLOY.md)** for full steps. Short version:

1. **On the board (SSH):** Create `~/akida_deployment/{server,models}`, activate `venv_akida`.
2. **On the PC:** Copy the inference server and **this folder’s** `akida_model.fbz` to the board (see deploy doc for `scp` commands).
3. **On the board:** Start `server/inference_server.py` with `--model-path models/akida_model.fbz --input-shape "1,1,1,5"`.
4. **On the PC:** Run Phase 5 or TD-HIL:
   - **PVP Phase 5:**  
     `poetry run python embark-evaluation/pvp/run_all_phases.py --run my_pvp --hil-host 10.42.0.1`
   - **TD-HIL only:**  
     `poetry run python embark-evaluation/run_models_benchmark_tdhil.py --host 10.42.0.1 --port 5000`

Input shape for this PMSM controller is **1,1,1,5** (batch, height, width, 5 features: normalized i_d, i_q, i_d_ref, i_q_ref, n).
