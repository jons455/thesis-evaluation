# Deploy inference server and .fbz to the Akida board

Use these steps to copy the inference script and your Akida model (`.fbz`) to the board, then run the TD-HIL benchmark from your PC.

**Prerequisites:** Akida board powered and reachable (e.g. Wi‑Fi `akida-devkit-*`, password `Demo123Demo123`). Default IP used below: `10.42.0.1`. User: `bcdev`, password: `Demo123`.

**Building the .fbz:** You don’t run anything on the board to “deploy” the model. The `.fbz` is built once (e.g. with `evaluation/akida/core/deploy.py` on a PC that has TensorFlow/quantizeml/cnn2snn). You only copy the built `.fbz` and the inference server to the board.

---

## 1. On the Akida board (SSH)

```bash
# Connect
ssh bcdev@10.42.0.1
# Password: Demo123

# Activate Akida environment (required)
source venv_akida/bin/activate

# Create deployment dirs
mkdir -p ~/akida_deployment/{server,models}
cd ~/akida_deployment
```

Leave this terminal open; you’ll start the server here in step 3.

---

## 2. On your PC – copy inference server and .fbz to the board

From the **project root** in a **new** terminal (not inside SSH):

**Windows (PowerShell):**
```powershell
cd C:\Users\Jonas\projects\thesis-nueromophic-controller-benchmark

# Inference server
scp evaluation/akida/server/inference_server.py bcdev@10.42.0.1:~/akida_deployment/server/

# Your .fbz model (adjust path to your model)
scp evaluation/trained_models/v4/learned_linear/akida/akida_model.fbz bcdev@10.42.0.1:~/akida_deployment/models/
```

**Linux / Mac:**
```bash
cd /path/to/thesis-nueromophic-controller-benchmark

scp evaluation/akida/server/inference_server.py bcdev@10.42.0.1:~/akida_deployment/server/
scp path/to/akida_model.fbz bcdev@10.42.0.1:~/akida_deployment/models/
```

If your `.fbz` lives elsewhere, change the path accordingly.

**Optional – inspect script (only if you want to check hardware backend / SDK statistics on the board):**
```bash
scp evaluation/akida/scripts/inspect_akida_statistics.py bcdev@10.42.0.1:~/akida_deployment/
# Then on the board: cd ~/akida_deployment && python3 inspect_akida_statistics.py models/akida_model.fbz
```

---

## 3. On the Akida board – start the inference server

In the SSH session (Terminal 1):

```bash
source venv_akida/bin/activate
cd ~/akida_deployment

python3 server/inference_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --model-path models/akida_model.fbz \
  --input-shape "1,1,1,5"
```

Leave this running. You should see something like: `Listening on 0.0.0.0:5000 (echo=False)` and then `Connected by ...` when the benchmark connects.

---

## 4. On your PC – run TD-HIL benchmark

In a PC terminal (Terminal 2), from the project root:

```bash
poetry run python embark-evaluation/run_models_benchmark_tdhil.py --host 10.42.0.1 --port 5000
```

Optional: `--quick` for fewer scenarios, `--run my_run` to save under `embark-evaluation/models_for_evaluation/results/my_run/`.

---

## Quick reference

| Step | Where   | Command |
|------|---------|--------|
| Connect to board | PC | `ssh bcdev@10.42.0.1` |
| Activate env     | Board | `source venv_akida/bin/activate` |
| Copy server      | PC | `scp evaluation/akida/server/inference_server.py bcdev@10.42.0.1:~/akida_deployment/server/` |
| Copy .fbz        | PC | `scp path/to/akida_model.fbz bcdev@10.42.0.1:~/akida_deployment/models/` |
| Start server     | Board | `python3 server/inference_server.py --host 0.0.0.0 --port 5000 --model-path models/akida_model.fbz --input-shape "1,1,1,5"` |
| Run benchmark    | PC | `poetry run python embark-evaluation/run_models_benchmark_tdhil.py --host 10.42.0.1 --port 5000` |
