# DGX Spark deployment

Verified 2026-10-19 on `spark-f2fa` (NVIDIA GB10, aarch64), using Python 3.12.14 and PyTorch 2.11.0+cu130. The deployment directory is `/home/qiker/NeuroSteer`, with an isolated `.venv`. The existing research environment was not modified.

## Running services

- `neurosteer-backend.service`: loopback-only `127.0.0.1:18000`.
- `neurosteer-frontend.service`: `0.0.0.0:18501`.
- Both are user systemd services, enabled at login/startup; user lingering was already enabled.
- Logs: `journalctl --user -u neurosteer-backend.service -u neurosteer-frontend.service`.
- Stop: `systemctl --user stop neurosteer-frontend.service neurosteer-backend.service`.
- Restart: `systemctl --user restart neurosteer-backend.service neurosteer-frontend.service`.

In the same LAN, try `http://192.168.1.13:18501`. From a computer with the configured `dgx` SSH alias:

```bash
ssh -N -L 127.0.0.1:18501:127.0.0.1:18501 dgx
```

Then open `http://127.0.0.1:18501`. A tunnel was started on the user's Mac during verification; if it closes, rerun this command. The local address is not a permanent public URL. No new VPS service, public DNS record, router rule, or Internet reverse proxy was configured for this deployment.

## Verified behavior

- All four recovered task caches deserialize on CUDA, contain 24,576-feature frequency/mean vectors and have finite values.
- Public GPT-2 and matching `gpt2-small-res-jb` layer-6 SAE weights load successfully.
- Model preprocessing follows the SAE's `center_writing_weights` setting, avoiding mismatched default folding.
- Seven API/feature-operation tests pass in the actual inference environment.
- Ten real API generations pass: baseline, eight task directions, and combined sentiment/politeness; test calls generated 30 new tokens each, taking approximately 0.19–0.55 seconds per request in this run. This is a smoke-test observation, not a benchmark.
- Streamlit tests using the real backend pass for combined generation, baseline/steered comparison, and history reset.
- API health returns 200/ready; Streamlit `/_stcore/health` returns `ok`.

No claim is made that task quality or paper metrics were reproduced. The model is a GPT-2 continuation model; outputs can be inaccurate, incoherent or offensive depending on intervention.

## Reinstalling

Copy the maintained source and four task caches to `~/NeuroSteer`. Prepare a Python 3.12 environment and install the compatible packages in this directory's `requirements.txt`. This runtime depends on a GB10-compatible CUDA PyTorch build; check an actual CUDA tensor operation before loading models.

Copy the two `.service` files into `~/.config/systemd/user/`, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now neurosteer-backend.service
# Confirm backend /health returns 200 before starting the frontend.
systemctl --user enable --now neurosteer-frontend.service
```

Public model weights are obtained using Hugging Face identifiers, while task statistics come from this repository's recovered files. The first model download can take several minutes. The installed test environment pins NumPy 1.26.4 with PyArrow 19.0.1: newer PyArrow required NumPy 2 and failed to import with the old SAE Lens dependency stack.
