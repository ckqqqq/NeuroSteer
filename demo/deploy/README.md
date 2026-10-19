# VPS deployment preparation

This configuration exposes Streamlit through Caddy on ports 80/443. The inference API has no host port. Caddy supports the WebSocket connection required by Streamlit.

The default image is **UI/API only**: it does not install PyTorch, TransformerLens, or SAE Lens. Backend health intentionally remains 503 until a separately prepared inference environment and four valid cache files are available. This is not a live-generation deployment or an experiment reproduction.

Once VPS access and hardware are confirmed:

1. Place the repository on the VPS and check whether ports 80/443 are already used. Integrate with an existing proxy instead of replacing it if necessary.
2. In `demo/deploy`, copy `.env.example` to `.env` and set `NEUROSTEER_SITE` to the real domain, or `http://VPS_IP` for HTTP-only access. For HTTPS, set the domain DNS record to the VPS first.
3. Create the cache directory (empty is acceptable for a UI-only deployment): `mkdir -p ../../results/demo/demo_v1`. If caches are elsewhere, configure `NEUROSTEER_CACHE_PATH` and make them readable by container user 10001.
4. Run `docker compose config` and review the configuration, then `docker compose up -d --build`.
5. Inspect `docker compose logs --tail=100`. Verify the public page and `docker compose exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read())"`. In UI-only mode this last command will raise HTTP 503, as expected; the browser being reachable does not mean inference is ready.
6. Stop with `docker compose down`; persistent TLS certificates remain in named volumes.

Configure the VPS firewall to allow the chosen SSH port and public ports 80/443. No need to expose 8000/8501. Container logs use Docker logging; original local launcher logs remain under `logs/demo/`.

Real inference needs a hardware-specific image/environment, cached Hugging Face model assets, and four task caches. Determine CPU/GPU, RAM, available disk and original artifact locations before selecting dependencies. Do not install the repository's CUDA-specific requirements unmodified on a CPU-only VPS.

Deployment status: configuration prepared locally; no VPS has been connected and no remote resources have been changed. Docker is unavailable on the current workstation, so the Compose build and reverse proxy have not been tested here.
