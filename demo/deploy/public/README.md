# Public deployment: DGX + VPS + Cloudflare

Verified public URL: **https://neurosteer.keqichen.top/** (2026-10-19).

Request path:

```text
Browser HTTPS
  -> Cloudflare public hostname and named Tunnel
  -> VPS cloudflared
  -> VPS 127.0.0.1:18502 (reverse SSH listener)
  -> DGX 127.0.0.1:18501 (Streamlit)
  -> DGX 127.0.0.1:18000 (FastAPI and GB10 inference)
```

No model or inference process runs on the VPS. Its existing port-443 service remains in place. Cloudflared makes outbound HTTP/2 connections, so no inbound 80/443 firewall or router change was needed.

## Services

On DGX (`ssh dgx`):

```bash
systemctl --user status neurosteer-backend neurosteer-frontend neurosteer-vps-tunnel
journalctl --user -u neurosteer-vps-tunnel -n 50
```

On VPS (`ssh vps`):

```bash
systemctl status neurosteer-cloudflared
journalctl -u neurosteer-cloudflared -n 50
curl -fsS http://127.0.0.1:18502/_stcore/health
```

DGX initiates the SSH tunnel using a dedicated local key `~/.ssh/id_ed25519_neurosteer`, to the dedicated `neurosteer-tunnel` account on the VPS. The server key is checked against known_hosts. Its authorized key is restricted to the 127.0.0.1:18502 endpoint and has a forced false command; it cannot open an interactive shell. SSH keepalives and systemd restart reconnect the tunnel when the link drops.

Cloudflare uses the dedicated `neurosteer-demo` Tunnel. Only the `neurosteer.keqichen.top` CNAME was added; existing records/tunnels were not replaced. Runtime files are `/etc/neurosteer/cloudflared.yml` and `/etc/neurosteer/tunnel.json`. The latter is a private credential; it is not included in this repository. Cloudflared runs as `neurosteer-cloudflared`, separately from root.

The corresponding unit templates are in this directory and `../spark/`. Authentication credentials and private SSH keys are intentionally excluded. Domain authorization was performed through the owner's authenticated Cloudflare browser session and official `cloudflared tunnel login`.

## Verification

- Both public DNS resolvers 1.1.1.1 and 8.8.8.8 resolved the new hostname.
- Public HTTPS `/_stcore/health` returned HTTP/2 200 and `ok`, with Cloudflare response headers and valid certificate verification.
- A real browser loaded the public Streamlit UI and submitted `The movie was`; a generated continuation appeared over the public WebSocket session.
- All three DGX services and the VPS Cloudflare service were active after setup.
- Existing VPS 443 listener and the pre-existing DGX SSH-management tunnel remained running.

If the site returns 502, first check the DGX frontend, then the VPS loopback health endpoint, then the named Cloudflare Tunnel. DGX and VPS must remain powered on and connected. The public URL does not depend on the Mac's local SSH port forwarding.

## Stop the public route

Stop `neurosteer-cloudflared` on the VPS and `neurosteer-vps-tunnel` on DGX. The local/LAN research demo can remain running. Delete the dedicated DNS route/Tunnel through Cloudflare only when retiring the deployment permanently.
