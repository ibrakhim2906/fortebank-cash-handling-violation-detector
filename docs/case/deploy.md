# Deployment (R9, R26, R27, R51)

The jury checks remotely via a **public URL**, after the 12:00 deadline. A dead service = 0.

## Run locally (WSL, CPU)
```
bash scripts/get_models.sh            # once, needs internet
uv sync
export API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"   # save it!
uv run uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1
```
One uvicorn worker (the job queue lives in-process). Check: http://localhost:8000 and `/health`.

## Public URL (laptop / CPU mode)
- Install once in WSL: `curl -L -o ~/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/cloudflared`
- Quick: `~/cloudflared tunnel --url http://localhost:8000` → prints `https://<random>.trycloudflare.com`. The URL changes on restart, so **don't restart it after handing in**.
- Better (stable URL): a named Cloudflare tunnel on your own domain, or ngrok with a reserved domain.
- Test from a phone on mobile data: open the UI, run the §10 curl with the key.

## Keep it alive until the jury finishes
- Laptop on power, sleep/hibernate off (Windows: Power settings → Never; also disable "sleep when lid closed"), Windows Update paused.
- Run uvicorn and cloudflared in `tmux` (or two terminals) with an auto-restart loop:
  `while true; do uv run uvicorn service.app:app --host 0.0.0.0 --port 8000; sleep 2; done`
- Jobs persist on disk, so a restart re-queues unfinished jobs.
- Someone checks `/health` every hour overnight.

## GPU mode on RunPod (jury deployment)
Human (10 min, web UI): runpod.io → add $40–60 credit → Pods → Deploy → **RTX 4090 (24 GB)** (fallback A5000/L4/A40; for 32B model: A100 80 GB) → **On-Demand** → template "RunPod PyTorch" → container disk 40 GB, volume 80 GB → **Expose HTTP ports: 8000**, **Expose TCP ports: 22** (needed for scp) → Deploy. Settings → SSH keys: paste `~/.ssh/id_ed25519.pub`. Copy the **"SSH over exposed TCP"** command from Connect (the `ssh.runpod.io` proxy variant does not support scp). If scp is impossible, move data with `runpodctl send data.zip` (laptop) → `runpodctl receive <code>` (pod).
Codex does the rest with `scripts/pod_setup.sh` (run on the pod over SSH):
1. install uv, ffmpeg, tmux; clone the repo (private GitHub, fine-grained read-only token) into `/workspace/datathon-kit`; `uv sync` (project env = CPU libraries only);
2. `bash scripts/get_models.sh`; create the **separate vLLM env** and download the VLM to `/workspace/models/` exactly as in `auditor.md` §1;
3. start vLLM (127.0.0.1:8001) and the service (0.0.0.0:8000, `AUDITOR=on`, `VLLM_URL=http://127.0.0.1:8001`) in two `tmux` windows with restart loops; both with `HF_HUB_OFFLINE=1`;
4. copy data for eval from the laptop: `scp -P <port> -r ~/datathon-kit/data/raw/open root@<ip>:/workspace/datathon-kit/data/raw/`.
Everything lives in `/workspace` (the persistent volume) so a pod restart keeps weights and data.
Note: the RunPod HTTP proxy closes requests after ~100 s — fine, because POST /analyze only uploads and returns a job_id; never make a request wait for analysis.
Public URL: **`https://<POD_ID>-8000.proxy.runpod.net`** (no cloudflared). Update code: `git pull` on the pod + restart the service.
Keep the pod **running** (not stopped) until the jury finishes. Our pod: A40 48 GB, $0.59/h, $15 budget, auto top-up off → ~24 h of runtime from 20:30 Oct 17. **If the jury may test after ~18:00 Oct 18, add credit or switch the jury URL to the laptop before the credit runs out** (a stopped pod = every video unprocessed). Laptop CPU mode stays ready as a fallback URL.

## Docker (portability story for the bank's closed network)
`Dockerfile`: `python:3.12-slim`, install uv, copy repo, `uv sync --frozen --no-dev`, run `scripts/get_models.sh` at build time, `CMD uvicorn service.app:app --host 0.0.0.0 --port 8000`. Runtime needs no internet. (Docker is for the pitch and the submission; the live demo can run without it.)

## WSL notes
- Codex sandbox needs `bubblewrap` (`sudo apt install -y bubblewrap`).
- If OpenCV can't decode a .mov, install ffmpeg (`sudo apt install -y ffmpeg`) and use the ffmpeg fallback in frames.py.
