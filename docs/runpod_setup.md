# Approved Runpod setup — one48GB Pod,$15 total cash

17 October2026. The human explicitly approves the48GB Runpod option and$15 total cap and reports that this deployment is permitted. Record this as user-reported approval, not an independently obtained organizer statement. This authorizes the proposed GPU runtime/model setup and scoped case processing inside the team-managed Pod; external hosted AI APIs remain excluded.

## 1. Account and prepaid funding — human browser steps

Open https://console.runpod.io/ and create/sign into your account. Browser control from this WSL task fails before account access (`sandboxCwd is not a local file URI`); login/payment must be completed by the human. No password or API key is needed in chat.

Use prepaid credit with auto top-up disabled. Prefer a custom$12 credit addition if the final checkout including tax/fees is at most$15. If only a$10 choice fits, record both credited balance and actual charge; the operating window will be shortened accordingly. Do not choose a$25 top-up, subscription or savings plan. Total cash paid across all top-ups must remain at most$15.

## 2. Prepared SSH key — registered through MCP

The authenticated MCP connection has registered the prepared public key and confirmed it by reading the account key list back. No manual key upload is needed. The public key is recorded at `data/processed/runpod-setup-v1/ssh-public-key.txt`; its private key stays locally in the ignored setup directory and must never enter a bundle or chat.

## 3. Exact Pod configuration

| Field | Selection |
|---|---|
| Product | **Pods**, not Serverless or a hosted model endpoint |
| Cloud | Secure Cloud,on-demand |
| GPUs | **1× A40 or RTX A6000,48GB** |
| Public quote ceiling | **$0.59/GPU-hour**, recheck the final configuration total |
| Template/image | Official Runpod Base,Ubuntu24.04/CUDA12.8.1; exact image below |
| Container disk |20GB |
| Persistent volume |60GB,mounted at`/workspace` |
| Total disk |80GB; do not accidentally allocate80GB twice |
| HTTP port |8000 only, for the application after setup |
| TCP port |22 for direct SSH/file transfer |
| Model port |8003 on loopback only; do not expose publicly |
| GPU count |1 |
| Auto top-up |Off |
| Notebook |Not needed; no public Jupyter port |

Pinned base image, verified via Docker Hub's official`runpod/base`metadata:

`runpod/base:1.4.1-cuda1281-ubuntu2404`

Digest:`sha256:fe1287968710a5cb5dc8835f1818856ecfa20057eec8b46a76495f81182788e5`.

Where supported, use`runpod/base@sha256:fe1287968710a5cb5dc8835f1818856ecfa20057eec8b46a76495f81182788e5`as the image reference. Keep the image's default startup command. Verify the live configuration shows SSH access; do not replace the startup command with a generic sleep command that prevents SSH initialization. If the console cannot configure this base image or only offers a different hourly price, share the non-secret configuration values before deployment rather than improvising a larger instance.

The prepared repository setup installs the pinned standalone CUDA llama.cpp runtime; PyTorch/vLLM/Transformers installation is not required. Base-image infrastructure software is recorded by image digest; Python project packages still use the existing uv lock.

## 4. Assistant provisions through authenticated MCP

OAuth and live account access are verified. The account currently has no Pods and no recorded usage. The human reports it is not funded yet and is adding funds. Before allocating paid compute, record credited balance, actual total cash charged including tax, and confirmation that auto top-up is off. The total charge must stay within$15; the assistant cannot read the personal prepaid wallet through the available MCP tools.

After funding is confirmed, the assistant rechecks live stock and price, creates one approved Pod through MCP, then reads it back to verify GPU, price, image, storage and access. MCP manages the resource; the prepared local SSH key handles code transfer and remote commands. No manual deployment or copying of SSH details is needed while MCP works. Rental begins at allocation and includes setup/download time.

Compute a conservative operating deadline from the actual credited balance and live rate; preserve serving time for judging. No second Pod, auto top-up or unsupported provider-enforced cap is assumed.

## 5. Assistant-owned work after SSH is available

1. Verify the host key/SSH connection, GPU model/VRAM,driver,OS,disk and uv availability. Record actual price/balance/time rather than a planning quote.
2. Transfer the code-only bundle, not the private SSH key,API credentials,entire worktree,raw data or old model caches. Verify bundle hashes.
3. Install the locked Python environment through uv and download the pinned GPU runtime/model assets directly on the Pod,verifying byte counts and SHA-256. Preserve model/runtime licenses.
4. Verify actual CUDA device listing/offload. Run public/synthetic input controls first, then the approved bounded bank-video qualification. GPU availability is not a semantic pass.
5. Keep model port private. Publish the keyed asynchronous application on8000 once the selected recognition engine is qualified; preserve the five-minute deadline. The Runpod proxy's100-second connection limit is compatible with quick job-ID responses/status polling,not one long synchronous request.
6. Keep the same funded host available for jury evaluation. Prepaid credit/no auto top-up is the primary financial boundary; stopping only the model process does not stop Pod billing. A stopped Pod can retain billable storage. Verify provider stop/cleanup when the approved judging window ends.

No GPU allocation or remote execution is claimed until provider readback and a real SSH command confirm them. The current setup remains preparation while funding is pending.

Official references: [SSH keys and direct connection](https://docs.runpod.io/pods/configuration/use-ssh),[port exposure](https://docs.runpod.io/pods/configuration/expose-ports),[billing](https://docs.runpod.io/accounts-billing/billing),[Pod/storage pricing](https://docs.runpod.io/pods/pricing).
