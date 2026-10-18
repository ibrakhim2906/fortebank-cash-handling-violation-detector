# GPU access and bounded deployment recommendation

**Subsequent action:** the human approves the48GB Runpod configuration and$15 total cap and reports it is allowed. Setup is now authorized as described in `docs/runpod_setup.md`. Dedicated SSH access and code-only deployment assets are prepared; official hosted Runpod MCP configuration is added but OAuth is pending. No paid Pod or remote model execution is claimed. The research-only status and approval prerequisites below describe the state before this explicit approval.

Checked17 October2026. Research and plan update only: no account created, payment, machine allocation, model download or bank-media transfer. Public quotes are not live available inventory. Private account balance could not be read.

## Decision

Prefer one **Runpod on-demand Secure Cloud A40 or RTX A6000,48GB VRAM**, if a live checkout quote fits the proposed **$15 total cash cap including tax/storage**. The public pricing page lists both at$0.59/hour. A single host can perform development, verification and jury serving; its uptime must cover actual judging, not just the submission deadline. Do not choose a short-lived free notebook as the final API host.

If only$10 is available, or48GB inventory is absent, the price-first fallback is an RTX A5000,24GB, advertised at$0.27/hour, with the8B model. One host only; do not allocate both. RTX4090,24GB, is listed at$0.89/hour and is not the budget-first choice for continuous operation. Prices must be rechecked at checkout. [Provider pricing](https://www.runpod.io/pricing).

## NVIDIA credits: question resolved at program level

The human confirms the credit page is **NVIDIA Build / API Catalog**, not Brev Billing. NVIDIA's API trial terms describe these credits as consumption of API Services. No documented conversion to general GPU-machine credit was found. Do not count them as rental funds or send case footage to hosted recognition endpoints. [API credit terms,section1.4](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf).

Brev is a separate resource console whose organization balance pays compute/storage and can receive credit codes. Only an actual Brev balance or applicable compute voucher would establish usable machine credit; the user has identified Build instead. [Brev Billing](https://docs.nvidia.com/brev/guides/console-reference#billing--usage).

The Computer Use initialization was attempted and failed before account inspection with `sandboxCwd is not a local file URI: file:///home/ibrak/datathon-kit`. This is a tool-access failure, not evidence that the user's NVIDIA account is empty or lacks a session. No password/API key was requested or read. Program classification is based on the user's answer and public terms, not an authenticated balance lookup.

## Free resources that actually exist

**Lightning AI:** published offer is5 free credits on registration and25 more for adding a card, subject to account eligibility. Its headline of up to80 GPU hours is not80 hours on any chosen GPU. The free tier advertises4-hour Studio restarts. Use it for a short public/synthetic benchmark if credits are immediately available; account verification, allocation and card eligibility are not established here. Do not rely on it for uninterrupted jury serving. Older docs describe different monthly grants; use the current pricing/account balance and confirm live terms. [Current pricing and offer](https://lightning.ai/pricing), [workspace credit documentation](https://api.lightning.ai/docs/platform/build/ai-studio).

**Google Colab:** free GPU access exists, but resource availability is unguaranteed. The official FAQ restricts unrelated web-service hosting and free-tier remote-control/UI bypass workloads. It is a notebook benchmark option, not our final keyed public service. [Official FAQ](https://research.google.com/colaboratory/faq.html).

**Vast.ai:** documented$5 minimum deposit is a lower entry point, not free compute. Marketplace offers vary with host, reliability, networking and additional charges. A current acceptable host/quote was not selected. Keep it as a purchase fallback, not a second paid experimental platform. [Official quickstart](https://docs.vast.ai/guides/get-started/quickstart).

Bound free-access investigation to15–20minutes once the user can sign in. Do not spend the remaining datathon waiting for an unconfirmed grant.

## Proposed exact rental configuration

- One on-demand Secure Cloud A40 or RTX A6000,48GB; no spot/preemptible instance, subscription or savings plan.
- Public GPU quote ceiling$0.59/hour for this recommendation; reprice the whole plan if the actual quote is higher.
- Approximately50GB host RAM as listed, x86_64 Ubuntu24.04-compatible CUDA environment; verify the actual host/driver before executable install.
- Total disk allocation80GB, avoiding accidental double allocation of80GB container plus80GB volume. A suggested split is20GB temporary container and60GB persistent work volume. Model and code copies remain reproducible locally.
- The required public API/UI uses one exposed application port with HTTPS. Keep the model endpoint on loopback, protect the application API and do not expose notebook/terminal services as the product. Verify real automated API requests without browser-login redirects. [Runpod port exposure](https://docs.runpod.io/pods/configuration/expose-ports).
- No auto top-up or recurring purchase. Track running charges and a shutdown deadline; do not assume an application timer alone is a provider-enforced spending cap.
- Confirm paid uptime through the actual judging/pitch window before allocating. A stopped Pod releases GPU capacity; persistent storage can still cost money, and restart capacity must not be assumed. Preserve approved cleanup/recovery procedures. [Billing and storage](https://docs.runpod.io/pods/pricing).

## Budget arithmetic, not a checkout quote

Computed from$0.59/GPU-hour and80GB at$0.10/GB/month, using a30-day month for illustration. The actual provider billing basis, disk allocation, taxes and checkout totals prevail.

| Runtime | GPU compute | Estimated running storage | Before tax |
|---|---:|---:|---:|
|18hours|$10.62|$0.20|$10.82|
|20hours|$11.80|$0.22|$12.02|
|24hours|$14.16|$0.27|$14.43|

Plan18hours initially; extend toward20 only if the tax-inclusive total remains within$15 and the judging window requires it.24hours leaves inadequate contingency under$15. On the cheaper A5000,$0.27×24hours=$6.48 compute before storage/tax, suitable for the$10 fallback. These are estimates, not incurred charges or guaranteed allocations.

Runpod says a newcomer can start with as little as$10; deposited credits are non-refundable. Deposit and spend are different: **total out-of-pocket funding including checkout tax must also stay within$15**. Do not deposit$15 and then add an unexpected tax charge above the user's budget. [Funding/refund documentation](https://docs.runpod.io/accounts-billing/billing#refunds-and-credits).

## Stronger recognition resources and one bounded selection

On48GB, qualify **Qwen3-VL-30B-A3B-Instruct Q4_K_M** as the primary observation candidate. Its sparse architecture is a capacity/speed hypothesis, not a bank-recognition benchmark. Fall back once to the already specified8B model if quality, latency, memory or runtime compatibility fails. No larger-model sweep and no full external-dataset training. All existing observation controls, human reference separation, source evidence and rule prerequisites remain required. Only one model is resident at a time.

Official30B source is Apache2.0 and documents CUDA/llama.cpp compatibility. [Model release](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF).

Pinned revision:`f54435e6cc31258f04b0969105c3f6badb197931`.

| Asset | Bytes | SHA-256 |
|---|---:|---|
|`Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf`|18,556,687,168|`87bb374d849f80ebdfabb304189fac9e0bd35a0f74506e6a59c51b206cbe863b`|
|`mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf`|1,083,499,584|`cae72cf123cc9e08d553cd5a5055d6d3cf0f82652aa41c3e4aa424cda9a26f7f`|

Total19,640,186,752bytes, roughly19.64GB. Weight-file size is not peak VRAM/RAM. Metadata was fetched from the revision-specific official Hugging Face API; no weights were downloaded. The existing8B candidate remains pinned at6,186,814,624bytes. Machine-readable URLs/hashes and price arithmetic are in `data/processed/gpu-resource-audit-v1/manifest.json`.

Proposed GPU runtime is the CUDA12.8 Linux x86_64 variant of the **same llama.cpp b11707** release, rather than introducing a Transformers/vLLM dependency stack:

| Official release asset | Bytes | SHA-256 |
|---|---:|---|
|`llama-b11707-bin-ubuntu-cuda-12.8-x64.tar.gz`|172,731,212|`208d228a2e752b91ebb15717f9116915a1704f2af644ee278d4b4d7911199e6a`|
|`cudart-llama-b11707-bin-ubuntu-cuda-12.8-x64.tar.gz`|594,377,569|`a0698562c748ea4ffcf2631f13ca1429c3b794e67272066053c21c607ade5ebf`|

Release metadata/digests were verified through the official GitHub API. Preserve the existing MIT license and exact binaries' provenance. These CUDA binaries/libraries are a concrete proposed additional runtime install requiring approval; current CPU-runtime approval alone does not claim they are installed. Check actual OS/driver/library compatibility and record the selected container image digest before use. [Official release](https://github.com/ggml-org/llama.cpp/releases/tag/b11707).

MIT covers llama.cpp; do not label bundled NVIDIA CUDA runtime libraries as MIT. Preserve and verify their actual bundled license notices during approved asset preparation. Confirm real GPU execution using the model server's device listing and NVIDIA runtime utilization, including vision-projector offload; do not accidentally benchmark a CPU fallback on a paid GPU. The existing CPU launcher's explicit offload-disabling flags require a deliberate GPU configuration change before use.

Cap qualification and any single8B fallback at**$2 compute** and one fixed reference pack. The GPU's benefit should be measured as sharper/denser chronological coverage and event verification within240seconds, not larger legal explanations. Before any bank inference, use public/synthetic controls and verify all model computation is inside the team's instance with outbound inference traffic disabled after asset preparation.

## Eligibility/data boundary and next actions

The brief defines local processing as all models running inside the team-deployed service and forbids external cloud AI APIs. A team-managed GPU VM differs technically from a hosted model endpoint, but the brief does not explicitly authorize third-party hosting of supplied/hidden bank media. **Organizer confirmation and the human's scoped data-transfer authorization are still needed before bank videos go onto a rented machine.** Record that confirmation rather than asserting compliance from architecture alone. No organizer message was sent.

Question ready for the human to ask: “May we run our own downloaded models and service on a rented GPU VM, with no calls to external AI APIs, process jury videos only inside that instance and keep the system portable/offline after weights are downloaded?”

Research and the requested plan update are complete. Actual next steps require a usable provider account/live quote, approval of the exact runtime/configuration and paid allocation, and the data-hosting clarification before bank transfer. No private credential should be pasted into chat. Prepare/configure access locally or through the provider's secure login flow.
