# Approved local video decoder

Status: **approved and installed on 2026-10-17** after the human delegated full local implementation. The root agent downloaded and verified the pinned archive and selected members. Synthetic provisioning tests use tiny fake archives. AGENTS.md rule 9 required approval for this native dependency; any different dependency remains a separate decision.

The installed llama.cpp runtime advertises native video support. The required `ffmpeg`/`ffprobe` tools were absent during inspection and are now installed to enable that path. They do not provide an action model or establish detection quality.

## Exact proposed assets

The [official FFmpeg download page](https://ffmpeg.org/download.html) links Linux static builds to maintainer BtbN. Use its [dated release](https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-10-16-13-04), not the floating `latest` release.

| Field | Pinned value |
|---|---|
| Release | `autobuild-2026-10-16-13-04`, release ID `414971902` |
| Builder revision | `63ac3265b0789ea16c7b1f8351cd2ad195a28722` |
| FFmpeg source revision | `46d899d7a02bc63237472cc4897258d5ddf274ac` |
| Archive | `ffmpeg-n8.1.3-18-g46d899d7a0-linux64-lgpl-8.1.tar.xz` |
| Asset ID | `647361913` |
| Archive bytes | **137,124,016** |
| Archive SHA-256 | `70df7305a38b8dc518e35e44d1f165edc74033082ae53b5d977acfa2849fe8fb` |
| Incorporated GPL-v3 license text | **35,147 bytes**, SHA-256 `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` |
| LGPL-v3 license inside archive | **7,651 bytes**, SHA-256 `da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768` |

Exact asset bytes/digest/IDs were read from the [public GitHub release API](https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/autobuild-2026-10-16-13-04). The [expanded asset listing](https://github.com/BtbN/FFmpeg-Builds/releases/expanded_assets/autobuild-2026-10-16-13-04) independently displays the digest. The archive download URL is fixed in the provisioner and contains this dated release and exact filename. Daily-release retention is limited; a future missing asset fails instead of silently selecting another build. See the [maintainer retention policy](https://raw.githubusercontent.com/BtbN/FFmpeg-Builds/63ac3265b0789ea16c7b1f8351cd2ad195a28722/README.md).

Exact proposed network payload is **137,159,163 bytes**, including the separate GPL-v3 license text. No Python package, model weight, GPU runtime or system package is added.

The first approved installation established the exact selected binary bytes, now independently pinned in the provisioner:

| Installed executable | Bytes | SHA-256 |
|---|---:|---|
| `bin/ffmpeg` | 142,166,568 | `9087520002e707613b9c20fd3e251474bf85eaa16156ba4df42652231f7b1046` |
| `bin/ffprobe` | 141,933,032 | `db87fc6e0f78cf83b7aacb2d21f7658216f0f861834f3612c9931728256e026e` |

Selected executables and both license texts total **284,142,398 bytes**. Verified total disk use including retained archive and manifest is **421,268,198 bytes**. The installer still imposes a **1,210,966,523-byte maximum disk budget**, including the archive, at most 1 GiB of uncompressed archive contents, the additional license and a bounded manifest. Before approval, installed binary sizes were explicitly unknown rather than estimated as facts.

## License and compatibility

This is the maintainer's **LGPL static variant**, excluding GPL-only dependencies such as x264/x265; it is not the GPL or nonfree variant. The [pinned build configuration](https://raw.githubusercontent.com/BtbN/FFmpeg-Builds/63ac3265b0789ea16c7b1f8351cd2ad195a28722/variants/defaults-lgpl.sh) enables version 3 and selects `COPYING.LGPLv3`. The [packaging script](https://raw.githubusercontent.com/BtbN/FFmpeg-Builds/63ac3265b0789ea16c7b1f8351cd2ad195a28722/build.sh) copies that source license into archive `LICENSE.txt`.

LGPL permits commercial use subject to its terms. Preserve the LGPL text and incorporated GPL-v3 text, provenance and source links. Invoke standalone executables as subprocesses; this proposal does not link project Python code against FFmpeg libraries. Shipping binaries to third parties requires the corresponding-source and notice obligations to be addressed; a source URL alone is not claimed to complete redistribution compliance. [FFmpeg legal guidance](https://ffmpeg.org/legal.html), [pinned LGPL-v3 text](https://raw.githubusercontent.com/FFmpeg/FFmpeg/46d899d7a02bc63237472cc4897258d5ddf274ac/COPYING.LGPLv3).

The [maintainer platform contract](https://raw.githubusercontent.com/BtbN/FFmpeg-Builds/63ac3265b0789ea16c7b1f8351cd2ad195a28722/README.md) targets Linux x86_64 with glibc >=2.28 and kernel >=4.18. Local version checks passed for both executables (`n8.1.3-18-g46d899d7a0-20261016`), and decoder listings confirm H.264/MJPEG support. No bank footage was used. Readiness hashing itself does not execute the tools. The LGPL variant omits external x264/x265 encoders.

## Reviewable commands

Safe inspection now, with no network, installation or binary execution:

```bash
uv run python -m scripts.provision_video_decoder
uv run pytest -q tests/test_video_decoder_provision.py
```

Installation is now approved for this exact asset plan; repeat invocation verifies the existing files:

```bash
uv run python -m scripts.provision_video_decoder --install
```

All writes remain under `tools/local-video-decoder/`: retained archive, `bin/ffmpeg`, `bin/ffprobe`, license texts and a manifest. No system PATH changes occur. Consumers call `verify_ready()` to obtain explicit paths and recheck pinned archive/selected-file hashes before use. Readiness checks hash the archive, executables and licenses directly, without reopening or decompressing the tar/xz archive. They reject a modified binary even when its local manifest is edited to match it. Full archive metadata validation and selected-member hash derivation still run during fresh installation, and derived hashes must match the independent pins before publication.

Downloads have exact size/SHA-256 checks. Every tar path, member count, total size, file type and link target is checked; selected executables/licenses must be regular files. Only the fixed allowlist is materialized, without archive links. Existing mismatches are preserved and cause failure. Neither installation nor verification executes downloaded binaries.

Readiness now completes in **0.270473 seconds**, compared with approximately two minutes for repeated tar/xz validation. Fifteen synthetic provisioning tests pass. Next, run the prepared isolated synthetic native-video protocol check and measure local timing after the existing model-server comparison is complete. It tests basic motion/order with helper timestamp text disabled; precise event-time alignment is a separate test. Action recognition, bank-rule grounding and improved held-out detection remain unproven; no bank media or descriptions may leave the machine.
