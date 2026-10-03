# Interactive model setup and tuner

Strata Unleashed **0.1.38-r4** uses the upstream Strata 0.1.38 engine version. The CLI runs on x86-64 Windows and Linux. Its measured self-tuner currently uses **one NVIDIA CUDA GPU**; it does not promise AMD, CPU-only, or multi-GPU tuning.

## Start

Linux:

```bash
./cli-unleashed.sh
```

Windows:

```bat
CLI-UNLEASHED.bat
```

Running `start-unleashed.sh` or `START-UNLEASHED.bat` without arguments also opens this wizard. Explicit `--model` and `--config` arguments keep the direct launch behavior.

The scripts create a local Python environment and install the project's Python dependencies. Python 3.10+ must already be installed. Source builds need CUDA Toolkit, CMake, Ninja and a C++20 compiler. On Windows, run from an **x64 Visual Studio C++ developer terminal**. The wizard detects missing tools before starting a large download for a model it cannot run.

The Linux release includes the previously tested Ada/sm89 binary. It requires glibc 2.43, GLIBCXX_3.4.32, AVX2 and CUDA 13 libraries. The wizard requests a local source build when the bundled binary cannot execute or does not target the selected GPU. The Windows release supplies source/build scripts, not a Windows binary tested by this project.

## What happens

1. Scan OS, CPU, installed/available RAM, free disk, NVIDIA GPU(s), free VRAM, compute capability, driver, engine and build tools.
2. Choose compact/general, coding, larger model, or uncensored use. Select a GPU when several are present.
3. Set extra VRAM and system RAM to keep free for other applications; context remains 262,144. Automatic safety headroom is separate.
4. Compare supported model/quant choices, largest CPU footprint first. Each shows download size, SSD engram table, CPU MEM and GPU MEM at FP16/Q8 KV. Memory estimates are informational.
5. Choose download/setup/chat (the default), download/setup/API, setup only, or download only. API mode prompts for an available local port. Nothing downloads until you select an action.
6. Download every shard from a pinned Hugging Face revision, resume partial files, verify published SHA256 hashes, and check actual headers before packing.
7. Run the procedural tuner: fit the VRAM budget, optimize 512-input/512-output decode, then prefill while retaining at least 90% of the measured decode reference. Save the result for later reuse.

Downloaded weights stay in `models/<provider>/<quant>/<revision>/`. Catalog updates, headers, temporary files and selections stay in `work/`; the scanner never searches unrelated folders for models. `--local-model` explicitly opens existing weights read-only, and writes its prepared pack and tuning results inside the project.

After setup, terminal chat uses no system prompt: only user/assistant turns, with template thinking instructions disabled. The rendered prompt is checked before inference. `/new`, `/tokens N` and `/quit` control the test session. API mode keeps the ordinary web/API flow.

The scan identifies the storage device and takes an uncached, read-only sequential sample from an existing file (up to 32 MiB). The selected model's volume is checked again before launch. Non-NVMe/unknown storage receives an experience-may-vary notice; samples below 500 MiB/s receive a slow-read notice. Neither is a hard compatibility block or a guarantee of inference speed. If uncached reads are unsupported, no speed is claimed.

## Catalog and recommendation policy

The bundled catalog was checked against the publishers on **2026-10-03** and includes **22 complete model variants**. All shard directories were inspected, including tensors split across files. Sizes, SHA256 hashes and source revisions are recorded in [../data/unleashed-models.json](../data/unleashed-models.json).

| Publisher | Quants | Notes |
|---|---|---|
| [ISTA Flash Next Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | IQ1_M (expert-pruned Coder release) | Smallest option; coding focused |
| [ISTA Flash Next](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S | Small general models |
| [Atomic Chat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF) | AD-3.84bpw-IQ4_XS-M64, AD-4.27bpw-Q4_K_M-M64, AD-5.00bpw-Q5_K_M-M64 | Main pick for larger models |
| [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | Q8_0, UD-IQ1_M, UD-IQ3_XXS, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL | Larger quants available; optional, not the first recommendation |
| [HuiHui](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) | Q8_0, UD-Q4_K_XL | Uncensored / abliterated |

“Up to 3-bit” includes ISTA's published IQ3_S family (approximately 3.5-bit in upstream's description), and the Coder's IQ1_M filename is not a claim that every tensor is one-bit. Mixed quantizations are checked by their actual tensor types, not filenames. Atomic currently publishes three Flash Next variants; imatrix, vision-projector and MTP artifacts are not presented as independent language models. This catalog covers **Flash Next**, not unrelated 27B or 2.4T architectures.

The BF16 expert models from HuiHui and Unsloth, and Unsloth UD-IQ1_S, are hidden because this engine does not implement their expert encodings. Other variants passed header/layout checks; this is **not a claim of inference or quality validation for all 19 compatible variants**. The recorded full-model throughput validation remains the Atomic Q5 run.

The uncensored recommendation never silently substitutes a different provider. If no HuiHui candidate fits, it says so. Publisher preferences are user-requested defaults, not a measured quality ranking. A larger file is not automatically smarter or faster.

## RAM and VRAM reservations

Interactive setup asks separately for extra VRAM and system RAM to leave available, in MiB. The equivalent flags are `--reserve-vram-mib 2048 --reserve-ram-mib 8192`. Both default to zero additional reservation. RAM/VRAM estimates do not reject models before preparation, launch or cached-profile reuse. The tuner samples available system RAM and rejects measured trials below the reserve. Both reservations are recorded and change the tuning-cache identity. This is measured headroom, not an OS-enforced limit during subsequent API/chat sessions.

## KV defaults and native context

User guidance: **FP16/FP16 is ideal; FP16/Q8 is recommended; Q8/Q5 is a last resort only when no other viable option fits.** These recommendations do not replace the capacity-based defaults below.

Defaults use total card capacity, independently of model weight quantization: FP16/FP16 at 16 GiB+, FP16/Q8 at 12–16 GiB, Q8/Q8 at 8–12 GiB, and Q8/Q6 below 8 GiB. A 64 MiB tolerance accounts for driver-reported overhead. `--kv` overrides the default explicitly; tuning never changes it automatically.

Every trial and terminal inference test allocates 262,144 tokens. The long workload is 261,624 prompt tokens plus 512 generated tokens and the upstream engine’s eight-token internal margin. Imported extended-context/RoPE settings are normalized to native context with scaling disabled. A matching capacity does not guarantee a fit on small GPUs.

## Fit estimates and limitations

The wizard estimates this tuner's **resident-expert mode**, not upstream's separate SSD-streaming low-RAM configurations. It budgets:

- CPU MEM: host expert bytes + host token embedding + 6 GiB runtime allowance. PLE/engram bytes stay SSD-backed; main KV stays on GPU. GPU expert caches retain their host copies.
- Disk: remaining model download bytes + converted pack bytes + 2 GiB working headroom. Existing complete or partial download sizes reduce the estimate; every complete file is still hashed before reuse.
- GPU MEM: GPU dense bytes + 4.59375 GiB FP16/Q8 KV at 262,144 context + 2 GiB workspace, before the adaptive expert cache. Available allocation space is current free VRAM minus the user reserve and automatic `max(512 MiB, 2% total VRAM)` headroom.

RAM/VRAM estimates are informational and never block model selection or launch. Actual allocation failures and measured RAM/VRAM reservation checks still apply. Unsupported model encodings are omitted. See [the complete per-model memory chart and Q5 measurements](MEMORY.md).

No throughput or quality is invented for an untested model. Tuning and inference use native 262,144-token context. Long prefill validation uses 261,624 input tokens plus 512 generated tokens. The tuner never falls back to a shorter context. MTP is optional; `--mtp` accepts an already prepared runtime folder and the CLI does not automatically download or convert a draft head.

## Commands

Use the same arguments after `CLI-UNLEASHED.bat` on Windows.

```bash
# Hardware only; no model/network downloads (apart from first-run Python dependency setup).
./cli-unleashed.sh --scan

# All models, fit reasons and the larger-model recommendation.
./cli-unleashed.sh --list --intent large

# Machine-readable report, using only the bundled catalog.
./cli-unleashed.sh --list --offline --json

# Refresh publisher files, immutable revisions, SHA256s and all shard headers.
./cli-unleashed.sh --refresh-catalog --list

# Download a specific model without prompts, verify and tune with 2 GiB extra reserve.
./cli-unleashed.sh --model-id atomic/AD-5.00bpw-Q5_K_M-M64 \
  --reserve-vram-mib 2048 --tune-only --yes

# Use existing weights; no copies/downloads, another local HTTP port.
./cli-unleashed.sh --local-model /path/to/model-00001-of-00033.gguf --port 8101
```

| Flag | Meaning |
|---|---|
| `--scan` | Print the hardware scan and exit |
| `--list` | Print all catalog choices and fit reasons, without downloading weights |
| `--json` | JSON output with `--scan` or `--list` |
| `--offline` | Use bundled metadata; allows listing/scanning and existing local models, no downloads |
| `--refresh-catalog` | Update metadata and header checks into `work/model-catalog.json`; retain the last catalog if refresh fails |
| `--intent small\|coder\|large\|uncensored` | Recommendation preference |
| `--model-id ID` | Exact ID from `--list` |
| `--local-model PATH` | Existing complete shard family; any shard can identify it |
| `--gpu N` | Physical NVIDIA GPU index, default 0; one GPU at a time |
| `--allow-over-budget` | Compatibility option; memory estimates no longer block launch |
| `--reserve-ram-mib N` | Extra available system RAM to retain, in MiB; default 0 |
| `--reserve-vram-mib N` | Extra VRAM for other applications, default 0 |
| `--context N` | Native 262,144 tokens; this is the only accepted tuning capacity |
| `--kv PAIR` | Explicit KV override; by default 16 GiB+: FP16/FP16, 12–16: FP16/Q8, 8–12: Q8/Q8, below 8: Q8/Q6 |
| `--port N` | Local HTTP port, default 8100 |
| `--mtp PATH` | Optional already prepared MTP runtime |
| `--chat` | Open terminal inference test chat with no system prompt |
| `--build` | Rebuild the engine for this machine |
| `--retune` | Repeat measurements even if a cached result exists |
| `--tune-only` | Tune and save configuration without starting the HTTP server |
| `--download-only` | Verify/download the selected model and exit |
| `--yes` | Execute an explicitly selected model without prompts; never selects a model silently |
| `--version` | Print the version shared with CMake/upstream |

A second download process is refused to protect partial files. Interrupted downloads resume; invalid byte ranges and SHA256 mismatches are rejected. Corrupt partials are preserved with `.bad-*` names so they do not cause an endless failed resume. Existing invalid completed files are left untouched for the user to move aside.

## Validation

The release checks the live Linux scan, all 22 pinned remote variants' headers, the SSD Atomic Q5 local inspection, scripted interactive exit and CLI JSON output. Automated tests exercise Windows-style scan/build requirements, recommendations, insufficient resources, command forwarding, complete shard grouping, download resume and integrity failures. Windows script/native execution still needs validation on Windows; it is not claimed from these platform simulations.

Reservation inputs mean **memory to leave free**, not memory available to the model. Interactive prompts accept units and echo the converted value:

| Input | Reserved amount |
| --- | --- |
| `4096` or `4096 MiB` | 4096 MiB = 4 GiB |
| `8192` or `8 GiB` | 8192 MiB = 8 GiB |
| `4096mb` | 4096 decimal MB, rounded up to 3907 MiB |
| `8192mb` | 8192 decimal MB, rounded up to 7813 MiB |

MB/GB use decimal units; MiB/GiB use binary units. Rounding to whole MiB never reserves less than requested. Command-line flags named `--reserve-*-mib` still take integer MiB. With `--reserve-vram-mib 8192 --reserve-ram-mib 4096`, the GPU reserve is 8 GiB and system RAM reserve is 4 GiB; they are not swapped or added to each other.
