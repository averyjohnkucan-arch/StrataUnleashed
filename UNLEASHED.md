# Strata Unleashed

For interactive hardware scanning, model/quant recommendations and downloads, start `cli-unleashed.sh` (Linux) or `CLI-UNLEASHED.bat` (Windows). See [the complete CLI guide](docs/UNLEASHED-CLI.md). The original direct launch commands below still work. Add `--chat` for terminal inference testing with no system prompt, or `--chat --chat-prompt "Reply with 4"` for a single test. The README also provides one-command app installation and model download.

A local fork of Niko1221/Strata with the retained SSD changes, broader native weight formats, optional MTP, independent K/V precisions, and measured self-tuning. The supported model family is **Qwen3.8-Flash-Next / `qwen4exp`**, including its compatible quantized and pruned variants. This is not a generic engine for unrelated architectures or arbitrary attention dimensions.

## Start on Linux

```bash
./start-unleashed.sh --model "/path/to/model-00001-of-00033.gguf"
```

## Start on Windows

Open an **x64 Visual Studio developer terminal**, then:

```powershell
.\START-UNLEASHED.bat --model "D:\Models\model-00001-of-00033.gguf"
```

The launchers create `.venv` inside the extracted folder, prepare the model pack there, tune automatically, and serve the local API. Keep every shard together; any shard filename may be supplied. Original weights are opened read-only. The bundled expert profile is used only when its layer and expert counts match the model; otherwise a matching initial profile is generated locally. Logs, packs, caches and tuning results go under `work/` in this folder.

Python 3.10+, an NVIDIA GPU and a compatible driver are required. A source build also needs CMake, Ninja, a C++20 compiler and CUDA Toolkit (13.1 was used for the Linux validation). Windows source builds use Visual Studio's C++ toolchain; Linux uses GCC. `--build` rebuilds for the current GPU. The launchers do not install system drivers or toolchains.

Add `--mtp /path/to/prepared-mtp-runtime` to let the tuner compare speculative decoding with a truly unloaded MTP head. Without that option no draft weights are required. A prepared runtime contains `dense.bin`, `dense.txt`, `experts.bin` and, where applicable, `draft_vocab.bin`. Existing upstream MTP preparation tools remain available.

Useful options:

- `--context 262144`: native context for every tuning trial and inference session. Imported configurations are normalized to this capacity; reduced-context tuning is not accepted.
- `--kv PAIR`: explicit cache-precision override. Otherwise GPU capacity chooses FP16/FP16 (16 GiB+), FP16/Q8 (12–16 GiB), Q8/Q8 (8–12 GiB), or Q8/Q6 (below 8 GiB). The choice stays fixed during tuning. Driver-reported totals within 64 MiB below a tier count as that nominal tier.
- `--reserve-vram-mib 2048`: keep an additional 2 GiB free for other applications; default `0`. This is in addition to the automatic safety margin. Changing it creates a separate tuning result.
- `--retune`: measure again, even when a validated result is cached.
- `--tune-only`: save the result without starting the API.
- `--config path.json`: tune an existing local engine configuration. It supplies context, port and optional `tune_mtp`; `--model`, `--context` and `--mtp` are for new configurations.
- `--gpu N`: select one physical NVIDIA GPU; defaults to 0 for new model configurations.
- `--port 8100`: local API port; also overrides an existing configuration without retuning.

## How self-tuning works

The launcher fingerprints the engine, tuner, server, GPU/driver, CPU, model shards, pack, expert profile, requested VRAM reserve and relevant environment settings. A different fingerprint gets its own tuning directory. A validated matching result is reused on later launches. First tuning can take tens of minutes or longer because it measures complete generations and repeatedly reloads the model; later launches reuse the saved decision.

The default fills as much VRAM as practical, keeping **max(512 MiB, 2% of total VRAM)** as automatic safety headroom. `--reserve-vram-mib N` adds N MiB of free space for other applications. The hard measured peak limit is `total VRAM - safety headroom - N`; it includes memory already occupied by the desktop and other processes. There is no budget overshoot allowance. For example, on a 16,376 MiB GPU, the default limit is 15,864 MiB; reserving 2,048 MiB lowers it to 13,816 MiB.

```bash
./start-unleashed.sh --model "/path/to/model.gguf" --reserve-vram-mib 2048
```

```powershell
.\START-UNLEASHED.bat --model "D:\Models\model.gguf" --reserve-vram-mib 2048
```

The search starts with the highest GPU work share and an automatically sized GPU expert cache. It refits residency after changing MTP. Allocation failures restart only the trial engine with a larger reserve; unrelated format errors are not mistaken for memory shortages. Each fit has at most eight attempts. GPU work share is then measured against alternatives to retain the throughput objective: forcing every possible operation onto the GPU is not necessarily fastest. Current kernels still use CPU work and host RAM for nonresident experts. Cache residency and PCIe share are separate settings: resident experts always run on the GPU; `--pcie-frac` controls only the fraction of cache misses copied from host RAM to the GPU. The remaining misses run on the CPU. A small PCIe share therefore does not mean the GPU cache is left empty.

The reserve is a measured tuning constraint, not an OS reservation or a promise that arbitrary future workloads cannot exhaust memory. It does not automatically evict another application's memory. Retune after increasing context, changing the reserve, or changing your workload substantially. Saved configurations do not continuously resize themselves during serving. Within the chosen cache size, the upstream adaptive expert policy can still replace resident experts as routing changes.

It holds the selected cache precision fixed and evaluates machine-relative CPU worker counts (including low counts), optional MTP window lengths, PCIe work share and draft probability thresholds. The broad worker sweep runs after the GPU work split, so CPU performance differences are not hidden by an all-GPU trial. It then tests neighboring worker/share values and reduces the step when improvements stop. Imported configurations are normalized to automatic expert-cache sizing, disabled prompt reuse and disabled suffix drafting for comparable fresh-prompt timings. Candidates that crash, cannot fit or exceed the memory budget are rejected. Each search step refreshes its baseline before comparing candidates, so an old result cannot hide a current improvement after conditions change. Interleaved repeated measurements must confirm a speedup before replacing an incumbent.

Decode uses exactly **512 prompt tokens and 512 generated tokens**, on two tutorial workloads. Benchmark-only `ignore_eos` ensures equal work; ordinary API requests still honor stop tokens. Prefill chunk candidates are 256, 512, 1,024, 2,048, 4,096 and 8,192 tokens. Every trial allocates **262,144 context tokens**. Prefill is measured with **261,624 input tokens and 512 generated tokens**, leaving the engine’s eight-token internal margin, followed by another 512/512 decode check. A prefill setting is accepted only if decode retains at least **90%** of the measured decode baseline. Between engine runs, telemetry polling pauses and a five-second quiet interval allows the CUDA driver to settle. Trial records include GPU telemetry before and after each generation. Fresh-engine final measurements recheck the memory budget and decode floor before publishing `best-config.json`. If the leading prefill candidate fails that check, the tuner tries the other eligible candidates in prefill-speed order. It publishes no configuration if none meets the limits.

This is a bounded empirical search, not a proof of a global optimum. Thermal, power, workload and background-process changes can change the winner. Failed trials, generated text, token counts, timings and GPU telemetry (including PCIe link generation and width) remain in the result directory. The tuner refuses another visible GPU compute process and does not kill other applications.

## Formats and compatibility

Main-model cache modes:

| CLI `--kv` | K / V storage | Bytes per cell per QSA layer* |
|---|---|---:|
| `FP16/FP16` | FP16 / FP16 | 2048 |
| `FP16/Q8` | FP16 / 8-bit | 1568 |
| `Q8/Q8` | 8-bit / 8-bit | 1056 |
| `Q8/Q6` | 8-bit / 6-bit | 960 |
| `Q8/Q5` | 8-bit / 5-bit | 896 |
| `Q5/Q5` | 5-bit / 5-bit | 704 |
| `Q5/Q4` | 5-bit / 4-bit | 640 |
| `Q4/Q4` | 4-bit / 4-bit | 576 |

*Two KV heads, 256 values per head. Excludes indexer, page tables, graph scratch and other model memory. New mixed modes use FP16 scales per 32 values and packed symmetric codes after the existing Hadamard rotation. Q5/Q6 here name cache precision, not GGUF Q5_K/Q6_K weights. The MTP drafter retains a supported existing cache layout.

The new mixed modes currently require fully resident KV and do not support `--kv-resident` or parked conversation snapshots (`--conversation-cache-mib`). Existing uniform modes keep their upstream capabilities. No-MTP serving currently disables parked conversation snapshots and suffix drafting.

Native expert paths cover Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, IQ1_M and Q2_0, plus **NVFP4 and MXFP4**. GGUF format block sizes still constrain valid row widths; malformed or incompatible tensors are rejected. This is a defined list, not a claim that every GGUF type can serve as an expert tensor. Q2_K, Q3_K and IQ1_S expert tensors are not currently in the native expert dispatch list. Dense tensors supported by the packer can be converted separately. NVFP4 uses CUDA decoding and quantized activation arithmetic; this does not claim a hardware FP4 tensor-core speedup.

Split discovery, metadata checks and per-role shard offsets support layers divided between shards. The Q5 validation model has 33 shards and 20 layers whose gate/up/down roles cross shard boundaries. A layer's gate and up formats must match, as required by the existing expert layout. Q4/Q5/Q6/Q8 labels describe weight quantization, independently of cache precision.

PLE table rows support IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, BF16, F32 and the retained tagged FP8 representation, up to **640 bytes per row**. Direct I/O and mapped reads use 64-bit file offsets. The new alignment check accepts legal final-tensor padding while rejecting truncation.

## Validation and releases

See `releases/` for the packaged artifacts and their manifests. The Linux engine is tested on the local RTX 5000 Ada 16 GB GPU with the SSD Q5 model. The Windows archive is a source/build release: Windows execution is not available in this environment, so it is not labeled as a tested Windows binary.

Numerical checks compare CUDA dequantization/experts against ggml, and mixed-cache storage/attention against CPU references. Real Q5 layer checks include shard boundaries. The large-table fixture exercises a 204.8 GB logical F32 PLE table with sparse disk allocation.

Source and runtime licenses are preserved. Model weights and an MTP head are not bundled.

## Tuning command reference

The normal launcher tunes automatically. Advanced use with an existing engine JSON:

```bash
python tools/unleashed_tune.py work/q5-config.json --out work/tuning-custom --reserve-vram-mib 2048
```

- `config`: engine JSON containing `exe`, `args` and tokenizer information; optional `tune_mtp` enables draft-head candidates.
- `--out`: result directory inside the project; previous measurements are archived on rerun.
- `--reserve-vram-mib`: additional free space for other apps, nonnegative integer MiB.
- `--target-vram .8`: optional additional cap at 80% of total memory, for reproducing the earlier policy; the stricter cap wins.
- `--repeats 2`: measurements per candidate; interleaved and final checks add further repetitions.
- `--smoke`: one diagnostic 512/512 run, not a completed tuning decision.

`memory-budget.json` records the computed budget; `measurements.json` contains each trial and generated response; `telemetry.json` records sampled memory, power, temperature and clocks. `RESULTS.json` and `best-config.json` are published only after final validation. The engine's lower-level `--vram-reserve-mib` is an internal allocation allowance adjusted by the tuner, **not** the launcher's `--reserve-vram-mib` guarantee of measured free headroom. Other engine settings are documented in [docs/DETAILS.md](docs/DETAILS.md).

## Validation commands

Build the parity tools with `BUILD-UNLEASHED.sh` or `BUILD-UNLEASHED.ps1`, then run these with the launcher's local Python environment. Do not run CUDA checks concurrently with throughput tuning.

```bash
.venv/bin/python -B tools/unleashed_test.py --model "/path/to/model-00001-of-00033.gguf"
.venv/bin/python -B -m unittest tools.test_unleashed tools.test_unleashed_ple
```

On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`. The large sparse-file addressing test is Linux-only; ordinary PLE format/padding tests are portable. The numerical driver without `--model` runs synthetic expert/cache checks only. Logs and return codes are saved under `work/`.

For direct engine use, omitting `--mtp` serves without a draft head; `--no-mtp` also clears a preceding `--mtp` argument. Missing MTP no longer prevents the server from starting. The launcher omits MTP unless an optional runtime is supplied and measured to improve performance.

## System RAM headroom

`--reserve-ram-mib N` reserves additional available system RAM in the model-fit estimate and tuning acceptance checks. Interactive setup asks for it alongside the existing VRAM reserve. The launcher rechecks current RAM before preparation and cached-profile reuse. Both reservations enter the cache fingerprint and selection receipt. This does not lock RAM or impose an operating-system cap during later inference.

KV guidance shown in setup: **FP16/FP16 is ideal; FP16/Q8 is recommended; Q8/Q5 is a last resort only when no other viable option fits.** Card-capacity defaults remain unchanged.
