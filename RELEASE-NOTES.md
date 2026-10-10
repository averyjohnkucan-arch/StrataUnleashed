# Strata Unleashed 0.1.41-r1

Merged upstream Strata 0.1.41 (fb58e0d, October 8, 2026), retaining Unleashed's model picker, downloads, tuning policy, extended native weight formats, independent K/V precision, optional MTP, and PLE format support.

The engine now includes upstream Intel AVX-VNNI and per-core gathered expert decoding, interleaved CUDA verify projections, short-prompt CPU/GPU expert sharing, and the updated expert I/O and prefill planner. Existing tuning results should be measured again with this engine. No whole-engine speedup is promised from kernel benchmarks.

The installed media extension is retained: SVE1 image and SVE2 video embedding records use their respective image/video padding tokens, with context bounds and media-kind checks. Existing image/video encoders remain compatible.

Merge integration keeps mixed KV on its resident layout and per-token append path, prevents it from entering elastic KV or ordinary block movers, extends upstream's unified PLE registry with Q4_1/F16/F32, preserves legal GGUF padding, and retains failed-start child cleanup in the server. CUDA MMQ exports its cuBLAS/driver link dependencies for standalone tests.

Experimental pinned stage buffers remain off by default, following upstream's IQ3_S corruption finding. Model precision, context capacity, and routing are not reduced by this revision.

Validation on Linux/Ada: 107 Python regression tests passed; the native suite passed 97 cases and skipped its AVX-512-only expert test. Its memory-lock test passed separately with the required limit raised. Its artifact-dependent PLE block test could not run because its captured model fixtures are not installed. Five mixed-KV storage/attention checks and additional NVFP4/MXFP4/Q4_1/Q6_K synthetic expert checks passed. Installed-model deployment results are reported separately; Windows execution is not verified.

# Strata Unleashed 0.1.38-r6

- Reservation prompts accept explicit MiB/GiB and decimal MB/GB units and echo the interpreted free-memory amount. Bare 4096/8192 remain 4/8 GiB. Inputs such as `4096mb` and `8192mb` are now accepted as decimal MB, rounded up to whole MiB.
- Removed model-size RAM/VRAM admission checks, memory-based unavailable labels and memory-based strikethroughs. Memory columns remain informational. The engine attempts the selected supported model.
- Removed the tuner's estimated 1 GiB startup-space rejection. Actual allocation failures and measured user-selected reservations still apply; reservations mean memory to leave free, not memory to allocate to the model.
- 66 regression tests cover exact inputs, independent RAM/VRAM forwarding, low-memory selection, units and existing tuning reservations. Native inference code is unchanged; r5 full-context Q5 measurements still apply.
- Linux includes the validated engine and source-build fallback. Windows is a source/build package; native Windows execution remains unvalidated.

## Previous revision

# Strata Unleashed 0.1.38-r5

- Corrected memory accounting from each pinned model's shard headers: SSD engram table, host expert/embedding memory and GPU dense/KV memory. All picker estimates use FP16/Q8 at 262,144 context; runtime capacity defaults remain unchanged.
- Added explicit Engram table, CPU MEM and GPU MEM columns, largest CPU footprint first. Unsupported BF16/IQ1_S expert variants are hidden; 19 supported choices remain.
- Over-budget choices are struck through in supporting terminals and remain selectable. `--allow-over-budget` overrides estimated admission checks; actual allocation errors and measured tuning reservation checks still apply.
- Atomic Q5 CPU planning corrected from 79.3 to 53.8 GiB. Real tests with preallocated native context measured 50.0 GiB peak engine RSS at 512/512 FP16/FP16 and 50.07 GiB at 261,624/512 FP16/Q8. [Per-model chart and evidence](docs/MEMORY.md).
- 65 regression tests passed, including stale-catalog replacement, memory placement, fixed FP16/Q8 estimates, sorted supported choices and over-budget selection.
- The native engine is unchanged. Linux includes the validated engine with source-build fallback; Windows remains a source/build release without native Windows execution validation.

## Previous revision

# Strata Unleashed 0.1.38-r4

- Interactive setup asks separately for extra VRAM and system RAM headroom. `--reserve-ram-mib` is available in the Windows/Linux CLI and launcher.
- System RAM reservation reduces model-fit budgets, is rechecked against current available RAM before launch, and is included in tuning-cache identity. Tuning trials below the sampled RAM reserve are rejected. This does not impose an OS memory cap during subsequent inference.
- Setup explains that FP16/FP16 is ideal, FP16/Q8 is recommended, and Q8/Q5 is a last resort when no other viable option fits. Capacity-based defaults and native 262,144-token context remain unchanged.
- 62 regression tests passed, including interactive prompts, argument forwarding, RAM-budget rejection and current-memory launch checks. The inference engine is unchanged from r3; its recorded native-context GPU validation still applies to that same binary. Windows remains a source/build package without native Windows execution validation.

## Previous revision

# Strata Unleashed 0.1.38-r3

Based on upstream Strata 0.1.38. This revision adds capacity-based KV defaults and native-context tuning to the installation/chat improvements below.

- Default K/V cache precision: 16 GiB+ FP16/FP16; 12–16 GiB FP16/Q8; 8–12 GiB Q8/Q8; below 8 GiB Q8/Q6. Totals within 64 MiB below a tier account for driver overhead. Weight quantization is independent.
- Added Q8/Q6 mixed KV parsing and numerical parity coverage. The tuner holds the chosen KV pair fixed; `--kv` is an explicit override.
- Every tuning trial allocates 262,144 context tokens. Short decode remains 512/512; prefill candidates and final validation use 261,624/512. A failed full-context trial cannot produce an accepted tuning profile.
- Imported configurations and cache fingerprints enforce the new policy. No automatic context reduction, KV precision reduction or RoPE extension occurs.
- Fit estimates count the supported architecture's 12 full-attention layers and the selected KV precision. Small cards can still be rejected when the selected model does not fit.

58 CPU regression tests and CUDA mixed-KV storage/attention parity passed, including Q8/Q6. Q8/Q6 also passed real Q5 short inference and zero-system-prompt chat with 262,144-token allocation; its near-full-context prompt was not separately tested. Q5 FP16/FP16 passed 512/512 and 261,624/512 generation at native context without MTP. These are single validation samples, not a completed tuning sweep. Native-context GPU validation is recorded in [the native-context validation record](docs/UNLEASHED-NATIVE-VALIDATION.json); earlier performance results below are historical and do not validate this policy. Windows remains a source/build release without native Windows execution testing.

## Installation and chat improvements (0.1.38-r2)


This packaging revision keeps the Strata 0.1.38 engine and adds a simpler install-to-chat experience.

- **Model selection and download are the starting point.** The README provides one command each for Windows and Linux; installers verify the release archive and start guided setup.
- **Terminal inference test chat has no system prompt.** Only user/assistant messages enter the model template; hidden template system turns are rejected. Generation statistics count token IDs, not streamed chunks.
- **Storage checks run on the user's machine.** Drive type and bounded uncached read speed are measured read-only. Non-NVMe/unknown storage and slow reads produce advisory warnings.
- Hardware, download, storage, catalog and chat code now have separate responsibilities. Binary compatibility comes from a release manifest or local build receipt, not an assumed development GPU.
- Installation, model selection, troubleshooting and contributor docs were rewritten around the fork. An engineering note is provided as Markdown and PDF; original upstream papers/references remain attributed.

The engine and procedural tuning search are unchanged. Earlier Q5 throughput results below remain historical measurements of their recorded binary/hardware, not a guarantee on arbitrary machines. The installers do not install GPU drivers or compiler/CUDA toolchains. Windows remains a source/build release without native Windows validation.

Validation: 53 regression tests passed, including 24 simulated machine profiles. Real Q5 terminal chat returned the requested answer with no rendered system turn. Read-only storage checks and document links passed. See [the validation record](docs/UNLEASHED-EXPERIENCE-VALIDATION.json).

## Previous release: 0.1.38

## Interactive setup release

Version 0.1.38 matches the current upstream engine release. New Linux `cli-unleashed.sh` and Windows `CLI-UNLEASHED.bat` entry points scan hardware, recommend models by use case, explain fit/format limits, verify resumable downloads and start procedural tuning. The ordinary launchers open the wizard when called without arguments.

The pinned catalog includes 22 variants: four ISTA general, one ISTA Coder, three Atomic Chat, three HuiHui and eleven Unsloth. All shard headers were checked against the engine contract; 19 passed and three unsupported expert encodings are explicitly blocked. Full inference testing of every catalog model is not claimed.

ISTA is preferred for compact/coding use, Atomic Chat for larger choices, and HuiHui for uncensored use. Recommendations account for installed and available RAM, disk, GPU, reserved VRAM, context and build tools. See [docs/UNLEASHED-CLI.md](docs/UNLEASHED-CLI.md) for estimates, exact variants and every flag. Windows hardware/tool behavior is covered by simulated unit tests; Windows execution remains unvalidated.

This update passed 39 CLI/launcher/tuner regression tests, a real resumed Hugging Face download with SHA256 verification, a real Linux system scan, and read-only inspection of all 33 local Q5 shards. All 170 catalog shard headers were inspected. Details: [docs/UNLEASHED-CLI-VALIDATION.json](docs/UNLEASHED-CLI-VALIDATION.json).

The inference engine and self-tuning search are unchanged from the first Unleashed release; the Q5 measurements below carry forward for that same engine.

Based on upstream Strata `99f3dbd`. Local SSD changes from `Strata`, `Strata-q6` and `Strata-atomic` were integrated into the newer upstream source: Q6 expert/prefill support, shard metadata tolerance, Q8/Q5 PLE tables, MXFP4, native PLE keys and the on-demand server alias. The newer upstream lifecycle behavior is retained, with cleanup of failed or interrupted engine starts.

Unleashed adds NVFP4, broader native Q4/Q5 dispatch, independent K/V precision, optional MTP serving, larger PLE rows, measured self-tuning and local Linux/Windows launch/build scripts. See [UNLEASHED.md](UNLEASHED.md) for the exact format list, arguments and limitations.

## Available validation

Local hardware: RTX 5000 Ada, 16,376 MiB VRAM, CUDA Toolkit 13.1, NVIDIA driver 595.91.07; 32 logical CPU threads and 128 GB RAM. Linux build uses the AVX2 CPU baseline and CUDA architecture 89, built on Ubuntu 26.04.1. Its ELF dependencies require glibc 2.43 and libstdc++ providing GLIBCXX_3.4.32. Rebuild from source on older Linux distributions. Other GPUs need a source rebuild. CUDA runtime and cuBLAS libraries are required and are not bundled.

The SSD Q5 model is `Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64`, 33 GGUF shards. Its actual tensor encodings are mixed; the Q5 family label is not a promise that every tensor is Q5. Weights remain read-only.

- Synthetic CUDA expert/dequantization checks passed for Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, NVFP4 and MXFP4 combinations. Larger FF=768 K-quant checks passed.
- Mixed K/V storage and attention checks passed for FP16/Q8, Q8/Q5, Q5/Q5 and Q5/Q4; attention maximum absolute error was below 0.0001 against the CPU reference.
- Real Q5 expert checks passed for layers 0, 2, 11, 20, 41 and 47, covering split role locations. Real Q8 PLE direct reads matched ggml exactly.
- All seven requested KV pairs completed 512-input/512-output Q5 benchmark trials during the earlier 80% policy sweep. These were intermediate trials, not a final selected profile.
- PLE regression tests passed across nine formats in mapped and direct modes, including legal padding and truncated-file rejection. The sparse 204.8 GB F32 fixture uses distinct nonzero rows beyond 32-bit offsets.
- 23 packer tests, 15 server lifecycle/monitor tests, 16 launcher/tuner tests and 2 PLE regression tests passed.

## Release status

Final Q5 medians from three measurements per workload:

| Workload | Decode tokens/s | Prefill tokens/s | Peak VRAM |
|---|---:|---:|---:|
| 512 input / 512 output | 51.64 | 230.13 | 15844 MiB |
| 8,192 input / 512 output | 52.00 | 1644.02 | 15844 MiB |

Selected: `FP16/FP16`, 23 CPU workers, MTP window 3, draft threshold 0.7, PCIe cache-miss share 0.1, prefill chunk 8192. The engine allocation reserve is 456 MiB; the user's extra reserve is 0 MiB, with a separate 512 MiB safety margin.

The final decode reference was 53.37 tokens/s and the acceptance floor was 48.03. The selected prefill setting changed short-workload decode by -3.24%. Observed peak usage was 96.75% of VRAM, within the measured limit.

The explicit `--reserve-vram-mib 2048` check also passed: peak 13794 MiB, limit 13816 MiB. This validates the reserve behavior; it is not a separate complete optimization sweep for that reserve.

Both ordinary and streaming HTTP completions passed with the selected configuration and with the MTP head absent. The launcher reused the validated profile without retuning. 56 CPU unit tests and all recorded CUDA/reference checks passed.

Measurements, flags and validation details are included in [docs/UNLEASHED-VALIDATION.json](docs/UNLEASHED-VALIDATION.json). Earlier stages and generated responses remain in the local `work/` directory. PCIe link speed varied during development; the tuner refreshes baseline measurements and checks the final configuration on fresh engine starts. These are local workload measurements, not a guarantee of a global optimum or identical speeds on another machine.

The release archives contain source and dependency licenses; the Linux archive also includes the validated local binary. SHA256SUMS and manifests identify their contents. Model weights and machine-specific configurations are excluded.

Windows is a source/build package. This Linux environment has not compiled or executed a Windows binary. The Windows scripts require Python, an x64 Visual Studio C++ developer terminal, CMake, Ninja and CUDA Toolkit. The source package includes the ggml/gguf dependencies and licenses.
