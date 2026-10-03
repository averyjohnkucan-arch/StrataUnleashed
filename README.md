# Strata Unleashed

A Windows/Linux fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) for the Qwen3.8-Flash-Next (`qwen4exp`) model family.

- Retains the local Q6, mixed-shard, PLE and server-lifecycle fixes.
- Adds NVFP4/MXFP4 native expert support and broader Q4/Q5/Q6/Q8 paths.
- Serves without an MTP head; optionally measures speculative decoding when one is supplied.
- Supports FP16/FP16, FP16/Q8, Q8/Q8, Q8/Q5, Q5/Q5, Q5/Q4 and Q4/Q4 caches.
- Self-tunes for maximum practical GPU residency with configurable VRAM reserved for other apps, maximizes measured 512/512 decode throughput, then chooses prefill settings within a 10% decode-speed loss limit.

Interactive setup (hardware scan, model/quant choice, download and self-tuning):

Linux:

```bash
./cli-unleashed.sh
```

Windows, from an x64 Visual Studio developer terminal for the first source build:

```powershell
.\CLI-UNLEASHED.bat
```

The catalog includes ISTA Flash Next and Coder, all published Atomic Chat Flash Next quants, HuiHui abliterated, and Unsloth. It recommends ISTA for compact models, Atomic for larger models, and HuiHui for uncensored use, while explaining unsupported formats and resource limits. See **[the CLI guide](docs/UNLEASHED-CLI.md)** for every option.

[Download release **0.1.38**](https://github.com/averyjohnkucan-arch/StrataUnleashed/releases/tag/unleashed-v0.1.38), matching upstream Strata.

See **[UNLEASHED.md](UNLEASHED.md)** for setup, self-tuning, compatibility and current limitations. **[RELEASE-NOTES.md](RELEASE-NOTES.md)** records validation and release status. The original project documentation remains in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

Model weights are not included. Windows is supplied as a source/build release; Linux is validated locally. The new self-tuner currently targets one NVIDIA GPU.
