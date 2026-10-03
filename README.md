# Strata Unleashed

A Windows/Linux fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) for the Qwen3.8-Flash-Next (`qwen4exp`) model family.

- Retains the local Q6, mixed-shard, PLE and server-lifecycle fixes.
- Adds NVFP4/MXFP4 native expert support and broader Q4/Q5/Q6/Q8 paths.
- Serves without an MTP head; optionally measures speculative decoding when one is supplied.
- Supports FP16/FP16, FP16/Q8, Q8/Q8, Q8/Q5, Q5/Q5, Q5/Q4 and Q4/Q4 caches.
- Self-tunes for maximum practical GPU residency with configurable VRAM reserved for other apps, maximizes measured 512/512 decode throughput, then chooses prefill settings within a 10% decode-speed loss limit.

Linux:

```bash
./start-unleashed.sh --model "/path/to/model.gguf"
```

Windows, from an x64 Visual Studio developer terminal for the first source build:

```powershell
.\START-UNLEASHED.bat --model "D:\Models\model.gguf"
```

See **[UNLEASHED.md](UNLEASHED.md)** for setup, self-tuning, compatibility and current limitations. **[RELEASE-NOTES.md](RELEASE-NOTES.md)** records validation and release status. The original project documentation remains in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

Model weights are not included. Windows is supplied as a source/build release; Linux is validated locally. The new self-tuner currently targets one NVIDIA GPU.
