# Help with Strata Unleashed

## No models are available

Read the reason beside each choice. Check **available** RAM, not just installed RAM; close other large applications. Lower context or extra VRAM reservation if those are the constraint. A missing compiler or CUDA Toolkit is a setup issue, not a model-size issue. Unsupported expert encodings cannot be fixed by adding RAM.

The guided tuner supports one NVIDIA GPU. AMD, CPU-only and multi-GPU configurations described in some upstream references are not available through this workflow.

## “Not identified as NVMe” or a slow-drive warning

Your experience may vary. SATA/USB SSDs, hard drives and network or unidentified storage can load weights more slowly and affect disk-backed PLE reads. Move the model's entire shard family to an NVMe drive and use `--local-model` with its new path.

The speed check reads up to 32 MiB from an existing file, bypassing the OS cache when supported. It writes no model data or disk-test files. Results vary with load and controller caching, and do not measure random-I/O inference throughput. When uncached reads are unavailable, setup says so rather than presenting cached RAM speed as SSD speed. A sample below 500 MiB/s produces an advisory notice even on NVMe.

## Windows cannot find Python, CMake, Ninja or the compiler

The installer can install Python with WinGet. Source builds require Visual Studio C++ Build Tools, CUDA Toolkit, CMake and Ninja. Open an **x64 developer PowerShell**, then run `CLI-UNLEASHED.bat`. Ordinary PowerShell may not have the compiler environment loaded.

Windows is distributed as source/build scripts. Native Windows execution has not yet been validated by this project.

## Linux reports missing libraries or an unsupported GPU architecture

The bundled engine targets Ada/sm89 and a recent Linux runtime. Rebuild with `./BUILD-UNLEASHED.sh` after installing the compiler/CUDA prerequisites, or use `--build`. The wizard checks whether the existing binary can execute before treating it as ready.

## Download interrupted, checksum failed, or a shard is missing

Choose the same model again to resume `.part` files. A checksum mismatch never becomes a completed model file. Corrupt partials get `.bad-*` names; remove them manually if you need the disk space. An invalid already completed file is left untouched; move it aside and retry.

Keep every shard in the same directory with its original filename. Do not mix revisions. `--local-model` reads the weights in place without copying them.

## First setup takes a long time

A model may be over 100 GB. Download, checksum verification, packing and measured tuning are separate stages. During engine loading, the PC can become less responsive as tens of gigabytes enter RAM. Tuning tests multiple configurations; later starts reuse a matching result. The logs and measurements are under `work/`.

## Another app needs VRAM

Use `--reserve-vram-mib 2048` to leave an extra 2 GiB. This is additional to automatic safety headroom. Changing the reserve creates a separate tuning result. Another application can still allocate memory later; no initial scan can prevent that.

## Port already in use

Terminal test chat needs no HTTP port. For browser/API mode choose a free one, for example `--port 8101`. Do not stop an unrelated server just to free the default port.

## Test chat says the template adds a system message

Test chat intentionally refuses hidden system turns. It uses the model's template with thinking instructions off, checks the result, and passes only user/assistant messages. Do not “fix” this by silently inserting a different system prompt. Report the model ID and template revision.

## The conversation is too long

Type `/new` or lower the response limit with `/tokens 512`. Model context includes both your messages and generated replies. `/quit` shuts down the test-chat engine cleanly.

## Reporting a problem

Include your OS, GPU/VRAM, RAM, model ID or shard names, the command used, and the relevant error. `./cli-unleashed.sh --scan --json` produces a useful hardware/storage report. Review paths and other personal details before sharing logs. Do not post access tokens or private prompts.

[Open an issue](https://github.com/averyjohnkucan-arch/StrataUnleashed/issues) · [Installation](INSTALL.md)
