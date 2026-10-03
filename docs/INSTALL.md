# Install Strata Unleashed

Pick a model and setup handles the download, preparation and tuning. You can chat when it finishes.

## What you need

- Windows 10/11 or Linux on an x86-64 CPU with AVX2.
- One NVIDIA CUDA GPU, RTX 20 series or newer. The picker checks available VRAM, not just the card's advertised capacity.
- Enough free RAM for the selected model. This workflow keeps its experts resident; smaller choices generally need substantially less RAM than Q8. Use the scan rather than assuming every quant fits a given RAM size.
- Space for the model, converted pack and working files. Downloads are roughly 54 GiB to more than 300 GiB across the catalog; unavailable formats remain visible with an explanation.
- An SSD, preferably NVMe. Drive type and bounded uncached read speed are checked. A non-NVMe or unknown drive produces an advisory notice, not a blanket rejection.
- A current NVIDIA driver and Python 3.10+. Windows setup can install Python 3.12 through WinGet; on Ubuntu/Debian use `sudo apt install python3 python3-venv curl` if these are missing.

### Source builds

Windows currently needs an **x64 Visual Studio C++ developer terminal**, CUDA Toolkit, CMake and Ninja. Install Visual Studio Build Tools with the C++ desktop workload, then open its developer PowerShell and run setup. Install these first. The setup command does not install them or change your drivers.

Linux includes the validated Ada/sm89 binary, requiring glibc 2.43, GLIBCXX_3.4.32, AVX2 and CUDA 13 runtime/cuBLAS libraries. If it cannot run, or you have a different GPU architecture, setup builds locally when the required tools are available. Install a C++20 compiler, CUDA Toolkit, CMake and Ninja first. `./BUILD-UNLEASHED.sh` also builds explicitly.

## One command

Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install.ps1 | iex
```

Linux:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install.sh)"
```

The installer gets the latest release, verifies SHA256SUMS and extracts it into Documents/StrataUnleashed. Python packages live in its `.venv`. It refuses to overwrite an unrelated nonempty folder; an existing Unleashed installation is reused without changing your source or model files.

To choose a different destination, set `STRATA_INSTALL_DIR` before running the command. If you cloned the repo, use `git pull` to update it.

## First setup

1. Read the hardware and drive scan. A speed result is a short sequential read sample, not a guarantee of inference performance.
2. Choose general/compact, coding, larger, or uncensored use.
3. Set optional VRAM and system RAM headroom for other applications, separately in MiB (1024 MiB = 1 GiB). Context stays at native 262,144 tokens; KV precision defaults by card capacity (see the README).
4. Choose a model/quant. Setup explains unavailable choices and displays the download size.
5. Choose **Download, set up and chat** for terminal inference testing, **open the API** for browser/API use, **set up only**, or **download only**.
6. Wait for the download, setup and tuning. Large models take a while to load.

The final terminal chat uses no system message. `/new` clears the conversation, `/tokens 512` changes the reply limit, and `/quit` closes the engine. Chat does not save your conversation to a transcript by default.

## Start again

From the installation folder:

```text
Windows: CLI-UNLEASHED.bat
Linux:   ./cli-unleashed.sh
```

Select the same model to reuse its files and matching tuning result. Completed files are verified; partial downloads resume. For existing weights, avoid a duplicate download:

```bash
./cli-unleashed.sh --local-model "/path/to/model-00001-of-00033.gguf" --chat --yes
```

Use any shard to identify the family; all shards must be together. The weights remain read-only. The pack and tuning results go inside the app folder.

For an already prepared engine configuration:

```bash
./start-unleashed.sh --config work/my-config.json --chat
```

Omit `--chat` to run the local web/API server. If its port is occupied, use `--port 8101`. Terminal test chat does not need an HTTP port.

## Where files go

| Folder | Contents |
| --- | --- |
| `models/` | Verified downloaded model shards |
| `work/packs/` | Prepared model data and tokenizer |
| `work/autotune/` | Measurements and selected configurations |
| `work/installer/` | Release download and checksum metadata |
| `work/storage-check.json` | Most recent selected-model drive/read-speed check |
| `.venv/` | Python packages |
| `build/` | Inference engine |

The model weights, local configuration, caches and credentials are excluded from Git. The app does not upload prompts to a hosted model. Initial installation and downloads require network access; local inference does not.

[Model choices](MODELS.md) · [Every CLI flag](UNLEASHED-CLI.md) · [Troubleshooting](TROUBLESHOOTING.md)

Full-context tuning processes a near-262K prompt for each prefill candidate and can take a long time. Progress reports elapsed time and GPU memory every 30 seconds during generation. Keep the terminal open; subsequent starts reuse a matching validated profile.

For an existing Git checkout, use `git pull --ff-only` on the `unleashed` branch, then rerun the CLI. The one-command installer reuses an existing installation; it does not overwrite it as an updater. A release archive can instead be extracted into a fresh directory. Changes to native-context policy invalidate old tuning profiles.
