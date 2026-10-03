# Strata Unleashed

**Choose a model, download it, and run it on your own PC.**

Windows or Linux · NVIDIA graphics card · local chat and API · free and open source

Strata Unleashed is a fork of [Strata](https://github.com/Niko1221/Strata). It adds a model picker, verified downloads and automatic tuning: setup checks your PC, helps you choose a model that fits, downloads every file, and finds settings for your hardware. Then you can chat in the same terminal.

[Download](https://github.com/averyjohnkucan-arch/StrataUnleashed/releases/latest) · [Models](docs/MODELS.md) · [Installation](docs/INSTALL.md) · [Help](docs/TROUBLESHOOTING.md) · [How it works](docs/HOW_IT_WORKS.md)

## Download, install, and set up a model with one command

**Windows — paste into PowerShell:**

```powershell
irm https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install.ps1 | iex
```

**Linux — paste into a terminal:**

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install.sh)"
```

The command downloads the app into **Documents/StrataUnleashed**, checks the release checksum, installs its Python packages, and opens setup. Choose a model and press Enter for the recommended settings. Setup downloads the weights, prepares the model, tunes it, and opens terminal chat. Model downloads can be tens or hundreds of gigabytes; setup shows the size before you start. Interrupted downloads resume.

**Before you start:** install a current NVIDIA driver. Linux needs Python 3.10+ with `venv` and `curl`; Windows setup can install Python through WinGet. A source build also needs CUDA Toolkit, CMake, Ninja and a C++ compiler. Windows currently uses a source build; run from an x64 Visual Studio developer PowerShell. The installer does not install GPU drivers or the compiler/CUDA toolchain. [Requirements and build help](docs/INSTALL.md#what-you-need).

Prefer to download a ZIP or tar file yourself? [Get the latest release](https://github.com/averyjohnkucan-arch/StrataUnleashed/releases/latest), extract it, then run **`CLI-UNLEASHED.bat`** on Windows or **`./cli-unleashed.sh`** on Linux.

## Pick the model you want

| You want… | Setup prefers… |
| --- | --- |
| A smaller general model | ISTA Flash Next |
| A model for coding | ISTA Flash Next Coder |
| A larger model | Atomic Chat |
| An uncensored/abliterated model | HuiHui |
| An alternative quantization | Unsloth is also available |

The catalog includes **22 model/quant choices**. Setup explains which ones fit its resource estimates and which cannot run in this engine. It checks actual tensor types, not just the quant name. Three choices have unsupported expert encodings and are shown as unavailable. [Exact model list and limits](docs/MODELS.md).

Setup also checks your drive and samples read speed without changing your model files. **NVMe is recommended.** SATA, USB, hard disks and unidentified storage show an “experience may vary” notice; slow reads can affect loading and disk-backed inference.

## Talk to it, right there

Choose **Download, set up and chat**. The terminal opens a simple test conversation:

```text
you> Explain this Python function.
model> ...

/new       Start a fresh conversation
/tokens N  Set the maximum reply length
/quit      Close the model and leave
```

Test chat has **no system prompt**: only your messages and the model's replies. It disables template-added thinking instructions and checks that the rendered prompt contains no system turn. It still uses the model's normal chat format. Token counts and generation speed appear after each reply.

Prefer a browser or another app? Choose **Download, set up and open the API**. The local web chat and OpenAI-compatible API stay available while that terminal is open.

## Leave room for other apps

Setup asks how much extra graphics memory to leave free. For example:

```bash
./cli-unleashed.sh --reserve-vram-mib 2048
```

This leaves an additional 2 GiB, on top of automatic safety headroom. Tuning first optimizes short-prompt generation, then prompt processing while keeping at least 90% of the measured generation speed. Later starts reuse a matching tuning result.

## What has been tested?

On the local RTX 5000 Ada 16 GB laptop GPU, the Atomic Q5 model reached **51.64 generated tokens/s** on a 512-token prompt/reply and **1,644 prompt tokens/s** on an 8,192-token prompt. Those measurements are hardware-specific. [Results and limitations](RELEASE-NOTES.md).

Linux execution is tested. Windows launch/build scripts and simulated platform checks are included, but Windows execution has not been verified. The self-tuner currently supports **one NVIDIA GPU**. The bundled Linux engine targets Ada and recent Linux libraries; other machines may need a source build.

## Read more

- [Installation and first chat](docs/INSTALL.md)
- [All setup options](docs/UNLEASHED-CLI.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Documentation index](docs/README.md)
- [Technical note and measurements](docs/paper/Strata-Unleashed.md) · [PDF](docs/paper/Strata-Unleashed.pdf)
- [Contributing](CONTRIBUTING.md)

Based on **Strata 0.1.38**, with credits to its developers, llama.cpp/ggml, Qwen and the model publishers. [MIT license](LICENSE); model files retain their publishers' licenses. The original [Strata paper](docs/paper/Strata-Paper.pdf) remains available and is separate from this fork's technical note.
