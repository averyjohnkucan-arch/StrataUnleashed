# Strata Unleashed

**Download a model and run it on your PC.**

Windows or Linux · NVIDIA graphics card · local chat and API · free and open source

A fork of [Strata](https://github.com/Niko1221/Strata) with model selection, downloads and auto-tuning. Pick a model, let it download and set itself up, then chat in the terminal. Setup checks your hardware first.

[Download](https://github.com/averyjohnkucan-arch/StrataUnleashed/releases/latest) · [Models](docs/MODELS.md) · [Installation](docs/INSTALL.md) · [Help](docs/TROUBLESHOOTING.md) · [How it works](docs/HOW_IT_WORKS.md)

## Download, install and set up a model

**Windows — paste into PowerShell:**

```powershell
irm https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install.ps1 | iex
```

**Linux — paste into a terminal:**

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install.sh)"
```

This installs into **Documents/StrataUnleashed** and opens setup. Pick your model and follow the prompts. It downloads the files, checks them, tunes the settings and opens chat. Downloads are large, so check the size first. If a download stops, you can resume it.

**Before you start:** install a current NVIDIA driver. Linux needs Python 3.10+ with `venv` and `curl`; Windows setup can install Python through WinGet. A source build also needs CUDA Toolkit, CMake, Ninja and a C++ compiler. Windows currently uses a source build; run from an x64 Visual Studio developer PowerShell. The installer does not install GPU drivers or the compiler/CUDA toolchain. [Requirements and build help](docs/INSTALL.md#what-you-need).

Prefer to download a ZIP or tar file yourself? [Get the latest release](https://github.com/averyjohnkucan-arch/StrataUnleashed/releases/latest), extract it, then run **`CLI-UNLEASHED.bat`** on Windows or **`./cli-unleashed.sh`** on Linux.

## Models

| Model | Notes |
| --- | --- |
| ISTA Flash Next Coder | Smallest option. Coding focused. |
| ISTA Flash Next | Small general models. |
| Atomic Chat | The next step up. Main pick for larger models. |
| Unsloth | Larger quants if you want them. Not the first recommendation; don't expect a big jump just from a larger quant. |
| HuiHui | For uncensored chat. Abliterated models. |

There are **22 model/quant options**. Setup shows what fits and what it can't run. Three options are listed but currently unsupported. [Model list](docs/MODELS.md).

Setup checks your drive and read speed too. **NVMe is recommended.** Other drives get an “experience may vary” warning. Slow storage can mean slower loading and inference.

## Chat

Choose **Download, set up and chat**. The terminal opens a simple test conversation:

```text
you> Explain this Python function.
model> ...

/new       Start a fresh conversation
/tokens N  Set the maximum reply length
/quit      Close the model and leave
```

Test chat has **no system prompt**. Just your messages and the model's replies, using its normal chat format. It shows token counts and speed after each reply.

For browser chat or another app, choose **Download, set up and open the API**. Leave the terminal open while you use it.

## Leave room for other apps

Setup asks how much extra graphics memory to leave free. For example:

```bash
./cli-unleashed.sh --reserve-vram-mib 2048
```

This leaves an additional 2 GiB, on top of automatic safety headroom. Tuning first optimizes short-prompt generation, then prompt processing while keeping at least 90% of the measured generation speed. Later starts reuse a matching tuning result.

## Testing

On the local RTX 5000 Ada 16 GB laptop GPU, the Atomic Q5 model reached **51.64 generated tokens/s** on a 512-token prompt/reply and **1,644 prompt tokens/s** on an 8,192-token prompt. Those measurements are hardware-specific. [Results and limitations](RELEASE-NOTES.md).

Linux execution is tested. Windows launch/build scripts and simulated platform checks are included, but Windows execution has not been verified. The self-tuner currently supports **one NVIDIA GPU**. The bundled Linux engine targets Ada and recent Linux libraries; other machines may need a source build.

## Docs

- [Installation and first chat](docs/INSTALL.md)
- [All setup options](docs/UNLEASHED-CLI.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Documentation index](docs/README.md)
- [Technical note and measurements](docs/paper/Strata-Unleashed.md) · [PDF](docs/paper/Strata-Unleashed.pdf)
- [Contributing](CONTRIBUTING.md)

Based on **Strata 0.1.38**, with credits to its developers, llama.cpp/ggml, Qwen and the model publishers. [MIT license](LICENSE); model files retain their publishers' licenses. The original [Strata paper](docs/paper/Strata-Paper.pdf) remains available and is separate from this fork's technical note.
