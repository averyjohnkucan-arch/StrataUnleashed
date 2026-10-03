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

## Native context and KV defaults

Every tuning trial and inference session uses **262,144 tokens (256 Ki tokens)** of context capacity. Short generation tests use 512 input and 512 output tokens; long prefill validation fills the context with 261,624 input and 512 output tokens, leaving the engine’s required eight-token safety margin.

| GPU capacity | Default key / value cache |
| --- | --- |
| 16 GiB or more | FP16 / FP16 |
| 12–16 GiB | FP16 / Q8 |
| 8–12 GiB | Q8 / Q8 |
| Below 8 GiB | Q8 / Q6 |

**KV guidance:** FP16/FP16 is ideal; FP16/Q8 is recommended. Q8/Q5 is a last resort, only when no other viable option fits. These are setup recommendations, not a claim of measured quality for every model.

These are **KV cache** settings, separate from model weight quantization. The tuner keeps the selected precision fixed; `--kv Q8/Q8`, for example, explicitly overrides it. Capacity tiers allow 64 MiB for driver-reported overhead. A default does not guarantee that a model fits: setup reports insufficient resources and never silently reduces context or precision.

## Leave room for other apps

Setup asks how much extra **VRAM and system RAM** to leave available for other applications. Enter each amount in MiB (1024 MiB = 1 GiB). For example:

```bash
./cli-unleashed.sh --reserve-vram-mib 2048 --reserve-ram-mib 8192
```

This requests 2 GiB of extra VRAM headroom and 8 GiB of available system RAM. RAM reservation reduces the model-selection budget, is rechecked before launch, and rejects tuning trials that dip below it. It is a headroom target, not an operating-system memory lock; other applications can still consume memory. VRAM safety headroom remains separate. Tuning first optimizes short-prompt generation, then prompt processing while keeping at least 90% of the measured generation speed. Later starts reuse a matching tuning result.

## Testing

The native-context revision passes 58 regression tests and CUDA mixed-cache checks, including Q8/Q6. [Native-context validation record](docs/UNLEASHED-NATIVE-VALIDATION.json).

On the local RTX 5000 Ada 16 GB laptop GPU, the Atomic Q5 model reached **51.64 generated tokens/s** on a 512-token prompt/reply and **1,644 prompt tokens/s** on an 8,192-token prompt. Those are historical reduced-context measurements, not validation or speed claims for the new native-context policy. [Results and limitations](RELEASE-NOTES.md).

Linux execution is tested. Windows launch/build scripts and simulated platform checks are included, but Windows execution has not been verified. The self-tuner currently supports **one NVIDIA GPU**. The bundled Linux engine targets Ada and recent Linux libraries; other machines may need a source build.

## Docs

- [Installation and first chat](docs/INSTALL.md)
- [All setup options](docs/UNLEASHED-CLI.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Documentation index](docs/README.md)
- [Technical note and measurements](docs/paper/Strata-Unleashed.md) · [PDF](docs/paper/Strata-Unleashed.pdf)
- [Contributing](CONTRIBUTING.md)

Based on **Strata 0.1.38**, with credits to its developers, llama.cpp/ggml, Qwen and the model publishers. [MIT license](LICENSE); model files retain their publishers' licenses. The original [Strata paper](docs/paper/Strata-Paper.pdf) remains available and is separate from this fork's technical note.
