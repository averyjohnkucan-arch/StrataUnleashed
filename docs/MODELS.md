# Choose and download a model

Run `CLI-UNLEASHED.bat` on Windows or `./cli-unleashed.sh` on Linux. Pick a model. Setup downloads all its files and checks them for you.

## Models

| Model | Notes |
| --- | --- |
| ISTA Flash Next Coder | Smallest option. Mainly for coding. |
| ISTA Flash Next | Small general models. |
| Atomic Chat | Larger models. The main recommendation once you have the RAM. |
| Unsloth | Larger quants are available. Optional, not the first pick. Don't expect a big jump just from using more bits. |
| HuiHui | Uncensored / abliterated. Pick this if that's what you want. |

The picker sorts by CPU memory from largest to smallest; look near the bottom if RAM is tight. This is a rough guide, not a quality benchmark. Check the size of the actual quant before downloading. The picker has separate small, coding, large and uncensored modes; HuiHui stays separate from the general recommendations.

## Included choices

- **ISTA Coder:** IQ1_M. Not every tensor is one-bit.
- **ISTA general:** Q2_0, IQ2_XS, IQ3_XXS, IQ3_S.
- **Atomic Chat:** AD-3.84bpw-IQ4_XS-M64, AD-4.27bpw-Q4_K_M-M64, AD-5.00bpw-Q5_K_M-M64.
- **HuiHui:** Q8_0, UD-Q4_K_XL.
- **Unsloth:** Q8_0, UD-IQ1_M, UD-IQ3_XXS, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL.

The bundled snapshot contains 22 variants and 170 shards, checked on 2026-10-03. HuiHui/Unsloth BF16 expert models and Unsloth UD-IQ1_S are excluded from the picker because of unsupported expert encodings. The remaining 19 passed header/layout checks; they have not all been inference-tested. The full-model benchmark used Atomic Q5.

This is the **Qwen3.8-Flash-Next family**. Other architectures such as 27B and 2.4T models are not interchangeable with this engine. MTP draft files, vision projectors and imatrix files are not standalone chat models.

## What “fits” means

The picker checks available RAM, free disk, GPU/VRAM, context, reserved VRAM and engine/build tools. The estimates leave some room. Over-budget rows are struck through in supporting terminals but remain selectable. Missing engine/build requirements still need resolving. The table always uses FP16/Q8 KV at 262,144 context; actual runs use the capacity default or `--kv`. See [per-model CPU/GPU/engram accounting](MEMORY.md).

PLE lookup-table files are mapped from disk rather than permanently resident in RAM. The type and speed of the drive holding them can matter during inference. NVMe is preferred; a slower or unidentified drive receives a warning.

## Check the list

```bash
./cli-unleashed.sh --list
./cli-unleashed.sh --refresh-catalog --list
./cli-unleashed.sh --list --offline --json
```

Refresh retrieves the publishers' file lists and bounded GGUF headers, pins immutable revisions, and records sizes/hashes. Downloads verify complete-file SHA256 hashes before use. On Windows, replace `./cli-unleashed.sh` with `CLI-UNLEASHED.bat`.

[Full catalog, sources and flags](UNLEASHED-CLI.md) · [Machine-readable catalog](../data/unleashed-models.json)
