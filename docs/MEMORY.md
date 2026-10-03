# Memory estimates

The model picker uses **FP16/Q8 KV at 262,144 context tokens** for every estimate, on Windows and Linux. Actual runs retain the card-capacity KV default unless you choose `--kv`. All table values are GiB (2³⁰ bytes); download publishers may use decimal GB.

## Where memory goes

The guided workflow uses the engine's full host expert arena. Every expert weight remains in system RAM. GPU expert caches copy selected experts, so their VRAM **does not subtract from that host arena**. The SSD-backed `per_layer_token_embd.weight` tensor is the PLE/engram table. Its full size is excluded from permanent RAM; accessed pages and staging still consume some memory. The token embedding remains host-mapped; dense projections and the output head reside on the GPU.

- **CPU MEM** = exact host expert bytes + host token embedding bytes + 6 GiB planning allowance.
- **Engram table** = exact SSD-backed PLE tensor bytes across all shards.
- **GPU MEM** = GPU dense weights + 4.59375 GiB main-model FP16/Q8 KV + 2 GiB workspace allowance, before the automatic expert cache.

The 6 GiB allowance covers host runtime/driver, loading buffers, hot PLE pages and temporary allocations. The 2 GiB GPU workspace is also an allowance, not a measured universal constant. KV storage uses 12 full-attention layers, two KV heads, 256 values per head and the engine's quantization scales. FP16/FP16 would use 6 GiB of main-model KV. KV is on the GPU in this workflow; adding a second context-sized CPU allocation would double-count it.

These estimates apply to the guided no-MTP, full-arena, GPU-KV configuration. Imported/manual configurations using host KV, RAM PLE tables, MTP or the separate resident-expert/complement mode need their own accounting. Available system RAM already excludes current OS/application usage; the user's additional RAM reserve is subtracted separately.

## Per-model estimates

Derived from tensor headers in the [pinned catalog](../data/unleashed-models.json), including every shard. Unsupported BF16 expert and IQ1_S expert variants are omitted. Header compatibility does not establish inference validation for every model.

| Provider / quant | Download | Engram table (SSD) | Host experts | Host embedding | CPU MEM | GPU MEM* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| huihui/Q8_0 | 175.30 | 50.66 | 119.53 | 0.63 | 126.16 | 11.51 |
| unsloth/Q8_0 | 175.30 | 50.66 | 119.53 | 0.63 | 126.16 | 11.51 |
| unsloth/UD-Q6_K_XL | 157.55 | 50.66 | 101.75 | 0.63 | 108.37 | 11.51 |
| unsloth/UD-Q5_K_XL | 147.42 | 50.66 | 91.61 | 0.63 | 98.24 | 11.51 |
| huihui/UD-Q4_K_XL | 103.69 | 26.82 | 71.73 | 0.63 | 78.36 | 11.51 |
| unsloth/UD-Q4_K_XL | 103.69 | 26.82 | 71.73 | 0.63 | 78.36 | 11.51 |
| unsloth/UD-IQ4_XS | 87.25 | 26.82 | 55.43 | 0.63 | 62.06 | 11.37 |
| unsloth/UD-Q3_K_XL | 83.81 | 26.82 | 51.99 | 0.63 | 58.62 | 11.37 |
| atomic/AD-4.27bpw-Q4_K_M-M64 | 88.03 | 35.76 | 47.17 | 0.63 | 53.80 | 11.51 |
| atomic/AD-5.00bpw-Q5_K_M-M64 | 102.93 | 50.66 | 47.17 | 0.63 | 53.80 | 11.51 |
| ista/IQ3_S | 77.88 | 26.82 | 46.84 | 0.31 | 53.15 | 10.48 |
| unsloth/UD-IQ3_XXS | 76.33 | 26.82 | 45.29 | 0.49 | 51.77 | 10.74 |
| unsloth/UD-Q2_K_XL | 73.45 | 26.82 | 42.92 | 0.41 | 49.33 | 10.31 |
| ista/IQ3_XXS | 70.63 | 26.82 | 39.97 | 0.25 | 46.22 | 10.17 |
| unsloth/UD-IQ1_M | 69.42 | 26.82 | 38.96 | 0.33 | 45.30 | 10.30 |
| atomic/AD-3.84bpw-IQ4_XS-M64 | 79.10 | 35.76 | 38.23 | 0.63 | 44.86 | 11.51 |
| ista/IQ2_XS | 63.35 | 26.82 | 33.02 | 0.31 | 39.33 | 9.78 |
| ista/Q2_0 | 61.86 | 26.82 | 31.64 | 0.25 | 37.90 | 9.73 |
| ista-coder/IQ1_M | 54.40 | 26.82 | 23.42 | 0.31 | 29.73 | 10.42 |

*GPU expert caching grows into spare VRAM during tuning; GPU MEM is a startup planning estimate, not the final VRAM target. Rows appear in descending CPU MEM order. Over-budget rows are struck through in supporting terminals and remain selectable. Redirected output uses an explicit status instead. Selecting one attempts setup despite the estimate; actual allocation failures and measured reservation checks still apply. The direct launcher provides `--allow-over-budget` for the same estimate override. Unsupported encodings stay hidden and cannot be selected.

## Atomic Q5: corrected example and measurement

The 33-shard Atomic Q5 download is **110.525 GB / 102.935 GiB**. Its PLE table is **54.400 GB / 50.664 GiB**, host experts are **47.168 GiB**, host embedding is **0.629 GiB**, and GPU dense weights are **4.918 GiB**. CPU planning is therefore **47.168 + 0.629 + 6 = 53.797 GiB**, approximately **57.765 GB**. The reference GPU estimate is **4.918 + 4.594 + 2 = 11.512 GiB**, before expert caching. Atomic Q4 and Q5 have the same expert payload; much of their download-size difference is the PLE table.

The earlier 79.3 GiB CPU estimate incorrectly duplicated dense storage and added an extra host context allowance. Subtracting the entire GPU allocation would also be wrong: KV and workspace are not host expert weights, and the default cache retains host expert copies.

Real Atomic Q5 runs on Linux with a 16 GiB RTX 5000 Ada laptop GPU measured **50.00 GiB peak engine RSS** for 512/512 inference with FP16/FP16 KV, and **50.07 GiB** for **261,624 input / 512 output tokens with FP16/Q8 KV**. Both preallocated a 262,144-token context. The full-context sample measured 829.6 prefill tokens/s and 28.1 decode tokens/s, with 15,552 MiB total GPU memory in use. These are single validation runs, not a tuning sweep or a cross-platform minimum-RAM guarantee. RSS does not include every OS/driver allocation; keep the planning allowance and compare it with available RAM. See the [memory measurements](UNLEASHED-MEMORY-VALIDATION.json).

The engine calls `cudaMalloc(session_bytes(g, o.max_context, ...))` during startup and initializes the full session before sizing the expert cache. Context capacity is therefore preallocated, including during the short benchmark; it does not grow from a 512-token cache.

## Implementation evidence

The accounting was checked against the pinned Strata 0.1.38 engine, not generic model-size heuristics:

- [ArenaExpertSource allocation/loading](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/src/core/expert_source.cpp): full host expert arena and separate GPU copies.
- [Generation setup](../src/program/generate.cpp): default arena selection, host embedding, disk-backed PLE and GPU KV allocation.
- [Native dense weights](../src/core/native_dense.cpp), [canonical weights](../src/core/weights.cpp) and [output head](../src/core/native_head.cpp): device allocations and bounded upload staging.
- [Catalog tensor classification](../tools/unleashed_catalog.py) and [shared memory policy](../tools/unleashed_policy.py): the executable calculation behind this chart.
