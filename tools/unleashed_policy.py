"""Shared KV defaults and native-context tuning policy."""

NATIVE_CONTEXT = 262144
REPLY_TOKENS = 512
CONTEXT_SLACK = 8  # Upstream --serve reserves eight tokens for internal decode work.
PREFILL_TOKENS = NATIVE_CONTEXT - REPLY_TOKENS - CONTEXT_SLACK
KV_FORMATS = (
    "FP16/FP16",
    "FP16/Q8",
    "Q8/Q8",
    "Q8/Q6",
    "Q8/Q5",
    "Q5/Q5",
    "Q5/Q4",
    "Q4/Q4",
)


def default_kv(total_mib):
    """Use nominal card capacity, allowing 64 MiB for driver-reported overhead."""
    if total_mib <= 0:
        raise ValueError("GPU memory capacity must be positive")
    for capacity, pair in ((16, "FP16/FP16"), (12, "FP16/Q8"), (8, "Q8/Q8")):
        if total_mib >= capacity * 1024 - 64:
            return pair
    return "Q8/Q6"


def native_args(args):
    """Normalize imported configs too; never reuse a reduced-context tuning run."""
    from tools.calibrate import with_arg

    args = with_arg(list(args), "--max-context", str(NATIVE_CONTEXT))
    return with_arg(args, "--rope-scaling", "none")


KV_GUIDANCE = (
    "KV cache guidance: FP16/FP16 is ideal; FP16/Q8 is recommended. "
    "Q8/Q5 is a last resort, only when no other viable option fits. "
    "Card-capacity defaults still apply; model weight quantization is separate."
)


def ram_breakdown(inspection, context=NATIVE_CONTEXT):
    """Default full-arena mode: GPU cache duplicates experts; KV stays on GPU.

    Six GiB is planning headroom for runtime/driver, loader, PLE hot pages and
    temporary buffers, not an exact allocation or a second dense-weight copy.
    """
    if inspection.get("memory_accounting_version") != 2:
        raise ValueError("Model memory accounting is outdated; refresh the catalog")
    return {
        "host_experts_bytes": inspection["expert_bytes"],
        "host_embedding_bytes": inspection["host_embedding_bytes"],
        "runtime_allowance_bytes": 6 * 1024**3,
        "ssd_ple_bytes": inspection["ple_bytes"],
        "gpu_dense_bytes": inspection["gpu_dense_bytes"],
        "host_kv_bytes": 0,
        "gpu_expert_cache_host_savings_bytes": 0,
        "context_tokens": context,
    }


def ram_estimate(inspection, context=NATIVE_CONTEXT):
    budget = ram_breakdown(inspection, context)
    return sum(
        budget[key]
        for key in (
            "host_experts_bytes",
            "host_embedding_bytes",
            "runtime_allowance_bytes",
            "host_kv_bytes",
        )
    )


def kv_cache_bytes(context, pair):
    """Main model only: 12 QSA layers, two KV heads, 256 values/head.

    Uniform Q8 uses scales per 64 values; mixed formats use groups of 32.
    Device indexers, RoPE, scratch and optional MTP are accounted separately.
    """
    if pair not in KV_FORMATS:
        raise ValueError(f"Unsupported KV format: {pair}")
    if pair == "Q8/Q8":
        per_head = 2 * (256 + 4 * 2)
    else:
        bits = [16 if side == "FP16" else int(side[1:]) for side in pair.split("/")]
        per_head = sum(512 if b == 16 else 8 * (2 + 4 * b) for b in bits)
    return context * 12 * 2 * per_head


def reservation_mib(value):
    """Parse explicit SI/binary units; unsuffixed values retain the CLI's MiB unit."""
    import re
    from decimal import Decimal, ROUND_CEILING

    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*(mib|gib|mb|gb)?", str(value).strip(), re.I
    )
    if not match:
        raise ValueError(
            "Use a nonnegative amount, e.g. 4096, 4096 MiB, 4 GiB or 4096 MB"
        )
    amount, unit = match.groups()
    scale = {"mib": 1024**2, "gib": 1024**3, "mb": 1000**2, "gb": 1000**3}[
        unit.lower() if unit else "mib"
    ]
    return int(
        (Decimal(amount) * scale / (1024**2)).to_integral_value(rounding=ROUND_CEILING)
    )
