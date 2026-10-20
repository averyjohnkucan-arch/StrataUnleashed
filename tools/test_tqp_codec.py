"""Compare Strata's pair codec with independently built vendor dequantizers.

Pass --vendor-library and --probe-library explicitly; nothing is downloaded.
This checks CPU state extraction, the codebook and FP16 scaling. GPU dot
products and inference require separate qualification (see tqp_parity).
"""

import argparse
import ctypes
import json
import random
import struct


def check_codec(vendor, probe):
    """Return per-format results, raising on the first bitwise mismatch."""
    probe.tqp_probe.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    probe.tqp_probe.restype = None

    # One deterministic stream across all formats makes failures reproducible.
    rng = random.Random(20261019)
    rows = []
    for k in [5, 6, 7]:
        reference = getattr(vendor, f"dequantize_row_tqp{k}")
        reference.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        reference.restype = None
        count = 0

        # Cover signed and zero scales, an ordinary fraction, a large scale,
        # and the smallest positive normal FP16 scale. Each block has 128 values.
        for scale in [0.0, 1.0, -1.0, 0.125, 8.0, 2**-14]:
            for block_index in range(32):
                data = struct.pack("<e", scale) + bytes(
                    rng.getrandbits(8) for _ in range(8 * k)
                )
                buffer = ctypes.create_string_buffer(data)
                expected = (ctypes.c_float * 128)()
                actual = (ctypes.c_float * 128)()
                reference(buffer, expected, 128)
                probe.tqp_probe(k, buffer, actual)

                # Compare bytes to catch signed-zero differences too. Keep this
                # check active under python -O, which removes assert statements.
                if bytes(expected) != bytes(actual):
                    max_error = max(abs(x - y) for x, y in zip(expected, actual))
                    raise AssertionError(
                        f"TQP{k} scale={scale} block={block_index}: "
                        f"bitwise mismatch, max absolute error={max_error}"
                    )
                count += 1

        rows.append(
            {
                "format": f"TQP{k}",
                "randomBlocks": count,
                "values": count * 128,
                "bitIdentical": True,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-library", required=True)
    parser.add_argument("--probe-library", required=True)
    args = parser.parse_args()

    vendor = ctypes.CDLL(args.vendor_library)
    probe = ctypes.CDLL(args.probe_library)
    rows = check_codec(vendor, probe)
    print(
        json.dumps(
            {
                "state": "passed",
                "cases": rows,
                "scope": "CPU state extraction, codebook and FP16 scaling; "
                "not GPU expert/inference parity",
            }
        )
    )


if __name__ == "__main__":
    main()
