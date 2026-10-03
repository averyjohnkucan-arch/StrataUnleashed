"""PLE type, alignment and >4GB addressing regressions; no model or GPU required.
Requires build/ple_q8_parity and gguf-py; all fixtures live below work/tmp.
"""

import json, os, struct, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants

BIN = ROOT / "build" / ("ple_q8_parity.exe" if os.name == "nt" else "ple_q8_parity")


def probe(path, mode):
    return subprocess.run(
        [str(BIN), str(path)] + (["direct"] if mode == "direct" else []),
        capture_output=True,
        text=True,
        timeout=30,
    )


@unittest.skipUnless(BIN.is_file(), "build ple_q8_parity first")
class PleFormats(unittest.TestCase):
    def test_formats_and_truncation(self):
        x = (np.sin(np.arange(103 * 160, dtype=np.float32) * 0.371) * 2).reshape(
            103, 160
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            for kind in (
                Q.F32,
                Q.F16,
                Q.BF16,
                Q.Q4_0,
                Q.Q4_1,
                Q.Q5_0,
                Q.Q5_1,
                Q.Q8_0,
                Q.IQ4_NL,
            ):
                with self.subTest(format=kind.name):
                    path = Path(td) / (kind.name + ".gguf")
                    if kind == Q.IQ4_NL:
                        raw = np.zeros((103, 5, 18), np.uint8)
                        raw[:, :, 0:2] = np.array([0, 60], np.uint8)
                        raw[:, :, 2:] = np.arange(16, dtype=np.uint8) * 17
                        raw = raw.reshape(103, 90)
                    else:
                        raw = quants.quantize(x, kind)
                    w = GGUFWriter(path, "qwen4exp")
                    w.add_tensor("per_layer_token_embd.weight", raw, raw_dtype=kind)
                    w.write_header_to_file()
                    w.write_kv_data_to_file()
                    w.write_tensors_to_file()
                    w.close()
                    for mode in ("mmap", "direct"):
                        r = probe(path, mode)
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    with path.open("r+b") as f:
                        f.truncate(path.stat().st_size - 64)
                    r = probe(path, "mmap")
                    self.assertNotEqual(r.returncode, 0, "truncated tensor accepted")

    def test_sparse_f32_table_beyond_32bit_offsets(self):
        # 204.8 GB logical tensor, almost no physical storage; first, interior and final reads.
        if os.name == "nt":
            self.skipTest(
                "sparse-file allocation setup is platform-specific; regular row tests cover Windows"
            )
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            path = Path(td) / "large.gguf"
            name = b"per_layer_token_embd.weight"
            rows = 320001536
            head = (
                struct.pack("<IIQQ", 0x46554747, 3, 1, 0)
                + struct.pack("<Q", len(name))
                + name
                + struct.pack("<IQQIQ", 2, 160, rows, 0, 0)
            )
            head += b"\0" * ((-len(head)) % 32)
            with path.open("wb") as f:
                f.write(head)
                f.truncate(len(head) + rows * 640)
                for row in (0, 1, 12345, 20000003, rows - 1):
                    f.seek(len(head) + row * 640)
                    values = (np.arange(160, dtype=np.float32) + (row % 97)) / 16
                    f.write(values.tobytes())
            for mode in ("mmap", "direct"):
                r = probe(path, mode)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
