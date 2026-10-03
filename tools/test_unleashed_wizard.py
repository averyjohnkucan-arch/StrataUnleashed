"""Hardware/model advice, shard grouping, download integrity and CLI regression tests."""

import copy
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools import unleashed_wizard as W
from tools import unleashed_download as D
from tools import unleashed_hardware as H
from tools import unleashed_catalog as C

GIB = W.GIB


def hardware(os_name="Linux"):
    return {
        "os": os_name,
        "platform": "test",
        "machine": "AMD64" if os_name == "Windows" else "x86_64",
        "cpu": {"avx2": True, "logical_cpus": 16},
        "ram_total": 128 * GIB,
        "ram_available": 120 * GIB,
        "disk_free": 500 * GIB,
        "storage_path": "models",
        "errors": [],
        "tools": {
            n: n
            for n in ("cmake", "ninja", "nvcc", "cl" if os_name == "Windows" else "c++")
        },
        "engine_runs": True,
        "engine_built_for": "test GPU",
        "gpus": [
            {
                "index": 0,
                "name": "test GPU",
                "total_mib": 24576,
                "free_mib": 23000,
                "compute_capability": 8.9,
                "driver": "test",
            }
        ],
    }


class Response:
    def __init__(self, body, status=200, headers=None):
        self.body, self.status_code, self.headers = body, status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.body


class WizardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(C.CATALOG.read_text())
        cls.entries = cls.catalog["models"]

    def model(self, mid="atomic/AD-5.00bpw-Q5_K_M-M64"):
        return copy.deepcopy(next(e for e in self.entries if e["id"] == mid))

    def fit(self, e=None, s=None, **kw):
        return W.assess(e or self.model(), s or hardware(), download_bytes=0, **kw)

    def test_system_ram_reservation_changes_model_fit(self):
        system = hardware()
        needed = self.fit()["ram_estimate_bytes"]
        system["ram_available"] = needed + 4 * GIB
        self.assertEqual(self.fit(s=system)["status"], "candidate")
        reserved = self.fit(s=system, reserve_ram_mib=8192)
        self.assertEqual(reserved["status"], "cannot-run-now")
        self.assertEqual(
            reserved["ram_budget_bytes"], system["ram_available"] - 8 * GIB
        )
        self.assertTrue(any("reserving 8.0 GiB" in r for r in reserved["reasons"]))

    def test_interactive_reservations_and_kv_guidance(self):
        output = io.StringIO()
        with patch.object(W, "scan_system", return_value=hardware()), patch.object(
            W, "ask_int", side_effect=[1, 2048, 8192, 0]
        ) as ask, patch.object(
            W, "rows_for", wraps=W.rows_for
        ) as rows, redirect_stdout(
            output
        ):
            self.assertEqual(W.main(["--offline"]), 0)
        args = rows.call_args.args[2]
        self.assertEqual((args.reserve_vram_mib, args.reserve_ram_mib), (2048, 8192))
        self.assertIn("system RAM", ask.call_args_list[2].args[0])
        for guidance in (
            "FP16/FP16 is ideal",
            "FP16/Q8 is recommended",
            "Q8/Q5 is a last resort",
        ):
            self.assertIn(guidance, output.getvalue())

    def test_published_catalog_complete_and_pinned(self):
        self.assertEqual(len(self.entries), 22)
        self.assertEqual({e["provider"] for e in self.entries}, set(C.SOURCES))
        for e in self.entries:
            self.assertRegex(e["revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(e["download_bytes"], sum(f["size"] for f in e["files"]))
            for f in e["files"]:
                self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len([e for e in self.entries if e["provider"] == "atomic"]), 3)
        self.assertFalse(self.model("unsloth/UD-IQ1_S")["inspection"]["compatible"])

    def test_actual_types_not_quant_label(self):
        # These mixed models have supported expert encodings despite Q2_K/Q3_K labels.
        self.assertTrue(self.model("unsloth/UD-Q2_K_XL")["inspection"]["compatible"])
        self.assertTrue(self.model("unsloth/UD-Q3_K_XL")["inspection"]["compatible"])
        self.assertIn(
            "Unsupported expert encoding: BF16",
            self.model("huihui/BF16")["inspection"]["reasons"],
        )

    def test_preferences_and_no_uncensored_substitution(self):
        rows = [{"model": e, "assessment": self.fit(e)} for e in self.entries]
        for goal, provider in [
            ("small", "ista"),
            ("coder", "ista-coder"),
            ("large", "atomic"),
            ("uncensored", "huihui"),
        ]:
            self.assertEqual(
                W.recommendations(rows, goal)[0]["model"]["provider"], provider
            )
        for row in rows:
            if row["model"]["provider"] == "huihui":
                row["assessment"]["status"] = "cannot-run-now"
        self.assertEqual(W.recommendations(rows, "uncensored"), [])

    def test_low_ram_and_disk_are_explained(self):
        s = hardware()
        s.update(ram_total=16 * GIB, ram_available=8 * GIB, disk_free=1)
        reasons = self.fit(s=s)["reasons"]
        self.assertTrue(any("RAM" in r for r in reasons))
        self.assertTrue(any("disk" in r for r in reasons))

    def test_available_ram_not_just_installed(self):
        s = hardware()
        s["ram_available"] = 12 * GIB
        self.assertTrue(
            any("currently available" in r for r in self.fit(s=s)["reasons"])
        )

    def test_reservation_and_context_reduce_fit(self):
        base = self.fit()
        large = self.fit(reserve_mib=20000, context=262144)
        self.assertEqual(base["status"], "candidate")
        self.assertEqual(large["status"], "cannot-run-now")
        self.assertLess(large["vram_budget_bytes"], base["vram_budget_bytes"])

    def test_no_gpu_old_gpu_and_wrong_gpu_blocked(self):
        s = hardware()
        s["gpus"] = []
        self.assertEqual(self.fit(s=s)["status"], "cannot-run-now")
        s = hardware()
        s["gpus"][0]["compute_capability"] = 6.1
        self.assertTrue(any("7.5" in r for r in self.fit(s=s)["reasons"]))
        self.assertEqual(self.fit(gpu_index=99)["status"], "cannot-run-now")

    def test_windows_requires_build_tools_without_binary(self):
        s = hardware("Windows")
        s["engine_runs"] = False
        s["tools"]["cl"] = None
        self.assertTrue(any("Visual Studio" in r for r in self.fit(s=s)["reasons"]))
        s["tools"]["cl"] = "cl.exe"
        self.assertTrue(self.fit(s=s)["needs_build"])
        self.assertEqual(self.fit(s=s)["status"], "candidate")

    def test_resume_checks_range_and_hash(self):
        data = b"the complete file"
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory(dir=W.ROOT / "work/tmp") as td:
            p = Path(td) / "file.gguf"
            p.with_suffix(".gguf.part").write_bytes(data[:4])
            with patch.object(
                D.requests,
                "get",
                return_value=Response(
                    data[4:],
                    206,
                    {"Content-Range": f"bytes 4-{len(data)-1}/{len(data)}"},
                ),
            ) as get:
                D.download_file(
                    "https://example.test/model", p, len(data), digest, progress=False
                )
                self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=4-")
            self.assertEqual(p.read_bytes(), data)

    def test_ignored_range_restarts(self):
        data = b"correct"
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory(dir=W.ROOT / "work/tmp") as td:
            p = Path(td) / "x.gguf"
            p.with_suffix(".gguf.part").write_bytes(b"old")
            with patch.object(D.requests, "get", return_value=Response(data)):
                D.download_file("url", p, len(data), digest, progress=False)
            self.assertEqual(p.read_bytes(), data)

    def test_bad_range_and_digest_never_publish(self):
        with tempfile.TemporaryDirectory(dir=W.ROOT / "work/tmp") as td:
            p = Path(td) / "x.gguf"
            with patch.object(
                D.requests,
                "get",
                return_value=Response(b"bad", 206, {"Content-Range": "bytes 1-3/3"}),
            ):
                with self.assertRaisesRegex(ValueError, "range"):
                    D.download_file("url", p, 3, "0" * 64, progress=False)
            with patch.object(D.requests, "get", return_value=Response(b"bad")):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    D.download_file("url", p, 3, "0" * 64, progress=False)
            self.assertFalse(p.exists())
            self.assertEqual(len(list(Path(td).glob("*.bad-*"))), 1)

    def test_truncated_download_retains_resume_partial(self):
        with tempfile.TemporaryDirectory(dir=W.ROOT / "work/tmp") as td:
            p = Path(td) / "x.gguf"
            with patch.object(D.requests, "get", return_value=Response(b"abc")):
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    D.download_file("url", p, 6, "0" * 64, progress=False)
            self.assertEqual(p.with_suffix(".gguf.part").read_bytes(), b"abc")

    def test_existing_bad_file_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=W.ROOT / "work/tmp") as td:
            p = Path(td) / "x.gguf"
            p.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "Existing"):
                D.download_file("url", p, 3, "0" * 64, progress=False)
            self.assertEqual(p.read_bytes(), b"bad")

    def test_verified_existing_file_is_reused(self):
        with tempfile.TemporaryDirectory(dir=W.ROOT / "work/tmp") as td:
            p = Path(td) / "x.gguf"
            p.write_bytes(b"ok")
            with patch.object(D.requests, "get") as get:
                D.download_file(
                    "url", p, 2, hashlib.sha256(b"ok").hexdigest(), progress=False
                )
                get.assert_not_called()

    def test_scan_windows_and_multiple_gpus(self):
        result = subprocess.CompletedProcess(
            [],
            0,
            "0, GPU A, 8192, 7000, 8.6, 600\n1, GPU B, 24576, 23000, 8.9, 600\n",
            "",
        )
        with patch.object(H.platform, "system", return_value="Windows"), patch.object(
            W.subprocess, "run", return_value=result
        ):
            s = W.scan_system()
        self.assertEqual(len(s["gpus"]), 2)
        self.assertEqual(s["gpus"][1]["index"], 1)

    def test_offline_json_list_never_downloads(self):
        with patch.object(W, "scan_system", return_value=hardware()), patch.object(
            W, "download_model"
        ) as download, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(W.main(["--list", "--json", "--offline"]), 0)
            download.assert_not_called()
            self.assertEqual(len(json.loads(out.getvalue())["models"]), 22)

    def test_blocked_selection_never_downloads(self):
        with patch.object(W, "scan_system", return_value=hardware()), patch.object(
            W, "download_model"
        ) as download, redirect_stdout(io.StringIO()):
            self.assertEqual(W.main(["--model-id", "huihui/BF16", "--yes"]), 2)
            download.assert_not_called()

    def test_cli_passes_gpu_reserve_and_tuning_controls(self):
        with patch.object(W, "scan_system", return_value=hardware()), patch.object(
            W, "download_model", return_value=Path("/readonly/model.gguf")
        ), patch.object(
            W.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ) as run, redirect_stdout(
            io.StringIO()
        ), patch.object(
            Path, "write_text"
        ), patch.object(
            W.unleashed, "split_paths", return_value=[Path(__file__)]
        ), patch.object(
            W,
            "scan_storage",
            return_value={"path": "test", "transport": "nvme", "warnings": []},
        ):
            self.assertEqual(
                W.main(
                    [
                        "--model-id",
                        "ista/Q2_0",
                        "--yes",
                        "--tune-only",
                        "--retune",
                        "--reserve-vram-mib",
                        "2048",
                        "--reserve-ram-mib",
                        "8192",
                        "--port",
                        "8101",
                    ]
                ),
                0,
            )
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--reserve-vram-mib") + 1], "2048")
            self.assertEqual(cmd[cmd.index("--gpu") + 1], "0")
            self.assertEqual(cmd[cmd.index("--reserve-ram-mib") + 1], "8192")
            self.assertIn("--retune", cmd)
            self.assertIn("--tune-only", cmd)

    def test_grouping_does_not_mix_shards_or_artifacts(self):
        def file(n):
            return {"rfilename": n, "size": 32, "lfs": {"sha256": "a" * 64}}

        info = {
            "sha": "b" * 40,
            "siblings": [
                file("Q4/model-00001-of-00002.gguf"),
                file("Q4/model-00002-of-00002.gguf"),
                file("imatrix.gguf"),
                file("MTP/mtp.gguf"),
            ],
        }
        self.assertEqual(len(C.variants("atomic", info)), 1)
        info["siblings"].pop(1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            C.variants("atomic", info)

    def test_unsafe_paths_and_unpinned_revisions_rejected(self):
        for path in ("../file.gguf", "/file.gguf", "C:\\file.gguf"):
            with self.assertRaises(ValueError):
                C.safe_path(path)
        with self.assertRaises(ValueError):
            C.resolve_url("owner/repo", "main", "file.gguf")

    def test_second_downloader_is_rejected(self):
        if D.os.name == "nt":
            self.skipTest(
                "Same-process Windows byte-lock behavior differs; exercised by implementation"
            )
        with D.download_lock():
            with self.assertRaisesRegex(ValueError, "Another"):
                with D.download_lock():
                    pass

    def test_occupied_port_is_detected(self):
        import socket

        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            self.assertFalse(W.port_available(occupied.getsockname()[1]))

    def test_header_mismatch_and_unsupported_experts(self):
        # Reuse tiny real GGUF-header descriptors without allocating model weights.
        from types import SimpleNamespace

        g = SimpleNamespace(
            path=Path("single.gguf"),
            metadata={"general.architecture": "llama"},
            tensors=[],
            data_start=0,
        )
        result = C.inspect_headers([g], [0])
        self.assertFalse(result["compatible"])
        self.assertIn("Architecture is not qwen4exp", result["reasons"])


if __name__ == "__main__":
    unittest.main()
