"""Install safety, read-only storage checks and no-system-prompt terminal inference."""

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import install_unleashed as I
from tools import unleashed_chat as C
from tools import unleashed_storage as S
from serve.frontend import ChatTemplate
from tools.unleashed_catalog import ROOT


class ExperienceTests(unittest.TestCase):
    def test_fit_uses_each_machine_not_the_development_gpu(self):
        from tools.test_unleashed_wizard import hardware
        from tools.unleashed_hardware import assess

        entries = json.loads((ROOT / "data/unleashed-models.json").read_text())[
            "models"
        ]
        model = next(e for e in entries if e["id"] == "ista/Q2_0")
        for system_name in ("Linux", "Windows"):
            for capability in (7.5, 8.6, 8.9, 12.0):
                for vram in (6144, 12288, 24576):
                    with self.subTest(os=system_name, capability=capability, vram=vram):
                        machine = hardware(system_name)
                        machine["engine_built_for"] = ""
                        machine["engine_architectures"] = [86]
                        machine["gpus"][0].update(
                            compute_capability=capability,
                            total_mib=vram,
                            free_mib=vram - 256,
                        )
                        result = assess(model, machine, download_bytes=0)
                        self.assertEqual(result["needs_build"], capability != 8.6)
                        self.assertLess(result["vram_budget_bytes"], vram * 1024**2)
        machine = hardware()
        machine["engine_built_for"] = ""
        machine["engine_architectures"] = []
        self.assertTrue(assess(model, machine, download_bytes=0)["needs_build"])

    def test_zero_system_render_using_real_model_template(self):
        path = ROOT / "work/pack-q5/tokenizer/chat_template.jinja"
        if not path.is_file():
            path = ROOT / "serve/chat_template.jinja"
        rendered = C.render_prompt(
            ChatTemplate(path), [{"role": "user", "content": "Reply with 4."}]
        )
        self.assertNotIn("<|im_start|>system", rendered)
        self.assertNotIn("Reasoning effort is set", rendered)
        self.assertIn("<|im_start|>user", rendered)

    def test_hidden_system_template_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            p = Path(td) / "template"
            p.write_text("<|im_start|>system\nHidden instructions")
            with self.assertRaisesRegex(ValueError, "system message"):
                C.render_prompt(ChatTemplate(p), [{"role": "user", "content": "hello"}])

    def test_system_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only user and assistant"):
            C.render_prompt(None, [{"role": "system", "content": "x"}])

    def test_reply_uses_actual_tokens_and_closes_generator(self):
        closed = []

        def generate(*args):
            try:
                yield 1
                yield None
                yield 2
            finally:
                closed.append(True)

        engine = SimpleNamespace(
            generate=generate, max_context=20, last={"decode_ms": 100, "prompt_ms": 5}
        )
        tokenizer = SimpleNamespace(
            encode=lambda *a, **kw: [1, 2, 3], decode=lambda ids, **kw: "a" * len(ids)
        )
        template = SimpleNamespace(render=lambda *a, **kw: "user only")
        text, stats = C.reply(
            engine,
            tokenizer,
            template,
            [{"role": "user", "content": "hi"}],
            4,
            output=lambda *a, **kw: None,
        )
        self.assertEqual(text, "aa")
        self.assertEqual(stats["output_tokens"], 2)
        self.assertEqual(stats["decode_tokens_per_second"], 20)
        self.assertEqual(stats["system_messages"], 0)
        self.assertTrue(closed)

    def test_context_limit_checked_before_generation(self):
        engine = SimpleNamespace(max_context=5)
        tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "too long"):
            C.reply(
                engine,
                tokenizer,
                SimpleNamespace(render=lambda *a, **kw: "text"),
                [{"role": "user", "content": "hi"}],
                4,
            )

    def test_non_nvme_and_unknown_warn_without_blocking(self):
        for bus in ("sata", "usb", "unknown"):
            with patch.object(
                S, "linux_drive", return_value={"transport": bus, "nvme": False}
            ), patch.object(S.os, "name", "posix"):
                result = S.scan_storage(ROOT)
            self.assertTrue(any("experience may vary" in w for w in result["warnings"]))

    def test_nvme_but_slow_read_warns(self):
        with patch.object(
            S, "linux_drive", return_value={"transport": "nvme", "nvme": True}
        ), patch.object(
            S, "direct_reads", return_value={"read_mib_s": 120}
        ), patch.object(
            S.os, "name", "posix"
        ):
            result = S.scan_storage(ROOT, "some-file")
        self.assertTrue(any("below 500" in w for w in result["warnings"]))

    def test_uncached_unavailable_not_reported_as_a_speed(self):
        with patch.object(
            S, "linux_drive", return_value={"transport": "nvme", "nvme": True}
        ), patch.object(
            S, "direct_reads", side_effect=OSError("not supported")
        ), patch.object(
            S.os, "name", "posix"
        ):
            result = S.scan_storage(ROOT, "some-file")
        self.assertIsNone(result["read_mib_s"])
        self.assertIn("not supported", result["speed_note"])

    def test_linux_drive_walks_partition_parents(self):
        results = [
            subprocess.CompletedProcess(
                [],
                0,
                json.dumps(
                    {"filesystems": [{"source": "/dev/nvme0n1p2", "fstype": "ext4"}]}
                ),
            ),
            subprocess.CompletedProcess(
                [],
                0,
                json.dumps(
                    {
                        "blockdevices": [
                            {
                                "name": "nvme0n1p2",
                                "type": "part",
                                "tran": None,
                                "children": [
                                    {
                                        "name": "nvme0n1",
                                        "type": "disk",
                                        "tran": "nvme",
                                        "rota": False,
                                    }
                                ],
                            }
                        ]
                    }
                ),
            ),
        ]
        with patch.object(S.subprocess, "run", side_effect=results):
            self.assertTrue(S.linux_drive(ROOT)["nvme"])

    def test_read_test_does_not_modify_file(self):
        if not hasattr(os, "O_DIRECT"):
            self.skipTest("Linux O_DIRECT check")
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            p = Path(td) / "sample"
            p.write_bytes(b"a" * (4 * S.MIB))
            before = p.stat()
            try:
                r = S.direct_reads(p)
            except OSError as exc:
                self.skipTest(str(exc))
            self.assertEqual(r["sample_bytes"], 4 * S.MIB)
            self.assertTrue(r["cache_bypassed"])
            self.assertEqual(p.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(p.stat().st_size, before.st_size)

    def test_installer_rejects_traversal(self):
        for name in ("../file", "/file", "C:\\file"):
            with self.assertRaises(ValueError):
                I.safe_members([name])

    def test_installer_refuses_existing_unrelated_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            p = Path(td)
            (p / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "nothing was overwritten"):
                I.install(p, {})
            self.assertEqual((p / "keep.txt").read_text(), "keep")

    def test_installer_checks_release_hash_and_installs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            p = Path(td)
            name = "StrataUnleashed-test-linux-x86_64.tar.gz"
            archive = p / name
            with tarfile.open(archive, "w:gz") as t:
                for file in ("unleashed.py", "cli-unleashed.sh"):
                    data = b"# test\n"
                    info = tarfile.TarInfo("StrataUnleashed-test/" + file)
                    info.size = len(data)
                    info.mode = 0o755
                    t.addfile(info, io.BytesIO(data))
            sums = p / "SHA256SUMS"
            sums.write_text(
                hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + name + "\n"
            )
            release = {
                "tag_name": "test",
                "assets": [
                    {"name": name, "browser_download_url": archive.as_uri()},
                    {"name": "SHA256SUMS", "browser_download_url": sums.as_uri()},
                ],
            }
            dest = p / "installed"
            with patch.object(I.os, "name", "posix"):
                I.install(dest, release)
            self.assertTrue((dest / "unleashed.py").is_file())
            sums.write_text("0" * 64 + "  " + name + "\n")
            with patch.object(I.os, "name", "posix"), self.assertRaisesRegex(
                ValueError, "checksum"
            ):
                I.install(p / "bad", release)
            self.assertFalse((p / "bad/unleashed.py").exists())


if __name__ == "__main__":
    unittest.main()
