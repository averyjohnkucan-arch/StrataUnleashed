"""Self-tuner selection, model identity, and release launcher regression tests (no GPU)."""

import copy, json, os, struct, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import unleashed
from unleashed_tune import Tuner


class SelectionTests(unittest.TestCase):
    def bare(self):
        t = Tuner.__new__(Tuner)
        t.cfg = {
            "args": ["--native", "weights.gguf", "--pack", "pack", "--mtp", "original"],
            "tune_mtp": "draft",
        }
        return t

    def test_mtp_absence_releases_head_and_config_input_is_immutable(self):
        t = self.bare()
        before = copy.deepcopy(t.cfg)
        state = dict(
            kv="Q5/Q4", workers=7, prefill=512, reserve=2000, pcie=0.3, minp=0.7, spec=0
        )
        args = t.args(state)
        self.assertNotIn("--mtp", args)
        self.assertEqual(args[args.index("--expert-cache") + 1], "auto")
        self.assertEqual(args[args.index("--prompt-cache") + 1], "0")
        self.assertEqual(t.cfg, before)
        state["spec"] = 3
        args = t.args(state)
        self.assertEqual(args[args.index("--mtp") + 1], "draft")

    def test_rejects_faster_over_budget_candidate(self):
        t = self.bare()
        best = dict(state={"workers": 4}, decode=50, valid=True)
        t.evaluate = lambda st, label: dict(
            state=st, decode=50 if st["workers"] == 4 else 100, valid=st["workers"] == 4
        )
        self.assertEqual(t.coordinate(best, "workers", [8]), best)

    def test_confirmation_rejects_transient_speedup(self):
        t = self.bare()
        best = dict(state={"workers": 4}, decode=50, valid=True)

        def evaluate(st, label):
            return dict(
                state=st,
                decode=(
                    100 if label == "workers-8" else (45 if st["workers"] == 8 else 50)
                ),
                valid=True,
            )

        t.evaluate = evaluate
        self.assertEqual(t.coordinate(best, "workers", [8]), best)

    def test_confirmation_accepts_repeatable_gain(self):
        t = self.bare()
        best = dict(state={"workers": 4}, decode=50, valid=True)
        t.evaluate = lambda st, label: dict(
            state=st, decode=60 if st["workers"] == 8 else 50, valid=True
        )
        self.assertEqual(t.coordinate(best, "workers", [8])["state"]["workers"], 8)

    def test_feedback_tracks_measured_peak(self):
        t = self.bare()
        t.total = 10000
        t.target = 0.8
        t.evaluate = lambda st, label, repeats: dict(
            state=st,
            peak=10000 - st["reserve"],
            decode=50,
            valid=10000 - st["reserve"] <= 8150,
        )
        r = t.fit(dict(reserve=1000))
        self.assertLessEqual(abs(r["peak"] - 8000), 120)

    def test_format_error_is_not_treated_as_vram_shortage(self):
        t = self.bare()
        t.total = 10000
        t.target = 0.8
        calls = []

        def evaluate(st, label, repeats):
            calls.append(st)
            return {"error": "unsupported tensor format", "valid": False}

        t.evaluate = evaluate
        t.fit(dict(reserve=1000))
        self.assertEqual(len(calls), 1)

    def test_allocation_failure_backs_off_without_retrying_failed_reserve(self):
        t = self.bare()
        t.total = 10000
        t.target = 0.95
        tried = []

        def evaluate(st, label, repeats):
            tried.append(st["reserve"])
            if st["reserve"] < 1000:
                return dict(state=st, error="CUDA out of memory", valid=False)
            return dict(state=st, peak=10000 - st["reserve"], decode=50, valid=True)

        t.evaluate = evaluate
        result = t.fit(dict(reserve=600))
        self.assertTrue(result["valid"])
        self.assertEqual(tried, [600, 1112])

    def test_over_budget_by_one_mib_is_rejected(self):
        t = self.bare()
        t.total = 10000
        t.target = 0.9
        t.repeats = 1
        t.load = lambda state: None
        t.request = lambda *args: dict(
            decode_tps=50, prefill_tps=200, peak_vram_mib=9001
        )
        self.assertFalse(t.evaluate({}, "limit")["valid"])

    def test_final_validation_falls_back_when_fast_prefill_loses_decode_speed(self):
        t = self.bare()
        t.close = lambda: None

        def evaluate(state, label, **kwargs):
            return dict(
                state=state, valid=True, decode=80 if state["prefill"] == 4096 else 100
            )

        t.evaluate = evaluate
        candidates = [
            dict(state={"prefill": 4096}, prefill=500),
            dict(state={"prefill": 512}, prefill=200),
        ]
        winner, short, long, floor = t.validate_final(
            candidates, dict(state={"prefill": 512}, decode=100)
        )
        self.assertEqual(winner["state"]["prefill"], 512)
        self.assertEqual(floor, 90)

    def test_search_finds_low_worker_optimum_visible_only_after_gpu_split(self):
        # At full GPU share every worker count ties. Once the GPU share is tuned,
        # eight workers win; a worker-first sweep plus local refinement misses it.
        t = self.bare()
        t.cfg.pop("tune_mtp")
        t.total = 10000
        t.target = 0.95
        t.reserve_vram_mib = 0
        t.safety_mib = 500
        t.background_mib = 0
        t.close = lambda: None

        def evaluate(state, label, **kwargs):
            split = abs(state["pcie"] - 0.1) < 1e-8
            speed = 100 if split else 50
            if split and state["workers"] == 8:
                speed += 50
            return dict(
                state=dict(state),
                decode=speed,
                prefill=state["prefill"],
                peak=9500,
                valid=True,
            )

        t.evaluate = evaluate
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            t.out = Path(td)
            with patch(
                "unleashed_tune.cpu_capacity", return_value=set(range(32))
            ), patch("unleashed_tune.physical_workers", return_value=23):
                report = t.tune()
            self.assertEqual(report["selected"]["workers"], 8)
            self.assertAlmostEqual(report["selected"]["pcie"], 0.1)
            self.assertGreaterEqual(
                report["final_short"]["decode"], report["decode_floor_tps"]
            )

    def test_stale_fast_baseline_cannot_hide_current_improvement(self):
        t = self.bare()
        stale = dict(state={"workers": 4}, decode=100, valid=True)
        t.evaluate = lambda st, label: dict(
            state=st, decode=50 if st["workers"] == 4 else 60, valid=True
        )
        self.assertEqual(t.coordinate(stale, "workers", [8])["state"]["workers"], 8)


class LauncherTests(unittest.TestCase):
    def test_any_shard_resolves_and_missing_shard_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            p = Path(td)
            paths = [p / f"model-{i:05}-of-00003.gguf" for i in range(1, 4)]
            for path in paths:
                path.touch()
            self.assertEqual(unleashed.split_paths(paths[2]), paths)
            paths[1].unlink()
            with self.assertRaises(FileNotFoundError):
                unleashed.split_paths(paths[2])

    def test_profile_header_and_rank_table_agree(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            p = Path(td) / "profile.bin"
            unleashed.make_profile(p, 3, 4)
            data = p.read_bytes()
            self.assertEqual(data[:4], b"STRP")
            self.assertEqual(struct.unpack_from("<5I", data, 4), (1, 3, 4, 12, 12))
            for rank in range(12):
                layer, expert = struct.unpack_from("<2H", data, 24 + rank * 4)
                self.assertEqual(
                    struct.unpack_from("<i", data, 24 + 48 + (layer * 4 + expert) * 4)[
                        0
                    ],
                    rank,
                )

    def test_profile_rejects_wrong_expert_count_and_truncation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            path = Path(td) / "profile.bin"
            unleashed.make_profile(path, 3, 4)
            self.assertTrue(unleashed.profile_matches(path, 3, 4))
            self.assertFalse(unleashed.profile_matches(path, 3, 2))
            path.write_bytes(path.read_bytes()[:-4])
            self.assertFalse(unleashed.profile_matches(path, 3, 4))

    def test_cached_launcher_selects_real_engine(self):
        import json

        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            root = Path(td)
            config = root / "config.json"
            config.write_text(json.dumps({"exe": sys.executable, "port": 8123}))
            out = root / "work/autotune/test"
            out.mkdir(parents=True)
            (out / "best-config.json").write_text("{}")
            (out / "RESULTS.json").write_text("{}")
            with patch.object(unleashed, "ROOT", root), patch.object(
                unleashed, "configure_environment"
            ), patch.object(unleashed, "signature", return_value=("test", {})), patch.object(
                unleashed, "run"
            ) as run, patch.object(sys, "argv", ["unleashed.py", "--config", str(config), "--port", "8124"]):
                self.assertEqual(unleashed.main(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--engine") + 1], "strata")
            self.assertEqual(command[command.index("--port") + 1], "8124")
            self.assertEqual(run.call_count, 1)

    def test_signature_invalidates_for_model_or_engine_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            p = Path(td)
            model = p / "model.gguf"
            model.write_bytes(b"weights")
            exe = p / "engine"
            exe.write_bytes(b"engine")
            pack = p / "pack"
            pack.mkdir()
            (pack / "dense.bin").write_bytes(b"pack")
            cfg = {
                "exe": str(exe),
                "args": ["--native", str(model), "--pack", str(pack)],
            }
            with patch.object(unleashed, "gpu_identity", return_value="test GPU"):
                key, _ = unleashed.signature(cfg)
                self.assertEqual(unleashed.signature(cfg)[0], key)
                model.write_bytes(b"changed weights")
                self.assertNotEqual(unleashed.signature(cfg)[0], key)
                key, _ = unleashed.signature(cfg)
                cfg["unleashed_tuning"] = {"reserve_vram_mib": 2048}
                self.assertNotEqual(unleashed.signature(cfg)[0], key)
                key, _ = unleashed.signature(cfg)
                exe.write_bytes(b"changed engine")
                self.assertNotEqual(unleashed.signature(cfg)[0], key)


if __name__ == "__main__":
    unittest.main()
