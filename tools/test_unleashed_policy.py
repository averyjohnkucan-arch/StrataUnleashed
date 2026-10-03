"""Regression checks for capacity defaults and full-context tuning acceptance."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.unleashed_policy import (
    NATIVE_CONTEXT,
    PREFILL_TOKENS,
    default_kv,
    native_args,
)
from tools.unleashed_tune import Tuner

ROOT = Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def test_nominal_card_sizes_and_driver_overhead(self):
        for mib, pair in (
            (24576, "FP16/FP16"),
            (16384, "FP16/FP16"),
            (16376, "FP16/FP16"),
            (12288, "FP16/Q8"),
            (12272, "FP16/Q8"),
            (10240, "Q8/Q8"),
            (8192, "Q8/Q8"),
            (8176, "Q8/Q8"),
            (6144, "Q8/Q6"),
            (4096, "Q8/Q6"),
        ):
            with self.subTest(mib=mib):
                self.assertEqual(default_kv(mib), pair)
        with self.assertRaises(ValueError):
            default_kv(0)

    def test_imported_short_context_is_normalized_without_mutating_input(self):
        args = ["--max-context", "16384", "--kv", "Q5/Q4", "--rope-scaling", "yarn"]
        result = native_args(args)
        self.assertEqual(result[result.index("--max-context") + 1], "262144")
        self.assertEqual(args[1], "16384")
        self.assertEqual(result[result.index("--rope-scaling") + 1], "none")
        from serve.server import CTX_SLACK

        self.assertEqual(PREFILL_TOKENS + 512 + CTX_SLACK, NATIVE_CONTEXT)

    def test_terminal_chat_rejects_old_reduced_context_profile(self):
        from tools.unleashed_chat import session

        with self.assertRaisesRegex(ValueError, "native 262144"):
            session({"args": ["--max-context", "16384"]}, prompt="Hello")

    def test_full_context_failure_never_creates_a_validated_profile(self):
        t = Tuner.__new__(Tuner)
        t.close = lambda: None
        calls = []

        def evaluate(state, label, **kwargs):
            calls.append(kwargs.get("n", 512))
            return dict(
                state=state, decode=100, valid=kwargs.get("n", 512) != PREFILL_TOKENS
            )

        t.evaluate = evaluate
        with self.assertRaises(RuntimeError):
            t.validate_final([dict(state={}, prefill=1000)], dict(state={}, decode=100))
        self.assertIn(PREFILL_TOKENS, calls)

    def test_search_keeps_capacity_precision_and_native_allocation(self):
        t = Tuner.__new__(Tuner)
        t.cfg = {"args": ["--max-context", "16384"]}
        t.total, t.target, t.safety_mib = 6144, 0.9, 512
        t.reserve_vram_mib = t.background_mib = 0
        t.close = lambda: None
        trials = []

        def evaluate(state, label, **kwargs):
            trials.append((dict(state), kwargs.get("n", 512)))
            args = t.args(state)
            self.assertEqual(args[args.index("--max-context") + 1], "262144")
            return dict(
                state=dict(state),
                valid=True,
                decode=100,
                prefill=state["prefill"],
                peak=5529,
            )

        t.evaluate = evaluate
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as td:
            t.out = Path(td)
            with patch("tools.unleashed_tune.cpu_capacity", return_value={0, 1}), patch(
                "tools.unleashed_tune.physical_workers", return_value=1
            ):
                result = t.tune()
        self.assertTrue(all(s["kv"] == "Q8/Q6" for s, _ in trials))
        self.assertTrue(any(n == PREFILL_TOKENS for _, n in trials))
        self.assertEqual(result["context_tokens"], NATIVE_CONTEXT)


if __name__ == "__main__":
    unittest.main()
