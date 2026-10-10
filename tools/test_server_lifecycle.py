"""Lazy engine admission and failed startup cleanup, using tiny portable children."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server


class LifecycleTests(unittest.TestCase):
    def spawn(self, script):
        children = []
        def popen(_label, _args, **kwargs):
            child = subprocess.Popen([sys.executable, '-c', script], **kwargs)
            children.append(child)
            return child
        return children, popen

    def test_lazy_restart_preserves_admission_condition(self):
        children, spawn = self.spawn('import sys\nprint("READY 4096 stop", flush=True)\nfor line in sys.stdin:\n if line.strip() == "QUIT": break\n')
        engine = server.StrataEngine('fixture', [], lazy=True)
        condition = engine.slot_cv
        with patch.object(server, 'popen', side_effect=spawn):
            try:
                engine.restart(tries=1)
                self.assertTrue(engine.alive())
                self.assertIs(engine.slot_cv, condition)
            finally:
                engine.close()
        self.assertIsNotNone(children[0].poll())

    def test_failed_startup_reaps_child(self):
        children, spawn = self.spawn('raise SystemExit(1)')
        with patch.object(server, 'popen', side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError, 'before it was ready'):
                server.StrataEngine('fixture', [])
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(children[0].stdin.closed)
        self.assertTrue(children[0].stdout.closed)

    def test_lazy_busy_failure_is_not_masked_or_retried(self):
        engine = server.StrataEngine('fixture', [], lazy=True)
        with patch.object(engine, '__init__', side_effect=server.GpuBusy('occupied')) as start:
            with self.assertRaisesRegex(server.GpuBusy, 'occupied'):
                engine.restart()
        start.assert_called_once()
        self.assertFalse(engine.starting)
        self.assertFalse(engine.alive())


if __name__ == '__main__':
    unittest.main()
