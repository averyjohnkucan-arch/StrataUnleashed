"""Measured coordinate search: maximize 512/512 decode, then prefill at >=90% decode.
All trial configurations, outputs, timings and telemetry are retained under --out.
"""

from __future__ import annotations
import argparse, copy, json, math, os, statistics, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.server import StrataEngine, child_env
from strata_tokenizer import Tokenizer
from calibrate import with_arg

from tools.unleashed_policy import (
    NATIVE_CONTEXT,
    PREFILL_TOKENS,
    KV_FORMATS,
    default_kv,
    native_args,
)

GPU_SELECTOR = "0"
KV = KV_FORMATS


def atomic(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def cpu_capacity():
    return (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set(range(os.cpu_count() or 2))
    )


def physical_workers():
    try:
        import psutil

        return max(
            1,
            min(
                len(cpu_capacity()) - 1,
                (psutil.cpu_count(logical=False) or len(cpu_capacity())) - 1,
            ),
        )
    except ImportError:
        return max(1, len(cpu_capacity()) - 1)


def gpu():
    r = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            GPU_SELECTOR,
            "--query-gpu=memory.total,memory.used,power.draw,clocks.sm,temperature.gpu,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    values = []
    for v in r.splitlines()[0].split(","):
        try:
            values.append(float(v.strip()))
        except ValueError:
            values.append(None)
    if len(values) < 2 or values[0] is None or values[1] is None:
        raise RuntimeError(
            "GPU memory telemetry is unavailable; cannot enforce the VRAM target"
        )
    return values


def exclusive(owner=None):
    r = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            GPU_SELECTOR,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    others = [int(p) for p in r.split() if p.isdigit() and int(p) != owner]
    if others:
        raise RuntimeError(f"GPU has other compute processes: {others}")


def prompt(tok, n, variant=0):
    start = tok.encode("<|im_start|>user\n", parse_special=True)
    task = (
        "Write a detailed tutorial on how operating systems manage virtual memory. Include worked examples and pseudocode. Continue for at least 1500 words. "
        if variant == 0
        else "Write a detailed Python programming tutorial on sorting and searching. Include several functions, tests, and explanations. Continue for at least 1500 words. "
    )
    end = tok.encode(
        "\nUse the notes above as background. Now write the tutorial in full.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        parse_special=True,
    )
    needed = n - len(start) - len(end)
    if needed < 1:
        raise ValueError("Benchmark prompt is shorter than its chat template")
    body = tok.encode(task)
    notes = tok.encode(
        "Background notes: caches trade space for speed; correctness should be tested; measure before optimizing. "
    )
    body += notes * max(0, math.ceil((needed - len(body)) / len(notes)))
    result = start + body[:needed] + end
    assert len(result) == n
    return result


class Tuner:
    def __init__(self, cfg, out, repeats=2, target=None, reserve_vram_mib=0):
        global GPU_SELECTOR
        GPU_SELECTOR = str(cfg.get("gpu", 0))
        if "," in GPU_SELECTOR or isinstance(cfg.get("gpu"), list):
            raise ValueError("Self-tuning currently supports one NVIDIA GPU at a time")
        from tools.gguf_reader import GGUFFile

        native = cfg["args"][cfg["args"].index("--native") + 1]
        metadata = GGUFFile(native).metadata
        trained_context = metadata.get("qwen4exp.context_length")
        if trained_context is None or int(trained_context) < NATIVE_CONTEXT:
            raise ValueError(
                "Model metadata does not support native 262144-token context; tuning will not reduce context or apply RoPE scaling"
            )
        self.lock = (ROOT / "work/gpu.lock").open("a+b")
        if os.name == "nt":
            import msvcrt

            self.lock.seek(0)
            self.lock.write(b"0")
            self.lock.flush()
            self.lock.seek(0)
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = dict(cfg, args=native_args(cfg["args"]))
        self.cfg = cfg
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        if (out / "measurements.json").exists():
            archive = out / "history" / str(time.time_ns())
            archive.mkdir(parents=True)
            for p in list(out.iterdir()):
                if p.is_file() and (
                    p.name.startswith("engine-")
                    or p.name
                    in (
                        "measurements.json",
                        "telemetry.json",
                        "decode-best.json",
                        "RESULTS.json",
                        "best-config.json",
                        "memory-budget.json",
                        "prefill-candidates.json",
                    )
                ):
                    p.rename(archive / p.name)
        self.repeats = repeats
        memory = gpu()
        self.total = memory[0]
        self.background_mib = memory[1]
        self.reserve_vram_mib = reserve_vram_mib
        self.safety_mib = max(512, math.ceil(self.total * 0.02))
        budget = self.total - reserve_vram_mib - self.safety_mib
        if target is not None:
            budget = min(budget, self.total * target)
        if budget - self.background_mib < 1024:
            raise ValueError("VRAM reservation leaves less than 1 GiB for the engine")
        self.target = budget / self.total
        atomic(
            out / "memory-budget.json",
            {
                "total_mib": self.total,
                "background_mib": self.background_mib,
                "reserved_for_other_apps_mib": reserve_vram_mib,
                "safety_mib": self.safety_mib,
                "peak_limit_mib": budget,
                "target_fraction": self.target,
            },
        )
        self.eng = None
        self.records = []
        self.seq = 0
        self.state = None
        self.peak = 0
        self.stopping = threading.Event()
        self.telemetry = []
        self.active_trial = None
        self.last_progress = 0.0
        self.tok = Tokenizer.from_gguf(cfg["args"][cfg["args"].index("--native") + 1])
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def monitor(self):
        while not self.stopping.wait(0.25):
            if self.eng is None:
                continue
            try:
                g = gpu()
                self.peak = max(self.peak, g[1])
                self.telemetry.append([time.time(), *g])
                trial = self.active_trial
                now = time.monotonic()
                if trial and now - self.last_progress >= 30:
                    label, started = trial
                    print(
                        f"{label}: running for {now - started:.0f}s; GPU memory {g[1]:.0f}/{self.total:.0f} MiB",
                        flush=True,
                    )
                    self.last_progress = now
            except Exception:
                pass

    def close(self):
        engine = self.eng
        self.eng = None
        if engine:
            engine.close()
            # Allow the CUDA driver to settle without telemetry polling between contexts.
            time.sleep(5)

    def finish(self):
        self.close()
        self.stopping.set()
        self.thread.join(timeout=5)
        atomic(self.out / "telemetry.json", self.telemetry)
        self.lock.close()

    def args(self, state):
        a = native_args([arg for arg in self.cfg["args"] if arg != "--no-mtp"])
        # Compare fresh prompts with adaptive cache sizing, including imported configs.
        for flag, value in (
            ("--max-context", str(NATIVE_CONTEXT)),
            ("--expert-cache", "auto"),
            ("--prompt-cache", "0"),
            ("--suffix-draft", "0"),
        ):
            a = with_arg(a, flag, value)
        for f, v in [
            ("--kv", state["kv"]),
            ("--pool-workers", state["workers"]),
            ("--prefill", state["prefill"]),
            ("--vram-reserve-mib", state["reserve"]),
            ("--pcie-frac", state["pcie"]),
            ("--spec-min-p", state["minp"]),
            ("--spec", max(2, state["spec"])),
        ]:
            a = with_arg(a, f, str(v))
        if state["spec"] == 0:
            a = with_arg(a, "--mtp", None)
        elif self.cfg.get("tune_mtp"):
            a = with_arg(a, "--mtp", self.cfg["tune_mtp"])
        else:
            raise ValueError("MTP candidate needs tune_mtp in config")
        return a

    def load(self, state):
        # Runtime-only coordinates use protocol overrides; changing allocations restarts.
        alloc = ("kv", "workers", "prefill", "reserve", "spec")
        if self.eng and self.state and all(state[k] == self.state[k] for k in alloc):
            return
        self.close()
        exclusive()
        self.seq += 1
        env = child_env(self.cfg)
        env.update(
            {
                k: os.environ[k]
                for k in (
                    "TMPDIR",
                    "CUDA_CACHE_PATH",
                    "XDG_CACHE_HOME",
                    "PYTHONDONTWRITEBYTECODE",
                )
            }
        )
        env["STRATA_DECODE_TIMING"] = "1"
        self.peak = 0
        print("LOAD", state, flush=True)
        self.eng = StrataEngine(
            self.cfg["exe"],
            self.args(state),
            cwd=str(ROOT),
            log=str(self.out / f"engine-{self.seq:03}.log"),
            env=env,
        )
        self.state = copy.deepcopy(state)
        self.request(state, 128, 32, "warmup", 0)

    def request(self, state, n, new, label, variant):
        exclusive(self.eng.proc.pid)
        ids = prompt(self.tok, n, variant)
        before = gpu()
        self.peak = before[1]
        t = time.monotonic()
        print(
            f"{label}: {n} input / {new} output tokens; context {NATIVE_CONTEXT}, KV {state['kv']}",
            flush=True,
        )
        self.active_trial = (label, t)
        self.last_progress = t
        try:
            tokens = [
                t
                for t in self.eng.generate(
                    ids,
                    new,
                    {
                        "temperature": 0,
                        "strata_tune": {
                            "pcie_frac": state["pcie"],
                            "spec_min_p": state["minp"],
                            "ignore_eos": 1,
                        },
                    },
                    threading.Event(),
                )
                if t is not None
            ]
        finally:
            self.active_trial = None
        last = dict(self.eng.last)
        after = gpu()
        self.peak = max(self.peak, after[1])
        exclusive(self.eng.proc.pid)
        if len(tokens) != new or not last.get("decode_ms", 0) > 0:
            raise RuntimeError(f"incomplete benchmark: {len(tokens)} != {new}; {last}")
        r = {
            "label": label,
            "state": dict(state),
            "context_tokens": NATIVE_CONTEXT,
            "input_tokens": n,
            "output_tokens": len(tokens),
            "decode_tps": len(tokens) * 1000 / last["decode_ms"],
            "prefill_tps": (n - 1) * 1000 / max(last.get("prompt_ms", 0), 0.001),
            "peak_vram_mib": self.peak,
            "vram_fraction": self.peak / self.total,
            "wall_s": time.monotonic() - t,
            "engine": last,
            "ids": tokens,
            "text": self.tok.decode(tokens),
            "variant": variant,
            "gpu_before": before,
            "gpu_after": after,
        }
        self.records.append(r)
        atomic(self.out / "measurements.json", self.records)
        print(
            label,
            round(r["decode_tps"], 2),
            "decode tok/s",
            round(r["prefill_tps"], 2),
            "prefill tok/s",
            round(r["vram_fraction"] * 100, 1),
            "% VRAM",
            flush=True,
        )
        return r

    def evaluate(self, state, label, n=512, repeats=None):
        try:
            self.load(state)
            rows = [
                self.request(state, n, 512, label, i % 2)
                for i in range(repeats or self.repeats)
            ]
            return {
                "state": dict(state),
                "decode": statistics.median(r["decode_tps"] for r in rows),
                "prefill": statistics.median(r["prefill_tps"] for r in rows),
                "peak": max(r["peak_vram_mib"] for r in rows),
                "valid": max(r["peak_vram_mib"] for r in rows)
                <= self.total * self.target,
            }
        except Exception as e:
            self.close()
            print("FAILED", label, str(e), flush=True)
            self.records.append(
                {"label": label, "state": dict(state), "error": repr(e)}
            )
            atomic(self.out / "measurements.json", self.records)
            return {
                "state": dict(state),
                "valid": False,
                "decode": 0,
                "prefill": 0,
                "error": repr(e),
            }

    def fit(self, state):
        # Fit the measured peak, retaining an allocation backoff floor after an OOM.
        state = dict(state)
        minimum = 256
        for i in range(8):
            r = self.evaluate(state, f"vram-fit-{i}", repeats=1)
            if "error" in r:
                if not any(
                    w in r["error"].lower() for w in ("memory", "fit", "alloc", "vram")
                ):
                    return r
                minimum = state["reserve"] + 512
                state["reserve"] = minimum
                continue
            delta = r["peak"] - self.total * self.target
            if r["valid"] and (-delta <= 128 or state["reserve"] <= minimum):
                return r
            adjusted = max(minimum, state["reserve"] + math.ceil(delta) + 64)
            if adjusted == state["reserve"]:
                return r
            state["reserve"] = adjusted
        return r

    def coordinate(self, best, key, values):
        # Hardware power/link conditions and adaptive residency may have changed.
        # Never let an old fast measurement suppress a faster current candidate.
        control = self.evaluate(best["state"], f"control-{key}")
        best = control
        candidates = [best] if best["valid"] else []
        for v in values:
            if v == best["state"][key]:
                continue
            state = dict(best["state"], **{key: v})
            if key in ("kv", "spec", "prefill"):
                fitted = self.fit(state)
                r = (
                    self.evaluate(fitted["state"], f"{key}-{v}-fitted")
                    if fitted["valid"]
                    else fitted
                )
            else:
                r = self.evaluate(state, f"{key}-{v}")
            if not r["valid"] and (
                "peak" in r
                or any(
                    w in r.get("error", "").lower() for w in ("memory", "alloc", "vram")
                )
            ):
                fitted = self.fit(state)
                if fitted["valid"]:
                    r = self.evaluate(fitted["state"], f"{key}-{v}-fitted")
            if r["valid"]:
                candidates.append(r)
        if not candidates:
            return best
        candidate = max(candidates, key=lambda r: r["decode"])
        if candidate is best:
            return best
        # Interleaved confirmation on both workloads guards order/warm-cache noise.
        scores = {0: [], 1: []}
        for i in (0, 1, 1, 0):
            scores[i].append(
                self.evaluate((best, candidate)[i]["state"], f"confirm-{key}-{i}")
            )
        if (
            all(r["valid"] for r in scores[1])
            and statistics.median(r["decode"] for r in scores[1])
            > statistics.median(r["decode"] for r in scores[0]) * 1.01
        ):
            candidate["decode"] = statistics.median(r["decode"] for r in scores[1])
            return candidate
        return best

    def tune(self, seed=None, coarse_workers=True):
        # A trusted caller may continue completed MTP comparisons on the same model.
        # The seed is remeasured; worker and PCIe refinement and all final checks still run.
        capacity = max(1, len(cpu_capacity()) - 1)
        initial = {
            "kv": self.cfg.get("unleashed_tuning", {}).get("kv")
            or default_kv(self.total),
            "workers": physical_workers(),
            "prefill": 512,
            "reserve": int(self.total * (1 - self.target)) + 600,
            "pcie": 1.0,
            "minp": 0.8,
            "spec": 0,
        }
        if seed is not None:
            initial = dict(seed, kv=initial["kv"])
        best = self.fit(initial)
        if best["valid"]:
            best = self.evaluate(best["state"], "baseline")
        if not best["valid"]:
            raise RuntimeError(
                "Could not fit the selected KV format at native 262144-token context within the VRAM budget; choose a smaller model, explicitly override --kv, or reduce the extra reservation"
            )
        coordinates = [
            (
                "workers",
                sorted(
                    {
                        1,
                        *[
                            max(1, round(capacity * f))
                            for f in (0.125, 0.25, 0.375, 0.5, 0.67, 0.83, 1.0)
                        ],
                    }
                ),
            ),
            ("spec", (0, 2, 3, 4) if self.cfg.get("tune_mtp") else (0,)),
            ("pcie", (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)),
            ("minp", (0.3, 0.5, 0.7, 0.8, 0.9) if self.cfg.get("tune_mtp") else (0.8,)),
        ]
        # Measure worker counts after the PCIe split: at 100% GPU share the CPU
        # workers may be idle, hiding a much faster low-thread-count setting.
        order = ("kv", "spec", "pcie", "workers", "minp")
        coordinates.sort(key=lambda item: order.index(item[0]))
        if seed is not None:
            coordinates = [
                item
                for item in coordinates
                if item[0] == "minp" or (item[0] == "workers" and coarse_workers)
            ]
        for key, values in coordinates:
            if key == "minp" and best["state"]["spec"] == 0:
                continue
            best = self.coordinate(best, key, values)
            atomic(self.out / "decode-best.json", best)
        # Procedural local refinement: halve each step after the neighborhood stops improving.
        for key, step, low, high in [
            ("workers", max(1, capacity // 8), 1, capacity),
            ("pcie", 0.1, 0.0, 1.0),
        ]:
            for iteration in range(3):
                center = best["state"][key]
                values = sorted(
                    {
                        max(low, min(high, center - step)),
                        max(low, min(high, center + step)),
                    }
                )
                previous = dict(best["state"])
                best = self.coordinate(best, key, values)
                if best["state"] == previous:
                    step = step // 2 if key == "workers" else step / 2
                    if step == 0:
                        break
        # Refine allocation after precision/drafter changes and revisit coupled coordinates.
        fitted = self.fit(best["state"])
        if fitted["valid"] and fitted["decode"] >= best["decode"] * 0.99:
            best = fitted
        best = self.coordinate(best, "pcie", (0.0, 0.05, 0.1, 0.2, 0.35, 0.5))
        baseline = self.evaluate(best["state"], "decode-baseline", repeats=3)
        atomic(self.out / "decode-best.json", baseline)
        # Tune on longer prompts but enforce the requested 512/512 decode floor after each.
        eligible = []
        for chunk in (256, 512, 1024, 2048, 4096, 8192):
            state = dict(baseline["state"], prefill=chunk)
            long = self.evaluate(state, f"prefill-{chunk}", n=PREFILL_TOKENS)
            if not long["valid"]:
                state["reserve"] += max(
                    512,
                    math.ceil(
                        long.get("peak", self.total * self.target)
                        - self.total * self.target
                    )
                    + 128,
                )
                long = self.evaluate(
                    state, f"prefill-{chunk}-backoff", n=PREFILL_TOKENS
                )
            short = self.evaluate(state, f"prefill-{chunk}-decode-check")
            if (
                long["valid"]
                and short["valid"]
                and short["decode"] >= baseline["decode"] * 0.9
            ):
                eligible.append(dict(short, prefill=long["prefill"]))
        if not eligible:
            raise RuntimeError(
                "No prefill candidate preserves the decode floor within the VRAM budget"
            )
        atomic(self.out / "prefill-candidates.json", eligible)
        winner, final, large, floor = self.validate_final(eligible, baseline)
        cfg = dict(
            self.cfg, args=self.args(winner["state"]), log=str(ROOT / "work/engine.log")
        )
        cfg.pop("tune_mtp", None)
        atomic(self.out / "best-config.json", cfg)
        report = {
            "context_tokens": NATIVE_CONTEXT,
            "prefill_input_tokens": PREFILL_TOKENS,
            "kv_policy": "explicit override or capacity default; fixed throughout tuning",
            "reserve_vram_mib": self.reserve_vram_mib,
            "safety_mib": self.safety_mib,
            "background_mib": self.background_mib,
            "target_vram_fraction": self.target,
            "decode_floor_tps": floor,
            "decode_baseline": baseline,
            "final_short": final,
            "final_long": large,
            "selected": winner["state"],
            "config": str(self.out / "best-config.json"),
        }
        atomic(self.out / "RESULTS.json", report)
        return report

    def validate_final(self, eligible, baseline):
        # Recheck the control, then try slower prefill candidates if the leader drifts.
        self.close()
        control = self.evaluate(baseline["state"], "final-control", repeats=3)
        if not control["valid"]:
            raise RuntimeError(
                "Final decode control failed; no configuration was accepted"
            )
        floor = 0.9 * max(control["decode"], baseline["decode"])
        for winner in sorted(eligible, key=lambda r: r["prefill"], reverse=True):
            self.close()
            final = self.evaluate(winner["state"], "final-512", repeats=3)
            if not final["valid"] or final["decode"] < floor:
                continue
            large = self.evaluate(
                winner["state"], "final-native-context", n=PREFILL_TOKENS, repeats=3
            )
            if large["valid"]:
                return winner, final, large, floor
        raise RuntimeError(
            "All final candidates failed the VRAM/decode floor; no configuration was accepted"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "work/tuning")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument(
        "--target-vram",
        type=float,
        help="Optional additional total-VRAM fraction cap (for example .8)",
    )
    ap.add_argument(
        "--reserve-vram-mib",
        type=int,
        default=0,
        help="Additional VRAM kept free for other apps, in addition to the automatic safety margin (default 0 MiB)",
    )
    a = ap.parse_args()
    out = a.out.resolve()
    out.relative_to(ROOT)
    if a.target_vram is not None and not 0.1 <= a.target_vram <= 1:
        ap.error("target VRAM must be between .1 and 1")
    if a.reserve_vram_mib < 0:
        ap.error("reserved VRAM must not be negative")
    if a.repeats < 1:
        ap.error("repeats must be positive")
    for d in ("tmp", "cache"):
        (ROOT / "work" / d).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(ROOT / "work/tmp"),
        CUDA_CACHE_PATH=str(ROOT / "work/cache"),
        XDG_CACHE_HOME=str(ROOT / "work/cache"),
        TEMP=str(ROOT / "work/tmp"),
        TMP=str(ROOT / "work/tmp"),
    )
    if os.name != "nt":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpu_capacity())
    cfg = json.loads(a.config.read_text())
    cfg["args"] = native_args(cfg["args"])
    t = Tuner(cfg, out, a.repeats, a.target_vram, a.reserve_vram_mib)
    try:
        if a.smoke:
            state = {
                "kv": cfg.get("unleashed_tuning", {}).get("kv") or default_kv(t.total),
                "workers": physical_workers(),
                "prefill": 512,
                "reserve": int(t.total * (1 - t.target)) + 600,
                "pcie": 0.0,
                "minp": 0.8,
                "spec": 0,
            }
            r = t.evaluate(state, "smoke", repeats=1)
            atomic(out / "RESULTS.json", r)
            if not r["valid"]:
                return 1
        else:
            t.tune()
    finally:
        t.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
