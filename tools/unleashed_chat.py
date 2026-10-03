"""Terminal inference testing with user/assistant messages and no system prompt."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import unleashed
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, child_env
from tools.strata_tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def render_prompt(template, messages):
    """Use the model's own turn format, but forbid added system instructions."""
    if any(message.get("role") not in ("user", "assistant") for message in messages):
        raise ValueError("Test chat accepts only user and assistant messages.")
    prompt = template.render(messages, tools=None, enable_thinking=False)
    if "<|im_start|>system" in prompt:
        raise ValueError(
            "This template inserted a system message. Test chat refuses to add hidden instructions."
        )
    return prompt


def reply(
    engine, tokenizer, template, messages, max_tokens=512, temperature=0.0, output=print
):
    prompt = render_prompt(template, messages)
    ids = tokenizer.encode(prompt, parse_special=True)
    if len(ids) + max_tokens > engine.max_context:
        raise ValueError(
            "The conversation is too long. Type /new or lower /tokens before trying again."
        )
    tokens, visible = [], ""
    started = time.monotonic()
    stream = engine.generate(
        ids, max_tokens, {"temperature": temperature}, threading.Event()
    )
    try:
        for token in stream:
            if token is None:
                continue
            tokens.append(token)
            text = tokenizer.decode(tokens, errors="ignore")
            for marker in END_TOKENS:
                text = text.replace(marker, "")
            if text.startswith(visible):
                output(text[len(visible) :], end="", flush=True)
            visible = text
    finally:
        stream.close()
    last = dict(engine.last)
    seconds = time.monotonic() - started
    # Engine timings distinguish generation from loading/prompt processing; chunks are not tokens.
    decode_ms = last.get("decode_ms") or 0
    stats = {
        "input_tokens": len(ids),
        "output_tokens": len(tokens),
        "elapsed_seconds": seconds,
        "decode_tokens_per_second": (
            len(tokens) * 1000 / decode_ms if decode_ms > 0 else None
        ),
        "prompt_ms": last.get("prompt_ms"),
        "finish_reason": last.get("finish_reason", "complete"),
        "system_messages": 0,
    }
    return visible, stats


def session(
    config, prompt=None, max_tokens=512, temperature=0.0, input_fn=input, output=print
):
    cfg = (
        json.loads(Path(config).read_text(encoding="utf-8"))
        if not isinstance(config, dict)
        else config
    )
    folder = Path(cfg["tokenizer"])
    tokenizer = Tokenizer.from_gguf(cfg["args"][cfg["args"].index("--native") + 1])
    template_path = folder / "chat_template.jinja"
    template = ChatTemplate(
        template_path if template_path.exists() else ROOT / "serve/chat_template.jinja"
    )
    render_prompt(
        template, [{"role": "user", "content": "Hello"}]
    )  # Fail before loading weights if template adds a system role.
    engine = StrataEngine(
        cfg["exe"],
        cfg["args"],
        cwd=cfg.get("cwd", str(ROOT)),
        log=str(ROOT / "work/chat-engine.log"),
        env=child_env(cfg),
    )
    messages = []
    output("\nReady to chat. No system prompt. Thinking instructions are off.")
    output(
        "Type a message. /new starts over, /tokens N sets the reply limit, /quit closes the model."
    )
    try:
        while True:
            try:
                text = prompt if prompt is not None else input_fn("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                output("")
                return 0
            if not text:
                if prompt is not None:
                    return 0
                continue
            if prompt is None and text in ("/quit", "/exit"):
                return 0
            if prompt is None and text in ("/new", "/reset"):
                messages.clear()
                output("New conversation.")
                continue
            if prompt is None and text.startswith("/tokens "):
                try:
                    value = int(text.split(maxsplit=1)[1])
                    if not 1 <= value <= engine.max_context:
                        raise ValueError()
                    max_tokens = value
                    output(f"Reply limit: {value} tokens.")
                except ValueError:
                    output(
                        "Use /tokens followed by a positive number within the context limit."
                    )
                continue
            if prompt is None and text == "/help":
                output(
                    "/new  /tokens N  /quit. Only your messages and the model replies enter the conversation."
                )
                continue
            current = messages + [{"role": "user", "content": text}]
            output("model> ", end="", flush=True)
            try:
                answer, stats = reply(
                    engine,
                    tokenizer,
                    template,
                    current,
                    max_tokens,
                    temperature,
                    output,
                )
            except ValueError as exc:
                output(str(exc))
                if prompt is not None:
                    return 2
                continue
            except KeyboardInterrupt:
                output("\nReply stopped.")
                return 130
            messages = current + [{"role": "assistant", "content": answer}]
            speed = stats["decode_tokens_per_second"]
            detail = f", {speed:.1f} tokens/s generation" if speed is not None else ""
            output(
                f'\n[{stats["output_tokens"]} tokens, {stats["elapsed_seconds"]:.2f}s total{detail}]'
            )
            if prompt is not None:
                return 0
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="A prepared best-config.json"
    )
    parser.add_argument("--prompt", help="Run one inference test and exit")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_tokens < 1 or not 0 <= args.temperature <= 2:
        parser.error("Require positive --max-tokens and --temperature between 0 and 2")
    unleashed.configure_environment()
    return session(args.config, args.prompt, args.max_tokens, args.temperature)


if __name__ == "__main__":
    raise SystemExit(main())
