"""nebula/commands.py: a tier that runs a command (docs/DESIGN.md, plan step F: external destinations).

A CLI installed on this machine, Claude Code, Codex or Grok, is a model the cockpit can route to like any other:
the conversation is rendered as a prompt, the command runs with it (its own account, its own tools, in a working
directory of the tier's), and what it prints is the reply. Each known CLI has a form here: how the prompt and the
system text are passed, what the model flag is, how the answer is read back. A command of the owner's own gets
the prompt on its standard input and its standard output is the answer.

A `CommandEngine` has the surface `cockpit.engine.Engine` has, so the router, the registry, admission and the
ledger see one more node: `/health` says whether the binary is there, `/props` reports its slots (how many may
run at once), a chat runs the command once and answers in the OpenAI shape. Nothing streams until the command
exits, since none of the CLIs prints a partial answer in a form worth parsing; the stream is the whole reply as
one chunk. Tools in the request are not passed on: the CLI has its own.

A command tier stays inside the owner's own Nebula: it runs under the owner's account, so it is never offered to a
paired machine and a paired machine's request that names one is refused (the owner's rule, 2026-10-16).
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from nebula import cmdline, names
from nebula.engine import EngineError
from nebula.sandbox import Sandbox

CLI_FORMS: dict[str, dict] = {
    "claude": {"label": "Claude Code", "binary": "claude"},
    "codex": {"label": "Codex CLI", "binary": "codex"},
    "grok": {"label": "Grok", "binary": "grok"},
    "jev": {"label": "Jev", "binary": "jev", "decisions_only": True},   # TypeSafe's Jev: typed decisions, never free text
}
DEFAULT_SLOTS = 2
STDERR_TAIL = 600


# ------------------------------------------------------------------------------------------------ the prompt
def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return "" if content is None else str(content)


def prompt_from_messages(messages: list[dict]) -> tuple[str | None, str]:
    """(system text, prompt): one user message is the prompt itself; a longer conversation is rendered as a
    transcript the command is asked to continue."""
    system, rest = [], []
    for m in messages or []:
        role = m.get("role")
        if role == "system" or role == "developer":
            system.append(_text_of(m.get("content")))
        else:
            rest.append(m)
    sys_text = "\n\n".join(s for s in system if s) or None
    if len(rest) == 1 and rest[0].get("role") == "user":
        return sys_text, _text_of(rest[0].get("content"))
    lines = ["The conversation so far; write the assistant's next turn and nothing else.", ""]
    for m in rest:
        role = m.get("role") or "user"
        if role == "tool":
            lines.append(f"### tool result ({m.get('name') or m.get('tool_call_id') or 'tool'})")
        else:
            lines.append(f"### {role}")
        text = _text_of(m.get("content"))
        if m.get("tool_calls"):
            text = (text + "\n" if text else "") + "(called tools: " + json.dumps(m["tool_calls"], ensure_ascii=False)[:2000] + ")"
        lines.append(text)
        lines.append("")
    lines.append("### assistant")
    return sys_text, "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the forms
def build_command(cli: str | None, command, model: str | None, system: str | None, prompt_file: str,
                  output_file: str) -> tuple[list[str], bool, str]:
    """(argv, prompt on stdin?, how the answer is read: "stdout" | "json" | "file"). `{prompt_file}`, `{output_file}`
    and `{model}` in an owner's command are filled in; without any, its prompt comes on stdin."""
    if cli == "claude":
        argv = ["claude", "-p", "--output-format", "json", "--no-session-persistence"]
        if model:
            argv += ["--model", model]
        if system:
            argv += ["--system-prompt", system]
        return argv, True, "json"
    if cli == "codex":
        argv = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "--color", "never", "-o", output_file]
        if model:
            argv += ["-m", model]
        return argv + ["-"], True, "file"             # the system text is in the prompt: codex has no flag for it
    if cli == "grok":
        argv = ["grok", "--prompt-file", prompt_file, "--output-format", "plain", "--permission-mode", "dontAsk"]
        if model:
            argv += ["-m", model]
        if system:
            argv += ["--system-prompt-override", system]
        return argv, False, "stdout"
    if cli == "jev":
        raise ValueError("Jev answers typed decisions only (a choice, a score, yes or no), never a chat: it is offered "
                         "as the smart router's judge (Routes), not as a model to talk to")
    if cli:
        raise ValueError(f"no form for the CLI {cli!r}: one of {sorted(CLI_FORMS)}")
    words = cmdline.split(command) if isinstance(command, str) else [str(w) for w in (command or [])]
    if not words:
        raise ValueError("a command tier needs a cli or a command")
    filled = [w.replace("{prompt_file}", prompt_file).replace("{output_file}", output_file).replace("{model}", model or "") for w in words]
    uses_file = any("{prompt_file}" in w for w in words)
    return filled, not uses_file, "file" if any("{output_file}" in w for w in words) else "stdout"


def parse_claude_json(out: str) -> tuple[str, dict | None]:
    """Claude Code's --output-format json: the result text and its usage, or the text as it is."""
    try:
        obj = json.loads(out)
    except ValueError:
        return out.strip(), None
    if isinstance(obj, list):
        obj = next((o for o in reversed(obj) if isinstance(o, dict) and o.get("type") == "result"), obj[-1] if obj else {})
    if not isinstance(obj, dict):
        return out.strip(), None
    text = obj.get("result")
    if not isinstance(text, str):
        text = out.strip()
    u = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    usage = None
    if u:
        usage = {"prompt_tokens": int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0),
                 "completion_tokens": int(u.get("output_tokens") or 0)}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        if obj.get("total_cost_usd") is not None:
            usage["cost_usd"] = obj["total_cost_usd"]
    return text, usage


# ------------------------------------------------------------------------------------------------ the engine
class CommandEngine:
    """A tier that runs a command. The same surface as Engine, for the cockpit's other parts."""

    def __init__(self, tier, timeout_s: float = 600.0, base_dir: str | None = None):
        self.tier = tier
        self.cli = tier.cli
        self.command = tier.command
        self.model = tier.model
        self.context = tier.context
        self.timeout_s = timeout_s
        self.slots = int(tier.slots or DEFAULT_SLOTS)
        self.cwd = tier.cwd or names.local_path(base_dir, "nebula-work", "cockpit-work")
        self.sandbox = Sandbox(tier.name, dict(tier.sandbox, cli=tier.cli), base_dir, timeout_s) if tier.sandbox else None
        self.base = tier.address or f"cli:{self.binary}"
        self.api_key = None
        self._props = None
        self._no_tokenize = True
        self.running = 0
        self.lock = threading.Lock()
        self.last: dict = {}

    @property
    def binary(self) -> str:
        if self.cli:
            return CLI_FORMS.get(self.cli, {}).get("binary", self.cli)
        words = cmdline.split(self.command) if isinstance(self.command, str) else list(self.command or [])
        return str(words[0]) if words else ""

    @property
    def label(self) -> str:
        return CLI_FORMS.get(self.cli, {}).get("label", self.binary) if self.cli else self.tier.name

    def installed(self) -> str | None:
        """Where the CLI is: on this machine's PATH, or (sandboxed) `openshell` on the PATH, the CLI being the
        image's; None when the tier cannot run."""
        if self.sandbox is not None:
            return self.sandbox.binary()
        b = self.binary
        if not b:
            return None
        return shutil.which(b) or (b if os.path.isabs(b) and os.access(b, os.X_OK) else None)

    def unavailable(self) -> str | None:
        """Why the tier is down, or None."""
        if self.sandbox is not None:
            st = self.sandbox.status()
            if not st["installed"]:
                return "openshell is not installed on this machine"
            if st["exists"] and not st["running"]:
                return f"the sandbox {self.sandbox.name} is stopped"
            return None
        return None if self.installed() else f"{self.binary!r} is not installed on this machine"

    # --- the Engine surface
    def get(self, path: str) -> tuple[int, str, bytes]:
        path = path.split("?", 1)[0]
        if path == "/health":
            why = self.unavailable()
            return (200 if why is None else 503), "application/json", json.dumps({"status": "ok" if why is None else why, "binary": self.binary}).encode()
        if path == "/props":
            p = {"model_alias": self.model or self.label, "total_slots": self.slots, "command": True, "binary": self.installed() or self.binary}
            if self.sandbox is not None:
                p["sandbox"] = {"name": self.sandbox.name, "image": self.sandbox.image}
            if self.context:
                p["default_generation_settings"] = {"n_ctx": self.context}
            return 200, "application/json", json.dumps(p).encode()
        if path == "/slots":
            return 200, "application/json", json.dumps([{"id": i, "is_processing": i < self.running} for i in range(self.slots)]).encode()
        if path in ("/v1/models", "/models"):
            return 200, "application/json", json.dumps({"object": "list", "data": [{"id": self.model or self.label, "object": "model"}]}).encode()
        return 404, "application/json", json.dumps({"error": {"message": f"a command tier has no {path}", "type": "invalid_request_error"}}).encode()

    def props(self) -> dict:
        if self._props is None:
            self._props = json.loads(self.get("/props")[2])
        return self._props

    def n_ctx(self) -> int | None:
        return self.context or None

    def busy(self) -> int:
        return self.running

    def tokenize(self, text: str):
        return None

    def run(self, messages: list[dict]) -> tuple[str, dict, float]:
        """Run the command once with the conversation: (reply text, usage, seconds)."""
        why = self.unavailable()
        if why is not None:
            raise EngineError(503, {"error": {"message": f"{self.label}: {why}", "type": "server_error"}})
        system, prompt = prompt_from_messages(messages)
        if self.sandbox is not None:
            return self._run_sandboxed(system, prompt)
        os.makedirs(self.cwd, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nebula-cmd-") as tmp:
            prompt_file, output_file = os.path.join(tmp, "prompt.txt"), os.path.join(tmp, "answer.txt")
            if self.cli == "codex" and system:
                prompt = f"System instructions:\n{system}\n\n{prompt}"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            argv, on_stdin, how = build_command(self.cli, self.command, self.model, system, prompt_file, output_file)
            if argv and not os.path.isabs(argv[0]):
                argv[0] = self.installed() or argv[0]
            env = dict(os.environ, NO_COLOR="1", TERM="dumb")
            started = time.monotonic()
            with self.lock:
                self.running += 1
            try:
                proc = subprocess.run(argv, input=prompt.encode("utf-8") if on_stdin else None, cwd=self.cwd, env=env,
                                      capture_output=True, timeout=self.timeout_s, check=False)
            except subprocess.TimeoutExpired:
                raise EngineError(504, {"error": {"message": f"{self.label} did not finish in {int(self.timeout_s)} s", "type": "server_error"}}) from None
            except OSError as e:
                raise EngineError(502, {"error": {"message": f"{self.label} could not run: {e}", "type": "server_error"}}) from None
            finally:
                with self.lock:
                    self.running -= 1
            seconds = time.monotonic() - started
            out = proc.stdout.decode("utf-8", "replace")
            err = proc.stderr.decode("utf-8", "replace")
            self.last = {"argv": argv[:6], "rc": proc.returncode, "seconds": round(seconds, 2), "stderr": err[-STDERR_TAIL:]}
            if proc.returncode != 0:
                raise EngineError(502, {"error": {"message": f"{self.label} exited {proc.returncode}: {(err or out).strip()[-STDERR_TAIL:]}", "type": "server_error"}})
            usage = None
            if how == "json":
                text, usage = parse_claude_json(out)
            elif how == "file":
                try:
                    with open(output_file, encoding="utf-8") as f:
                        text = f.read()
                except OSError:
                    text = out
            else:
                text = out
            text = text.strip()
            if usage is None:
                usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4, "estimated": True}
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            return text, usage, seconds

    def _run_sandboxed(self, system: str | None, prompt: str) -> tuple[str, dict, float]:
        """The same call inside the tier's OpenShell sandbox: the prompt and answer files live in the sandbox's /tmp."""
        work = "/tmp/nebula-" + uuid.uuid4().hex[:10]
        prompt_file, output_file = work + "/prompt.txt", work + "/answer.txt"
        if self.cli == "codex" and system:
            prompt = f"System instructions:\n{system}\n\n{prompt}"
        argv, on_stdin, how = build_command(self.cli, self.command, self.model, system, prompt_file, output_file)
        started = time.monotonic()
        with self.lock:
            self.running += 1
        try:
            rc, out, err = self.sandbox.run(argv, prompt, on_stdin, how, prompt_file, output_file, timeout=self.timeout_s)
        finally:
            with self.lock:
                self.running -= 1
        seconds = time.monotonic() - started
        self.last = {"argv": argv[:6], "rc": rc, "seconds": round(seconds, 2), "stderr": err[-STDERR_TAIL:], "sandbox": self.sandbox.name}
        if rc != 0:
            raise EngineError(502, {"error": {"message": f"{self.label} (in {self.sandbox.name}) exited {rc}: {(err or out).strip()[-STDERR_TAIL:]}", "type": "server_error"}})
        usage = None
        if how == "json":
            text, usage = parse_claude_json(out)
        else:
            text = out
        text = text.strip()
        if usage is None:
            usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4, "estimated": True}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return text, usage, seconds

    def chat(self, body: dict) -> dict:
        text, usage, seconds = self.run(body.get("messages") or [])
        return {"id": "chatcmpl-cmd-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                "model": self.model or self.label,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": usage, "timings": {"predicted_ms": round(seconds * 1000)}}

    def chat_stream(self, body: dict):
        obj = self.chat(body)
        msg = obj["choices"][0]["message"]
        yield {"id": obj["id"], "object": "chat.completion.chunk", "created": obj["created"], "model": obj["model"],
               "choices": [{"index": 0, "delta": {"role": "assistant", "content": msg["content"]}, "finish_reason": "stop"}],
               "usage": obj["usage"], "timings": obj["timings"]}
