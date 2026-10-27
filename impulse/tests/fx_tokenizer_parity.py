#!/usr/bin/env python3
"""One Engine P1, gate P1a: llama.cpp's tokenizer (impulse-server's, a vocab-only load of the pack's dense GGUF) against
Strata's Python tokenizer (flashnext/tools/strata_tokenizer.py reading the pack's tokenizer/ folder, exactly as
serve/server.py builds it) on the same texts: the ids must be identical.  No GPU, no weights.

    python fx_tokenizer_parity.py --gguf DENSE_OR_META.gguf --tokenizer PACK/tokenizer --fx-tokenize build/bin/fx-tokenize
        [--mb 2] [--out summary.json]

The GGUF may be tools/gguf_meta_only.py's metadata-only copy of the dense GGUF (the tokenizer is all in its metadata).
Strata's tokenizer needs the `regex` module.  The corpus: strata_tokenizer.py's own round-trip corpus, edge cases
(scripts, combining marks, emoji sequences, whitespace runs, digits, control characters), at least --mb MB of real text
from this repository (docs, code, the translated READMEs), and chat-formatted prompts with special tokens (ChatML turns,
<think>, <tool_call>) built from that text.  Every text is tokenized with parse_special off and on.
"""
import argparse
import ast
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
IMPULSE = HERE.parent
FLASHNEXT = IMPULSE / "flashnext"


def strata_tokenizer(tdir: pathlib.Path):
    sys.path.insert(0, str(FLASHNEXT / "tools"))
    import strata_tokenizer as ST   # noqa: E402
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)   # serve/server.py's construction (main, --engine strata)


def roundtrip_corpus() -> list[str]:
    """The list strata_tokenizer.py's --check round-trips (read from its source, so it stays that list)."""
    tree = ast.parse((FLASHNEXT / "tools" / "strata_tokenizer.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "corpus" for t in node.targets):
            # a literal list but for one `"x" * 5000`: evaluated with no names in scope
            return list(eval(compile(ast.Expression(node.value), "strata_tokenizer.py", "eval"), {"__builtins__": {}}))
    raise SystemExit("no corpus list in strata_tokenizer.py")


EDGE = [
    "Hello, world!", " Hello", "  two spaces", "\tTab\tseparated\tvalues\n", "trailing spaces   ", "\n\n\n\n", "\r\n\r\nx\r\n",
    "a b c　d", "​‌‍﻿ zero widths", "é ä ô combining", "क्ष नमस्ते",
    "สวัสดี", "שלום مرحبا بكم", "한국어 텍스트",
    "日本語のテキスト。カタカナ", "中文，标点！“引号”",
    "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 family", "\U0001f44d\U0001f3fd skin tone", "\U0001f1e9\U0001f1ea\U0001f1ef\U0001f1f5 flags",
    "❤️ variation", "\U0001d400\U0001d401 math bold", "½ ⅓ ⅔ fractions", "ⅠⅡⅢ roman",
    "0", "007", "3.14159265358979", "1,234,567.89", "-42e-7", "2026-10-25T02:24:00Z", "0xDEADBEEF", "١٢٣ arabic digits",
    "don't won't it's I'M YOU'RE we'LL they'D", "'s 's 'S", "C++ C# F# .NET node.js", "snake_case camelCase PascalCase kebab-case",
    "x" * 300, "ab" * 200, " " * 64 + "x", "\n" * 33 + "y", "!!!???...,,,;;;", "<<<>>>&&&|||", "~`@#$%^&*()_+-=[]{}\\|;:'\",.<>/?",
    "\x00\x01\x02\x7f", "\u0080\u0081\u009f", "\U0010fffd", "� replacement",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
    "text<|im_end|>more", "a<think>b</think>c", "<|im_start|>user\nhi<|im_end|>", "<|im_start <|im_end <think",
    "<|vision_start|><|image_pad|><|vision_end|>", "<|im_start|><|im_start|><|im_end|>", "</think>\n\n", "<think>\n\n</think>\n\n",
]


def real_texts(mb: float) -> list[str]:
    """Paragraph-sized pieces (<= 6 KB) of the repository's text files until `mb` MB, in a fixed order."""
    exts = {".md", ".py", ".cpp", ".hpp", ".h", ".cu", ".cuh", ".txt", ".json", ".sh", ".cmake", ".yaml", ".toml", ".jinja"}
    roots = [IMPULSE.parent / "docs", IMPULSE / "docs", FLASHNEXT / "docs", FLASHNEXT, IMPULSE / "src", IMPULSE / "tests",
             IMPULSE.parent / "nebula", IMPULSE / "tools"]
    seen, files = set(), []
    for r in roots:
        if not r.exists():
            continue
        for p in sorted(r.rglob("*")):
            if p.is_file() and p.suffix in exts and p not in seen and p.stat().st_size < 4_000_000 and "third_party" not in p.parts:
                seen.add(p)
                files.append(p)
    out, total, limit = [], 0, int(mb * 1024 * 1024)
    # the translated READMEs first (CJK, accents), then everything in order
    files.sort(key=lambda p: (0 if p.name.startswith("README.") and p.name != "README.md" else 1, str(p)))
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        piece = ""
        for para in text.split("\n\n"):
            if len(piece) + len(para) + 2 > 6000 and piece:
                out.append(piece)
                total += len(piece.encode("utf-8"))
                piece = ""
            piece += (("\n\n" if piece else "") + para)[:6000]
        if piece:
            out.append(piece)
            total += len(piece.encode("utf-8"))
        if total >= limit:
            break
    return out


def chat_prompts(texts: list[str], n: int) -> list[str]:
    """ChatML prompts as serve/server.py renders them for Qwen3.8 (system, user, assistant turns, thinking on and off,
    a tool call and its response), the messages taken from the real text."""
    rnd = random.Random(1225)
    out = []
    for i in range(n):
        a, b, c = (texts[rnd.randrange(len(texts))][:rnd.randrange(50, 1500)] for _ in range(3))
        p = f"<|im_start|>system\n{a}<|im_end|>\n<|im_start|>user\n{b}<|im_end|>\n"
        if i % 3 == 0:
            p += (f"<|im_start|>assistant\n<think>\nLet me look.\n</think>\n\n<tool_call>\n"
                  f"{json.dumps({'name': 'read_file', 'arguments': {'path': 'src/x.cpp', 'note': c[:80]}})}\n</tool_call><|im_end|>\n"
                  f"<|im_start|>user\n<tool_response>\n{c}\n</tool_response><|im_end|>\n")
        elif i % 3 == 1:
            p += f"<|im_start|>assistant\n{c}<|im_end|>\n<|im_start|>user\nAnd then?<|im_end|>\n"
        p += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if i % 2 else "<think>\n")
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True, help="the pack's tokenizer/ folder")
    ap.add_argument("--fx-tokenize", required=True)
    ap.add_argument("--mb", type=float, default=2.0)
    ap.add_argument("--chats", type=int, default=600)
    ap.add_argument("--out")
    a = ap.parse_args()

    t0 = time.time()
    tok = strata_tokenizer(pathlib.Path(a.tokenizer))
    real = real_texts(a.mb)
    groups = {"roundtrip": roundtrip_corpus(), "edge": EDGE, "real": real, "chat": chat_prompts(real, a.chats)}
    cases = [(g, s, ps) for g, items in groups.items() for s in items for ps in (False, True)]
    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "in.bin"), os.path.join(td, "out.txt")
        with open(inp, "wb") as f:
            for _, s, ps in cases:
                b = s.encode("utf-8", "surrogatepass")
                f.write(struct.pack("<BI", 1 if ps else 0, len(b)) + b)
        subprocess.run([a.fx_tokenize, a.gguf, inp, outp], check=True)
        cpp = [[int(x) for x in line.split()] for line in open(outp, encoding="utf-8").read().split("\n")[:len(cases)]]
    t_cpp = time.time() - t0
    stats = {g: {"texts": 0, "bytes": 0, "tokens": 0, "mismatches": 0} for g in groups}
    bad, roundtrip_bad = [], 0
    for (g, s, ps), ids_cpp in zip(cases, cpp):
        ids_py = tok.encode(s, parse_special=ps)
        st = stats[g]
        st["texts"] += 1
        st["bytes"] += len(s.encode("utf-8", "surrogatepass"))
        st["tokens"] += len(ids_py)
        if ids_py != ids_cpp:
            st["mismatches"] += 1
            k = next((i for i, (x, y) in enumerate(zip(ids_py, ids_cpp)) if x != y), min(len(ids_py), len(ids_cpp)))
            bad.append({"group": g, "parse_special": ps, "text": s[:200], "first_diff": k,
                        "python": ids_py[max(0, k - 3):k + 5], "llama": ids_cpp[max(0, k - 3):k + 5]})
        if tok.decode(ids_cpp) != s:
            roundtrip_bad += 1
    total = {k: sum(st[k] for st in stats.values()) for k in ("texts", "bytes", "tokens", "mismatches")}
    summary = {"gguf": a.gguf, "tokenizer": a.tokenizer, "groups": stats, "total": total,
               "llama_ids_decode_back_differs": roundtrip_bad, "first_mismatches": bad[:20],
               "seconds": round(time.time() - t0, 1), "seconds_llama_incl_load": round(t_cpp, 1),
               "note": "each text tokenized twice (parse_special off, on); bytes and tokens count both passes"}
    print(json.dumps({k: v for k, v in summary.items() if k != "first_mismatches"}, indent=1))
    for b in bad[:10]:
        print("MISMATCH", json.dumps(b, ensure_ascii=False))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("PASS: identical ids" if total["mismatches"] == 0 else f"FAIL: {total['mismatches']} texts differ")
    return 0 if total["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
