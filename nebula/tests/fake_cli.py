"""A stand-in for a CLI such as Grok, Codex or Claude Code: the prompt comes on stdin or from --prompt-file, the
answer is "cli says: <the prompt's last line>" on stdout or into the -o file. `--fail` exits 3; `--slow S` waits."""
import os
import sys
import time


def main(argv):
    prompt_file = out_file = None
    fail = False
    slow = 0.0
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--prompt-file":
            prompt_file = args.pop(0)
        elif a == "-o":
            out_file = args.pop(0)
        elif a == "--fail":
            fail = True
        elif a == "--slow":
            slow = float(args.pop(0))
    prompt = open(prompt_file, encoding="utf-8").read() if prompt_file else sys.stdin.read()
    if slow:
        time.sleep(slow)
    if fail:
        sys.stderr.write("the fake cli failed as asked\n")
        return 3
    lines = [ln for ln in prompt.splitlines() if ln.strip() and not ln.startswith("###")]
    answer = "cli says: " + (lines[-1].strip() if lines else "") + (f" [model {os.environ['FAKE_MODEL']}]" if os.environ.get("FAKE_MODEL") else "")
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(answer + "\n")
    else:
        sys.stdout.write(answer + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
