"""Halo as the smart router, on a live Halo: 21 requests and the route each should get (everyday, larger, writing,
critique). Run against Halo's own server port: python3 greenfield/halo_router_eval.py <port>. Its cases tuned the
prompt and the checks in nebula/stage.py (2026-10-22): read the score with that in mind."""
import json, sys, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from types import SimpleNamespace as S
from nebula import stage
from nebula.engine import Engine
PORT = int(sys.argv[1])
own = S(ready=lambda: True, tier=S(address=f"http://127.0.0.1:{PORT}", model="halo", context=4096))
stage.HALO = lambda: own
CASES = [("Good morning! What's a quick breakfast idea?", "everyday"), ("Hi Halo, how are you tonight?", "everyday"),
         ("What's the capital of Australia?", "everyday"), ("Thanks, that's all for now.", "everyday"),
         ("Write a short scene where two rivals share an umbrella in the rain.", "writing"),
         ("Write the opening paragraph of a gothic romance set in a lighthouse.", "writing"),
         ("Continue the chapter: Mara opens the cellar door and hears breathing.", "writing"),
         ("Debug this: my Python script says 'list index out of range' in a loop over range(len(xs)+1). Walk me through why.", "larger"),
         ("Explain step by step how a B-tree split works, with the edge cases.", "larger"),
         ("Compare three database designs for a booking system and their trade-offs.", "larger"),
         ("Critique this opening line: 'It was a dark and stormy night, and Ellen was sad.'", "critique"),
         ("Here is my scene: 'She left the letter on the table and did not look back.' What works and what doesn't?", "critique"),
         ("Review my chapter's pacing: the first half is all description and the fight is two lines.", "critique"),
         ("What's 17 times 23?", "everyday"), ("Translate 'good night' into French.", "everyday"),
         ("Write a Python function that merges two sorted lists.", "larger"), ("Write a poem about autumn leaves.", "writing"),
         ("Why is my SQL query slow when I join three tables?", "larger"), ("Give me feedback on this paragraph: 'The rain fell. He waited.'", "critique"),
         ("Recommend a movie for tonight.", "everyday"), ("Draft dialogue for a tense interrogation scene.", "writing")]
ok = 0
for text, want in CASES:
    h = stage.ask_halo({"messages": [{"role": "user", "content": text}]}, ["everyday", "larger", "writing", "critique"])
    got = h and h["route"]
    ok += got == want
    print(f"{'OK ' if got == want else 'BAD'} want {want:9s} got {str(got):9s} why={h and h['why']!r}  {h and h['seconds']}s  | {text[:60]}")
print(f"{ok}/{len(CASES)}")
