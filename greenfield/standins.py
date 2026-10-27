"""Stand-ins for a from-scratch run (greenfield/run.sh), started inside the container from the cloned repo. No model
runs; generic names only.

    python greenfield/standins.py

- :8080  a pretend Impulse engine (the tests' fake strata-lite-server: /health, /props, /v1/models, chat, streams)
- :8081  a second pretend engine, a second lane
- :18300 a pretend Switchyard to watch (optional): its status, callers and plan under /switchyard/
- :18280 a pretend Gravity router on this machine, over two pretend machines ("alpha": tiny, chat; "beta": chat)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nebula.examples.fake_router import FakeRouter  # noqa: E402
from nebula.tests.fake_engine import FakeEngine     # noqa: E402
from nebula.tests.fake_pair import FakeRouter as GravityRouter, Node   # noqa: E402

engines = [FakeEngine(alias="local-small", port=8080), FakeEngine(alias="lane-two", port=8081)]
router = FakeRouter(port=18300, every=4).start()
gravity = GravityRouter([Node("alpha", ["tiny", "chat"]), Node("beta", ["chat"])], port=18280)
print("stand-ins up: engines", [e.url for e in engines], "router", router.url, flush=True)
while True:
    time.sleep(3600)
