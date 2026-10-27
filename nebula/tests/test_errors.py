"""The error log behind the System tab's button: what it keeps, the page's own reports, and clearing it."""
import json
import logging
import unittest

from nebula.errors import ErrorLog
from nebula.tests.test_escalation import Rig
from nebula.tests.test_ui import req


class ErrorLogTest(unittest.TestCase):
    def test_keeps_warnings_and_errors_newest_last_up_to_its_capacity(self):
        log = ErrorLog(capacity=3).attach()
        self.addCleanup(log.detach)
        lg = logging.getLogger("cockpit.test")
        lg.info("not kept")
        for i in range(4):
            lg.warning("warning %d", i)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            lg.error("it broke", exc_info=True)
        got = log.list()
        self.assertEqual([e["message"] for e in got["errors"]], ["warning 2", "warning 3", "it broke"])
        self.assertEqual([e["level"] for e in got["errors"]], ["warning", "warning", "error"])
        self.assertIn("RuntimeError: boom", got["errors"][-1]["detail"])
        self.assertEqual(got["last_id"], 5)
        self.assertEqual(log.clear(), 3)
        self.assertEqual(log.list()["count"], 0)


class ErrorApiTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def errors(self):
        return json.loads(req(self.rig.base, "/api/errors")[2])

    def test_a_failed_engine_call_the_page_s_report_and_clearing(self):
        self.assertEqual(self.errors()["count"], 0)
        self.rig.b.stop()                        # the big tier's engine is gone
        status, _h, _raw = req(self.rig.base, "/v1/chat/completions", {"model": "cockpit/big", "messages": [{"role": "user", "content": "hi"}]})
        self.assertGreaterEqual(status, 500)
        logged = self.errors()["errors"]
        self.assertEqual(len(logged), 1)
        self.assertEqual((logged[0]["level"], logged[0]["source"]), ("warning", "cockpit"))
        self.assertIn("POST /v1/chat/completions: the engine answered", logged[0]["message"])
        status, _h, raw = req(self.rig.base, "/api/errors", {"message": "room: the turn failed", "detail": "at send"})
        self.assertEqual(status, 200, raw)
        self.assertEqual(req(self.rig.base, "/api/errors", {"message": "  "})[0], 400)
        got = self.errors()
        self.assertEqual([(e["source"], e["message"]) for e in got["errors"]][-1], ("page", "room: the turn failed"))
        self.assertEqual(json.loads(req(self.rig.base, "/api/monitor")[2])["errors"], 2)
        self.assertEqual(json.loads(req(self.rig.base, "/api/errors/clear", {})[2]), {"cleared": 2})
        self.assertEqual(self.errors()["count"], 0)

    def test_a_closed_server_leaves_no_handler_behind(self):
        before = len(logging.getLogger("cockpit").handlers)
        rig = Rig()
        self.assertEqual(len(logging.getLogger("cockpit").handlers), before + 1)
        rig.stop()
        self.assertEqual(len(logging.getLogger("cockpit").handlers), before)


if __name__ == "__main__":
    unittest.main()
