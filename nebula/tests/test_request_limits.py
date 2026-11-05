"""A Content-Length that is not a whole number of bytes is refused at once (it used to hold a thread until the client
hung up: read(-1) reads to the end of the stream, and the sign-in route answers before anyone signs in)."""
import socket
import time
import unittest

from nebula.tests.test_escalation import Rig


class RequestLengthTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()
        self.addCleanup(self.rig.stop)

    def send(self, path: str, length: str) -> tuple[bytes, float]:
        port = self.rig.srv.server_port
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            t0 = time.monotonic()
            s.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {length}\r\n\r\n{{}}".encode("utf-8"))
            return s.recv(200), time.monotonic() - t0          # the client keeps its side open
        finally:
            s.close()

    def test_a_negative_or_odd_length_is_refused_at_once(self):
        for path in ("/api/signin/pin", "/v1/chat/completions", "/api/look"):
            for length in ("-1", "abc", "1.5", "\u00b2", "9" * 5000):
                head, took = self.send(path, length)
                self.assertTrue(head.startswith(b"HTTP/1.0 400") or head.startswith(b"HTTP/1.1 400"), (path, length, head))
                self.assertLess(took, 3, (path, length))


if __name__ == "__main__":
    unittest.main()
