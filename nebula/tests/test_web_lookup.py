"""Lookup's read_page reads the public web only: this machine, the home network and a redirect to either are refused
(a page a model reads could send it to Nebula's own API here, which opens without a PIN, or to a router)."""
import http.server
import threading
import unittest
from unittest import mock

from nebula.lookup import web


class _Here(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/private")
            self.end_headers()
            return
        body = b"<p>PRIVATE-MARKER</p>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class ReadPageTest(unittest.TestCase):
    def setUp(self):
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Here)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.base = f"http://127.0.0.1:{self.srv.server_port}"

    def test_this_machine_and_the_home_network_are_refused(self):
        w = web.Web("duckduckgo")
        for url in (self.base + "/private", f"http://localhost:{self.srv.server_port}/private", "http://172.16.0.1/",
                    "http://10.0.0.1/", "http://[::1]/", "http://169.254.169.254/latest/meta-data/", "file:///etc/passwd"):
            with self.assertRaises(ValueError, msg=url):
                w.read(url)

    def test_a_redirect_to_a_private_page_is_refused(self):
        with self.assertRaises(ValueError):
            web._PUBLIC_OPENER.open(self.base + "/redirect", timeout=5)    # what read_page uses after its first check
        req = web.urllib.request.Request(self.base + "/redirect")
        handler = web._CheckedRedirects()
        with self.assertRaises(web.PrivateAddress):
            handler.redirect_request(req, None, 302, "Found", {}, self.base + "/private")

    def test_a_name_that_answers_public_then_private_is_refused(self):
        # DNS rebinding: the early check sees a public address, the connection would get this machine; the address is
        # checked again where it is connected to, so the private page is never read
        real = web.socket.getaddrinfo
        answers = iter(["93.184.216.34", "127.0.0.1"])

        def flip(host, port, *a, **k):
            if host == "rebind.test":
                ip = next(answers, "127.0.0.1")
                return [(web.socket.AF_INET, web.socket.SOCK_STREAM, 6, "", (ip, self.srv.server_port))]
            return real(host, port, *a, **k)
        with mock.patch.object(web.socket, "getaddrinfo", flip):
            with self.assertRaises(ValueError):
                web.Web("duckduckgo").read(f"http://rebind.test:{self.srv.server_port}/private")

    def test_allow_private_reads_them(self):
        self.assertIn("PRIVATE-MARKER", web.Web("duckduckgo", allow_private=True).read(self.base + "/redirect"))


if __name__ == "__main__":
    unittest.main()
