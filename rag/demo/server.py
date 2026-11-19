"""Serve the read-only dashboard on the local loopback interface."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    page = Path(__file__).with_name("index.html")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route == "/":
                content, mime = page.read_bytes(), "text/html; charset=utf-8"
            elif route == "/events":
                try:
                    lines = args.events.read_bytes().splitlines(keepends=True)
                except FileNotFoundError:
                    lines = []
                content = json.dumps([json.loads(line) for line in lines if line.endswith(b"\n")]).encode()
                mime = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
