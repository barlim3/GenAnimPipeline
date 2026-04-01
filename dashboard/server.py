"""
Tiny dev server for GenAnim Dashboard.
Serves the dashboard static files and exposes /api/fbx-files
to list all .fbx files found under the project root.
"""

import http.server
import json
import os
from pathlib import Path

PORT = 8080
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        # API: list .fbx files relative to project root
        if self.path == "/api/fbx-files":
            fbx_files = []
            for root, _, files in os.walk(PROJECT_ROOT):
                for f in files:
                    if f.lower().endswith(".fbx"):
                        rel = os.path.relpath(os.path.join(root, f), PROJECT_ROOT)
                        fbx_files.append(rel.replace("\\", "/"))
            fbx_files.sort()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(fbx_files).encode())
            return

        # Serve FBX files from project root via /assets/ prefix
        if self.path.startswith("/assets/"):
            rel_path = self.path[len("/assets/"):]
            file_path = PROJECT_ROOT / rel_path
            if file_path.exists() and file_path.suffix.lower() == ".fbx":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "File not found")
                return

        # Everything else: serve from dashboard/
        super().do_GET()


if __name__ == "__main__":
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Dashboard dir: {DASHBOARD_DIR}")
    print(f"  Serving on:    http://localhost:{PORT}")
    with http.server.HTTPServer(("", PORT), DashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
