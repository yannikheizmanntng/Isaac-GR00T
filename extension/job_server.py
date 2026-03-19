from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8765

_REPO_ROOT = Path(__file__).parent.parent
MAIN_IG = Path(__file__).parent / "main.py"
LOGS_DIR = _REPO_ROOT / "logs"

_jobs: dict[str, subprocess.Popen] = {}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _job_status(name: str) -> dict:
    proc = _jobs.get(name)
    if proc is None:
        return {"gone": True, "exit_code": None}
    exit_code = proc.poll()
    if exit_code is not None:
        del _jobs[name]
        return {"gone": True, "exit_code": exit_code}
    return {"gone": False, "exit_code": None}


def _launch(name: str, args: list[str], log: Path) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = open(log, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(MAIN_IG)] + args,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    _jobs[name] = proc


class _JobHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/jobs":
            jobs = {name: "running" for name, proc in list(_jobs.items()) if proc.poll() is None}
            self._respond(200, {"jobs": jobs})
        else:
            match = re.fullmatch(r"/screen_gone/(.+)", self.path)
            if match:
                self._respond(200, _job_status(match.group(1)))
            else:
                self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body: dict = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/run":
            self._handle_run(body)
        elif self.path == "/kill_screen":
            self._handle_kill_screen(body)
        else:
            self._respond(404, {"error": "not found"})

    def _handle_run(self, body: dict) -> None:
        name: str = body["name"]
        proc = _jobs.get(name)
        if proc is not None and proc.poll() is None:
            print(f"[job_server] rejected '{name}' — already running (pid {proc.pid})")
            self._respond(409, {"status": "already_running", "name": name})
            return
        args = shlex.split(body["args_str"])
        log = LOGS_DIR / f"{name}_{_ts()}.log"
        _launch(name, args, log)
        print(f"[job_server] started '{name}' → {log}")
        self._respond(200, {"status": "started", "name": name, "log": str(log)})

    def _handle_kill_screen(self, body: dict) -> None:
        name: str = body["name"]
        proc = _jobs.get(name)
        if proc is not None and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            del _jobs[name]
            print(f"[job_server] killed '{name}'")
            self._respond(200, {"status": "killed", "name": name})
        else:
            self._respond(200, {"status": "not_found", "name": name})

    def _respond(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), _JobHandler)
    print(f"[job_server] listening on :{PORT}")
    server.serve_forever()
