"""Loopback-only, read-only V216 monitor."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "campaign_20261104"
RESOURCES = ("events", "observed_bytes", "target_sites")
ARMS = ("FEW", "MANY")


def read(name, default=None):
    try:
        return json.loads((RUN / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def snapshot():
    lock = read("LOCK.json", {})
    status = read("STATUS.json", {})
    committed = set(status.get("committed", []))
    summary = read("SUMMARY.json", {}) if status.get("state") == "COMPLETED" else {}
    totals = dict.fromkeys(RESOURCES, 0)
    rows = []
    for index in lock.get("worlds", []):
        receipts = {arm: read(f"progress_{index}_{arm}.json", {}) for arm in ARMS}
        is_committed = index in committed
        full = read(f"family_{index}.json", {}) if is_committed else {}
        # Progress tallies are cumulative within a family, including across arms.
        newest = receipts["MANY"] or receipts["FEW"]
        tally = full.get("tally", newest.get("tally", {}))
        for key in RESOURCES:
            totals[key] += tally.get(key, 0)
        pid = status.get("active", {}).get(str(index))
        rows.append(dict(index=index, committed=is_committed, pid=pid,
                         checkpoints={k:v.get("checkpoint",0) for k,v in receipts.items()}))
    if summary:
        totals.update(summary["totals"])
    try:
        heartbeat = (RUN / "STATUS.json").stat().st_mtime
    except FileNotFoundError:
        heartbeat = None
    return dict(state=status.get("state","WAITING"), error=status.get("error"),
                heartbeat=heartbeat, now=time.time(), worlds=rows, totals=totals,
                budgets=lock.get("budgets",{}), projected=lock.get("projected",{}),
                metrics=summary.get("metrics"), eligible=summary.get("eligible_for_followup"))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}",
                                            f"localhost:{self.server.server_port}"}:
            return self.send_error(403)
        if self.path == "/api/status":
            content = json.dumps(snapshot()).encode()
            return self.send(content,"application/json")
        if self.path == "/":
            return self.send((HERE/"index.html").read_bytes(),"text/html; charset=utf-8")
        allowed = {"/report": ("REPORT.md","text/plain; charset=utf-8"),
                   "/bundle": ("RESULT_BUNDLE.zip","application/zip")}
        if self.path in allowed and read("STATUS.json",{}).get("state") == "COMPLETED":
            name, kind = allowed[self.path]
            try:
                return self.send((RUN/name).read_bytes(),kind,name)
            except FileNotFoundError:
                pass
        self.send_error(404)

    def send(self, content, kind, name=None):
        self.send_response(200)
        self.send_header("Content-Type",kind)
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Content-Security-Policy","default-src 'none'; connect-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
        if name:
            self.send_header("Content-Disposition",f'attachment; filename="{name}"')
        self.send_header("Content-Length",str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8216)
    args=parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as server:
        print(f"V216 monitor: http://127.0.0.1:{server.server_port}/",flush=True)
        server.serve_forever()
