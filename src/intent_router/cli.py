"""Command-line entry point.

    python -m src.intent_router.cli train                 # fit + compile production router
    python -m src.intent_router.cli evaluate [--quick]    # full benchmark -> artifacts/results.md
    python -m src.intent_router.cli predict "query" ...   # route queries (JSON lines)
    python -m src.intent_router.cli demo                  # route a few sample queries
    python -m src.intent_router.cli serve --port 8080     # tiny HTTP service: POST /route
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .data import ROOT

ART = ROOT / "artifacts" / "router"

DEMO = [
    "what is this podcast about",
    "show me the moment the goalkeeper saves the penalty",
    "what jersey number is the player who scores at 12:40",
    "how many times does the streamer die in the whole run",
    "why did the chef add vinegar to the sauce",
    "what does the sign at the entrance say in the opening shot",
    "Which of the following events happened after the storm in the video?",
]


def _train() -> Path:
    from .data import load_videomme_labeled
    from .fast import FastRouter
    from .models import TfidfCueLR
    from .synth import synth_queries

    import numpy as np

    from .data import LABEL2ID
    from .evaluate import fit_temperature, oof_predictions

    t0 = time.perf_counter()
    vm = load_videomme_labeled()
    data = vm + synth_queries()
    m = TfidfCueLR().fit([e.text for e in data], [e.label for e in data])
    # calibrate: temperature fitted on grouped out-of-fold predictions over VideoMME
    oof = oof_predictions("tfidf_cue_lr", True, vm, repeats=1, calibrate=False)[0]
    T = fit_temperature(oof, np.array([LABEL2ID[e.label] for e in vm]))
    ART.parent.mkdir(exist_ok=True)
    FastRouter.from_sklearn(m, temperature=T).save(ART)
    print(f"[train] {len(data)} examples, temperature={T}, compiled router -> {ART}.npz/.vocab.json "
          f"({time.perf_counter() - t0:.1f}s)")
    return ART


def _load():
    from .fast import FastRouter

    if not ART.with_suffix(".npz").exists():
        _train()
    return FastRouter.load(ART)


def cmd_predict(queries):
    from .routing import RoutingPolicy

    r, pol = _load(), RoutingPolicy()
    for q in queries:
        print(json.dumps(r.route(q, pol), ensure_ascii=False))


def cmd_serve(port: int):
    from .routing import RoutingPolicy

    router, pol = _load(), RoutingPolicy()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._send(200, {"status": "ok"} if self.path == "/healthz" else {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/route":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                qs = payload.get("queries") or [payload["query"]]
                self._send(200, {"results": [router.route(str(q), pol) for q in qs[:256]]})
            except Exception as e:  # bad input -> 400, never crash the server
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    print(f"[serve] listening on :{port}  (POST /route {{\"query\": ...}}, GET /healthz)")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


def main(argv=None):
    p = argparse.ArgumentParser(prog="intent_router")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    e = sub.add_parser("evaluate")
    e.add_argument("--quick", action="store_true", help="1 CV repeat, fewer bootstrap draws (~1 min)")
    pr = sub.add_parser("predict")
    pr.add_argument("queries", nargs="+")
    sub.add_parser("demo")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)
    if a.cmd == "train":
        _train()
    elif a.cmd == "evaluate":
        from .evaluate import run

        run(quick=a.quick)
    elif a.cmd == "predict":
        cmd_predict(a.queries)
    elif a.cmd == "demo":
        cmd_predict(DEMO)
    elif a.cmd == "serve":
        cmd_serve(a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
