import json
import threading
import time
import urllib.request

from src.intent_router import cli


def test_predict_prints_json(capsys):
    cli.main(["predict", "what is the score at halftime", "summarize the match"])
    out = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines() if l.startswith("{")]
    assert [o["intent"] for o in out] == ["frame", "whole_video"]
    assert {"probs", "route"} <= set(out[0])


def test_http_service_routes_and_rejects_bad_input():
    port = 18765
    threading.Thread(target=cli.cmd_serve, args=(port,), daemon=True).start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/route", data=json.dumps({"query": "why did he fall"}).encode(),
                                 headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert res["results"][0]["intent"] == "clip"
    bad = urllib.request.Request(f"http://127.0.0.1:{port}/route", data=b"not json")
    try:
        urllib.request.urlopen(bad, timeout=5)
        assert False, "expected HTTP 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
