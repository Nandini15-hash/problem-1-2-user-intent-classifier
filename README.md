# Problem 1.2: User Intent Classifier (video retrieval router)

This classifies a user's question about a video corpus into the retrieval granularity it needs:

| intent | goes to | example |
|---|---|---|
| `frame` | keyframe + OCR index | "what is the score shown at halftime" |
| `clip` | segment index + short VLM call | "why did the driver swerve" |
| `whole_video` | video-level summaries / map-reduce | "summarize the podcast" |

**Production model:** word and char TF-IDF plus 34 hand-built cue features, then a temperature-calibrated logistic regression, compiled to numpy-only inference. It is 136 KB with p50 latency of 0.15 ms per query on one CPU thread. A Bayes decision layer picks the route that minimises *routing regret* (lost answers plus compute) rather than the argmax class.

**Write-up (2 pages):** [`docs/WRITEUP.md`](docs/WRITEUP.md) · labeling guide: [`docs/labeling_guide.md`](docs/labeling_guide.md) · latest numbers: [`artifacts/results.md`](artifacts/results.md) · diagram: [`docs/architecture.drawio`](docs/architecture.drawio)

## Run

The exact commands the graders run are in [`run.yaml`](run.yaml):

```bash
python -m pip install -r requirements.txt
python -m src.intent_router.cli train       # ~5 s, writes artifacts/router.npz + router.vocab.json
python -m src.intent_router.cli evaluate    # ~2 min, writes artifacts/results.md / results.json
python -m src.intent_router.cli demo
python -m src.intent_router.cli predict "show me when the goalkeeper saves the penalty"
python -m src.intent_router.cli serve --port 8080   # POST /route {"query": "..."} ; GET /healthz
python -m pytest -q
```

Sample output:

```json
{"query": "what jersey number is the player who scores at 12:40", "intent": "frame",
 "probs": {"frame": 0.98, "clip": 0.01, "whole_video": 0.01}, "route": "frame",
 "temporal_hint": {"kind": "timestamp", "timestamp_s": 760}}
```

## Layout

```
src/intent_router/
  data.py       VideoMME download (HF, sha256-checked) + vendored fallback, gold labels
  features.py   normalisation, 18 cue lexicons, temporal-anchor slot extraction
  synth.py      seeded template grammar (450 realistic search-style queries)
  models.py     T0 rules / T1 cue-LR / T2 TF-IDF+cue-LR
  routing.py    answerability x cost model, Bayes-optimal routing, routing regret
  evaluate.py   grouped CV, cluster bootstrap CIs, calibration, OOD, latency/size
  fast.py       compiled numpy-only router (matches sklearn to 1e-7)
  cli.py        train / evaluate / predict / demo / serve
data/
  videomme_sample10_questions.jsonl   question text (offline fallback, verified against HF)
  videomme_intent_gold.csv            our labels for all 270 questions (+ acceptable alt label)
  realworld_queries_gold.csv          100 hand-written OOD search queries (never trained on)
tests/                                18 tests (data integrity, features, routing maths, model parity, CLI/HTTP)
```

Set `INTENT_OFFLINE=1` to skip the Hugging Face download and use the vendored copy.
