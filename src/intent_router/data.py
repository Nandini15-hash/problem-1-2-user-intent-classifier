"""Dataset loading.

VideoMME questions are fetched from the Hugging Face dataset named in the brief
(`aintropy-ai/video-mme-sample10`, file `qa/test.jsonl`). Only the question
text is needed for intent routing, so we never download the 11 GB of video.

If the network is unavailable, we fall back to a vendored copy of the question
text in `data/` (verified byte-for-byte against the HF file, see README).
Labels (`frame` / `clip` / `whole_video`) are ours - VideoMME has no routing
labels - and live in `data/videomme_intent_gold.csv` with an annotation guide
in `docs/labeling_guide.md`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
HF_URL = "https://huggingface.co/datasets/aintropy-ai/video-mme-sample10/resolve/main/qa/test.jsonl"
HF_SHA256 = "609e529faa2533ac35efca5ed409b47a7605a97f36b15f906819b2e3277473a5"
LABELS = ("frame", "clip", "whole_video")
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


@dataclass
class Example:
    text: str
    label: str
    group: str  # video id (for grouped CV) or source type
    alt: Optional[str] = None  # acceptable secondary label for ambiguous items
    meta: Optional[dict] = None


def _download_videomme(timeout: float = 20.0) -> Optional[Path]:
    CACHE.mkdir(exist_ok=True)
    dst = CACHE / "videomme_sample10_test.jsonl"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        req = urllib.request.Request(HF_URL, headers={"User-Agent": "intent-router/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec - fixed public URL
            body = r.read()
        digest = hashlib.sha256(body).hexdigest()
        if digest != HF_SHA256:
            print(f"[data] warning: qa/test.jsonl sha256 {digest[:12]}... differs from the reviewed version")
        dst.write_bytes(body)
        print(f"[data] downloaded VideoMME questions from Hugging Face ({len(body)} bytes)")
        return dst
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[data] HF download failed ({e.__class__.__name__}); using vendored copy")
        return None


def load_videomme_questions(offline: bool = False) -> List[dict]:
    path = None if offline or os.environ.get("INTENT_OFFLINE") else _download_videomme()
    src = path or DATA / "videomme_sample10_questions.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def load_videomme_labeled(offline: bool = False) -> List[Example]:
    qs = {q["question_id"]: q for q in load_videomme_questions(offline)}
    out: List[Example] = []
    with open(DATA / "videomme_intent_gold.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = qs[r["question_id"]]
            out.append(
                Example(
                    text=q["question"],
                    label=r["label"],
                    alt=r["alt"] or None,
                    group=q["video_id"],
                    meta={k: q[k] for k in ("question_id", "task_type", "duration", "domain")},
                )
            )
    return out


def load_realworld() -> List[Example]:
    with open(DATA / "realworld_queries_gold.csv", newline="", encoding="utf-8") as f:
        return [Example(text=r["query"], label=r["label"], group=r["source_type"]) for r in csv.DictReader(f)]
