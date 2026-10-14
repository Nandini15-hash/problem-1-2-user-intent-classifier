"""Inference-only "compiled" router: numpy + stdlib, no scikit-learn.

The trained TfidfCueLR is a linear model, so serving it only needs:
  token -> (idf, 3 class weights)    for word and char vocabularies
  cue scaler mean/scale + 3 x n_cue weights + intercept.

We store those in one compressed .npz (+ a JSON vocab), and score a query with
plain dict lookups. This replaces scikit-learn's per-call input validation,
which dominates single-query latency, and lets the serving image ship
without scikit-learn/scipy. `tests/test_fast.py` checks the compiled router
matches scikit-learn probabilities to 1e-6.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import LABELS
from .features import cue_vector, normalize, temporal_hint

_TOKEN = re.compile(r"(?u)\b\w\w+\b")
_WS = re.compile(r"\s\s+")


def _word_ngrams(text: str, lo: int, hi: int) -> List[str]:
    toks = _TOKEN.findall(text)
    out = []
    for n in range(lo, hi + 1):
        out += [" ".join(toks[i: i + n]) for i in range(len(toks) - n + 1)]
    return out


def _char_wb_ngrams(text: str, lo: int, hi: int) -> List[str]:
    text = _WS.sub(" ", text)
    out = []
    for w in text.split():
        w = " " + w + " "
        wl = len(w)
        for n in range(lo, hi + 1):
            off = 0
            out.append(w[off: off + n])
            while off + n < wl:
                off += 1
                out.append(w[off: off + n])
            if off == 0:
                break
    return out


class FastRouter:
    def __init__(self, blocks: Dict, cue_mean, cue_scale, cue_coef, intercept, cue_weight: float,
                 temperature: float = 1.0):
        self.temperature = float(temperature)
        self.blocks = blocks  # name -> {"vocab": {tok: idx}, "idf": arr, "coef": arr (n,3), "ngram": (lo,hi)}
        self.cue_mean, self.cue_scale = cue_mean, cue_scale
        self.cue_coef, self.intercept, self.cue_weight = cue_coef, intercept, cue_weight

    # ---- build / persist -------------------------------------------------
    @classmethod
    def from_sklearn(cls, m, temperature: float = 1.0) -> "FastRouter":
        coef = m.clf.coef_  # (3, n_features)
        blocks, off = {}, 0
        for name, vec in (("word", m.word), ("char", m.char)):
            n = len(vec.vocabulary_)
            blocks[name] = {"vocab": {k: int(v) for k, v in vec.vocabulary_.items()}, "idf": vec.idf_.astype(np.float64),
                            "coef": coef[:, off: off + n].T.copy(), "ngram": tuple(vec.ngram_range)}
            off += n
        return cls(blocks, m.scaler.mean_, m.scaler.scale_, coef[:, off:].T.copy(), m.clf.intercept_.copy(),
                   m.cue_weight, temperature)

    def save(self, path: Path) -> None:
        path = Path(path)
        arrays = {"cue_mean": self.cue_mean, "cue_scale": self.cue_scale, "cue_coef": self.cue_coef,
                  "intercept": self.intercept, "cue_weight": np.array([self.cue_weight]),
                  "temperature": np.array([self.temperature])}
        meta = {}
        for name, b in self.blocks.items():
            arrays[f"{name}_idf"] = b["idf"].astype(np.float32)
            arrays[f"{name}_coef"] = b["coef"].astype(np.float32)
            inv = sorted(b["vocab"], key=b["vocab"].get)
            meta[name] = {"vocab": inv, "ngram": list(b["ngram"])}
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".vocab.json").write_text(json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "FastRouter":
        path = Path(path)
        z = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".vocab.json").read_text(encoding="utf-8"))
        blocks = {}
        for name, m in meta.items():
            blocks[name] = {"vocab": {t: i for i, t in enumerate(m["vocab"])}, "idf": z[f"{name}_idf"].astype(np.float64),
                            "coef": z[f"{name}_coef"].astype(np.float64), "ngram": tuple(m["ngram"])}
        return cls(blocks, z["cue_mean"], z["cue_scale"], z["cue_coef"], z["intercept"], float(z["cue_weight"][0]),
                   float(z["temperature"][0]))

    # ---- inference -------------------------------------------------------
    def _block_score(self, name: str, grams: List[str]) -> np.ndarray:
        b = self.blocks[name]
        vocab, idf, coef = b["vocab"], b["idf"], b["coef"]
        cnt = Counter(g for g in grams if g in vocab)
        if not cnt:
            return np.zeros(3)
        idx = np.fromiter((vocab[g] for g in cnt), dtype=np.int64, count=len(cnt))
        tf = 1.0 + np.log(np.fromiter(cnt.values(), dtype=np.float64, count=len(cnt)))
        w = tf * idf[idx]
        w /= math.sqrt(float(w @ w))
        return w @ coef[idx]

    def logits(self, text: str) -> np.ndarray:
        t = normalize(text)
        s = self.intercept.astype(np.float64).copy()
        s += self._block_score("word", _word_ngrams(t, *self.blocks["word"]["ngram"]))
        s += self._block_score("char", _char_wb_ngrams(t, *self.blocks["char"]["ngram"]))
        cues = (cue_vector(t).astype(np.float64) - self.cue_mean) / self.cue_scale * self.cue_weight
        return s + cues @ self.cue_coef

    def predict_proba(self, texts, calibrated: bool = True) -> np.ndarray:
        z = np.vstack([self.logits(t) for t in texts])
        if calibrated:
            z = z / self.temperature
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def route(self, text: str, policy=None) -> Dict:
        p = self.predict_proba([text])[0]
        out = {"query": text, "intent": LABELS[int(p.argmax())],
               "probs": {l: round(float(x), 4) for l, x in zip(LABELS, p)}}
        if policy is not None:
            out["route"] = LABELS[int(policy.decide(p[None])[0])]
        hint = temporal_hint(normalize(text))
        if hint:
            out["temporal_hint"] = hint.as_dict()
        return out
