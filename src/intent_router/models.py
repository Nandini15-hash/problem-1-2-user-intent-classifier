"""Model tiers, from free to slightly-less-free.

    tier 0  RuleRouter    hand-weighted cue scores, no training      ~0 KB
    tier 1  CueLR         logistic regression on ~40 cue features    ~2 KB
    tier 2  TfidfCueLR    word+char TF-IDF + cue features -> LR      ~few hundred KB

All tiers expose `fit(texts, labels)` and `predict_proba(texts) -> (n, 3)`
with columns ordered as `data.LABELS`.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import LABEL2ID, LABELS
from .features import CUE_FEATURE_NAMES, CUE_NAMES, cue_vector, normalize


def _cues(texts: Sequence[str]) -> np.ndarray:
    return np.vstack([cue_vector(normalize(t)) for t in texts])


def _y(labels: Sequence[str]) -> np.ndarray:
    return np.asarray([LABEL2ID[l] for l in labels])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class RuleRouter:
    """Tier 0: a transparent weighted vote over cue lexicons. No training."""

    name = "rules"
    # weights[cue] = (frame, clip, whole_video)
    W = {
        "global_topic": (0, 0, 3.0), "global_scope": (0, 0.3, 1.5), "ordering": (0, 0.5, 2.0),
        "exhaustive_negation": (0, 0.3, 1.5), "count_how_many": (0.5, 0, 0.8), "count_repeated": (0, 0, 1.5),
        "inference": (0, 0.5, 1.0), "event_verbs": (0, 1.0, 0), "how_why": (0, 1.5, 0.2),
        "temporal_relation": (0, 1.0, 0), "segment_request": (0, 1.5, 0), "first_last": (0.4, 0.6, 0),
        "appearance": (1.8, 0, 0), "text_on_screen": (1.5, 0, 0), "spatial_static": (1.2, 0, 0),
        "frame_request": (2.0, 0, 0), "timestamp": (1.5, 0.5, 0), "what_object": (1.2, 0, 0),
    }
    BIAS = np.array([0.0, 0.4, 0.2])  # mild prior towards clip, the "safe middle"

    def fit(self, texts, labels):  # noqa: D401 - nothing to learn
        return self

    def decision_function(self, texts: Sequence[str]) -> np.ndarray:
        X = _cues(texts)[:, : len(CUE_NAMES)]
        W = np.array([self.W.get(n, (0, 0, 0)) for n in CUE_NAMES])
        return np.minimum(X, 2) @ W + self.BIAS

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return _softmax(self.decision_function(texts))


class CueLR:
    """Tier 1: logistic regression on the dense, interpretable cue vector."""

    name = "cue_lr"

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, texts, labels):
        X = _cues(texts)
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")
        self.clf.fit(self.scaler.transform(X), _y(labels))
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.scaler.transform(_cues(texts)))

    def top_features(self, k: int = 5):
        out = {}
        for i, lab in enumerate(LABELS):
            idx = np.argsort(-self.clf.coef_[i])[:k]
            out[lab] = [CUE_FEATURE_NAMES[j] for j in idx]
        return out


class TfidfCueLR:
    """Tier 2 (production): sparse word + char n-grams stacked with cue features."""

    name = "tfidf_cue_lr"

    def __init__(self, C: float = 4.0, cue_weight: float = 1.0):
        self.C = C
        self.cue_weight = cue_weight

    def _X(self, texts, fit: bool = False):
        norm = [normalize(t) for t in texts]
        if fit:
            self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=20000)
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                        max_features=30000)
            self.word.fit(norm)
            self.char.fit(norm)
            C = _cues(texts)
            self.scaler = StandardScaler().fit(C)
        C = self.scaler.transform(_cues(texts)) * self.cue_weight
        return sparse.hstack([self.word.transform(norm), self.char.transform(norm), sparse.csr_matrix(C)]).tocsr()

    def fit(self, texts, labels):
        X = self._X(texts, fit=True)
        self.clf = LogisticRegression(C=self.C, max_iter=4000, class_weight="balanced")
        self.clf.fit(X, _y(labels))
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._X(texts))


MODELS = {"rules": RuleRouter, "cue_lr": CueLR, "tfidf_cue_lr": TfidfCueLR}


def build(name: str):
    return MODELS[name]()


def predict_labels(model, texts: List[str]) -> List[str]:
    return [LABELS[i] for i in model.predict_proba(texts).argmax(1)]
