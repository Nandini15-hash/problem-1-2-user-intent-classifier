"""Feature engineering for query-intent routing.

Two kinds of signal are combined:

1. Lexical n-grams (TF-IDF over words and character n-grams) - learned from data.
2. Hand-built "retrieval-granularity" cues - small, interpretable lexicons that
   encode *why* a query needs a frame, a clip or the whole video. These carry
   the model when training data is tiny (270 labelled questions) and make
   errors explainable.

Everything here is pure Python + regex so feature extraction costs
microseconds and has no model download.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, NFKC-normalise and collapse whitespace.

    Enumerated option lists in VideoMME questions (e.g. "(a) ... (b) ...",
    circled digits) are kept as a single token "<enum>" because their
    *presence* is a strong whole-video signal (ordering several events), while
    their content is noise.
    """
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"(\n\s*(\([a-h]\)|[①②③④⑤⑥⑦⑧⑨]|[1-9]\.)\s*[^\n]*)+", " <enum> ", t)
    t = t.replace("’", "'")
    return _WS.sub(" ", t).strip()


# --------------------------------------------------------------------------
# Cue lexicons (regexes over normalised text)
# --------------------------------------------------------------------------
def _rx(*patterns: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b")


CUES: Dict[str, re.Pattern] = {
    # --- whole-video cues: need global aggregation -------------------------
    "global_topic": _rx(
        r"main (?:idea|theme|topic|focus|plot|point|argument|message)",
        r"(?:central|overall|primary|recurring) (?:theme|message|focus|idea|mood)",
        r"mainly about", r"what is (?:this|the) (?:video|episode|podcast|film|movie|vlog|lecture|talk) about",
        r"about overall", r"summar(?:y|ise|ize)", r"tl;?dr", r"gist", r"overview", r"takeaways?",
        r"best title", r"storyline", r"(?:genre|kind|type) of (?:video|game|show)",
        r"what does the video (?:depict|record|cover)", r"what (?:topics?|subjects?)",
        r"chapter", r"structured?", r"from start to finish", r"what happens in the",
    ),
    "global_scope": _rx(
        r"throughout", r"entire", r"whole (?:video|match|game|episode|film|movie|footage|thing)",
        r"in total", r"overall", r"all the", r"every", r"across the", r"at any point", r"ever",
        r"most(?:ly)? often", r"mostly", r"most of", r"over the (?:episode|video|course)",
    ),
    "ordering": _rx(
        r"order", r"sequence", r"chronolog\w*", r"sort\w*", r"rank\w*", r"<enum>",
    ),
    "exhaustive_negation": _rx(
        r"not (?:mentioned|shown|included|used|discussed|true|correct|caused|one of|appear\w*|displayed)",
        r"(?:wasn't|isn't|doesn't|didn't|don't|never) (?:mentioned|shown|show up|appear|included|used|discussed)",
        r"absent", r"except", r"left undiscussed", r"skipped", r"can not describe", r"cannot describe",
        r"doesn't contribute", r"does not",
    ),
    "count_how_many": _rx(r"how many", r"number of", r"count"),
    "count_repeated": _rx(r"how many times", r"how often", r"each time"),
    "inference": _rx(r"inferr?\w*", r"assess\w*", r"evaluat\w*", r"relationship", r"similarit\w*",
                     r"characters?", r"opinion", r"recommend\w*", r"beliefs?"),

    # --- clip cues: an event / action / explanation over a time span --------
    "event_verbs": _rx(
        r"happen\w*", r"do(?:es)? (?:the|he|she|they)", r"doing", r"react\w*", r"escape\w*",
        r"chas\w*", r"catch\w*", r"enter\w*", r"leav\w*", r"walk\w*", r"fall\w*", r"falls?",
        r"scor\w*", r"collect\w*", r"cool\w*", r"fold\w*", r"replace\w*", r"demonstrat\w*",
        r"explain\w*", r"talk\w* about", r"say about", r"said", r"answer\w*", r"argu\w*",
        r"perform\w*", r"weav\w*", r"make\w*", r"made", r"take\w*", r"took",
    ),
    "how_why": _rx(r"how (?:do|does|did|is|are|was|were)", r"why", r"what (?:is|was) the (?:purpose|reason)",
                   r"reason", r"purpose", r"what would happen", r"process"),
    "temporal_relation": _rx(r"after", r"before", r"during", r"while", r"when", r"until", r"then",
                              r"right after", r"first time"),
    "segment_request": _rx(r"part where", r"segment", r"section", r"moment", r"scene where",
                            r"clip", r"replay", r"find the part", r"show me", r"locate",
                            r"the part", r"in the (?:first|second) half", r"episode"),
    "first_last": _rx(r"first", r"last", r"final(?:ly)?", r"at last", r"begin\w*", r"start",
                      r"end(?:s|ing)?", r"opening", r"in the middle"),

    # --- frame cues: answer visible in one image ----------------------------
    "appearance": _rx(
        r"colou?rs?", r"wear\w*", r"dress\w*", r"clothing", r"clothes", r"shirt", r"pants",
        r"jacket", r"coat", r"tie", r"hat", r"shoes?", r"hair\w*", r"look like", r"resemble\w*",
        r"shape", r"icon", r"logo", r"brand", r"uniform", r"jersey", r"beak", r"tag",
    ),
    "text_on_screen": _rx(
        r"written", r"say(?:s)?\b(?! about)", r"text", r"title", r"caption", r"sign", r"label\w*",
        r"score(?:board)?", r"leaderboard", r"price", r"plate", r"url", r"name tag", r"poster",
        r"what time", r"clock", r"temperature", r"speed", r"statistics", r"listed", r"shown on",
        r"dashboard", r"screen", r"health bar", r"number",
    ),
    "spatial_static": _rx(r"where is", r"located", r"location of", r"placed on", r"position",
                           r"holding", r"sitting", r"standing", r"on the (?:table|desk|wall|ground)"),
    "frame_request": _rx(r"frame", r"screenshot", r"thumbnail", r"still", r"shot", r"image",
                          r"picture", r"visible"),
    "timestamp": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    "what_object": _rx(r"what (?:is|are) the (?:man|woman|person|player|people|boy|girl)\w* (?:wearing|eating|holding)",
                       r"which (?:item|object|animal|card|weapon|food)", r"what (?:animal|object|item|card|kind of)"),
}

CUE_NAMES: List[str] = list(CUES.keys())

_WH = ["what", "which", "how", "why", "when", "where", "who", "is", "does", "did", "show", "find", "list", "give"]


def cue_vector(text_norm: str) -> np.ndarray:
    """Dense, interpretable feature vector (cue counts + wh-word + length)."""
    feats = [len(CUES[name].findall(text_norm)) for name in CUE_NAMES]
    first = text_norm.split(" ", 1)[0] if text_norm else ""
    feats += [1.0 if first == w else 0.0 for w in _WH]
    n_tok = len(text_norm.split())
    feats += [min(n_tok, 60) / 60.0, 1.0 if "?" in text_norm else 0.0]
    return np.asarray(feats, dtype=np.float32)


CUE_FEATURE_NAMES: List[str] = CUE_NAMES + [f"wh={w}" for w in _WH] + ["len", "has_qmark"]


# --------------------------------------------------------------------------
# Temporal-anchor extraction (slot filling for the router)
# --------------------------------------------------------------------------
@dataclass
class TemporalHint:
    """Where in the video to look, as a fraction of duration [start, end]."""

    kind: str
    start: Optional[float] = None
    end: Optional[float] = None
    timestamp_s: Optional[int] = None

    def as_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


_ANCHORS = [
    (re.compile(r"\b(?:first few seconds|opening (?:shot|scene)|at the (?:very )?(?:start|beginning)|beginning of|start of)\b"), "start", 0.0, 0.1),
    (re.compile(r"\b(?:first half)\b"), "first_half", 0.0, 0.5),
    (re.compile(r"\b(?:second half)\b"), "second_half", 0.5, 1.0),
    (re.compile(r"\b(?:in the middle|middle (?:of|section))\b"), "middle", 0.35, 0.65),
    (re.compile(r"\b(?:at the end|end of the video|end card|by the end|at last|finally|final (?:scene|shot|lap|battle))\b"), "end", 0.9, 1.0),
]


def temporal_hint(text_norm: str) -> Optional[TemporalHint]:
    m = CUES["timestamp"].search(text_norm)
    if m:
        parts = [int(p) for p in m.group(0).split(":")]
        secs = 0
        for p in parts:
            secs = secs * 60 + p
        return TemporalHint(kind="timestamp", timestamp_s=secs)
    for rx, kind, s, e in _ANCHORS:
        if rx.search(text_norm):
            return TemporalHint(kind=kind, start=s, end=e)
    return None
