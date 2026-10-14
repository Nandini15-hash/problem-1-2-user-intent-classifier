import numpy as np
import pytest

from src.intent_router.data import load_videomme_labeled
from src.intent_router.fast import FastRouter
from src.intent_router.models import MODELS, build
from src.intent_router.synth import synth_queries


@pytest.fixture(scope="module")
def data():
    vm = load_videomme_labeled(offline=True)
    return vm + synth_queries(n_per_class=60)


@pytest.mark.parametrize("key", list(MODELS))
def test_models_output_valid_probabilities(key, data):
    m = build(key).fit([e.text for e in data], [e.label for e in data])
    p = m.predict_proba(["what is this video about", "what color is the car"])
    assert p.shape == (2, 3) and np.allclose(p.sum(1), 1)


def test_compiled_router_matches_sklearn(tmp_path, data):
    m = build("tfidf_cue_lr").fit([e.text for e in data], [e.label for e in data])
    f = FastRouter.from_sklearn(m)
    f.save(tmp_path / "r")
    g = FastRouter.load(tmp_path / "r")
    texts = [e.text for e in data[:200]] + ["", "???", "Ünïcödé query ①"]
    assert np.abs(m.predict_proba(texts) - g.predict_proba(texts, calibrated=False)).max() < 1e-5


def test_trained_router_gets_obvious_cases_right(data):
    m = build("tfidf_cue_lr").fit([e.text for e in data], [e.label for e in data])
    labels = ["frame", "clip", "whole_video"]
    pred = [labels[i] for i in m.predict_proba([
        "summarize the whole lecture", "show me when the dog jumps into the pool", "what does the sign say"]).argmax(1)]
    assert pred == ["whole_video", "clip", "frame"]


def test_synthetic_generator_is_deterministic():
    a = [e.text for e in synth_queries(n_per_class=20, seed=1)]
    b = [e.text for e in synth_queries(n_per_class=20, seed=1)]
    assert a == b and len(set(a)) == len(a)
