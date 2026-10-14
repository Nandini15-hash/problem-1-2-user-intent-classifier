import csv
from collections import Counter

from src.intent_router.data import DATA, LABELS, load_realworld, load_videomme_labeled, load_videomme_questions


def test_every_question_has_exactly_one_gold_label():
    qs = load_videomme_questions(offline=True)
    with open(DATA / "videomme_intent_gold.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(qs) == 270 and len(rows) == 270
    assert {q["question_id"] for q in qs} == {r["question_id"] for r in rows}
    assert all(r["label"] in LABELS for r in rows)
    assert all(r["alt"] in ("",) + LABELS and r["alt"] != r["label"] for r in rows)


def test_sample_structure_matches_dataset_card():
    ex = load_videomme_labeled(offline=True)
    assert len({e.group for e in ex}) == 90
    assert Counter(e.meta["duration"] for e in ex) == {"short": 90, "medium": 90, "long": 90}


def test_realworld_set_is_balanced_enough():
    c = Counter(e.label for e in load_realworld())
    assert set(c) == set(LABELS) and min(c.values()) >= 25
