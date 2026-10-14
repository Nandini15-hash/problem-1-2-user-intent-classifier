from src.intent_router.features import CUES, cue_vector, normalize, temporal_hint


def test_enumerated_options_collapse_to_token():
    t = normalize("In which order?\n(a) Lungs.\n(b) Protein.\n(c) Cells.")
    assert "<enum>" in t and "lungs" not in t


def test_timestamp_hint_in_seconds():
    h = temporal_hint(normalize("what is on screen at 1:02:03"))
    assert h.kind == "timestamp" and h.timestamp_s == 3723


def test_relative_hints():
    assert temporal_hint(normalize("At the end of the video, who wins?")).kind == "end"
    assert temporal_hint(normalize("in the first half of the video")).kind == "first_half"
    assert temporal_hint(normalize("what is this video about")) is None


def test_cues_fire_on_expected_text():
    assert CUES["global_topic"].search(normalize("What is the main idea of the video?"))
    assert CUES["appearance"].search(normalize("What color is the dragon?"))
    assert CUES["exhaustive_negation"].search(normalize("Which element is not mentioned?"))
    assert cue_vector("x").shape == cue_vector(normalize("a much longer query here")).shape
