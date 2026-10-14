import numpy as np

from src.intent_router.routing import RoutingPolicy, routing_metrics


def test_oracle_routing_has_zero_regret():
    pol = RoutingPolicy()
    y = ["frame", "clip", "whole_video"]
    m = routing_metrics(y, [0, 1, 2], pol)
    assert m["routing_regret"] == 0 and m["answerability"] == 1


def test_sending_whole_video_question_to_frame_is_worst():
    pol = RoutingPolicy()
    bad = routing_metrics(["whole_video"], [0], pol)["routing_regret"]
    mild = routing_metrics(["frame"], [1], pol)["routing_regret"]
    assert bad > 0.5 > mild and routing_metrics(["whole_video"], [0], pol)["catastrophic_rate"] == 1


def test_bayes_escalates_when_uncertain():
    pol = RoutingPolicy()
    # argmax would pick frame (0.45) but there is real mass on whole_video -> safer coarser route
    p = np.array([[0.45, 0.15, 0.40]])
    assert p.argmax() == 0 and pol.decide(p)[0] != 0
    # confident frame stays frame (cheap)
    assert pol.decide(np.array([[0.95, 0.04, 0.01]]))[0] == 0
