"""From class probabilities to a routing decision - and how to score it.

Plain accuracy treats every mistake as equal. For a retrieval router they are
not:

* Sending a *frame* question to the *clip* index usually still works (the clip
  contains the frame) - it just costs more.
* Sending a *whole-video* question to the *frame* index fails: no single frame
  holds "the main idea".

We therefore model two matrices (configurable, see `RoutingPolicy`):

    A[t, r]  probability the downstream QA can still answer a query of true
             type t when routed to index r ("answerability").
    cost[r]  relative compute cost of serving a query from index r.

Utility of routing a true-t query to r:  U[t, r] = A[t, r] - mu * cost[r] / max(cost)

Two things follow:

1. **Bayes-optimal routing.** Given calibrated p(t | q), pick
   r* = argmax_r  sum_t p(t | q) U[t, r]. Under uncertainty this naturally
   "escalates" to a coarser, safer index instead of gambling on argmax.
2. **Routing regret** (our headline metric). Regret = U[t, t] - U[t, r_hat],
   averaged over queries. 0 means oracle routing; it is in "answerability
   units", so 0.05 ~ "5 % of queries lose their answer, or pay equivalent
   compute". It rewards near-misses that are cheap and punishes the ones that
   kill the answer.

The default numbers below are assumptions, stated openly (docs/README), and
the evaluation reports a sensitivity analysis over mu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .data import LABEL2ID, LABELS

# rows = true type, cols = routed index; order = LABELS = (frame, clip, whole_video)
DEFAULT_A = np.array(
    [
        [1.00, 0.90, 0.55],  # frame question: clip index still finds it; video summary often drops fine detail
        [0.35, 1.00, 0.70],  # clip question: a single frame rarely captures an action / cause
        [0.10, 0.40, 1.00],  # whole-video question: needs global aggregation
    ]
)
# relative cost per query: ANN over keyframes < segment retrieval + short VLM call < map-reduce over all chunks
DEFAULT_COST = np.array([1.0, 3.0, 10.0])


@dataclass
class RoutingPolicy:
    A: np.ndarray = field(default_factory=lambda: DEFAULT_A.copy())
    cost: np.ndarray = field(default_factory=lambda: DEFAULT_COST.copy())
    mu: float = 0.15  # how many answerability-points a full "max cost" query is worth

    @property
    def U(self) -> np.ndarray:
        return self.A - self.mu * self.cost[None, :] / self.cost.max()

    def decide(self, proba: np.ndarray) -> np.ndarray:
        """Bayes-optimal route index for each row of class probabilities."""
        return (proba @ self.U).argmax(axis=1)


def routing_metrics(y_true: Sequence[str], routed: Sequence[int], policy: RoutingPolicy) -> Dict[str, float]:
    t = np.array([LABEL2ID[y] for y in y_true])
    r = np.asarray(routed)
    U = policy.U
    regret = U[t, t] - U[t, r]
    return {
        "routing_regret": float(regret.mean()),
        "answerability": float(policy.A[t, r].mean()),
        "rel_cost_vs_oracle": float(policy.cost[r].sum() / policy.cost[t].sum()),
        "catastrophic_rate": float(((r == 0) & (t == 2)).mean()),  # whole-video question sent to frame index
        "under_route_rate": float((r < t).mean()),  # routed finer than needed (answer at risk)
        "over_route_rate": float((r > t).mean()),   # routed coarser than needed (costs more)
    }


def route_name(i: int) -> str:
    return LABELS[i]


def describe_policy(policy: RoutingPolicy) -> List[str]:
    lines = ["| true \\ routed | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    for i, l in enumerate(LABELS):
        lines.append(f"| {l} | " + " | ".join(f"{policy.A[i, j]:.2f}" for j in range(3)) + " |")
    lines.append(f"\ncost per route = {dict(zip(LABELS, policy.cost.tolist()))}, mu = {policy.mu}")
    return lines
