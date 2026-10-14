"""Template-based synthetic queries.

Why: VideoMME questions are exam-style multiple-choice stems ("Which of the
following ..."). Real users of a video search engine type short imperative
queries ("show me when the dog jumps"). Training only on VideoMME would bake in
that style. A small, seeded template grammar covering the corpus "universe"
(gameplay, egocentric, podcasts, wildlife, sports, meetings, surveillance,
lectures, cooking ...) widens the style coverage at zero labelling cost.

The hand-written real-world test set (`data/realworld_queries_gold.csv`) was
written separately and is never used for training, so it measures how well
this augmentation generalises rather than how well it memorises templates.
"""
from __future__ import annotations

import random
from typing import List

from .data import Example

SUBJECTS = {
    "gameplay": ["the player", "the streamer", "the squad", "the main character", "the speedrunner"],
    "sports": ["the striker", "the home team", "the referee", "the sprinter", "the point guard", "the coach"],
    "wildlife": ["the cheetah", "the herd", "the eagle", "the octopus", "the wolves"],
    "meeting": ["the product manager", "the ceo", "the team", "the presenter", "the client"],
    "podcast": ["the host", "the guest", "the panel"],
    "egocentric": ["the vlogger", "the hiker", "the cyclist", "the person wearing the camera"],
    "surveillance": ["the intruder", "the delivery driver", "the customer", "the security guard"],
    "lecture": ["the professor", "the instructor", "the speaker"],
    "cooking": ["the chef", "the baker", "the home cook"],
}
EVENTS = ["scores", "falls", "opens the door", "starts the engine", "picks up the box", "jumps",
          "loses the round", "enters the room", "drops the ball", "hands over the baton", "attacks",
          "explains the formula", "shares the screen", "lights the stove", "crosses the street"]
TOPICS = ["pricing", "the roadmap", "the injury", "the new patch", "climate change", "the recipe",
          "the migration route", "security", "hiring", "the final score"]
OBJECTS = ["the jersey", "the car", "the backpack", "the mug", "the helmet", "the banner",
           "the laptop", "the bird", "the sign", "the jacket", "the trophy", "the screen"]
ATTRS = ["color", "brand", "number", "text", "shape", "logo"]
VIDEO = ["video", "clip", "stream", "match", "episode", "recording", "footage", "vlog", "talk", "game"]
ANCHORS = ["at the start", "in the opening shot", "at the end", "in the final frame", "at 03:15",
           "at 1:02:40", "when the timer hits zero", "in the thumbnail", ""]

T_WHOLE = [
    "what is this {v} about", "summarize the {v}", "give me a summary of the whole {v}",
    "what's the main point of the {v}", "list all the {o_pl} that appear in the {v}",
    "how many times does {s} {e} in the {v}", "in what order do the events happen in the {v}",
    "what topics come up across the {v}", "which of these is never mentioned in the {v}",
    "how does {s} change over the course of the {v}", "rank the moments of the {v} by importance",
    "overall, what does {s} think about {t}", "is {t} discussed anywhere in the {v}",
    "how many different people appear in the {v}", "what's the overall tone of the {v}",
    "describe the plot of the {v}", "outline the chapters of the {v}", "what kind of {v} is this",
    "tl;dr of the {v}", "which segments of the {v} cover {t}", "total number of times {s} {e}",
]
T_CLIP = [
    "show me when {s} {e}", "find the moment {s} {e}", "what happens after {s} {e}",
    "why does {s} {e}", "how does {s} {e}", "the part where {s} talks about {t}",
    "what did {s} say about {t}", "clip of {s} explaining {t}", "replay the bit where {s} {e}",
    "when does {s} {e}", "what does {s} do right before the break", "how did {s} react when {s2} {e}",
    "find the segment about {t}", "what happens during the second half", "show the sequence where {s} {e}",
    "how does {s} prepare {o}", "what was the response to the question on {t}",
    "jump to where {s} {e}", "what led to {s} {e}",
]
T_FRAME = [
    "what {a} is {o}", "what is the {a} of {o} {anc}", "read the {a} on {o}",
    "what is written on {o}", "what does the sign say {anc}", "what is {s} wearing {anc}",
    "what is the score on the scoreboard {anc}", "what time is shown on the clock {anc}",
    "screenshot of {o}", "frame showing {o}", "where is {o} in the room {anc}",
    "what is {s} holding {anc}", "how many people are in the shot {anc}", "what logo is on {o}",
    "what color is {s}'s shirt", "what does the caption say {anc}", "what is on the screen {anc}",
    "what number is on {o}", "what does the title card say", "what brand is {o}",
]


def _fill(tpl: str, rng: random.Random) -> str:
    dom = rng.choice(list(SUBJECTS))
    s = rng.choice(SUBJECTS[dom])
    s2 = rng.choice(SUBJECTS[rng.choice(list(SUBJECTS))])
    o = rng.choice(OBJECTS)
    out = tpl.format(
        s=s, s2=s2, e=rng.choice(EVENTS), t=rng.choice(TOPICS), o=o,
        o_pl=o.replace("the ", "") + "s", a=rng.choice(ATTRS), v=rng.choice(VIDEO), anc=rng.choice(ANCHORS),
    )
    out = " ".join(out.split())
    if rng.random() < 0.3:
        out += "?"
    if rng.random() < 0.2:
        out = out.capitalize()
    return out


def synth_queries(n_per_class: int = 150, seed: int = 13) -> List[Example]:
    rng = random.Random(seed)
    out: List[Example] = []
    for label, tpls in (("whole_video", T_WHOLE), ("clip", T_CLIP), ("frame", T_FRAME)):
        seen = set()
        tries = 0
        while len([x for x in out if x.label == label]) < n_per_class and tries < n_per_class * 20:
            tries += 1
            q = _fill(rng.choice(tpls), rng)
            if q.lower() in seen:
                continue
            seen.add(q.lower())
            out.append(Example(text=q, label=label, group=f"synth-{label}-{len(seen) % 25}"))
    return out
