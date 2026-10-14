# Labeling guide: retrieval intent

VideoMME has no routing labels, so we labelled all 270 sample questions ourselves (`data/videomme_intent_gold.csv`). The question we ask for each one is:

> **What is the smallest piece of video that is guaranteed to contain the answer?**

| Label | Smallest evidence unit | Typical cues | Examples from the sample |
|---|---|---|---|
| `frame` | A single moment. One keyframe (plus OCR) is enough. | colour, clothing, appearance, on-screen text, scores, clocks, logos, where something *is*, a count within one shot | "What colour is the news presenter's coat?" · "What is the score at the half time?" · "At the end of the video, how many people are on the staircase?" |
| `clip` | One contiguous span (seconds to a few minutes): an action, an event, its cause or consequence, a statement someone makes | why / how did, what happens after / before, a process, "what did X say about Y", first / last event | "How do the geologists cool the lava down?" · "Why did the Canadian athlete suddenly collapse?" · "What happened to the team on the counterattack after Sabonis' first steal?" |
| `whole_video` | Evidence spread across the video, or a global judgement | main idea / theme / title, ordering several events, "which is NOT mentioned", counting across the video, overall assessment of a character or a performance, the video's genre or setting | "What is the main idea of the video?" · "What is the correct chronological order …?" · "Which element doesn't show up in the video?" |

## Tie-break rules

These keep the labels consistent.

1. **Temporal anchors narrow the label.** "At the beginning / at the end / at 89:17" plus a static attribute → `frame`. The same anchor plus an action → `clip`.
2. **Counting.** A count with no anchor ("how many bridges in the video") → `whole_video`, because you must scan everything. A count inside one shot or one short event → `frame` or `clip`.
3. **Ordinals.** "First" or "last" event → `clip` (the anchor is the start or end of the video). "Second" or "third" item → `whole_video` (you need the full sequence).
4. **Negation over a list** ("which is NOT shown or mentioned") → `whole_video`, because proving absence requires exhaustive coverage. Negation scoped to one segment ("not discussed *while driving to work*") → `clip`.
5. **Video-level identity** (what sport, which country, what kind of handicraft) → `whole_video`. A video-level summary answers it most cheaply, even though many single frames could too.

## Ambiguity is recorded, not hidden

48 of the 270 items (18%) have a second, acceptable label in the `alt` column. "What does the man use to clean the keyboard?", for example, is `frame` (the cloth is visible) or `clip` (the cleaning action). We report **lenient accuracy** alongside strict scores, so the model is not punished for disagreeing with one annotator on cases that really are borderline.

## Sanity check

A naive mapping from VideoMME's own `task_type` to granularity (Information Synopsis → whole_video, OCR → frame, Action Reasoning → clip, and so on) agrees with our gold labels on 65% of items (Cohen's κ = 0.48). The map is a useful prior, but `task_type` describes the *skill* a question tests, not *where the evidence lives*. "Counting Problem" questions alone fall into all three of our classes. That gap is why routing needs its own labels.

## Limitations

There is a single annotator, and the hand-written real-world set (`data/realworld_queries_gold.csv`) was written by the same person who wrote the rule lexicons and synthetic templates. A second annotator (for inter-annotator κ) and real query logs are the first next steps.
