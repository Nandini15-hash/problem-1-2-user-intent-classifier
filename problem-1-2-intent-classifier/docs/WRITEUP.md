# User Intent Classifier for Video Retrieval: write-up

**In one line:** a 136 KB, 0.15 ms-per-query router decides whether a question needs a **frame**, a **clip** or the **whole video**. It outputs calibrated probabilities, and a small decision layer turns them into the route that costs least in lost answers plus compute, rather than the most likely class.

## 1. Problem framing

A video RAG index is multi-resolution: keyframes (with OCR and captions), clip segments (with ASR and clip captions), and video-level summaries. Each query should go to the cheapest index that can still answer it. Getting this wrong costs in two different ways:

* **Under-routing.** Sending a whole-video question to the frame index. The answer is *lost*: no single frame holds "the main idea".
* **Over-routing.** Sending a frame question to the whole-video path. The answer usually survives, but the query pays roughly 10× the compute.

So we optimise **routing regret**, not raw accuracy (section 4).

```mermaid
flowchart LR
  Q[user query] --> N[normalise<br/>+ cue lexicons<br/>+ temporal-anchor slots]
  N --> M[linear model<br/>word+char TF-IDF + 34 cue feats<br/>temperature-calibrated]
  M -->|p frame, clip, whole| D{Bayes decision<br/>argmax_r sum_t p_t U_t,r}
  D -->|frame| F[keyframe + OCR ANN index]
  D -->|clip| C[segment index + short VLM call]
  D -->|whole_video| W[video summaries / map-reduce]
  N -. "at 12:40", "at the end" .-> F & C
```

## 2. Data and labels

* **VideoMME 10% sample:** 270 questions over 90 videos, downloaded from HF (text only, never the 11 GB of video). VideoMME has no routing labels, so we labelled every question using a written guide (`docs/labeling_guide.md`). The test: *what is the smallest unit of video guaranteed to contain the answer?* 48 borderline items keep a second acceptable label, and we report lenient accuracy for them.
* **Distribution shift is the real risk.** VideoMME stems are exam-style ("Which of the following…"), while search users type "show me when the dog jumps". We added **450 synthetic queries** from a seeded template grammar covering the corpus types the brief lists (gameplay, podcasts, egocentric, wildlife, sports, meetings, surveillance…). We test on **100 separately hand-written real-world queries** that are never used for training.

## 3. Feature engineering

The features are cheap, interpretable and need no model download.

1. **Normalisation:** NFKC, lower-casing, and collapsing enumerated option lists ("(a)… (b)…") into one `<enum>` token. The list's presence signals "order these events" (whole video); its content is noise.
2. **18 cue lexicons** (regex) that encode *why* a granularity is needed. Global: topic, scope, ordering, exhaustive negation, counting. Clip: event verbs, how/why, temporal relations, segment requests. Frame: appearance, on-screen text, static spatial position, frame requests, timestamps. Plus the wh-word and length, for **34 dense features**.
3. **Word 1–2-gram and char_wb 3–5-gram TF-IDF.** The char n-grams make it robust to typos and morphology ("colour/color", "scores/scored").
4. **Slot extraction for the router (not a classifier input):** timestamps ("12:40" → 760 s) and relative anchors ("first half" → [0, 0.5]) let the frame and clip indexes search a narrow window.

## 4. Metric: routing regret, plus Bayes-optimal routing

We define `A[t,r]`, the chance an answer survives when a true-`t` query is routed to `r`, and `cost[r]` = (1, 3, 10). Utility is `U = A − μ·cost/max(cost)`. **Regret** = `U[t,t] − U[t,r̂]`, averaged over queries. 0 means oracle routing, and it is measured in "answers lost or equivalent compute". Because the model outputs probabilities, we route to `argmax_r Σ_t p(t|q)·U[t,r]`, which **escalates to a safer, coarser index when unsure**. That only works if the probabilities are honest, so we fit **temperature scaling**, cross-fitted inside CV so no fold calibrates on itself. A, cost and μ are stated assumptions in config, and we report sensitivity to μ.

Supporting metrics: macro-F1, lenient accuracy, ECE, catastrophic rate (whole→frame), latency (p50/p95) and size. The protocol is **GroupKFold by video**, 5 folds × 5 repeats; the 3 questions per video share entities, so a random split would leak. CIs come from a **95% cluster bootstrap over videos**.

## 5. Results: the accuracy / latency / size trade-off

| Tier | VideoMME macro-F1 (95% CI) | lenient acc | regret ↓ | OOD macro-F1 | p50 latency | size |
|---|---|---|---|---|---|---|
| T0 rules (no training) | 0.685 [0.63–0.74] | 0.770 | 0.108 | 0.847 | 0.06 ms | 0.1 KB |
| T1 cue-LR | 0.726 | 0.821 | 0.103 | 0.694 | 0.26 ms | 2 KB |
| T2 TF-IDF + cues | **0.734** | 0.822 | 0.101 | 0.737 | 1.2 ms | 369 KB |
| **T2 + synthetic (prod)** | 0.723 [0.67–0.77] | 0.811 | **0.094** | **0.923** | 1.2 ms → **0.15 ms compiled** | **136 KB** |

* **Synthetic augmentation is the largest single gain.** In-domain macro-F1 does not change measurably (CIs overlap), while OOD macro-F1 rises from 0.74 to 0.92. Without it, the learned models overfit exam phrasing and do *worse* than the hand-written rules on real queries.
* **Bayes routing cuts in-domain regret by about 20%** (0.119 → 0.094) and raises answerability from 0.884 to 0.911, for 6% more compute than the oracle. Whole-video questions sent to the frame index fall to 0.4%. On OOD queries it helps the unsure model (T2 without synthetic data: 0.135 → 0.102). For the production model, which is already right 92% of the time there, the in-domain temperature under-states its confidence, so Bayes over-escalates slightly (0.051 vs 0.037 for argmax). **Recalibrate on live traffic.** Example: "what is the score shown at halftime" gets p(frame) = 0.63 and is escalated to the clip index, which is cheap insurance.
* **The largest error mode is frame↔clip** (35 of 71 in-domain errors), which is also where annotators disagree. The ceiling here is label ambiguity, not model capacity. Hyperparameters are flat across a 9-point grid (macro-F1 between 0.714 and 0.727), so we did not tune on test.
* **Compiling** the linear model to numpy dict lookups gives a probability difference of 1e-7 (tested), **8× lower latency** and no scikit-learn at serve time.

## 6. Deployment on paper

* **Where:** an in-process library call (or the included stdlib HTTP sidecar: `POST /route`) inside the query service, ahead of retrieval. The artifact is one `.npz` plus a vocab JSON, loaded once and served read-only, so it is thread-safe.
* **Scale:** one core does more than 6k queries/s single-threaded, and the service is stateless, so it scales horizontally. At 1k QPS this is a rounding error next to retrieval.
* **Cascade:** if max p < 0.5, the Bayes layer already escalates. Optionally, about 5% of traffic (the uncertain slice) could go to a small LLM classifier.
* **Feedback loop:** log the query, probabilities, route and whether the answer was found. Use answer failures on frame or clip routes as free labels, and re-fit plus re-calibrate weekly. Alert on drift in the cue-feature distribution and on the share of escalated routes.
* **Safety valve:** μ and A sit in config, so ops can trade compute for answer rate without retraining.

## 7. Limitations and next steps

There is one annotator, and the same author wrote the rules, templates and OOD queries, so the OOD numbers are optimistic. A second annotator plus real query logs come first. Multi-intent queries ("summarise, and show the goal") need multi-label routing. The A matrix should be measured from end-to-end QA runs, not assumed. A distilled MiniLM tier (about 22M params) is the next accuracy step if frame vs clip ambiguity turns out to be learnable.
