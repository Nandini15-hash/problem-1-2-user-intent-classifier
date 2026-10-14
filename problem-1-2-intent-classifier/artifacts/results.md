# Results (auto-generated)

VideoMME sample: 270 questions / 90 videos; label mix {'frame': 66, 'clip': 110, 'whole_video': 94}  
Real-world OOD set: 100 queries; synthetic train queries: 450

## Accuracy vs latency vs size

| config | VideoMME macro-F1 (CV, 95% CI) | lenient acc | ECE | Bayes routing regret (CI) | OOD macro-F1 | OOD regret (argmax / Bayes) | p50 ms | p95 ms | batch q/s | size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 rules | 0.685 [0.63-0.74] | 0.770 | 0.045 | 0.108 [0.088-0.128] | 0.847 | 0.073 / 0.080 | 0.061 | 0.119 | 23100 | 0.1 |
| T1 cue-LR | 0.726 [0.66-0.76] | 0.821 | 0.072 | 0.103 [0.080-0.130] | 0.694 | 0.149 / 0.125 | 0.256 | 0.321 | 20359 | 2.3 |
| T1 cue-LR +synth | 0.707 [0.65-0.75] | 0.795 | 0.056 | 0.106 [0.078-0.123] | 0.903 | 0.043 / 0.055 | 0.252 | 0.301 | 23706 | 2.3 |
| T2 tfidf+cue-LR | 0.734 [0.66-0.77] | 0.822 | 0.071 | 0.101 [0.081-0.129] | 0.737 | 0.135 / 0.102 | 1.269 | 1.828 | 8652 | 369.1 |
| T2 tfidf+cue-LR +synth | 0.723 [0.67-0.77] | 0.811 | 0.041 | 0.094 [0.069-0.113] | 0.923 | 0.037 / 0.051 | 1.202 | 1.380 | 9162 | 483.5 |
| T2 +synth **compiled** (numpy-only, production) | same as above | | | | | | 0.152 | 0.265 | 7341 | 135.8 (on disk) |

Latency = single query, single thread, this machine; size = pickled model (compiled: .npz + vocab on disk).

## Argmax vs Bayes-optimal routing (VideoMME CV mean)

| config | argmax regret | Bayes regret | argmax answerability | Bayes answerability | Bayes rel. cost vs oracle | catastrophic (whole->frame) |
|---|---|---|---|---|---|---|
| T0 rules | 0.118 | 0.108 | 0.874 | 0.888 | 0.95 | 0.000 |
| T1 cue-LR | 0.123 | 0.103 | 0.879 | 0.897 | 1.00 | 0.007 |
| T1 cue-LR +synth | 0.124 | 0.106 | 0.878 | 0.895 | 1.01 | 0.004 |
| T2 tfidf+cue-LR | 0.120 | 0.101 | 0.882 | 0.902 | 1.04 | 0.007 |
| T2 tfidf+cue-LR +synth | 0.119 | 0.094 | 0.884 | 0.911 | 1.06 | 0.004 |

## Sensitivity to cost weight mu (production model)

| mu | argmax regret | Bayes regret | Bayes answerability | Bayes rel. cost |
|---|---|---|---|---|
| 0.0 | 0.111 | 0.086 | 0.914 | 1.15 |
| 0.1 | 0.113 | 0.090 | 0.913 | 1.08 |
| 0.15 | 0.114 | 0.091 | 0.913 | 1.05 |
| 0.25 | 0.116 | 0.097 | 0.902 | 0.99 |
| 0.4 | 0.119 | 0.095 | 0.891 | 0.93 |

## Confusion matrix, production model (rows = gold, cols = predicted)

| | frame | clip | whole_video |
|---|---|---|---|
| frame | 40 | 19 | 7 |
| clip | 16 | 80 | 14 |
| whole_video | 4 | 11 | 79 |

## Label sanity
Agreement of our gold labels with a naive Video-MME task_type -> granularity map: 0.65 (Cohen's kappa 0.48). 48 items carry an acceptable alternative label.

## Routing policy (answerability matrix A)

| true \ routed | frame | clip | whole_video |
|---|---|---|---|
| frame | 1.00 | 0.90 | 0.55 |
| clip | 0.35 | 1.00 | 0.70 |
| whole_video | 0.10 | 0.40 | 1.00 |

cost per route = {'frame': 1.0, 'clip': 3.0, 'whole_video': 10.0}, mu = 0.15
