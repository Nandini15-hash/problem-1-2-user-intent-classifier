"""Benchmark: accuracy, routing regret, calibration, latency and size per tier.

Protocol
--------
* In-domain: VideoMME-sample10, 270 questions from 90 videos. Each video
  contributes 3 questions that share entities and wording, so we use
  **GroupKFold by video_id** (5 folds x 5 seeded repeats) - a plain random
  split would leak and inflate scores.
* Out-of-domain: 100 hand-written, realistic search queries across corpus
  types the brief lists (gameplay, egocentric, podcasts, wildlife, sports,
  meetings, surveillance, ...). Models are trained on all of VideoMME (+
  synthetic templates when enabled) and never see these.
* Uncertainty: 95 % cluster-bootstrap CIs (resampling videos, not questions).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from .data import LABEL2ID, LABELS, ROOT, Example, load_realworld, load_videomme_labeled
from .models import build
from .routing import RoutingPolicy, describe_policy, routing_metrics
from .synth import synth_queries

CONFIGS = [
    # (display name, model key, use synthetic augmentation)
    ("T0 rules", "rules", False),
    ("T1 cue-LR", "cue_lr", False),
    ("T1 cue-LR +synth", "cue_lr", True),
    ("T2 tfidf+cue-LR", "tfidf_cue_lr", False),
    ("T2 tfidf+cue-LR +synth", "tfidf_cue_lr", True),
]
PRODUCTION = ("tfidf_cue_lr", True)

# Weak "second annotator": Video-MME task_type -> granularity. Used only to sanity-check our labels.
TASKTYPE_PRIOR = {
    "Information Synopsis": "whole_video", "Temporal Reasoning": "whole_video", "Counting Problem": "whole_video",
    "Attribute Perception": "frame", "OCR Problems": "frame", "Object Recognition": "frame",
    "Spatial Perception": "frame", "Action Recognition": "clip", "Action Reasoning": "clip",
    "Object Reasoning": "clip", "Temporal Perception": "clip", "Spatial Reasoning": "clip",
}


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(1)
    pred = proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def scores(y_true: List[str], proba: np.ndarray, alts: List = None, policy: RoutingPolicy = None) -> Dict:
    y = np.array([LABEL2ID[l] for l in y_true])
    pred = proba.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": ece(proba, y),
    }
    for i, l in enumerate(LABELS):
        out[f"f1_{l}"] = float(f1_score(y == i, pred == i))
    if alts is not None:
        ok = [(p == t) or (a is not None and LABELS[p] == a) for p, t, a in zip(pred, y, alts)]
        out["lenient_accuracy"] = float(np.mean(ok))
    if policy is not None:
        out.update({f"argmax_{k}": v for k, v in routing_metrics(y_true, pred, policy).items()})
        out.update({f"bayes_{k}": v for k, v in routing_metrics(y_true, policy.decide(proba), policy).items()})
    return out


def _group_folds(groups: List[str], k: int, seed: int):
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    for i in range(k):
        yield np.where(f != i)[0], np.where(f == i)[0]


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """1-parameter temperature scaling (Guo et al. 2017) by grid search on NLL."""
    z = np.log(np.clip(proba, 1e-12, 1.0))
    best = (np.inf, 1.0)
    for T in np.arange(0.5, 5.01, 0.1):
        q = apply_temperature(z, T, logits=True)
        nll = -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
        best = min(best, (nll, float(T)))
    return round(best[1], 2)


def apply_temperature(p: np.ndarray, T: float, logits: bool = False) -> np.ndarray:
    z = p if logits else np.log(np.clip(p, 1e-12, 1.0))
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def oof_predictions(model_key: str, use_synth: bool, data: List[Example], k=5, repeats=5, calibrate=True):
    """Out-of-fold probabilities. Temperature is cross-fitted: the T applied to
    fold i is fitted only on the other folds' out-of-fold predictions."""
    texts = [e.text for e in data]
    labels = [e.label for e in data]
    groups = [e.group for e in data]
    synth = synth_queries() if use_synth else []
    runs = []
    for rep in range(repeats):
        proba = np.zeros((len(data), 3))
        folds = list(_group_folds(groups, k, seed=rep))
        for tr, te in folds:
            m = build(model_key).fit([texts[i] for i in tr] + [s.text for s in synth],
                                     [labels[i] for i in tr] + [s.label for s in synth])
            proba[te] = m.predict_proba([texts[i] for i in te])
        if calibrate:
            y = np.array([LABEL2ID[l] for l in labels])
            cal = np.zeros_like(proba)
            for tr, te in folds:
                cal[te] = apply_temperature(proba[te], fit_temperature(proba[tr], y[tr]))
            proba = cal
        runs.append(proba)
    return runs


def cluster_bootstrap(data: List[Example], proba: np.ndarray, metric: str, policy, n=1000, seed=0):
    groups = np.array([e.group for e in data])
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        s = scores([data[i].label for i in idx], proba[idx], policy=policy)
        vals.append(s[metric])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def latency_and_size(model, sample_texts: List[str], n=2000):
    for t in sample_texts[:20]:
        model.predict_proba([t])
    times = []
    for i in range(n):
        t = sample_texts[i % len(sample_texts)]
        s = time.perf_counter()
        model.predict_proba([t])
        times.append(time.perf_counter() - s)
    batch = sample_texts * (2000 // len(sample_texts) + 1)
    s = time.perf_counter()
    model.predict_proba(batch[:2000])
    thr = 2000 / (time.perf_counter() - s)
    return {
        "p50_ms": float(np.percentile(times, 50) * 1e3),
        "p95_ms": float(np.percentile(times, 95) * 1e3),
        "batch_qps": float(thr),
        "size_kb": len(pickle.dumps(model)) / 1024,
    }


def run(out_dir: Path = None, repeats: int = 5, quick: bool = False) -> Dict:
    out_dir = out_dir or ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    policy = RoutingPolicy()
    vm = load_videomme_labeled()
    rw = load_realworld()
    synth = synth_queries()
    if quick:
        repeats = 1

    report = {"n_videomme": len(vm), "n_videos": len({e.group for e in vm}), "n_realworld": len(rw),
              "n_synthetic": len(synth), "label_dist_videomme": {l: sum(e.label == l for e in vm) for l in LABELS},
              "label_dist_realworld": {l: sum(e.label == l for e in rw) for l in LABELS}, "configs": {}}

    prior = [TASKTYPE_PRIOR[e.meta["task_type"]] for e in vm]
    report["label_sanity"] = {
        "agreement_with_tasktype_prior": float(np.mean([p == e.label for p, e in zip(prior, vm)])),
        "cohen_kappa_with_tasktype_prior": float(cohen_kappa_score([e.label for e in vm], prior)),
        "ambiguous_items_with_alt_label": int(sum(e.alt is not None for e in vm)),
    }

    for name, key, use_synth in CONFIGS:
        print(f"[eval] {name} ...", flush=True)
        runs = oof_predictions(key, use_synth, vm, repeats=repeats)
        per_rep = [scores([e.label for e in vm], p, [e.alt for e in vm], policy) for p in runs]
        mean = {k: float(np.mean([r[k] for r in per_rep])) for k in per_rep[0]}
        std = {k: float(np.std([r[k] for r in per_rep])) for k in per_rep[0]}
        ci = {m: cluster_bootstrap(vm, runs[0], m, policy, n=200 if quick else 1000)
              for m in ("macro_f1", "bayes_routing_regret")}
        full = build(key).fit([e.text for e in vm] + ([s.text for s in synth] if use_synth else []),
                              [e.label for e in vm] + ([s.label for s in synth] if use_synth else []))
        T = fit_temperature(oof_predictions(key, use_synth, vm, repeats=1, calibrate=False)[0],
                            np.array([LABEL2ID[e.label] for e in vm]))
        rw_proba = apply_temperature(full.predict_proba([e.text for e in rw]), T)
        rw_scores = scores([e.label for e in rw], rw_proba, policy=policy)
        perf = latency_and_size(full, [e.text for e in vm + rw], n=500 if quick else 2000)
        report["configs"][name] = {"model": key, "synthetic": use_synth, "videomme_cv_mean": mean,
                                   "videomme_cv_std": std, "videomme_ci95": ci, "realworld": rw_scores, "temperature": T,
                                   "perf": perf}

    # production model compiled to numpy-only inference (same predictions, see tests/test_fast.py)
    from .fast import FastRouter
    key, syn = PRODUCTION
    prod = build(key).fit([e.text for e in vm + synth], [e.label for e in vm + synth])
    T_prod = report["configs"][CONFIGS[-1][0]]["temperature"]
    fast = FastRouter.from_sklearn(prod, temperature=T_prod)
    fast.save(out_dir / "router")
    perf = latency_and_size(fast, [e.text for e in vm + rw], n=500 if quick else 2000)
    perf["size_kb"] = sum((out_dir / f).stat().st_size for f in ("router.npz", "router.vocab.json")) / 1024
    report["compiled_production_perf"] = perf

    # sensitivity of the headline decision metric to the cost weight mu (production model)
    runs = oof_predictions(key, syn, vm, repeats=1)
    sens = []
    for mu in (0.0, 0.1, 0.15, 0.25, 0.4):
        pol = RoutingPolicy(mu=mu)
        s = scores([e.label for e in vm], runs[0], policy=pol)
        sens.append({"mu": mu, "argmax_regret": s["argmax_routing_regret"], "bayes_regret": s["bayes_routing_regret"],
                     "bayes_answerability": s["bayes_answerability"], "bayes_rel_cost": s["bayes_rel_cost_vs_oracle"]})
    report["mu_sensitivity"] = sens

    # confusion matrix of production model (out-of-fold, repeat 0)
    y = np.array([LABEL2ID[e.label] for e in vm])
    pred = runs[0].argmax(1)
    report["confusion_videomme_prod"] = [[int(((y == i) & (pred == j)).sum()) for j in range(3)] for i in range(3)]
    report["errors_videomme_prod"] = [
        {"q": vm[i].text[:140], "gold": vm[i].label, "pred": LABELS[pred[i]], "p": round(float(runs[0][i].max()), 2)}
        for i in np.where(pred != y)[0][:25]
    ]
    report["policy"] = describe_policy(policy)

    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_markdown(report)
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return report


def render_markdown(r: Dict) -> str:
    L = [f"# Results (auto-generated)\n",
         f"VideoMME sample: {r['n_videomme']} questions / {r['n_videos']} videos; label mix {r['label_dist_videomme']}  ",
         f"Real-world OOD set: {r['n_realworld']} queries; synthetic train queries: {r['n_synthetic']}\n",
         "## Accuracy vs latency vs size\n",
         "| config | VideoMME macro-F1 (CV, 95% CI) | lenient acc | ECE | Bayes routing regret (CI) | OOD macro-F1 | OOD regret (argmax / Bayes) | p50 ms | p95 ms | batch q/s | size KB |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in r["configs"].items():
        m, ci, rw, p = c["videomme_cv_mean"], c["videomme_ci95"], c["realworld"], c["perf"]
        L.append(f"| {name} | {m['macro_f1']:.3f} [{ci['macro_f1'][0]:.2f}-{ci['macro_f1'][1]:.2f}] | "
                 f"{m['lenient_accuracy']:.3f} | {m['ece']:.3f} | {m['bayes_routing_regret']:.3f} "
                 f"[{ci['bayes_routing_regret'][0]:.3f}-{ci['bayes_routing_regret'][1]:.3f}] | "
                 f"{rw['macro_f1']:.3f} | {rw['argmax_routing_regret']:.3f} / {rw['bayes_routing_regret']:.3f} | {p['p50_ms']:.3f} | {p['p95_ms']:.3f} | "
                 f"{p['batch_qps']:.0f} | {p['size_kb']:.1f} |")
    p = r["compiled_production_perf"]
    L.append(f"| T2 +synth **compiled** (numpy-only, production) | same as above | | | | | | {p['p50_ms']:.3f} | "
             f"{p['p95_ms']:.3f} | {p['batch_qps']:.0f} | {p['size_kb']:.1f} (on disk) |")
    L.append("\nLatency = single query, single thread, this machine; size = pickled model (compiled: .npz + vocab on disk).")
    L += ["\n## Argmax vs Bayes-optimal routing (VideoMME CV mean)\n",
          "| config | argmax regret | Bayes regret | argmax answerability | Bayes answerability | Bayes rel. cost vs oracle | catastrophic (whole->frame) |",
          "|---|---|---|---|---|---|---|"]
    for name, c in r["configs"].items():
        m = c["videomme_cv_mean"]
        L.append(f"| {name} | {m['argmax_routing_regret']:.3f} | {m['bayes_routing_regret']:.3f} | "
                 f"{m['argmax_answerability']:.3f} | {m['bayes_answerability']:.3f} | "
                 f"{m['bayes_rel_cost_vs_oracle']:.2f} | {m['bayes_catastrophic_rate']:.3f} |")
    L += ["\n## Sensitivity to cost weight mu (production model)\n", "| mu | argmax regret | Bayes regret | Bayes answerability | Bayes rel. cost |", "|---|---|---|---|---|"]
    for s in r["mu_sensitivity"]:
        L.append(f"| {s['mu']} | {s['argmax_regret']:.3f} | {s['bayes_regret']:.3f} | {s['bayes_answerability']:.3f} | {s['bayes_rel_cost']:.2f} |")
    cm = r["confusion_videomme_prod"]
    L += ["\n## Confusion matrix, production model (rows = gold, cols = predicted)\n", "| | frame | clip | whole_video |", "|---|---|---|---|"]
    for lab, row in zip(LABELS, cm):
        L.append(f"| {lab} | " + " | ".join(map(str, row)) + " |")
    s = r["label_sanity"]
    L += [f"\n## Label sanity\nAgreement of our gold labels with a naive Video-MME task_type -> granularity map: "
          f"{s['agreement_with_tasktype_prior']:.2f} (Cohen's kappa {s['cohen_kappa_with_tasktype_prior']:.2f}). "
          f"{s['ambiguous_items_with_alt_label']} items carry an acceptable alternative label.",
          "\n## Routing policy (answerability matrix A)\n"] + r["policy"]
    return "\n".join(L) + "\n"
