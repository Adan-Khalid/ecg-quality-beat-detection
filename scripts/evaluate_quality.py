"""Leave-one-subject-out (LOSO) evaluation of the quality classifiers on BUT QDB.

Each of the 15 subjects is held out in turn (all of their records together, so a subject
recorded three times never appears on both sides). Models:

  rule         interpretable SQI thresholds fitted on the training subjects
  gbm_ecg      gradient boosting on ECG SQIs
  gbm_acc      gradient boosting on accelerometer features only (ablation)
  gbm_ecg_acc  gradient boosting on ECG SQIs + accelerometer features
  cnn_ecg      1D CNN on the raw 10 s ECG window
  cnn_ecg_acc  1D CNN on ECG + 3 accelerometer channels

Writes reports/metrics/quality_loso_{summary.json,per_subject.csv,predictions.csv.gz}.

Usage:
    python scripts/evaluate_quality.py --models rule gbm_ecg gbm_acc gbm_ecg_acc cnn_ecg cnn_ecg_acc
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import f1_score

from ecgq.metrics import quality_report
from ecgq.quality.dataset import load_labelled
from ecgq.quality.models import (
    RuleClassifier,
    balanced_weights,
    make_gbm,
    predict_cnn,
    train_cnn,
)
from ecgq.utils import REPO_ROOT, Timer, get_logger, load_config, save_json, set_seed

ALL_MODELS = ["rule", "gbm_ecg", "gbm_acc", "gbm_ecg_acc", "cnn_ecg", "cnn_ecg_acc"]


def subject_macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Macro-F1 over the classes that occur in this subject's labels."""
    labels = sorted(set(np.unique(y)))
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def paired_comparison(per_subj: pd.DataFrame, pred: pd.DataFrame, a: str, b: str,
                      seed: int = 0, n_boot: int = 2000) -> dict:
    """Is model b better than model a for unseen subjects?

    Per-subject paired differences with a Wilcoxon signed-rank test, plus a bootstrap over
    subjects (resampling whole subjects) of the pooled macro-F1 difference.
    """
    d = (per_subj[b] - per_subj[a]).to_numpy()
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    rng = np.random.default_rng(seed)
    subjects = pred.subject.unique()
    groups = {s: g for s, g in pred.groupby("subject")}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        g = pd.concat([groups[s] for s in pick])
        fa = f1_score(g.y, g[a], labels=[1, 2, 3], average="macro", zero_division=0)
        fb = f1_score(g.y, g[b], labels=[1, 2, 3], average="macro", zero_division=0)
        boots.append(fb - fa)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "a": a, "b": b, "subjects": int(len(d)),
        "mean_subject_diff": float(np.mean(d)), "median_subject_diff": float(np.median(d)),
        "subjects_improved": int(np.sum(d > 0)), "subjects_worse": int(np.sum(d < 0)),
        "wilcoxon_p": float(w.pvalue) if w is not None else 1.0,
        "pooled_diff_bootstrap_ci95": [float(lo), float(hi)],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "quality_butqdb.yaml"))
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    torch.set_num_threads(4)
    log = get_logger("evaluate_quality")
    data = load_labelled(REPO_ROOT / cfg["processed_dir"])
    log.info("labelled windows: %d  subjects: %d  records: %d  classes: %s", len(data),
             len(np.unique(data.subject)), len(np.unique(data.record)),
             dict(zip(*np.unique(data.y, return_counts=True), strict=True)))
    X_both = np.hstack([data.X_ecg, data.X_acc])
    feats = {"gbm_ecg": data.X_ecg, "gbm_acc": data.X_acc, "gbm_ecg_acc": X_both}
    cnn_ch = {"cnn_ecg": [0], "cnn_ecg_acc": [0, 1, 2, 3]}

    pred = pd.DataFrame({"record": data.record, "subject": data.subject, "start": data.start,
                         "y": data.y})
    timings = {m: 0.0 for m in args.models}
    rule_params = []
    for s in np.unique(data.subject):
        te = data.subject == s
        tr = ~te
        w = balanced_weights(data.y[tr], data.record[tr])
        log.info("fold: hold out subject %s (%d windows, classes %s)", s, te.sum(),
                 sorted(set(data.y[te])))
        for m in args.models:
            with Timer() as t:
                if m == "rule":
                    clf = RuleClassifier().fit(data.X_ecg[tr], data.y[tr], sample_weight=w)
                    pred.loc[te, m] = clf.predict(data.X_ecg[te])
                    rule_params.append({"subject": s, **clf.params()})
                elif m.startswith("gbm"):
                    clf = make_gbm(cfg["seed"]).fit(feats[m][tr], data.y[tr], sample_weight=w)
                    pred.loc[te, m] = clf.predict(feats[m][te])
                else:
                    ch = cnn_ch[m]
                    c = cfg["cnn"]
                    net = train_cnn(data.cnn_x[tr][:, ch].astype(np.float32), data.y[tr], w,
                                    epochs=c["epochs"], samples_per_epoch=c["samples_per_epoch"],
                                    batch=c["batch"], lr=c["lr"], seed=cfg["seed"])
                    p, prob = predict_cnn(net, data.cnn_x[te][:, ch].astype(np.float32))
                    pred.loc[te, m] = p
            timings[m] += t.seconds
            f = subject_macro_f1(data.y[te], pred.loc[te, m].to_numpy(int))
            log.info("  %-12s subject macro-F1 %.3f (%.0f s)", m, f, t.seconds)

    for m in args.models:
        pred[m] = pred[m].astype(int)
    out = REPO_ROOT / cfg["out_dir"] / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "quality_loso_predictions.csv.gz"
    if pred_path.exists():  # keep columns from earlier runs of other models
        old = pd.read_csv(pred_path, dtype={"record": str, "subject": str})
        if len(old) == len(pred) and (old.start.to_numpy() == pred.start.to_numpy()).all():
            for c in old.columns:
                if c not in pred.columns:
                    pred[c] = old[c]
    pred.to_csv(pred_path, index=False)

    models_done = [m for m in ALL_MODELS if m in pred.columns]
    per_subj = pred.groupby("subject").apply(
        lambda g: pd.Series({m: subject_macro_f1(g.y.to_numpy(), g[m].to_numpy())
                             for m in models_done} | {"windows": len(g)}),
        include_groups=False)
    per_subj.to_csv(out / "quality_loso_per_subject.csv", float_format="%.4f")

    summ_path = out / "quality_loso_summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    summary.update({
        "dataset": "BUT QDB 1.0.0", "protocol": "leave-one-subject-out (15 folds)",
        "window_s": cfg["window_s"], "n_windows": int(len(pred)),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(pred.y, return_counts=True),
                                                        strict=True)},
    })
    summary.setdefault("models", {})
    for m in models_done:
        rep = quality_report(pred.y, pred[m])
        rep["mean_subject_macro_f1"] = float(per_subj[m].mean())
        if m in timings:
            rep["train_eval_seconds"] = timings[m]
        summary["models"][m] = rep
        log.info("%-12s pooled macro-F1 %.3f  mean subject %.3f  per-class %s", m,
                 rep["macro_f1"], rep["mean_subject_macro_f1"], rep["f1_per_class"])
    if rule_params:
        summary["rule_thresholds_per_fold"] = rule_params
    comps = {}
    for a, b in [("gbm_ecg", "gbm_ecg_acc"), ("cnn_ecg", "cnn_ecg_acc"), ("rule", "gbm_ecg"),
                 ("gbm_ecg", "cnn_ecg")]:
        if a in pred.columns and b in pred.columns:
            comps[f"{b}_vs_{a}"] = paired_comparison(per_subj, pred, a, b, cfg["seed"])
            log.info("%s vs %s: %s", b, a, comps[f"{b}_vs_{a}"])
    summary.setdefault("comparisons", {}).update(comps)
    save_json(summary, summ_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
