"""Evaluate on the official BUT QDB train/test split (Smital et al., Sci Data 2026).

The split comes from TESTrun_v01.m in https://github.com/BioSig-BUT/BUT-QDB-supplement
(MIT licence): 11 training records from 8 subjects, 7 test records from 7 other subjects.
Their protocol scores every annotated sample, so window predictions are spread over the
samples of each 10 s window and compared with the sample-wise consensus labels. This makes
the numbers comparable with the SNR baseline reported in that paper (accuracy 74.08 %,
F1 Q1 83.04 %, Q2 43.39 %, Q3 81.91 %, quoted from the paper, not re-run here).

Usage:
    python scripts/evaluate_fixed_split.py --config configs/quality_butqdb.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ecgq.quality.dataset import load_labelled
from ecgq.quality.models import RuleClassifier, balanced_weights, make_gbm
from ecgq.utils import REPO_ROOT, get_logger, load_config, save_json, set_seed

TRAIN = ["100001", "100002", "103001", "103002", "103003", "104001", "111001", "113001",
         "114001", "115001", "126001"]
TEST = ["105001", "118001", "121001", "122001", "123001", "124001", "125001"]
PUBLISHED_SNR_BASELINE = {"accuracy": 0.7408, "f1_q1": 0.8304, "f1_q2": 0.4339, "f1_q3": 0.8191}


def scores_from_confusion(cm: np.ndarray) -> dict:
    """cm[true-1, pred-1] in samples."""
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"accuracy": float(tp.sum() / cm.sum()), "f1_q1": float(f1[0]),
            "f1_q2": float(f1[1]), "f1_q3": float(f1[2]), "macro_f1": float(f1.mean()),
            "confusion_samples": cm.tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "quality_butqdb.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("evaluate_fixed_split")
    proc = REPO_ROOT / cfg["processed_dir"]
    tr = load_labelled(proc, TRAIN)
    w = balanced_weights(tr.y, tr.record)
    models = {
        "rule": (RuleClassifier().fit(tr.X_ecg, tr.y, sample_weight=w), "ecg"),
        "gbm_ecg": (make_gbm(cfg["seed"]).fit(tr.X_ecg, tr.y, sample_weight=w), "ecg"),
        "gbm_ecg_acc": (make_gbm(cfg["seed"]).fit(np.hstack([tr.X_ecg, tr.X_acc]), tr.y,
                                                  sample_weight=w), "both"),
    }
    log.info("trained on %d windows from %d records", len(tr), len(TRAIN))

    results = {"split": {"train": TRAIN, "test": TEST},
               "published_snr_baseline_sample_level": PUBLISHED_SNR_BASELINE, "models": {}}
    for name, (clf, feats) in models.items():
        cm = np.zeros((3, 3), dtype=np.int64)
        for rec in TEST:
            d = np.load(Path(proc) / f"{rec}.npz")
            X = d["ecg_feats"] if feats == "ecg" else np.hstack([d["ecg_feats"], d["acc_feats"]])
            counts = d["label_counts"]
            annotated = counts[:, 1:].sum(axis=1) > 0
            pred = np.asarray(clf.predict(X[annotated])).astype(int)
            for c_true in (1, 2, 3):
                np.add.at(cm[c_true - 1], pred - 1, counts[annotated, c_true])
        res = scores_from_confusion(cm)
        results["models"][name] = res
        log.info("%-12s sample-level acc %.4f  F1 Q1 %.4f Q2 %.4f Q3 %.4f", name,
                 res["accuracy"], res["f1_q1"], res["f1_q2"], res["f1_q3"])
    save_json(results, REPO_ROOT / cfg["out_dir"] / "metrics" / "quality_fixed_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
