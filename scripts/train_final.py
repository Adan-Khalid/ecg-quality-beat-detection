"""Train the final quality models on all 15 BUT QDB subjects and save them to models/.

These models are used by the demo and by the noise-stress rejection experiment. Their
accuracy is the leave-one-subject-out estimate from scripts/evaluate_quality.py; nothing
here is evaluated on its own training data.

Usage:
    python scripts/train_final.py --config configs/quality_butqdb.yaml
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch

from ecgq.quality.dataset import load_labelled
from ecgq.quality.models import RuleClassifier, balanced_weights, make_gbm, train_cnn
from ecgq.utils import REPO_ROOT, get_logger, load_config, save_json, set_seed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "quality_butqdb.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    torch.set_num_threads(4)
    log = get_logger("train_final")
    out = REPO_ROOT / "models"
    out.mkdir(exist_ok=True)
    d = load_labelled(REPO_ROOT / cfg["processed_dir"])
    w = balanced_weights(d.y, d.record)
    log.info("training on %d windows from %d subjects", len(d), len(np.unique(d.subject)))

    rule = RuleClassifier().fit(d.X_ecg, d.y, sample_weight=w)
    log.info("rule thresholds: %s", rule.params())
    gbm_ecg = make_gbm(cfg["seed"]).fit(d.X_ecg, d.y, sample_weight=w)
    gbm_ecg_acc = make_gbm(cfg["seed"]).fit(np.hstack([d.X_ecg, d.X_acc]), d.y, sample_weight=w)
    with open(out / "sklearn_models.pkl", "wb") as f:
        pickle.dump({"rule": rule, "gbm_ecg": gbm_ecg, "gbm_ecg_acc": gbm_ecg_acc}, f)

    c = cfg["cnn"]
    for name, ch in (("cnn_ecg", [0]), ("cnn_ecg_acc", [0, 1, 2, 3])):
        log.info("training %s", name)
        net = train_cnn(d.cnn_x[:, ch].astype(np.float32), d.y, w, epochs=c["epochs"],
                        samples_per_epoch=c["samples_per_epoch"], batch=c["batch"], lr=c["lr"],
                        seed=cfg["seed"], log=log)
        torch.save({"state_dict": net.state_dict(), "channels": ch}, out / f"{name}.pt")
    save_json({"rule_thresholds": rule.params(), "n_windows": int(len(d)),
               "n_subjects": int(len(np.unique(d.subject)))},
              REPO_ROOT / cfg["out_dir"] / "metrics" / "final_models.json")
    log.info("saved models to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
