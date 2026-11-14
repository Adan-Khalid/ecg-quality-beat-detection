"""Held-out predictions for every 10 s window of every BUT QDB recording.

For each subject, a gradient-boosting model (ECG SQIs + accelerometer) is trained on the
other 14 subjects and then scores every window of that subject's recordings, labelled or
not. The demo and the day-long figures use these predictions, so no picture shows a model
judging a person it was trained on.

Writes data/processed/predictions/<record>.npz (pred, proba, starts) and
reports/metrics/holdout_daylong_agreement.csv (agreement with experts on labelled windows).

Usage:
    python scripts/predict_all.py --config configs/quality_butqdb.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecgq.io import official_window_mask, subject_of
from ecgq.quality.dataset import load_labelled
from ecgq.quality.models import balanced_weights, make_gbm
from ecgq.utils import REPO_ROOT, get_logger, load_config, set_seed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "quality_butqdb.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("predict_all")
    proc = REPO_ROOT / cfg["processed_dir"]
    out = proc.parent / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    d = load_labelled(proc)
    X = np.hstack([d.X_ecg, d.X_acc])
    rows = []
    for s in np.unique(d.subject):
        tr = d.subject != s
        clf = make_gbm(cfg["seed"]).fit(X[tr], d.y[tr],
                                        sample_weight=balanced_weights(d.y[tr], d.record[tr]))
        for f in sorted(Path(proc).glob(f"{s}*.npz")):
            z = np.load(f)
            Xa = np.hstack([z["ecg_feats"], z["acc_feats"]])
            ok = np.all(np.isfinite(Xa), axis=1)
            proba = np.full((len(Xa), 3), np.nan)
            proba[ok] = clf.predict_proba(Xa[ok])
            pred = np.where(ok, np.nanargmax(np.nan_to_num(proba, nan=-1), axis=1) + 1, 0)
            np.savez_compressed(out / f"{f.stem}.npz", pred=pred, proba=proba,
                                starts=z["starts"], fs=z["fs"])
            lab = z["labels"]
            m = (lab > 0) & official_window_mask(f.stem, z["starts"], float(z["fs"]),
                                                 float(z["win_s"]))
            rows.append({"record": f.stem, "subject": subject_of(f.stem), "windows": len(lab),
                         "labelled": int(m.sum()),
                         "agreement_labelled": float(np.mean(pred[m] == lab[m])) if m.any()
                         else np.nan,
                         "pred_class3_share_all": float(np.mean(pred == 3))})
            log.info("%s: %d windows, %.1f%% predicted class 3, agreement on labelled %.3f",
                     f.stem, len(lab), 100 * rows[-1]["pred_class3_share_all"],
                     rows[-1]["agreement_labelled"])
    pd.DataFrame(rows).to_csv(REPO_ROOT / cfg["out_dir"] / "metrics" /
                              "holdout_daylong_agreement.csv", index=False, float_format="%.4f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
