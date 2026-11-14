"""1280 x 640 social preview image (assets/social-preview.png) and a 320 x 160 thumbnail.

Numbers are read from reports/metrics, never typed in by hand.

Usage:
    python scripts/make_social_preview.py --record 100001
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from matplotlib.colors import ListedColormap

from ecgq.plotting import QUALITY_COLORS, plt, use_style
from ecgq.utils import REPO_ROOT

MET = REPO_ROOT / "reports" / "metrics"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default="100001")
    ap.add_argument("--model", default="gbm_ecg_acc")
    args = ap.parse_args(argv)
    use_style()
    qrs = json.loads((MET / "qrs_mitdb_summary.json").read_text())["all_records"]["ec57"]
    qual = json.loads((MET / "quality_loso_summary.json").read_text())["models"]
    pt = qrs["pan_tompkins"]
    m = qual[args.model]
    rule = qual["rule"]

    proc = REPO_ROOT / "data" / "processed"
    pred = np.load(proc / "predictions" / f"{args.record}.npz")["pred"]
    z = np.load(proc / "butqdb" / f"{args.record}.npz")
    win = float(z["win_s"])

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor("#0F172A")
    fig.text(0.05, 0.83, "ECG signal quality and beat detection", color="white", fontsize=34,
             fontweight="bold")
    fig.text(0.05, 0.74, "A subject-independent quality gate for wearable ECG", color="#CBD5E1",
             fontsize=19)
    lines = [
        f"Pan-Tompkins from the paper: Se {100 * pt['sensitivity']:.2f} %, "
        f"+P {100 * pt['ppv']:.2f} % on 48 MIT-BIH records",
        f"Quality (3 classes, unseen subjects): macro-F1 {m['macro_f1']:.2f} "
        f"vs {rule['macro_f1']:.2f} rule-based baseline",
    ]
    for i, t in enumerate(lines):
        fig.text(0.05, 0.62 - 0.07 * i, t, color="#E2E8F0", fontsize=16)
    ax = fig.add_axes((0.05, 0.12, 0.90, 0.24))
    cmap = ListedColormap([QUALITY_COLORS[k] for k in (0, 1, 2, 3)])
    ax.imshow(pred[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
              extent=(0, len(pred) * win / 3600, 0, 1))
    ax.set_yticks([])
    ax.tick_params(colors="#CBD5E1")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_facecolor("#0F172A")
    ax.set_xlabel(f"{len(pred) * win / 3600:.0f} hours of free-living ECG (BUT QDB "
                  f"{args.record}), quality predicted by a held-out model", color="#CBD5E1",
                  fontsize=12)
    fig.text(0.05, 0.035, "green: clean   orange: QRS only   red: unusable        "
             "Python, SciPy, scikit-learn, PyTorch", color="#94A3B8", fontsize=11)
    out = REPO_ROOT / "assets" / "social-preview.png"
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor(), bbox_inches=None)
    fig.savefig(REPO_ROOT / "assets" / "thumb.png", dpi=25, facecolor=fig.get_facecolor(),
                bbox_inches=None)
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
