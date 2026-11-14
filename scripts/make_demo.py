"""Demo visuals from real held-out model outputs (run scripts/predict_all.py first).

  day    reports/figures/day_quality_<record>.png: a whole recording coloured by predicted
         quality, the expert labels, the motion from the accelerometer and the heart rate
  gif    assets/hero.gif (README) and assets/demo.mp4: ECG scrolling from clean signal into
         motion artefact, with the model's quality call and the accelerometer underneath

Usage:
    python scripts/make_demo.py day --record 100001
    python scripts/make_demo.py gif --record 100001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ecgq.filters import preprocess
from ecgq.io import official_window_mask, read_butqdb_record
from ecgq.plotting import QUALITY_COLORS, QUALITY_NAMES, plt, savefig, use_style
from ecgq.quality.features import ACC_FEATURES, ECG_FEATURES
from ecgq.utils import REPO_ROOT

PROC = REPO_ROOT / "data" / "processed"
CMAP = ListedColormap([QUALITY_COLORS[k] for k in (0, 1, 2, 3)])


def load(record: str):
    z = np.load(PROC / "butqdb" / f"{record}.npz")
    p = np.load(PROC / "predictions" / f"{record}.npz")
    return z, p


def band(ax, values, hours, label):
    ax.imshow(values[None, :], aspect="auto", cmap=CMAP, vmin=0, vmax=3, interpolation="nearest",
              extent=(0, hours, 0, 1))
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    ax.grid(False)


def fig_day(record: str) -> Path:
    z, p = load(record)
    fs, win = float(z["fs"]), float(z["win_s"])
    pred, lab = p["pred"], z["labels"]
    n = len(pred)
    hours = n * win / 3600
    t = (np.arange(n) + 0.5) * win / 3600
    accm = z["acc_feats"][:, ACC_FEATURES.index("acc_mag_std")]
    hr = z["ecg_feats"][:, ECG_FEATURES.index("hr_pt")].copy()
    hr[pred == 3] = np.nan

    rec = read_butqdb_record(REPO_ROOT / "data" / "raw" / "butqdb", record, fs_out=fs)
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(6, 3, height_ratios=[0.35, 0.35, 1.1, 1.0, 0.25, 1.6], hspace=0.35,
                          wspace=0.15)
    ax_p = fig.add_subplot(gs[0, :])
    ax_l = fig.add_subplot(gs[1, :], sharex=ax_p)
    ax_a = fig.add_subplot(gs[2, :], sharex=ax_p)
    ax_h = fig.add_subplot(gs[3, :], sharex=ax_p)
    band(ax_p, pred, hours, "model\n(held out)")
    lab_show = np.where(official_window_mask(record, z["starts"], fs, win), lab, 0)
    band(ax_l, lab_show, hours, "experts")
    ax_p.set_title(f"BUT QDB record {record}: {hours:.1f} h of free-living ECG, quality "
                   "predicted by a model that never saw this person", loc="left")
    ax_a.semilogy(t, np.maximum(accm, 1), color="#555555", lw=0.6)
    for c in (2, 3):
        m = pred == c
        ax_a.semilogy(t[m], np.maximum(accm[m], 1), ".", ms=2.5, color=QUALITY_COLORS[c])
    ax_a.set_ylabel("motion\n(ACC std, mg)")
    ax_h.plot(t, hr, color="#0072B2", lw=0.6)
    ax_h.set(ylabel="heart rate\n(bpm)", ylim=(30, 200))
    ax_h.set_xlabel("hours since start of recording")
    for ax in (ax_p, ax_l, ax_a):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_h.set_xlim(0, hours)

    # Example strips: one window per class where model and experts agree, if any.
    rng = np.random.default_rng(0)
    for k, c in enumerate((1, 2, 3)):
        ax = fig.add_subplot(gs[5, k])
        cand = np.flatnonzero((pred == c) & (lab == c))
        if len(cand) == 0:
            cand = np.flatnonzero(pred == c)
        if len(cand) == 0:
            ax.axis("off")
            continue
        i = int(rng.choice(cand))
        s = int(z["starts"][i])
        w = int(win * fs)
        lo, hi = max(0, s - int(2 * fs)), min(len(rec.ecg), s + w + int(2 * fs))
        seg = preprocess(rec.ecg[lo:hi].astype(float), fs)[s - lo : s - lo + w] / 1000.0
        tt = np.arange(len(seg)) / fs
        ax.plot(tt, seg, color=QUALITY_COLORS[c], lw=0.8)
        ax.set_title(f"{QUALITY_NAMES[c]}, at {s / fs / 3600:.2f} h", fontsize=10)
        ax.set_xlabel("seconds")
        if k == 0:
            ax.set_ylabel("ECG (mV)")
        ax.set_yticks([])
        ax.axvspan(0, 10, color=QUALITY_COLORS[c], alpha=0.06)
    handles = [Patch(color=QUALITY_COLORS[k], label=QUALITY_NAMES[k]) for k in (1, 2, 3, 0)]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.06),
               frameon=False)
    return savefig(fig, REPO_ROOT / "reports" / "figures" / f"day_quality_{record}.png")


def pick_transition(pred: np.ndarray, accm: np.ndarray) -> int:
    """Window index where clean signal (class 1) turns into a run of class 3 with motion."""
    best, best_score = None, -1.0
    for i in range(8, len(pred) - 6):
        if np.all(pred[i - 6 : i] == 1) and np.sum(pred[i : i + 6] == 3) >= 3:
            score = float(np.nanmean(accm[i : i + 6]))
            if score > best_score:
                best, best_score = i, score
    if best is None:
        best = int(np.argmax(pred == 3)) if np.any(pred == 3) else len(pred) // 2
    return best


def make_gif(record: str, start_window: int | None = None, seconds: float = 60.0,
             fps_anim: int = 10, step_s: float = 0.5) -> tuple[Path, Path]:
    z, p = load(record)
    fs, win = float(z["fs"]), float(z["win_s"])
    pred, proba = p["pred"], p["proba"]
    lab = np.where(official_window_mask(record, z["starts"], fs, win), z["labels"], 0)
    accm = z["acc_feats"][:, ACC_FEATURES.index("acc_mag_std")]
    i0 = start_window if start_window is not None else pick_transition(pred, accm)
    t_start = max(0.0, (i0 - 4) * win)
    rec = read_butqdb_record(REPO_ROOT / "data" / "raw" / "butqdb", record, fs_out=fs)
    a, b = int(t_start * fs), int((t_start + seconds + 10) * fs)
    pad = int(5 * fs)
    lo = max(0, a - pad)
    ecg = preprocess(rec.ecg[lo : b + pad].astype(float), fs)[a - lo : a - lo + (b - a)] / 1000.0
    fa = rec.fs_acc
    acc = rec.acc[int(t_start * fa) : int((t_start + seconds + 10) * fa)].astype(float)
    acc_mag = np.linalg.norm(acc, axis=1)
    acc_mag -= np.median(acc_mag)
    beats = z["beats_pt"]
    beats = (beats[(beats >= a) & (beats < b)] - a) / fs
    t_ecg = np.arange(len(ecg)) / fs
    t_acc = np.arange(len(acc_mag)) / fa
    # Scale the ECG axis to the clean beats at the start; artefact larger than that is clipped
    # at the axis edge, which is itself the visual cue that the signal is unusable.
    lim = float(np.percentile(np.abs(ecg[: int(20 * fs)]), 99.5)) * 1.8
    ecg = np.clip(ecg, -lim * 0.98, lim * 0.98)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 5.4), height_ratios=[2.2, 1],
                                   sharex=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.11, hspace=0.35)
    title = fig.suptitle("", x=0.08, ha="left", fontsize=13, fontweight="bold")
    sub = fig.text(0.08, 0.895, "", fontsize=9.5, color="#444444")
    (l_ecg,) = ax1.plot([], [], color="#222222", lw=1.0)
    (l_beat,) = ax1.plot([], [], "v", color="#0072B2", ms=6)
    ax1.set(ylim=(-lim, lim), ylabel="ECG (mV)")
    (l_acc,) = ax2.plot([], [], color="#555555", lw=0.9)
    amax = max(50.0, float(np.percentile(np.abs(acc_mag), 99.5)) * 1.2)
    ax2.set(ylim=(-amax, amax), ylabel="motion (mg)", xlabel="seconds")
    spans: list = []

    def frame(k):
        t0 = k * step_s
        t1 = t0 + 10.0
        m = (t_ecg >= t0) & (t_ecg < t1)
        l_ecg.set_data(t_ecg[m], ecg[m])
        bm = (beats >= t0) & (beats < t1)
        l_beat.set_data(beats[bm], np.full(bm.sum(), 0.92 * lim))
        ma = (t_acc >= t0) & (t_acc < t1)
        l_acc.set_data(t_acc[ma], acc_mag[ma])
        for ax in (ax1, ax2):
            ax.set_xlim(t0, t1)
        for s_ in spans:
            s_.remove()
        spans.clear()
        first = int((t_start + t0) // win)
        for wi in range(first, first + 2):
            if wi >= len(pred):
                break
            ws = wi * win - t_start
            for ax in (ax1, ax2):
                spans.append(ax.axvspan(ws, ws + win, color=QUALITY_COLORS[int(pred[wi])],
                                        alpha=0.13, lw=0))
        cur = int((t_start + t0 + 5.0) // win)
        c = int(pred[min(cur, len(pred) - 1)])
        conf = proba[min(cur, len(pred) - 1)]
        conf_txt = f"  (p = {np.nanmax(conf):.2f})" if np.isfinite(conf).all() else ""
        title.set_text(f"Model: {QUALITY_NAMES[c]}{conf_txt}")
        title.set_color(QUALITY_COLORS[c])
        e = int(lab[min(cur, len(lab) - 1)])
        hh = (t_start + t0) / 3600
        sub.set_text(f"BUT QDB record {record} at {hh:.2f} h, model never trained on this person"
                     f"   |   experts: {QUALITY_NAMES[e]}")
        return [l_ecg, l_beat, l_acc, title, sub, *spans]

    n_frames = int(seconds / step_s)
    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    assets = REPO_ROOT / "assets"
    assets.mkdir(exist_ok=True)
    gif = assets / "hero.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=fps_anim), dpi=80)
    mp4 = assets / "demo.mp4"
    try:
        anim.save(mp4, writer=animation.FFMpegWriter(fps=fps_anim * 2, bitrate=2500), dpi=133)
    except (FileNotFoundError, RuntimeError):
        mp4 = None
    plt.close(fig)
    return gif, mp4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["day", "gif", "all"])
    ap.add_argument("--record", default="100001")
    ap.add_argument("--start-window", type=int, default=None,
                    help="10 s window index where the GIF should show the transition")
    ap.add_argument("--seconds", type=float, default=60.0, help="length of the scroll")
    ap.add_argument("--step", type=float, default=0.5, help="seconds of signal per frame")
    args = ap.parse_args(argv)
    use_style()
    if args.what in ("day", "all"):
        print(fig_day(args.record))
    if args.what in ("gif", "all"):
        print(make_gif(args.record, args.start_window, args.seconds, step_s=args.step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
