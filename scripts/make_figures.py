"""Generate every README and report figure from files in data/ and reports/.

Usage:
    python scripts/make_figures.py qrs         # detector figures (needs MIT-BIH)
    python scripts/make_figures.py quality     # LOSO figures (needs reports/metrics)
    python scripts/make_figures.py rejection   # noise-stress figure
    python scripts/make_figures.py all
"""

from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import pandas as pd
from scipy import signal

from ecgq.filters import highpass_baseline, lowpass, notch_powerline
from ecgq.io import read_beat_record
from ecgq.plotting import (
    DETECTOR_COLORS,
    DETECTOR_NAMES,
    QUALITY_COLORS,
    plt,
    savefig,
    use_style,
)
from ecgq.utils import REPO_ROOT

FIG = REPO_ROOT / "reports" / "figures"
MET = REPO_ROOT / "reports" / "metrics"
MITDB = REPO_ROOT / "data" / "raw" / "mitdb"
PT = importlib.import_module("ecgq.detectors.pan_tompkins")


# ------------------------------------------------------------------------------- QRS


def fig_pt_stages(record: str = "105", t0: float = 120.0, dur: float = 5.0) -> None:
    rec = read_beat_record(MITDB, record)
    qrs, tr = PT.pan_tompkins(rec.signal, rec.fs, return_trace=True)
    fs, f200 = rec.fs, PT.FS_PT
    a, b = int(t0 * fs), int((t0 + dur) * fs)
    a2, b2 = int(t0 * f200), int((t0 + dur) * f200)
    t = np.arange(a, b) / fs
    t2 = np.arange(a2, b2) / f200
    d = PT.LP_DELAY + PT.HP_DELAY
    fig, ax = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, rec.signal[a:b], color="#333333")
    ax[0].set_ylabel("ECG (mV)")
    ax[0].set_title(f"Pan-Tompkins stage by stage: MIT-BIH record {record}, {rec.channel_name}")
    ax[1].plot(t2, tr.bandpassed[a2 + d : b2 + d], color="#0072B2")
    ax[1].set_ylabel("band-pass\n5 to 15 Hz")
    ax[2].plot(t2, tr.derivative[a2 + d + 2 : b2 + d + 2], color="#56B4E9")
    ax[2].set_ylabel("derivative")
    ax[3].plot(t2, tr.squared[a2 + d + 2 : b2 + d + 2], color="#E69F00")
    ax[3].set_ylabel("squared")
    mwi_shift = d + 2 + PT.MWI_WIDTH // 2
    ax[4].plot(t2, tr.integrated[a2 + mwi_shift : b2 + mwi_shift], color="#009E73",
               label="moving-window integral (150 ms)")
    hist = np.array([(p, ti) for p, ti, _ in tr.thresholds])
    m = (hist[:, 0] >= a2 + mwi_shift) & (hist[:, 0] < b2 + mwi_shift)
    ax[4].step((hist[m, 0] - mwi_shift) / f200, hist[m, 1], where="post", color="#D55E00",
               label="adaptive threshold I1")
    ax[4].set_ylabel("integrated")
    ref = rec.beats[(rec.beats >= a) & (rec.beats < b)]
    det = qrs[(qrs >= a) & (qrs < b)]
    ax[0].plot(det / fs, rec.signal[det], "v", color="#0072B2", ms=8, label="detected R peak")
    ax[0].plot(ref / fs, rec.signal[ref] + 0.25, "|", color="#000000", ms=12, mew=2,
               label="expert beat label")
    lo, hi = ax[0].get_ylim()
    ax[0].set_ylim(lo, hi + 0.45 * (hi - lo))
    ax[0].legend(loc="upper center", ncol=2)
    lo, hi = ax[4].get_ylim()
    ax[4].set_ylim(lo, hi + 0.5 * (hi - lo))
    ax[4].legend(loc="upper center", ncol=2)
    ax[4].set_xlabel("time (s)")
    for x in ax:
        x.yaxis.set_label_coords(-0.07, 0.5)
    savefig(fig, FIG / "pan_tompkins_stages.png")


def fig_filter_responses() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    w, h_lp = signal.freqz(np.convolve(np.ones(6), np.ones(6)), worN=4096, fs=200)
    b_hp = -np.ones(33) / 32
    b_hp[32] = 0
    b_hp[16] += 1
    _, h_hp = signal.freqz(b_hp, worN=4096, fs=200)
    h_bp = h_lp * h_hp
    db = lambda h: 20 * np.log10(np.abs(h) / np.abs(h).max() + 1e-9)  # noqa: E731
    ax[0].plot(w, db(h_lp), label="low-pass (paper eq. 1)")
    ax[0].plot(w, db(h_hp), label="high-pass (paper eq. 3)")
    ax[0].plot(w, db(h_bp), color="#000000", lw=2, label="cascade (band-pass)")
    ax[0].axvspan(5, 15, color="#009E73", alpha=0.12, label="QRS energy band")
    ax[0].set(xlim=(0, 60), ylim=(-50, 3), xlabel="frequency (Hz)", ylabel="gain (dB)",
              title="Pan-Tompkins integer filters at 200 Hz")
    ax[0].legend(loc="upper right")

    # Analysis front end, measured by filtering an impulse with the actual functions.
    fs = 250.0
    n = 2**15
    imp = np.zeros(n)
    imp[n // 2] = 1
    f = np.fft.rfftfreq(n, 1 / fs)
    for name, y in [
        ("high-pass 0.5 Hz (baseline)", highpass_baseline(imp, fs)),
        ("notch 50 Hz (mains)", notch_powerline(imp, fs)),
        ("low-pass 40 Hz", lowpass(imp, fs)),
    ]:
        ax[1].semilogx(f[1:], 20 * np.log10(np.abs(np.fft.rfft(y))[1:] + 1e-9), label=name)
    ax[1].set(xlim=(0.05, 125), ylim=(-60, 5), xlabel="frequency (Hz)", ylabel="gain (dB)",
              title="Preprocessing filters (zero phase, 250 Hz)")
    ax[1].legend(loc="lower left")
    savefig(fig, FIG / "filter_responses.png")


def fig_qrs_per_record() -> None:
    df = pd.read_csv(MET / "qrs_mitdb_per_record.csv", dtype={"record": str})
    if "protocol" in df.columns:
        df = df[df.protocol == "ec57"]
    piv = df.pivot(index="record", columns="detector", values="f1")
    order = piv["pan_tompkins"].sort_values().index
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(order))
    for det, mk in (("wfdb_xqrs", "s"), ("elgendi", "D"), ("pan_tompkins", "o")):
        ax.plot(x, piv.loc[order, det], mk, ms=5, color=DETECTOR_COLORS[det],
                label=DETECTOR_NAMES[det])
    ax.set_xticks(x, order, rotation=90, fontsize=8)
    ax.set(ylabel="F1 per record", xlabel="MIT-BIH record (sorted by Pan-Tompkins F1)",
           title="QRS detection on all 48 MIT-BIH records (150 ms window, EC57 protocol)")
    ax.set_ylim(max(0.8, float(piv.min().min()) - 0.01), 1.003)
    ax.legend(loc="lower right")
    savefig(fig, FIG / "qrs_mitdb_per_record.png")


def fig_qrs_failure(record: str = "108", t0: float | None = None, dur: float = 8.0) -> None:
    from ecgq.metrics import match_beats

    rec = read_beat_record(MITDB, record)
    qrs = PT.pan_tompkins(rec.signal, rec.fs)
    tol = int(0.15 * rec.fs)
    # Pick the 8 s stretch with the most false positives if no time is given.
    _, found = match_beats(rec.beats, qrs, tol)
    used = np.zeros(len(qrs), bool)
    for r in rec.beats:
        j = np.flatnonzero(np.abs(qrs - r) <= tol)
        if len(j):
            used[j[np.argmin(np.abs(qrs[j] - r))]] = True
    fps = qrs[~used]
    if t0 is None:
        counts = np.histogram(fps, bins=np.arange(0, len(rec.signal), int(dur * rec.fs)))[0]
        t0 = float(np.argmax(counts) * dur)
    a, b = int(t0 * rec.fs), int((t0 + dur) * rec.fs)
    t = np.arange(a, b) / rec.fs
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t, rec.signal[a:b], color="#333333", lw=1)
    ref = rec.beats[(rec.beats >= a) & (rec.beats < b)]
    tp = qrs[used & (qrs >= a) & (qrs < b)]
    fp = fps[(fps >= a) & (fps < b)]
    miss = rec.beats[~found & (rec.beats >= a) & (rec.beats < b)]
    ax.plot(ref / rec.fs, np.full(len(ref), rec.signal[a:b].max() + 0.3), "|", color="black",
            ms=12, mew=2, label="expert beat")
    ax.plot(tp / rec.fs, rec.signal[tp], "o", mfc="none", color="#009E73", ms=9,
            label="correct detection")
    ax.plot(fp / rec.fs, rec.signal[fp], "x", color="#D55E00", ms=10, mew=2,
            label="false detection")
    ax.plot(miss / rec.fs, rec.signal[miss], "s", mfc="none", color="#E69F00", ms=10,
            label="missed beat")
    ax.set(xlabel="time (s)", ylabel="ECG (mV)",
           title=f"Failure case: Pan-Tompkins on MIT-BIH {record} (large P waves and noise)")
    ax.legend(loc="lower right", ncol=4)
    savefig(fig, FIG / f"qrs_failure_{record}.png")


# --------------------------------------------------------------------------- quality


def fig_quality() -> None:
    summ = json.loads((MET / "quality_loso_summary.json").read_text())
    per = pd.read_csv(MET / "quality_loso_per_subject.csv", dtype={"subject": str})
    models = [m for m in ["rule", "gbm_acc", "gbm_ecg", "gbm_ecg_acc", "cnn_ecg", "cnn_ecg_acc"]
              if m in summ["models"]]
    names = {"rule": "rule-based SQIs", "gbm_acc": "GBM, ACC only", "gbm_ecg": "GBM, ECG SQIs",
             "gbm_ecg_acc": "GBM, ECG + ACC", "cnn_ecg": "1D CNN, ECG",
             "cnn_ecg_acc": "1D CNN, ECG + ACC"}

    # 1) model comparison: pooled macro-F1 and per-class F1
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(models))
    wdt = 0.2
    for k, (key, col) in enumerate([("1", QUALITY_COLORS[1]), ("2", QUALITY_COLORS[2]),
                                    ("3", QUALITY_COLORS[3])]):
        ax.bar(x + (k - 1) * wdt, [summ["models"][m]["f1_per_class"][key] for m in models],
               wdt, color=col, label=f"F1 class {key}")
    ax.plot(x, [summ["models"][m]["macro_f1"] for m in models], "D", color="black", ms=8,
            label="macro-F1")
    for i, m in enumerate(models):
        ax.annotate(f"{summ['models'][m]['macro_f1']:.2f}", (i, summ["models"][m]["macro_f1"]),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontweight="bold")
    ax.set_xticks(x, [names[m] for m in models], rotation=12)
    ax.set(ylim=(0, 1.08), ylabel="F1 (pooled over 15 held-out subjects)",
           title="Quality classification, leave-one-subject-out on BUT QDB")
    ax.legend(loc="upper left", ncol=4)
    savefig(fig, FIG / "quality_model_comparison.png")

    # 2) confusion matrices of the best feature model and the best CNN
    show = [m for m in ("rule", "gbm_ecg_acc", "cnn_ecg_acc") if m in summ["models"]]
    fig, axs = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.8))
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, show, strict=True):
        cm = np.array(summ["models"][m]["confusion"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.55 else "black", fontsize=9)
        ax.set_xticks(range(3), ["1", "2", "3"])
        ax.set_yticks(range(3), ["1", "2", "3"])
        ax.set(xlabel="predicted class", ylabel="expert class", title=names[m])
    savefig(fig, FIG / "quality_confusion.png")

    # 3) research question: per-subject ECG vs ECG + ACC
    pairs = [(a, b) for a, b in (("gbm_ecg", "gbm_ecg_acc"), ("cnn_ecg", "cnn_ecg_acc"))
             if a in per.columns and b in per.columns]
    if pairs:
        fig, axs = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.6))
        axs = np.atleast_1d(axs)
        for ax, (a, b) in zip(axs, pairs, strict=True):
            c = summ["comparisons"].get(f"{b}_vs_{a}", {})
            ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
            ax.scatter(per[a], per[b], s=np.clip(per.windows / 40, 15, 200), color="#0072B2",
                       alpha=0.75, edgecolor="white")
            for _, r in per.iterrows():
                ax.annotate(r.subject, (r[a], r[b]), xytext=(4, -3), textcoords="offset points",
                            fontsize=7, color="#555555")
            ci = c.get("pooled_diff_bootstrap_ci95", [np.nan, np.nan])
            ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel=f"{names[a]}: subject macro-F1",
                   ylabel=f"{names[b]}: subject macro-F1",
                   title=f"{c.get('subjects_improved', '?')} of {c.get('subjects', '?')} subjects "
                         f"improve, Wilcoxon p = {c.get('wilcoxon_p', np.nan):.2f}")
            ax.text(0.02, 0.97, f"pooled macro-F1 change, 95% CI\n[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                    transform=ax.transAxes, va="top", fontsize=9)
        fig.suptitle("Do accelerometer features help on unseen subjects? (point size = windows)",
                     fontweight="bold")
        savefig(fig, FIG / "research_question_acc.png")


def fig_rejection() -> None:
    s = json.loads((MET / "rejection_nstdb_summary.json").read_text())["by_snr"]
    order = ["clean", "24", "18", "12", "06", "00", "-6"]
    labels = ["clean", "24", "18", "12", "6", "0", "-6"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    styles = {"none": ("no gate", "#7F7F7F", "o"), "rule": ("rule-based gate", "#CC79A7", "^"),
              "gbm_ecg": ("GBM gate", "#0072B2", "D"), "cnn_ecg": ("CNN gate", "#E69F00", "s"),
              "oracle": ("oracle (knows noise)", "#000000", "")}
    for g, (lab, col, mk) in styles.items():
        ls = ":" if g == "oracle" else "-"
        ax[0].plot(x, [s[o][g]["ppv"] for o in order], ls, marker=mk, color=col, label=lab)
        ax[1].plot(x, [s[o][g]["sensitivity"] for o in order], ls, marker=mk, color=col)
        ax[2].plot(x, [s[o][g]["coverage"] for o in order], ls, marker=mk, color=col)
    for a_, t in zip(ax, ["+P (precision) on kept windows", "Se (sensitivity) on kept windows",
                          "share of beats kept (coverage)"], strict=True):
        a_.set_xticks(x, labels)
        a_.set(xlabel="electrode-motion noise SNR (dB)", title=t)
    ax[0].legend(loc="lower left")
    fig.suptitle("Pan-Tompkins on noisy ECG (MIT-BIH NST DB), with and without the quality gate",
                 fontweight="bold")
    savefig(fig, FIG / "rejection_nstdb.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["qrs", "quality", "rejection", "all"])
    args = ap.parse_args(argv)
    use_style()
    if args.what in ("qrs", "all"):
        fig_pt_stages()
        fig_filter_responses()
        fig_qrs_per_record()
        fig_qrs_failure()
    if args.what in ("quality", "all"):
        fig_quality()
    if args.what in ("rejection", "all"):
        fig_rejection()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
