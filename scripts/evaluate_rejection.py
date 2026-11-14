"""Does rejecting class 3 windows improve beat detection? (MIT-BIH Noise Stress Test DB)

The NSTDB records 118eXX and 119eXX are MIT-BIH records 118 and 119 with calibrated
electrode-motion noise added in 2-minute blocks at SNRs from 24 dB down to -6 dB. Their
reference beat labels are the original expert labels, so beat detection can be scored
exactly on noisy ECG. The quality models were trained only on BUT QDB (different device,
lead and population), so this is also a cross-dataset transfer test.

For every record and gate (none, rule, gbm_ecg, cnn_ecg, and an oracle that knows where
noise was added) the script reports Pan-Tompkins Se, +P and F1 on the kept windows, the
share of reference beats kept (coverage), and how often noisy and clean windows are
rejected. The noise location is found exactly by subtracting the clean MIT-BIH record.

Usage:
    python scripts/evaluate_rejection.py --config configs/rejection_nstdb.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from ecgq.filters import resample
from ecgq.io import NSTDB_RECORDS, read_beat_record
from ecgq.metrics import BeatMatch, match_beats
from ecgq.quality.inference import load_sklearn_models, predict_windows
from ecgq.quality.windows import cnn_inputs, compute_windows
from ecgq.utils import REPO_ROOT, get_logger, load_config, save_json, set_seed

GATES = ["none", "rule", "gbm_ecg", "cnn_ecg", "oracle"]


def snr_of(name: str) -> str:
    return "clean" if "e" not in name else name.split("e")[1].replace("_", "-")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "rejection_nstdb.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    torch.set_num_threads(4)
    log = get_logger("evaluate_rejection")
    fs = float(cfg["fs"])
    w = int(cfg["window_s"] * fs)
    tol = int(round(cfg["tolerance_ms"] / 1000 * fs))
    models = load_sklearn_models()

    clean = {b: read_beat_record(REPO_ROOT / cfg["mitdb_dir"], b, cfg["channel"])
             for b in ("118", "119")}
    jobs = [(b, clean[b]) for b in ("118", "119")]
    jobs += [(n, read_beat_record(REPO_ROOT / cfg["nstdb_dir"], n, cfg["channel"]))
             for n in NSTDB_RECORDS]

    rows, win_rows = [], []
    for name, rec in jobs:
        base = name[:3]
        x = resample(rec.signal, rec.fs, fs)
        ref = np.round(rec.beats * fs / rec.fs).astype(int)
        ws = compute_windows(x, fs, win_s=cfg["window_s"], mains=cfg["mains_hz"])
        cx = cnn_inputs(x, fs, ws.starts, cfg["window_s"], fs_cnn=100, mains=cfg["mains_hz"])
        preds = {g: predict_windows(g, ws.ecg_feats, cnn_x=cx, models=models)
                 for g in ("rule", "gbm_ecg", "cnn_ecg")}
        # Exact noise location: NSTDB minus the clean record (same samples, same gain).
        nrec = len(x) // w
        noise = x - resample(clean[base].signal, clean[base].fs, fs)[: len(x)]
        noise_rms = np.sqrt(np.mean(noise[: nrec * w].reshape(nrec, w) ** 2, axis=1))
        sig_rms = np.sqrt(np.mean(x[: nrec * w].reshape(nrec, w) ** 2, axis=1))
        noisy = noise_rms > 0.01 * sig_rms
        preds["oracle"] = np.where(noisy, 3, 1)
        preds["none"] = np.ones(nrec, dtype=int)

        det = ws.beats_pt
        for g in GATES:
            keep_win = preds[g][:nrec] != 3
            ref_keep = keep_win[np.minimum(ref // w, nrec - 1)]
            det_keep = keep_win[np.minimum(det // w, nrec - 1)]
            m, _ = match_beats(ref[ref_keep], det[det_keep], tol)
            rows.append({
                "record": name, "snr_db": snr_of(name), "gate": g, **m.as_dict(),
                "coverage": float(ref_keep.mean()),
                "reject_rate_noisy": float(np.mean(~keep_win[noisy])) if noisy.any() else np.nan,
                "reject_rate_clean": float(np.mean(~keep_win[~noisy])),
            })
        for i in range(nrec):
            win_rows.append({"record": name, "start_s": i * cfg["window_s"],
                             "noisy": bool(noisy[i]), **{g: int(preds[g][i]) for g in
                                                         ("rule", "gbm_ecg", "cnn_ecg")}})
        r = {x_["gate"]: x_ for x_ in rows[-len(GATES):]}
        log.info("%-7s none F1=%.3f | gbm F1=%.3f cov=%.2f | cnn F1=%.3f cov=%.2f | oracle F1=%.3f",
                 name, r["none"]["f1"], r["gbm_ecg"]["f1"], r["gbm_ecg"]["coverage"],
                 r["cnn_ecg"]["f1"], r["cnn_ecg"]["coverage"], r["oracle"]["f1"])

    df = pd.DataFrame(rows)
    out = REPO_ROOT / cfg["out_dir"] / "metrics"
    df.to_csv(out / "rejection_nstdb_per_record.csv", index=False, float_format="%.5f")
    pd.DataFrame(win_rows).to_csv(out / "rejection_nstdb_windows.csv.gz", index=False)

    summary = {"dataset": "MIT-BIH Noise Stress Test Database 1.0.0 (+ clean 118, 119)",
               "channel": cfg["channel"], "tolerance_ms": cfg["tolerance_ms"],
               "detector": "pan_tompkins (blockwise, 250 Hz)", "by_snr": {}}
    order = ["clean", "24", "18", "12", "06", "00", "-6"]
    for snr in order:
        sub = df[df.snr_db == snr]
        summary["by_snr"][snr] = {}
        for g in GATES:
            s = sub[sub.gate == g]
            m = BeatMatch(int(s.tp.sum()), int(s.fp.sum()), int(s.fn.sum()))
            summary["by_snr"][snr][g] = {
                **m.as_dict(), "coverage": float(s.coverage.mean()),
                "reject_rate_noisy": float(s.reject_rate_noisy.mean()),
                "reject_rate_clean": float(s.reject_rate_clean.mean())}
    save_json(summary, out / "rejection_nstdb_summary.json")
    for snr in order:
        t = summary["by_snr"][snr]
        log.info("SNR %-5s F1 none %.3f  rule %.3f  gbm %.3f  cnn %.3f  oracle %.3f | coverage "
                 "gbm %.2f cnn %.2f", snr, t["none"]["f1"], t["rule"]["f1"], t["gbm_ecg"]["f1"],
                 t["cnn_ecg"]["f1"], t["oracle"]["f1"], t["gbm_ecg"]["coverage"],
                 t["cnn_ecg"]["coverage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
