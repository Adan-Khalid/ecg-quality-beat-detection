"""Validate the QRS detectors on the MIT-BIH Arrhythmia Database.

Writes:
  reports/metrics/qrs_mitdb_per_record.csv   one row per record and detector
  reports/metrics/qrs_mitdb_summary.json     gross Se, +P, F1 (all 48 and the 44 non-paced)
  reports/logs/evaluate_qrs.log

Usage:
    python scripts/evaluate_qrs.py --config configs/qrs_mitdb.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
import pandas as pd
import wfdb.processing as wp

from ecgq.detectors import elgendi, pan_tompkins
from ecgq.io import MITDB_PACED, MITDB_RECORDS, read_beat_record
from ecgq.metrics import BeatMatch, match_beats, timing_errors
from ecgq.utils import REPO_ROOT, Timer, get_logger, load_config, save_json, set_seed


def run_xqrs(x, fs):
    # XQRS prints progress to stdout; keep the log clean.
    with contextlib.redirect_stdout(io.StringIO()):
        return wp.xqrs_detect(sig=x, fs=fs, verbose=False)


DETECTORS = {"pan_tompkins": pan_tompkins, "elgendi": elgendi, "wfdb_xqrs": run_xqrs}


def summarise(df: pd.DataFrame, records: list[str]) -> dict:
    out = {}
    for (proto, det), g in df[df.record.isin(records)].groupby(["protocol", "detector"]):
        m = BeatMatch(int(g.tp.sum()), int(g.fp.sum()), int(g.fn.sum()))
        mae = float(np.average(g.timing_mae_ms, weights=g.tp))
        out.setdefault(proto, {})[det] = {**m.as_dict(), "records": len(g),
                                          "timing_mae_ms": mae, "seconds": float(g.seconds.sum())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "qrs_mitdb.yaml"))
    ap.add_argument("--records", nargs="*", help="subset of records (default: all 48)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("evaluate_qrs")
    data_dir = REPO_ROOT / cfg["data_dir"]
    out = REPO_ROOT / cfg["out_dir"]
    records = args.records or MITDB_RECORDS

    rows = []
    for name in records:
        rec = read_beat_record(data_dir, name, channel=cfg["channel"])
        tol = int(round(cfg["tolerance_ms"] / 1000 * rec.fs))
        for det_name in cfg["detectors"]:
            with Timer() as t:
                det = DETECTORS[det_name](rec.signal, rec.fs)
            det = np.asarray(det)
            learn = int(cfg["learning_period_s"] * rec.fs)
            for protocol, start in (("ec57", learn), ("whole_record", 0)):
                ref = rec.beats[rec.outside_vf(rec.beats) & (rec.beats >= start)]
                det_kept = det[rec.outside_vf(det) & (det >= start)]
                m, _ = match_beats(ref, det_kept, tol)
                err_ms = timing_errors(ref, det_kept, tol) / rec.fs * 1000
                rows.append({"record": name, "detector": det_name, "protocol": protocol,
                             "lead": rec.channel_name, "n_ref": len(ref), **m.as_dict(),
                             "timing_mae_ms": float(np.mean(np.abs(err_ms))),
                             "timing_bias_ms": float(np.mean(err_ms)),
                             "seconds": t.seconds,
                             "excluded_vf_detections": int(np.sum(~rec.outside_vf(det)))})
        msg = "  ".join(f"{r['detector']}: Se={r['sensitivity']:.4f} +P={r['ppv']:.4f}"
                        for r in rows[-2 * len(cfg["detectors"]):] if r["protocol"] == "ec57")
        log.info("record %s (%s)  %s", name, rec.channel_name, msg)

    df = pd.DataFrame(rows)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics" / "qrs_mitdb_per_record.csv", index=False, float_format="%.5f")
    non_paced = [r for r in records if r not in MITDB_PACED]
    summary = {
        "dataset": "MIT-BIH Arrhythmia Database 1.0.0",
        "channel": cfg["channel"],
        "tolerance_ms": cfg["tolerance_ms"],
        "evaluation": "one-to-one matching; ventricular flutter episodes excluded. Protocol "
                      "'ec57' also skips the first 5 min (learning period, as in WFDB bxb); "
                      "'whole_record' scores every beat.",
        "all_records": summarise(df, records),
        "non_paced_records": summarise(df, non_paced),
    }
    save_json(summary, out / "metrics" / "qrs_mitdb_summary.json")
    for k in ("all_records", "non_paced_records"):
        for proto, dets in summary[k].items():
            for det, s in dets.items():
                log.info("%s %s %-13s Se=%.4f +P=%.4f F1=%.4f (TP=%d FP=%d FN=%d) "
                         "timing MAE %.1f ms, %.1fs", k, proto, det, s["sensitivity"], s["ppv"],
                         s["f1"], s["tp"], s["fp"], s["fn"], s["timing_mae_ms"], s["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
