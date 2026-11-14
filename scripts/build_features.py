"""Build 10 s windows, SQI features, accelerometer features and CNN inputs for BUT QDB.

For each record writes data/processed/butqdb/<record>.npz with:
  starts, labels, ecg_feats, acc_feats   every window of the recording
  beats_pt, beats_el                     detector outputs (samples at 250 Hz)
  cnn_idx, cnn_x                         CNN inputs (ECG + 3 ACC channels) for labelled windows
and reports/metrics/butqdb_window_counts.csv.

Usage:
    python scripts/build_features.py --config configs/quality_butqdb.yaml [--records 100001 ...]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ecgq.io import butqdb_records, read_butqdb_record, subject_of
from ecgq.quality.windows import cnn_inputs, compute_windows
from ecgq.utils import REPO_ROOT, get_logger, load_config


def process(record: str, cfg: dict) -> dict:
    t0 = time.perf_counter()
    root = REPO_ROOT / cfg["data_dir"]
    out = REPO_ROOT / cfg["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    rec = read_butqdb_record(root, record, fs_out=cfg["fs"])
    ws = compute_windows(rec.ecg, rec.fs, labels=rec.labels, acc=rec.acc, fs_acc=rec.fs_acc,
                         win_s=cfg["window_s"], mains=cfg["mains_hz"])
    cnn_idx = np.flatnonzero(ws.labels > 0)
    cnn_x = cnn_inputs(rec.ecg, rec.fs, ws.starts[cnn_idx], cfg["window_s"], rec.acc,
                       rec.fs_acc, cfg["fs_cnn"], cfg["mains_hz"])
    np.savez_compressed(
        out / f"{record}.npz", starts=ws.starts, labels=ws.labels, ecg_feats=ws.ecg_feats,
        acc_feats=ws.acc_feats, beats_pt=ws.beats_pt, beats_el=ws.beats_el,
        cnn_idx=cnn_idx, cnn_x=cnn_x.astype(np.float16), fs=rec.fs, win_s=ws.win_s,
        label_counts=ws.label_counts,
    )
    counts = {c: int(np.sum(ws.labels == c)) for c in (0, 1, 2, 3)}
    return {"record": record, "subject": subject_of(record), "hours": len(rec.ecg) / rec.fs / 3600,
            "windows": len(ws.starts), "unlabelled": counts[0], "class1": counts[1],
            "class2": counts[2], "class3": counts[3], "seconds": time.perf_counter() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "quality_butqdb.yaml"))
    ap.add_argument("--records", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger("build_features")
    records = args.records or butqdb_records(REPO_ROOT / cfg["data_dir"])
    rows = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futs = {pool.submit(process, r, cfg): r for r in records}
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            log.info("%s  %.1f h  windows=%d  c1=%d c2=%d c3=%d  (%.0f s)", row["record"],
                     row["hours"], row["windows"], row["class1"], row["class2"], row["class3"],
                     row["seconds"])
    df = pd.DataFrame(rows).sort_values("record")
    path = REPO_ROOT / cfg["out_dir"] / "metrics" / "butqdb_window_counts.csv"
    if args.records and path.exists():  # partial rebuild: merge with the existing table
        old = pd.read_csv(path, dtype={"record": str, "subject": str})
        df = pd.concat([old[~old.record.isin(df.record)], df]).sort_values("record")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.2f")
    log.info("labelled windows: c1=%d c2=%d c3=%d", df.class1.sum(), df.class2.sum(),
             df.class3.sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
