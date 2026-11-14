"""Readers for the PhysioNet databases used here (WFDB format).

* MIT-BIH Arrhythmia Database (mitdb): 48 records, 360 Hz, two channels, beat labels.
* MIT-BIH Noise Stress Test Database (nstdb): records 118 and 119 with electrode-motion
  noise added at known SNRs, original beat labels kept.
* BUT QDB (butqdb): 18 long single-lead recordings at 1000 Hz with 3-axis accelerometer
  at 100 Hz and sample-wise quality labels from three experts plus a consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecgq.filters import resample

# Annotation symbols that mark a heartbeat (WFDB / MIT-BIH convention). Non-beat
# annotations such as rhythm changes ('+'), noise ('~') or comments are excluded.
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
]  # fmt: skip
MITDB_PACED = ("102", "104", "107", "217")

NSTDB_RECORDS = [f"{base}e{snr}" for base in ("118", "119") for snr in
                 ("24", "18", "12", "06", "00", "_6")]  # fmt: skip


@dataclass
class BeatRecord:
    name: str
    signal: np.ndarray  # (n_samples,) physical units (mV)
    fs: float
    beats: np.ndarray  # reference beat sample indices
    symbols: np.ndarray
    channel_name: str
    vf_intervals: np.ndarray  # (k, 2) ventricular flutter/fibrillation episodes ('[' to ']')

    def outside_vf(self, idx: np.ndarray) -> np.ndarray:
        """Boolean mask of samples outside VF episodes (excluded by ANSI/AAMI EC57)."""
        keep = np.ones(len(idx), dtype=bool)
        for a, b in self.vf_intervals:
            keep &= ~((idx >= a) & (idx <= b))
        return keep


def read_beat_record(root: Path, name: str, channel: int = 0) -> BeatRecord:
    """Read one MIT-BIH style record and its reference beat annotations (.atr)."""
    path = str(Path(root) / name)
    rec = wfdb.rdrecord(path, channels=[channel])
    ann = wfdb.rdann(path, "atr")
    sym = np.asarray(ann.symbol)
    mask = np.isin(sym, list(BEAT_SYMBOLS))
    samp = np.asarray(ann.sample)
    starts, ends = samp[sym == "["], samp[sym == "]"]
    if len(starts) > len(ends):  # episode running to the end of the record
        ends = np.append(ends, rec.sig_len)
    vf = np.column_stack([starts, ends[: len(starts)]]) if len(starts) else np.zeros((0, 2), int)
    return BeatRecord(
        name=name,
        signal=rec.p_signal[:, 0].astype(np.float64),
        fs=float(rec.fs),
        beats=np.asarray(ann.sample)[mask].astype(np.int64),
        symbols=sym[mask],
        channel_name=rec.sig_name[0],
        vf_intervals=vf,
    )


# ----------------------------------------------------------------------------- BUT QDB


def butqdb_records(root: Path) -> list[str]:
    """Record names. RECORDS lists signal paths such as '100001/100001_ECG', so take the folder."""
    names = [r.strip().split("/")[0] for r in (Path(root) / "RECORDS").read_text().split()]
    return sorted({n for n in names if n})


def subject_of(record: str) -> str:
    """BUT QDB record names: first three digits are the subject, last three the session."""
    return record[:3]


def read_butqdb_annotations(root: Path, record: str) -> pd.DataFrame:
    """Read the 12-column quality CSV (3 annotators + consensus, 3 columns each).

    Returns a tidy table with columns: who, start, end, cls. Sample indices refer to the
    1000 Hz ECG. Class 0 means "not annotated" and is dropped.
    """
    csv = Path(root) / record / f"{record}_ANN.csv"
    raw = pd.read_csv(csv, header=None)
    if raw.shape[1] != 12:
        raise ValueError(f"{csv} has {raw.shape[1]} columns, expected 12")
    who = ["A", "B", "C", "consensus"]
    frames = []
    for i, w in enumerate(who):
        part = raw.iloc[:, 3 * i : 3 * i + 3].dropna()
        part.columns = ["start", "end", "cls"]
        part = part.astype({"start": np.int64, "end": np.int64, "cls": np.int64})
        part.insert(0, "who", w)
        frames.append(part[part["cls"] > 0])
    return pd.concat(frames, ignore_index=True)


def consensus_label_array(ann: pd.DataFrame, n_samples: int, one_based: bool = True) -> np.ndarray:
    """Expand consensus segments into a per-sample label array (0 = unlabelled)."""
    lab = np.zeros(n_samples, dtype=np.int8)
    off = 1 if one_based else 0
    for row in ann[ann["who"] == "consensus"].itertuples():
        s = max(0, int(row.start) - off)
        e = min(n_samples, int(row.end) - off + 1)
        if e > s:
            lab[s:e] = int(row.cls)
    return lab


@dataclass
class QDBRecord:
    name: str
    ecg: np.ndarray  # ECG resampled to fs (mV)
    fs: float
    acc: np.ndarray  # (n, 3) accelerometer at fs_acc, original units
    fs_acc: float
    labels: np.ndarray  # per-sample consensus label at fs (0 unlabelled)
    n_samples_orig: int


def read_butqdb_record(
    root: Path, record: str, fs_out: float = 250.0, chunk_s: float = 3600.0
) -> QDBRecord:
    """Read a BUT QDB record, resampling the 1000 Hz ECG to ``fs_out`` in chunks.

    Reading a multi-day 1000 Hz record at once needs several GB of RAM, so the ECG is read
    and resampled one hour at a time with 5 s of overlap on each side to avoid edge effects.
    """
    base = Path(root) / record
    hdr = wfdb.rdheader(str(base / f"{record}_ECG"))
    fs_in = float(hdr.fs)
    n = int(hdr.sig_len)
    chunk = int(chunk_s * fs_in)
    pad = int(5 * fs_in)
    ratio = fs_out / fs_in
    parts = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        lo, hi = max(0, s - pad), min(n, e + pad)
        seg = wfdb.rdrecord(str(base / f"{record}_ECG"), sampfrom=lo, sampto=hi,
                            channels=[0], return_res=32).p_signal[:, 0]
        seg = np.nan_to_num(seg.astype(np.float64))
        r = resample(seg, fs_in, fs_out)
        a = int(round((s - lo) * ratio))
        b = a + int(round((e - s) * ratio))
        parts.append(r[a:b].astype(np.float32))
    ecg = np.concatenate(parts)

    acc_rec = wfdb.rdrecord(str(base / f"{record}_ACC"), return_res=32)
    acc = np.nan_to_num(acc_rec.p_signal.astype(np.float32))

    ann = read_butqdb_annotations(root, record)
    lab_full = consensus_label_array(ann, n)
    step = int(round(fs_in / fs_out))
    labels = lab_full[::step][: len(ecg)]
    if len(labels) < len(ecg):
        labels = np.pad(labels, (0, len(ecg) - len(labels)))
    return QDBRecord(record, ecg, fs_out, acc, float(acc_rec.fs), labels, n)


def read_butqdb_segment(root: Path, record: str, t0_s: float, dur_s: float,
                        fs_out: float = 250.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Read a short stretch of ECG (resampled to ``fs_out``, microvolts) and ACC (100 Hz).

    Used by the demo app so it never loads a whole 24 h record.
    """
    base = Path(root) / record
    hdr = wfdb.rdheader(str(base / f"{record}_ECG"))
    fs_in = float(hdr.fs)
    a = max(0, int(t0_s * fs_in))
    b = min(int(hdr.sig_len), int((t0_s + dur_s) * fs_in))
    ecg = wfdb.rdrecord(str(base / f"{record}_ECG"), sampfrom=a, sampto=b,
                        channels=[0]).p_signal[:, 0]
    ecg = resample(np.nan_to_num(ecg), fs_in, fs_out)
    ahdr = wfdb.rdheader(str(base / f"{record}_ACC"))
    fa = float(ahdr.fs)
    aa = max(0, int(t0_s * fa))
    ab = min(int(ahdr.sig_len), int((t0_s + dur_s) * fa))
    acc = wfdb.rdrecord(str(base / f"{record}_ACC"), sampfrom=aa, sampto=ab).p_signal
    return ecg, np.nan_to_num(acc), fa


# Annotated ranges used by the dataset authors' benchmark (1-based, inclusive, samples at
# 1000 Hz), copied from choseDATA('all') in TESTrun_v01.m,
# https://github.com/BioSig-BUT/BUT-QDB-supplement (MIT licence). The consensus CSV also
# labels the first 13.1 h of 105001 as class 3, a stretch where the accelerometer is static
# and the ECG is pure noise (device apparently off the body); the benchmark leaves it out,
# and so does this project.
OFFICIAL_RANGES: dict[str, list[tuple[int, int]]] = {
    "100001": [(1, 87087000)],
    "100002": [(28800001, 30000000), (57600001, 58800000)],
    "103001": [(28800001, 30000000), (57600001, 58800000)],
    "103002": [(28800001, 30000000), (57600001, 58800000)],
    "103003": [(28800001, 30000000), (57600001, 58800000)],
    "104001": [(28800001, 30000000), (57600001, 58800000)],
    "105001": [(46800001, 139147000)],
    "111001": [(1, 90645000)],
    "113001": [(28800001, 30000000), (36120000, 37319999), (57600001, 58800000)],
    "114001": [(11214751, 11334751), (11674751, 12874750), (28800001, 30000000),
               (57600001, 58800000)],
    "115001": [(28800001, 30000000), (57600001, 58800000)],
    "118001": [(28800001, 30000000), (57600001, 58800000)],
    "121001": [(28800001, 30000000), (57600001, 58800000)],
    "122001": [(28800001, 30000000), (57600001, 58800000)],
    "123001": [(28800001, 30000000), (57600001, 58800000)],
    "124001": [(28800001, 30000000), (33700000, 34899999), (57600001, 58800000),
               (65100000, 66299999)],
    "125001": [(28800001, 30000000), (57600001, 58800000)],
    "126001": [(28800001, 30000000), (57600001, 58800000)],
}


def official_window_mask(record: str, starts: np.ndarray, fs: float, win_s: float,
                         fs_orig: float = 1000.0) -> np.ndarray:
    """True for windows that lie fully inside the authors' annotated ranges."""
    s0 = np.asarray(starts, dtype=np.int64) * int(round(fs_orig / fs))
    s1 = s0 + int(round(win_s * fs_orig))
    keep = np.zeros(len(s0), dtype=bool)
    for a, b in OFFICIAL_RANGES.get(record, []):
        keep |= (s0 >= a - 1) & (s1 <= b + 1)
    return keep
