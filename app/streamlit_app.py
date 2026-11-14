"""Interactive ECG quality explorer for BUT QDB recordings.

Run (after scripts/build_features.py and scripts/predict_all.py):
    uv run streamlit run app/streamlit_app.py

Every prediction shown comes from a model trained without the subject on screen
(see scripts/predict_all.py). Research and education only, not for diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ecgq.filters import preprocess
from ecgq.io import official_window_mask, read_butqdb_segment
from ecgq.plotting import QUALITY_COLORS, QUALITY_NAMES
from ecgq.quality.features import ACC_FEATURES, ECG_FEATURES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "butqdb"
PROC = ROOT / "data" / "processed"
SHOW = ["bsqi", "tsqi", "ksqi", "psqi", "bassqi", "hr_pt", "rr_cv"]

st.set_page_config(page_title="ECG quality explorer", layout="wide")


@st.cache_data
def records() -> list[str]:
    return sorted(p.stem for p in (PROC / "predictions").glob("*.npz"))


@st.cache_data
def load(record: str):
    z = np.load(PROC / "butqdb" / f"{record}.npz")
    p = np.load(PROC / "predictions" / f"{record}.npz")
    return {k: z[k] for k in ("starts", "labels", "ecg_feats", "acc_feats", "beats_pt", "fs",
                              "win_s")} | {"pred": p["pred"], "proba": p["proba"]}


st.title("ECG signal quality explorer")
st.caption("Single-lead wearable ECG from BUT QDB, scored in 10 s windows as class 1 (clean), "
           "class 2 (QRS only) or class 3 (unusable). The model never saw the person shown. "
           "Research and education only, not for diagnosis.")

recs = records()
if not recs:
    st.error("No predictions found. Run scripts/build_features.py and scripts/predict_all.py.")
    st.stop()

with st.sidebar:
    st.header("Recording")
    # Optional deep link: ?record=114001&t=186.5 (start time in minutes)
    qp = st.query_params
    if qp.get("record") in recs:
        default = recs.index(qp["record"])
    else:
        default = recs.index("100001") if "100001" in recs else 0
    record = st.selectbox("BUT QDB record", recs, index=default)
    d = load(record)
    fs, win = float(d["fs"]), float(d["win_s"])
    n = len(d["pred"])
    hours = n * win / 3600
    st.write(f"Subject {record[:3]}, session {record[3:]}, {hours:.1f} h")
    st.markdown("**Model:** gradient boosting on 17 ECG quality indices and 7 accelerometer "
                "features, trained on the other 14 subjects.")

pred = d["pred"]
# Expert labels inside the dataset authors' annotated ranges (grey elsewhere).
lab = np.where(official_window_mask(record, d["starts"], fs, win), d["labels"], 0)
c1, c2, c3, c4 = st.columns(4)
for col, c in zip((c1, c2, c3), (1, 2, 3), strict=True):
    col.metric(f"Time in {QUALITY_NAMES[c]}", f"{100 * np.mean(pred == c):.1f} %")
m = lab > 0
c4.metric("Agreement with experts", f"{100 * np.mean(pred[m] == lab[m]):.1f} %" if m.any()
          else "no labels", help=f"on {int(m.sum())} expert-labelled windows")

# --- whole-recording timeline -------------------------------------------------------
t_h = (np.arange(n) + 0.5) * win / 3600
accm = d["acc_feats"][:, ACC_FEATURES.index("acc_mag_std")]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.16, 0.16, 0.68],
                    vertical_spacing=0.1,
                    subplot_titles=("Predicted quality", "Expert labels (grey = not labelled)",
                                    "Motion: accelerometer magnitude std per window (mg)"))
for row, vals in ((1, pred), (2, lab)):
    for c in (0, 1, 2, 3):
        sel = vals == c
        if sel.any():
            fig.add_trace(go.Bar(x=t_h[sel], y=np.ones(sel.sum()), width=win / 3600,
                                 marker_color=QUALITY_COLORS[c], name=QUALITY_NAMES[c],
                                 showlegend=row == 1 or c == 0, hoverinfo="x+name"),
                          row=row, col=1)
fig.add_trace(go.Scattergl(x=t_h, y=np.maximum(accm, 1), mode="lines", line=dict(width=0.7,
                           color="#555555"), name="motion", showlegend=False), row=3, col=1)
fig.update_yaxes(type="log", row=3, col=1)
fig.update_yaxes(showticklabels=False, row=1, col=1)
fig.update_yaxes(showticklabels=False, row=2, col=1)
fig.update_layout(height=520, bargap=0, margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(orientation="h", y=-0.12), plot_bgcolor="white")
fig.update_xaxes(title_text="hours since start", row=3, col=1)
st.plotly_chart(fig, width="stretch")

# --- zoom into one stretch ------------------------------------------------------------
st.subheader("Zoom in")
jump = st.radio("Jump to", ["choose a time", "a class 3 stretch", "a class 2 stretch",
                            "a clean stretch"], horizontal=True)
targets = {"a class 3 stretch": 3, "a class 2 stretch": 2, "a clean stretch": 1}
if jump in targets and np.any(pred == targets[jump]):
    idx = np.flatnonzero(pred == targets[jump])
    start_default = float(idx[len(idx) // 2] * win / 60)
else:
    try:
        start_default = float(st.query_params.get("t", 60.0))
    except ValueError:
        start_default = 60.0
    start_default = min(max(0.0, start_default), float(hours * 60 - 0.5))
t_min = st.slider("Start time (minutes)", 0.0, float(hours * 60 - 0.5), start_default, 0.5)
span = 30.0
t0 = t_min * 60
ecg, acc, fa = read_butqdb_segment(RAW, record, max(0.0, t0 - 5), span + 10, fs)
pad = int(min(5, t0) * fs)
clean = preprocess(ecg, fs)[pad : pad + int(span * fs)] / 1000.0
acc = acc[int(min(5, t0) * fa) : int((min(5, t0) + span) * fa)]
te = t0 + np.arange(len(clean)) / fs
ta = t0 + np.arange(len(acc)) / fa
beats = d["beats_pt"] / fs
beats = beats[(beats >= t0) & (beats < t0 + span)]

z = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                  vertical_spacing=0.06)
first, last = int(t0 // win), int((t0 + span) // win)
for wi in range(first, min(last + 1, n)):
    c = int(pred[wi])
    for row in (1, 2):
        z.add_vrect(x0=wi * win, x1=(wi + 1) * win, fillcolor=QUALITY_COLORS[c], opacity=0.18,
                    line_width=0, layer="below", row=row, col=1, exclude_empty_subplots=False)
z.add_trace(go.Scattergl(x=te, y=clean, mode="lines", line=dict(color="#222222", width=1),
                         name="ECG (mV)"), row=1, col=1)
bi = np.clip(((beats - t0) * fs).astype(int), 0, len(clean) - 1)
z.add_trace(go.Scatter(x=beats, y=clean[bi], mode="markers", marker=dict(color="#0072B2",
                       symbol="triangle-down", size=8), name="Pan-Tompkins beats"), row=1, col=1)
for k, name in enumerate(("x", "y", "z")):
    z.add_trace(go.Scattergl(x=ta, y=acc[:, k] - np.median(acc[:, k]), mode="lines",
                             line=dict(width=1), name=f"ACC {name} (mg)"), row=2, col=1)
z.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15))
z.update_xaxes(title_text="seconds since start of recording", row=2, col=1)
st.plotly_chart(z, width="stretch")

rows = []
for wi in range(first, min(last, n)):
    f = dict(zip(ECG_FEATURES, d["ecg_feats"][wi], strict=True))
    pr = d["proba"][wi]
    rows.append({"window start (s)": int(wi * win), "predicted": QUALITY_NAMES[int(pred[wi])],
                 "expert": QUALITY_NAMES[int(lab[wi])],
                 "p(class 1, 2, 3)": ", ".join(f"{v:.2f}" for v in pr) if np.isfinite(pr).all()
                 else "n/a",
                 **{k: round(float(f[k]), 3) for k in SHOW},
                 "motion std (mg)": round(float(accm[wi]), 1)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("bSQI: agreement of the two QRS detectors. tSQI: mean correlation of beats with "
           "their median template. kSQI: kurtosis. pSQI: QRS-band power share. basSQI: share of "
           "power above the baseline band.")
