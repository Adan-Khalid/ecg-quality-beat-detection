"""Render every results table as Markdown from the JSON/CSV files in reports/metrics.

The README and docs/EXPERIMENTS.md copy their numbers from reports/results_tables.md, so
every published number traces back to a generated file.

Usage:
    python scripts/make_tables.py
"""

from __future__ import annotations

import json

import pandas as pd

from ecgq.utils import REPO_ROOT

MET = REPO_ROOT / "reports" / "metrics"
NAMES = {"rule": "Rule-based SQI thresholds (baseline)", "gbm_acc": "Gradient boosting, ACC only",
         "gbm_ecg": "Gradient boosting, ECG SQIs", "gbm_ecg_acc": "Gradient boosting, ECG + ACC",
         "cnn_ecg": "1D CNN, ECG", "cnn_ecg_acc": "1D CNN, ECG + ACC"}
DET = {"pan_tompkins": "Pan-Tompkins (this repo)", "elgendi": "Elgendi two-MA (this repo)",
       "wfdb_xqrs": "WFDB XQRS (library baseline)"}


def pct(x: float) -> str:
    return f"{100 * x:.2f} %"


def qrs_table() -> str:
    s = json.loads((MET / "qrs_mitdb_summary.json").read_text())
    rows = ["| Detector | Records | Protocol | Se | +P | F1 | Timing MAE | Run time |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for subset, label in (("all_records", "all 48"), ("non_paced_records", "44 non-paced")):
        for proto in ("ec57", "whole_record"):
            for det in ("pan_tompkins", "elgendi", "wfdb_xqrs"):
                v = s[subset][proto][det]
                rows.append(f"| {DET[det]} | {label} | {proto} | {pct(v['sensitivity'])} | "
                            f"{pct(v['ppv'])} | {pct(v['f1'])} | {v['timing_mae_ms']:.1f} ms | "
                            f"{v['seconds']:.0f} s |")
    return "\n".join(rows)


def loso_table() -> str:
    s = json.loads((MET / "quality_loso_summary.json").read_text())
    dom = s.get("class3_dominant_subject", {}).get("subject")
    rows = ["| Model | Macro-F1 (pooled) | Mean subject macro-F1 | F1 class 1 | F1 class 2 | "
            f"F1 class 3 | Macro-F1 without subject {dom} |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in ("rule", "gbm_acc", "gbm_ecg", "gbm_ecg_acc", "cnn_ecg", "cnn_ecg_acc"):
        if m not in s["models"]:
            continue
        v = s["models"][m]
        ex = v.get(f"pooled_excluding_subject_{dom}", {}).get("macro_f1", float("nan"))
        f = v["f1_per_class"]
        rows.append(f"| {NAMES[m]} | {v['macro_f1']:.3f} | {v['mean_subject_macro_f1']:.3f} | "
                    f"{f['1']:.3f} | {f['2']:.3f} | {f['3']:.3f} | {ex:.3f} |")
    counts = s["class_counts"]
    rows.append("")
    rows.append(f"{s['n_windows']:,} labelled 10 s windows (class 1: {counts['1']:,}, class 2: "
                f"{counts['2']:,}, class 3: {counts['3']:,}), 15 subjects, {s['protocol']}.")
    return "\n".join(rows)


def comparisons_table() -> str:
    s = json.loads((MET / "quality_loso_summary.json").read_text())
    rows = ["| Comparison (B vs A) | Subjects improved / worse | Mean per-subject change | "
            "Wilcoxon p | Pooled macro-F1 change, 95 % CI (subject bootstrap) |",
            "| --- | --- | --- | --- | --- |"]
    for v in s.get("comparisons", {}).values():
        lo, hi = v["pooled_diff_bootstrap_ci95"]
        rows.append(f"| {NAMES[v['b']]} vs {NAMES[v['a']]} | {v['subjects_improved']} / "
                    f"{v['subjects_worse']} of {v['subjects']} | {v['mean_subject_diff']:+.3f} | "
                    f"{v['wilcoxon_p']:.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    return "\n".join(rows)


def fixed_split_table() -> str:
    s = json.loads((MET / "quality_fixed_split.json").read_text())
    b = s["published_snr_baseline_sample_level"]
    rows = ["| Method | Accuracy | F1 class 1 | F1 class 2 | F1 class 3 | Source |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| SNR method (Smital et al.), as published | {pct(b['accuracy'])} | "
            f"{pct(b['f1_q1'])} | {pct(b['f1_q2'])} | {pct(b['f1_q3'])} | "
            "quoted from the dataset paper, not re-run |"]
    for m in ("rule", "gbm_ecg", "gbm_ecg_acc"):
        v = s["models"][m]
        rows.append(f"| {NAMES[m]} | {pct(v['accuracy'])} | {pct(v['f1_q1'])} | "
                    f"{pct(v['f1_q2'])} | {pct(v['f1_q3'])} | this repo |")
    rows.append("")
    rows.append(f"Train: {len(s['split']['train'])} records; test: {len(s['split']['test'])} "
                "records from other subjects; scored per sample inside the authors' annotated "
                "ranges.")
    return "\n".join(rows)


def gate_table() -> str:
    s = json.loads((MET / "rejection_nstdb_summary.json").read_text())["by_snr"]
    rows = ["| SNR | No gate F1 | Rule gate F1 (beats kept) | GBM gate F1 (beats kept) | "
            "CNN gate F1 (beats kept) | Oracle F1 (beats kept) | GBM: noisy windows rejected | "
            "GBM: clean windows rejected |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for snr, label in (("clean", "clean"), ("24", "24 dB"), ("18", "18 dB"), ("12", "12 dB"),
                       ("06", "6 dB"), ("00", "0 dB"), ("-6", "-6 dB")):
        d = s[snr]

        def fc(g, d=d):
            return f"{d[g]['f1']:.3f} ({100 * d[g]['coverage']:.0f} %)"

        rn = d["gbm_ecg"]["reject_rate_noisy"]
        rows.append(f"| {label} | {d['none']['f1']:.3f} | {fc('rule')} | {fc('gbm_ecg')} | "
                    f"{fc('cnn_ecg')} | {fc('oracle')} | "
                    f"{'n/a' if rn != rn else f'{100 * rn:.0f} %'} | "
                    f"{100 * d['gbm_ecg']['reject_rate_clean']:.0f} % |")
    return "\n".join(rows)


def holdout_table() -> str:
    d = pd.read_csv(MET / "holdout_daylong_agreement.csv", dtype={"record": str})
    rows = ["| Record | Hours | Expert-labelled windows | Agreement with experts | "
            "Predicted class 3 (whole recording) |", "| --- | --- | --- | --- | --- |"]
    for r in d.itertuples():
        rows.append(f"| {r.record} | {r.windows * 10 / 3600:.1f} | {r.labelled} | "
                    f"{100 * r.agreement_labelled:.1f} % | {100 * r.pred_class3_share_all:.1f} % |")
    return "\n".join(rows)


def main() -> int:
    parts = [
        "# Results tables",
        "",
        "Generated by `scripts/make_tables.py` from `reports/metrics/`. Do not edit by hand.",
        "",
        "## QRS detection, MIT-BIH Arrhythmia Database (150 ms window)",
        "",
        qrs_table(),
        "",
        "## Quality classification, leave-one-subject-out on BUT QDB",
        "",
        loso_table(),
        "",
        "## Paired comparisons on unseen subjects",
        "",
        comparisons_table(),
        "",
        "## Official BUT QDB split, sample-level scores",
        "",
        fixed_split_table(),
        "",
        "## Quality gate before Pan-Tompkins, MIT-BIH Noise Stress Test Database",
        "",
        gate_table(),
        "",
        "## Held-out predictions over whole recordings",
        "",
        holdout_table(),
        "",
    ]
    out = REPO_ROOT / "reports" / "results_tables.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
