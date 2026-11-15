"""Count labelled 10 s windows per BUT QDB record, with and without the official ranges.

`all_*` columns count every pure consensus-labelled window. `official_*` columns count only
windows inside the dataset authors' annotated ranges (ecgq.io.OFFICIAL_RANGES), which is
what every model in this project is trained and evaluated on (DECISIONS D14).

Writes reports/metrics/butqdb_window_counts.csv (replacing the build-time version).

Usage:
    python scripts/count_windows.py --config configs/quality_butqdb.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ecgq.io import official_window_mask, subject_of
from ecgq.utils import REPO_ROOT, load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "quality_butqdb.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rows = []
    for f in sorted((REPO_ROOT / cfg["processed_dir"]).glob("*.npz")):
        z = np.load(f)
        lab = z["labels"]
        inside = official_window_mask(f.stem, z["starts"], float(z["fs"]), float(z["win_s"]))
        row = {"record": f.stem, "subject": subject_of(f.stem),
               "hours": len(lab) * float(z["win_s"]) / 3600, "windows": len(lab)}
        for c in (1, 2, 3):
            row[f"all_class{c}"] = int(np.sum(lab == c))
        for c in (1, 2, 3):
            row[f"official_class{c}"] = int(np.sum((lab == c) & inside))
        rows.append(row)
    df = pd.DataFrame(rows)
    total = {"record": "total", "subject": "", "hours": df.hours.sum(),
             "windows": df.windows.sum(), **{k: df[k].sum() for k in df.columns
                                             if k.startswith(("all_", "official_"))}}
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    out = REPO_ROOT / cfg["out_dir"] / "metrics" / "butqdb_window_counts.csv"
    df.to_csv(out, index=False, float_format="%.2f")
    print(df.tail(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
