"""Plot helpers sharing one style (assets/style.mplstyle) and one colour scheme."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecgq.utils import REPO_ROOT  # noqa: E402

STYLE = REPO_ROOT / "assets" / "style.mplstyle"
QUALITY_COLORS = {0: "#BBBBBB", 1: "#009E73", 2: "#E69F00", 3: "#D55E00"}
QUALITY_NAMES = {0: "unlabelled", 1: "class 1 (clean)", 2: "class 2 (QRS only)",
                 3: "class 3 (unusable)"}
DETECTOR_COLORS = {"pan_tompkins": "#0072B2", "elgendi": "#CC79A7", "wfdb_xqrs": "#7F7F7F"}
DETECTOR_NAMES = {"pan_tompkins": "Pan-Tompkins (this repo)",
                  "elgendi": "Elgendi two-MA (this repo)", "wfdb_xqrs": "WFDB XQRS (library)"}


def use_style() -> None:
    if STYLE.exists():
        plt.style.use(str(STYLE))


def savefig(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
