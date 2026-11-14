"""QRS detectors."""

from ecgq.detectors.elgendi import elgendi
from ecgq.detectors.pan_tompkins import pan_tompkins

__all__ = ["elgendi", "pan_tompkins"]
