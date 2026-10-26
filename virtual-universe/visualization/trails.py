"""Trail utilities (history sampling helpers)."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def downsample_trail(points: Iterable[np.ndarray], max_points: int = 400) -> List[np.ndarray]:
    pts = list(points)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
    return [pts[i] for i in idx]
