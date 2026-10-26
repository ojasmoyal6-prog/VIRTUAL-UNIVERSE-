"""Formatting helpers for scientific display."""

from __future__ import annotations


def sci(value: float, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "—"
    av = abs(value)
    if av == 0:
        return f"0 {unit}".strip()
    if 1e-2 <= av < 1e4:
        return f"{value:.{digits}g} {unit}".strip()
    return f"{value:.{digits}e} {unit}".strip()


def fmt_speed(mps: float) -> str:
    return f"{mps / 1000.0:.3f} km/s"


def fmt_distance(m: float) -> str:
    au = m / 1.495978707e11
    if abs(au) >= 0.01:
        return f"{au:.4f} AU ({sci(m, 'm')})"
    if abs(m) >= 1e3:
        return f"{m/1e3:.3f} km"
    return f"{m:.3f} m"


def fmt_mass(kg: float) -> str:
    return sci(kg, "kg")


def fmt_time_days(seconds: float) -> str:
    days = seconds / 86400.0
    if days >= 365.25:
        return f"{days/365.25:.3f} yr ({days:.2f} d)"
    if days >= 1:
        return f"{days:.3f} d"
    hours = seconds / 3600.0
    if hours >= 1:
        return f"{hours:.2f} h"
    return f"{seconds:.1f} s"
