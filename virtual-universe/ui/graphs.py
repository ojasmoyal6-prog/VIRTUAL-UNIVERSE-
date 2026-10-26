"""Graph colour constants for the diagnostics panel."""

from visualization.theme import THEME

SERIES_COLORS = {
    "total_energy": THEME.accent,
    "kinetic": THEME.success,
    "potential": THEME.warning,
    "momentum": (180, 140, 255),
}
