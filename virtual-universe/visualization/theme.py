"""Deep-space scientific UI theme."""

from dataclasses import dataclass
from typing import Tuple


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    bg: Color = (6, 8, 18)
    panel: Color = (12, 16, 32)
    panel_border: Color = (40, 60, 100)
    text: Color = (220, 230, 245)
    text_dim: Color = (130, 145, 170)
    accent: Color = (80, 180, 255)
    accent2: Color = (120, 220, 200)
    warning: Color = (255, 170, 70)
    danger: Color = (255, 90, 100)
    success: Color = (90, 220, 140)
    grid: Color = (30, 40, 70)
    trail: Color = (70, 120, 180)
    vector_v: Color = (80, 220, 140)
    vector_g: Color = (255, 120, 90)
    button: Color = (24, 36, 64)
    button_hover: Color = (36, 56, 96)
    button_active: Color = (50, 90, 150)


THEME = Theme()
