"""Lightweight immediate-mode UI widgets for pygame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import pygame

from visualization.theme import THEME

Color = Tuple[int, int, int]


@dataclass
class WidgetResult:
    clicked: bool = False
    hovered: bool = False
    value_changed: bool = False


class UIContext:
    def __init__(self, font: pygame.font.Font, font_small: pygame.font.Font):
        self.font = font
        self.font_small = font_small
        self.mouse_pos = (0, 0)
        self.mouse_down = False
        self.mouse_pressed = False  # rising edge
        self.mouse_released = False
        self.key_chars: List[str] = []
        self.backspace = False
        self.enter = False
        self.active_text_id: Optional[str] = None

    def begin_frame(self, events: List[pygame.event.Event]) -> None:
        self.mouse_pressed = False
        self.mouse_released = False
        self.key_chars = []
        self.backspace = False
        self.enter = False
        for e in events:
            if e.type == pygame.MOUSEMOTION:
                self.mouse_pos = e.pos
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self.mouse_down = True
                self.mouse_pressed = True
                self.mouse_pos = e.pos
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.mouse_down = False
                self.mouse_released = True
                self.mouse_pos = e.pos
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_BACKSPACE:
                    self.backspace = True
                elif e.key == pygame.K_RETURN:
                    self.enter = True
                elif e.unicode and e.unicode.isprintable():
                    self.key_chars.append(e.unicode)

    def button(self, surface, rect: pygame.Rect, label: str, active: bool = False) -> bool:
        hovered = rect.collidepoint(self.mouse_pos)
        if active:
            color = THEME.button_active
        elif hovered:
            color = THEME.button_hover
        else:
            color = THEME.button
        pygame.draw.rect(surface, color, rect, border_radius=6)
        pygame.draw.rect(surface, THEME.panel_border, rect, 1, border_radius=6)
        text = self.font_small.render(label, True, THEME.text)
        surface.blit(text, text.get_rect(center=rect.center))
        return hovered and self.mouse_pressed

    def toggle(self, surface, rect: pygame.Rect, label: str, value: bool) -> bool:
        box = pygame.Rect(rect.x, rect.y + 2, 16, 16)
        hovered = rect.collidepoint(self.mouse_pos)
        pygame.draw.rect(surface, THEME.button_active if value else THEME.button, box, border_radius=3)
        pygame.draw.rect(surface, THEME.accent if value else THEME.panel_border, box, 1, border_radius=3)
        if value:
            pygame.draw.line(surface, THEME.accent2, (box.x + 3, box.y + 8), (box.x + 7, box.y + 12), 2)
            pygame.draw.line(surface, THEME.accent2, (box.x + 7, box.y + 12), (box.x + 13, box.y + 4), 2)
        text = self.font_small.render(label, True, THEME.text)
        surface.blit(text, (box.right + 8, rect.y))
        if hovered and self.mouse_pressed:
            return not value
        return value

    def label(self, surface, pos, text: str, dim: bool = False, small: bool = False) -> None:
        font = self.font_small if small else self.font
        img = font.render(text, True, THEME.text_dim if dim else THEME.text)
        surface.blit(img, pos)

    def text_input(
        self,
        surface,
        rect: pygame.Rect,
        text_id: str,
        value: str,
        placeholder: str = "",
    ) -> str:
        hovered = rect.collidepoint(self.mouse_pos)
        if self.mouse_pressed:
            if hovered:
                self.active_text_id = text_id
            elif self.active_text_id == text_id:
                self.active_text_id = None
        active = self.active_text_id == text_id
        pygame.draw.rect(surface, THEME.bg, rect, border_radius=4)
        pygame.draw.rect(
            surface,
            THEME.accent if active else THEME.panel_border,
            rect,
            1,
            border_radius=4,
        )
        out = value
        if active:
            for ch in self.key_chars:
                out += ch
            if self.backspace:
                out = out[:-1]
        display = out if out else placeholder
        color = THEME.text if out else THEME.text_dim
        img = self.font_small.render(display[:48], True, color)
        surface.blit(img, (rect.x + 6, rect.y + (rect.height - img.get_height()) // 2))
        return out

    def slider(
        self,
        surface,
        rect: pygame.Rect,
        value: float,
        vmin: float,
        vmax: float,
        label: str,
    ) -> float:
        self.label(surface, (rect.x, rect.y - 16), f"{label}: {value:.4g}", small=True)
        pygame.draw.rect(surface, THEME.button, rect, border_radius=4)
        t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        knob_x = rect.x + int(t * rect.width)
        pygame.draw.circle(surface, THEME.accent, (knob_x, rect.centery), 7)
        if rect.collidepoint(self.mouse_pos) and self.mouse_down:
            t = (self.mouse_pos[0] - rect.x) / max(rect.width, 1)
            t = max(0.0, min(1.0, t))
            return vmin + t * (vmax - vmin)
        return value
