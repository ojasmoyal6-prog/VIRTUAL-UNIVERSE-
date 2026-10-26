"""Universe renderer: bodies, trails, vectors, grid, starfield."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pygame

from simulation.bodies import Body, BodyType
from simulation.constants import AU
from simulation.universe import Universe
from .camera import Camera
from .theme import THEME


class UniverseRenderer:
    def __init__(self) -> None:
        self.show_labels = True
        self.show_trails = True
        self.show_velocity = False
        self.show_gravity = False
        self.show_grid = True
        self.show_starfield = True
        self._starfield: List[Tuple[int, int, int]] = []
        self._starfield_seeded_for: Optional[Tuple[int, int]] = None

    def _ensure_starfield(self, size: Tuple[int, int]) -> None:
        if self._starfield_seeded_for == size:
            return
        rng = np.random.default_rng(7)
        w, h = size
        n = 180
        self._starfield = [
            (int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(80, 220)))
            for _ in range(n)
        ]
        self._starfield_seeded_for = size

    def draw(
        self,
        surface: pygame.Surface,
        universe: Universe,
        camera: Camera,
        selected_id: Optional[str] = None,
    ) -> Optional[str]:
        """Draw the universe. Returns body_id under last concept — picking separate."""
        rect = surface.get_rect()
        size = (rect.width, rect.height)
        surface.fill(THEME.bg)

        if self.show_starfield:
            self._ensure_starfield(size)
            for x, y, b in self._starfield:
                surface.set_at((x % size[0], y % size[1]), (b, b, min(255, b + 20)))

        if self.show_grid:
            self._draw_grid(surface, camera, size)

        # Depth sort for 3D
        bodies = [b for b in universe.bodies if b.alive]
        projected = []
        for b in bodies:
            sx, sy, depth = camera.project(b.position, size)
            projected.append((depth, b, sx, sy))
        if camera.mode == "3d":
            projected.sort(key=lambda t: -t[0])

        if self.show_trails:
            for _, b, _, _ in projected:
                self._draw_trail(surface, b, camera, size)

        for _, b, sx, sy in projected:
            self._draw_body(surface, b, sx, sy, camera, selected=(b.body_id == selected_id))
            if self.show_velocity:
                self._draw_vector(
                    surface, b.position, b.velocity, camera, size, THEME.vector_v, scale=2e5
                )
            if self.show_gravity:
                self._draw_vector(
                    surface,
                    b.position,
                    b.acceleration,
                    camera,
                    size,
                    THEME.vector_g,
                    scale=5e10,
                )
            if self.show_labels:
                self._draw_label(surface, b.name, sx, sy)

        # Axes hint
        self._draw_axes(surface, camera, size)
        return None

    def _draw_grid(self, surface, camera: Camera, size) -> None:
        w, h = size
        # Draw AU grid lines around camera center
        spacing = AU
        # Choose spacing based on zoom
        px_spacing = spacing * camera.scale
        while px_spacing < 40:
            spacing *= 2
            px_spacing = spacing * camera.scale
        while px_spacing > 200:
            spacing *= 0.5
            px_spacing = spacing * camera.scale

        origin_sx, origin_sy, _ = camera.project(camera.center, size)
        # Vertical / horizontal lines in 2D; in 3D still draw world XY grid
        extent = max(w, h)
        n = int(extent / max(px_spacing, 1)) + 3
        for i in range(-n, n + 1):
            # Lines parallel to Y (constant X)
            world_x = camera.center[0] + i * spacing
            p1 = camera.project(np.array([world_x, camera.center[1] - n * spacing, camera.center[2]]), size)
            p2 = camera.project(np.array([world_x, camera.center[1] + n * spacing, camera.center[2]]), size)
            pygame.draw.aaline(surface, THEME.grid, (p1[0], p1[1]), (p2[0], p2[1]))
            world_y = camera.center[1] + i * spacing
            p1 = camera.project(np.array([camera.center[0] - n * spacing, world_y, camera.center[2]]), size)
            p2 = camera.project(np.array([camera.center[0] + n * spacing, world_y, camera.center[2]]), size)
            pygame.draw.aaline(surface, THEME.grid, (p1[0], p1[1]), (p2[0], p2[1]))

    def _draw_axes(self, surface, camera: Camera, size) -> None:
        origin = camera.center
        axis_len = 0.4 * AU / max(camera.zoom, 0.01)
        ox, oy, _ = camera.project(origin, size)
        for vec, color in [
            (np.array([axis_len, 0, 0]), (200, 80, 80)),
            (np.array([0, axis_len, 0]), (80, 200, 100)),
            (np.array([0, 0, axis_len]), (80, 120, 220)),
        ]:
            ex, ey, _ = camera.project(origin + vec, size)
            pygame.draw.aaline(surface, color, (ox, oy), (ex, ey))

    def _draw_trail(self, surface, body: Body, camera: Camera, size) -> None:
        if len(body.trail) < 2:
            return
        pts = []
        for p in body.trail:
            sx, sy, _ = camera.project(p, size)
            pts.append((sx, sy))
        if len(pts) >= 2:
            # Fade by drawing segments with varying alpha via colour blend
            n = len(pts)
            for i in range(1, n):
                t = i / n
                col = (
                    int(THEME.trail[0] * t + body.color[0] * (1 - t) * 0.3),
                    int(THEME.trail[1] * t + body.color[1] * (1 - t) * 0.3),
                    int(THEME.trail[2] * t + body.color[2] * (1 - t) * 0.3),
                )
                pygame.draw.aaline(surface, col, pts[i - 1], pts[i])

    def _draw_body(
        self,
        surface,
        body: Body,
        sx: float,
        sy: float,
        camera: Camera,
        selected: bool = False,
    ) -> None:
        # Visual radius: physical scaled + minimum pixel size by type
        r_px = max(2.0, body.radius * camera.scale * body.display_scale)
        # Clamp huge stars
        r_px = min(r_px, 80.0)
        if body.body_type == BodyType.STAR:
            r_px = max(r_px, 6.0)
            # Glow
            for k, alpha_f in [(3.5, 20), (2.2, 40), (1.4, 70)]:
                glow = pygame.Surface((int(r_px * k * 2 + 2), int(r_px * k * 2 + 2)), pygame.SRCALPHA)
                pygame.draw.circle(
                    glow,
                    (*body.color, alpha_f),
                    (glow.get_width() // 2, glow.get_height() // 2),
                    int(r_px * k),
                )
                surface.blit(
                    glow,
                    (sx - glow.get_width() / 2, sy - glow.get_height() / 2),
                )
        elif body.body_type == BodyType.BLACK_HOLE:
            r_px = max(r_px, 8.0)
            pygame.draw.circle(surface, (10, 5, 20), (int(sx), int(sy)), int(r_px))
            pygame.draw.circle(surface, (120, 60, 180), (int(sx), int(sy)), int(r_px), 2)
            # Accretion hint
            pygame.draw.ellipse(
                surface,
                (180, 100, 40),
                pygame.Rect(sx - r_px * 2, sy - r_px * 0.4, r_px * 4, r_px * 0.8),
                1,
            )
        else:
            r_px = max(r_px, 3.0)

        pygame.draw.circle(surface, body.color, (int(sx), int(sy)), int(r_px))
        if selected:
            pygame.draw.circle(surface, THEME.accent, (int(sx), int(sy)), int(r_px + 4), 2)

    def _draw_vector(
        self,
        surface,
        origin: np.ndarray,
        vec: np.ndarray,
        camera: Camera,
        size,
        color,
        scale: float,
    ) -> None:
        if float(np.linalg.norm(vec)) < 1e-15:
            return
        tip = origin + vec * scale
        x0, y0, _ = camera.project(origin, size)
        x1, y1, _ = camera.project(tip, size)
        pygame.draw.aaline(surface, color, (x0, y0), (x1, y1))
        # Arrow head
        direction = np.array([x1 - x0, y1 - y0])
        ln = np.linalg.norm(direction)
        if ln > 8:
            d = direction / ln
            left = np.array([-d[1], d[0]])
            p = np.array([x1, y1])
            a1 = p - d * 8 + left * 4
            a2 = p - d * 8 - left * 4
            pygame.draw.aaline(surface, color, (x1, y1), (a1[0], a1[1]))
            pygame.draw.aaline(surface, color, (x1, y1), (a2[0], a2[1]))

    def _draw_label(self, surface, text: str, sx: float, sy: float) -> None:
        font = pygame.font.SysFont("menlo", 13)
        img = font.render(text, True, THEME.text_dim)
        surface.blit(img, (sx + 8, sy - 8))

    def pick_body(
        self,
        universe: Universe,
        camera: Camera,
        screen_size: Tuple[int, int],
        mx: float,
        my: float,
    ) -> Optional[str]:
        best_id = None
        best_d = 1e18
        for b in universe.live_bodies():
            sx, sy, _ = camera.project(b.position, screen_size)
            r_px = max(6.0, min(80.0, b.radius * camera.scale * b.display_scale))
            d = (sx - mx) ** 2 + (sy - my) ** 2
            if d <= (r_px + 6) ** 2 and d < best_d:
                best_d = d
                best_id = b.body_id
        return best_id
