"""2D / 3D camera for projecting simulation coordinates to screen space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from simulation.bodies import Body
from simulation.constants import AU


@dataclass
class Camera:
    """Orthographic/perspective hybrid camera.

    Modes:
        '2d' — top-down XY view (Z ignored for projection)
        '3d' — rotatable perspective-ish projection
    """

    mode: str = "2d"
    center: np.ndarray = None  # type: ignore[assignment]
    zoom: float = 1.0  # pixels per metre (scaled)
    # 3D orientation (radians)
    yaw: float = 0.6
    pitch: float = 0.4
    # Follow target
    follow_id: Optional[str] = None
    # Base scale: how many pixels for 1 AU at zoom=1
    pixels_per_au: float = 180.0

    def __post_init__(self) -> None:
        if self.center is None:
            self.center = np.zeros(3, dtype=np.float64)
        else:
            self.center = np.asarray(self.center, dtype=np.float64).reshape(3)

    @property
    def scale(self) -> float:
        """pixels per metre."""
        return self.zoom * self.pixels_per_au / AU

    def world_to_view(self, pos: np.ndarray) -> np.ndarray:
        rel = np.asarray(pos, dtype=np.float64) - self.center
        if self.mode == "2d":
            return np.array([rel[0], rel[1], rel[2]])
        # Rotate by yaw (about Y) then pitch (about X)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        # yaw about z for astronomy top-ish feel, then pitch
        x1 = rel[0] * cy - rel[1] * sy
        y1 = rel[0] * sy + rel[1] * cy
        z1 = rel[2]
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        return np.array([x1, y2, z2])

    def project(
        self, pos: np.ndarray, screen_size: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """Return (screen_x, screen_y, depth). Depth used for 3D sizing/ordering."""
        v = self.world_to_view(pos)
        w, h = screen_size
        cx, cy = w * 0.5, h * 0.5
        if self.mode == "2d":
            sx = cx + v[0] * self.scale
            sy = cy - v[1] * self.scale
            return sx, sy, v[2]
        # Simple perspective foreshortening
        focal = 2.5 * AU
        depth = v[2] + focal
        factor = focal / max(depth, focal * 0.2)
        sx = cx + v[0] * self.scale * factor
        sy = cy - v[1] * self.scale * factor
        return sx, sy, depth

    def screen_to_world_xy(self, sx: float, sy: float, screen_size: Tuple[int, int]) -> np.ndarray:
        """Inverse project for 2D picking / placement (z = center.z)."""
        w, h = screen_size
        cx, cy = w * 0.5, h * 0.5
        x = (sx - cx) / self.scale + self.center[0]
        y = (cy - sy) / self.scale + self.center[1]
        return np.array([x, y, self.center[2]])

    def zoom_at(self, factor: float, sx: float, sy: float, screen_size: Tuple[int, int]) -> None:
        before = self.screen_to_world_xy(sx, sy, screen_size)
        self.zoom = float(np.clip(self.zoom * factor, 1e-4, 1e4))
        after = self.screen_to_world_xy(sx, sy, screen_size)
        self.center[0] += before[0] - after[0]
        self.center[1] += before[1] - after[1]

    def pan_pixels(self, dx: float, dy: float) -> None:
        self.center[0] -= dx / self.scale
        self.center[1] += dy / self.scale

    def update_follow(self, body: Optional[Body]) -> None:
        if body is not None and body.alive:
            self.center = body.position.copy()

    def fit_bodies(self, bodies, screen_size: Tuple[int, int], margin: float = 0.85) -> None:
        live = [b for b in bodies if b.alive]
        if not live:
            return
        pos = np.array([b.position for b in live])
        self.center = pos.mean(axis=0)
        extent = np.max(np.linalg.norm(pos - self.center, axis=1))
        extent = max(extent, 0.1 * AU)
        w, h = screen_size
        target = margin * min(w, h) * 0.5
        # extent * scale = target, scale = zoom * pixels_per_au / AU
        self.zoom = target * AU / (extent * self.pixels_per_au)
        self.zoom = float(np.clip(self.zoom, 1e-4, 1e4))
