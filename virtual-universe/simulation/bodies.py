"""Celestial body data model.

Each body stores dynamical state (position, velocity, acceleration) plus
metadata used by visualisation and the Universe Builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple
from collections import deque
import uuid

import numpy as np


class BodyType(str, Enum):
    STAR = "star"
    PLANET = "planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    DWARF_PLANET = "dwarf_planet"
    BLACK_HOLE = "black_hole"  # Newtonian stand-in; not GR
    CUSTOM = "custom"


# Default RGBA-ish colours (0–255) keyed by body type
DEFAULT_COLORS = {
    BodyType.STAR: (255, 220, 120),
    BodyType.PLANET: (80, 160, 255),
    BodyType.MOON: (200, 200, 210),
    BodyType.ASTEROID: (160, 140, 120),
    BodyType.DWARF_PLANET: (180, 150, 200),
    BodyType.BLACK_HOLE: (40, 20, 60),
    BodyType.CUSTOM: (140, 220, 180),
}


@dataclass
class Body:
    """A single gravitating object in the simulation."""

    name: str
    mass: float  # kg
    radius: float  # m
    position: np.ndarray  # shape (3,), metres
    velocity: np.ndarray  # shape (3,), m/s
    body_type: BodyType = BodyType.CUSTOM
    color: Tuple[int, int, int] = (180, 180, 200)
    display_scale: float = 1.0  # visual radius multiplier (non-physical)
    parent_id: Optional[str] = None
    trail_maxlen: int = 800
    body_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    alive: bool = True
    fixed: bool = False  # if True, do not integrate (useful for experiments)

    # Motion history for trails: list of (x, y, z) samples
    trail: Deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3).copy()
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3).copy()
        self.acceleration = np.asarray(self.acceleration, dtype=np.float64).reshape(3).copy()
        if not self.color or self.color == (180, 180, 200):
            self.color = DEFAULT_COLORS.get(self.body_type, (180, 180, 200))
        if not isinstance(self.trail, deque):
            self.trail = deque(self.trail, maxlen=self.trail_maxlen)
        else:
            self.trail = deque(self.trail, maxlen=self.trail_maxlen)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def kinetic_energy(self) -> float:
        """Translational KE = ½ m |v|²."""
        return 0.5 * self.mass * float(np.dot(self.velocity, self.velocity))

    @property
    def momentum(self) -> np.ndarray:
        return self.mass * self.velocity

    def distance_to(self, other: "Body") -> float:
        return float(np.linalg.norm(self.position - other.position))

    def record_trail(self) -> None:
        if not self.alive:
            return
        self.trail.append(self.position.copy())

    def clear_trail(self) -> None:
        self.trail.clear()

    def copy(self, *, rename: Optional[str] = None) -> "Body":
        """Deep-ish copy with a new identity (for Duplicate in the builder)."""
        return Body(
            name=rename or f"{self.name} (copy)",
            mass=self.mass,
            radius=self.radius,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            body_type=self.body_type,
            color=self.color,
            display_scale=self.display_scale,
            parent_id=self.parent_id,
            trail_maxlen=self.trail_maxlen,
            acceleration=self.acceleration.copy(),
            alive=self.alive,
            fixed=self.fixed,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.body_id,
            "name": self.name,
            "mass": self.mass,
            "radius": self.radius,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
            "type": self.body_type.value,
            "color": list(self.color),
            "display_scale": self.display_scale,
            "parent_id": self.parent_id,
            "fixed": self.fixed,
            "alive": self.alive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Body":
        return cls(
            name=data["name"],
            mass=float(data["mass"]),
            radius=float(data["radius"]),
            position=np.array(data["position"], dtype=np.float64),
            velocity=np.array(data["velocity"], dtype=np.float64),
            body_type=BodyType(data.get("type", "custom")),
            color=tuple(data.get("color", (180, 180, 200))),  # type: ignore[arg-type]
            display_scale=float(data.get("display_scale", 1.0)),
            parent_id=data.get("parent_id"),
            body_id=data.get("id", str(uuid.uuid4())),
            fixed=bool(data.get("fixed", False)),
            alive=bool(data.get("alive", True)),
        )


def circular_orbit_velocity(
    central_mass: float,
    separation: float,
    G: float,
    plane: str = "xy",
) -> np.ndarray:
    """Return a tangential velocity for a circular orbit about a fixed centre.

    v = sqrt(G M / r)  (two-body, reduced mass ≈ m when M ≫ m)
    """
    speed = np.sqrt(G * central_mass / max(separation, 1.0))
    if plane == "xy":
        return np.array([0.0, speed, 0.0])
    if plane == "xz":
        return np.array([0.0, 0.0, speed])
    return np.array([0.0, speed, 0.0])
