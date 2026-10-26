"""Universe container: owns bodies, advances time, logs events."""

from __future__ import annotations

import copy
import time as wall_time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .bodies import Body
from .collisions import CollisionMode, resolve_collisions
from .constants import DAY, DEFAULT_SOFTENING, G
from .diagnostics import Diagnostics
from .integrator import IntegratorKind, step as integrate_step
from .physics import barnes_hut_accelerations, pairwise_accelerations


@dataclass
class SimEvent:
    sim_time: float
    message: str
    category: str = "info"  # info | collision | orbit | warning


class Universe:
    """Central simulation state machine."""

    def __init__(
        self,
        *,
        G_const: float = G,
        softening: float = DEFAULT_SOFTENING,
        dt: float = 3600.0,  # 1 hour base step
        integrator: IntegratorKind = IntegratorKind.VELOCITY_VERLET,
        collision_mode: CollisionMode = CollisionMode.MERGE,
        use_barnes_hut_threshold: int = 80,
        barnes_hut_theta: float = 0.6,
    ) -> None:
        self.bodies: List[Body] = []
        self.G = G_const
        self.softening = softening
        self.dt = dt
        self.integrator = integrator
        self.collision_mode = collision_mode
        self.use_barnes_hut_threshold = use_barnes_hut_threshold
        self.barnes_hut_theta = barnes_hut_theta

        self.sim_time = 0.0  # seconds
        self.running = False
        self.speed_multiplier = 1.0
        self.events: List[SimEvent] = []
        self.diagnostics = Diagnostics()
        self._snapshot: Optional[List[Body]] = None
        self._wall_start: Optional[float] = None
        self.wall_elapsed = 0.0
        self.trail_sample_every = 1
        self._step_count = 0
        self.selected_id: Optional[str] = None
        self.focus_id: Optional[str] = None
        self.learn_mode = False
        self.name = "Empty Universe"
        self.description = ""
        self._event_hooks: List[Callable[[SimEvent], None]] = []

    # ------------------------------------------------------------------
    # Body management
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.bodies.clear()
        self.sim_time = 0.0
        self.events.clear()
        self.diagnostics.clear()
        self.selected_id = None
        self.focus_id = None
        self._snapshot = None
        self._step_count = 0
        self.wall_elapsed = 0.0
        self._wall_start = None

    def add_body(self, body: Body) -> Body:
        self.bodies.append(body)
        return body

    def remove_body(self, body_id: str) -> None:
        self.bodies = [b for b in self.bodies if b.body_id != body_id]

    def get_body(self, body_id: str) -> Optional[Body]:
        for b in self.bodies:
            if b.body_id == body_id:
                return b
        return None

    def get_body_by_name(self, name: str) -> Optional[Body]:
        for b in self.bodies:
            if b.name == name and b.alive:
                return b
        return None

    def live_bodies(self) -> List[Body]:
        return [b for b in self.bodies if b.alive]

    def duplicate_body(self, body_id: str) -> Optional[Body]:
        src = self.get_body(body_id)
        if src is None:
            return None
        clone = src.copy()
        clone.position = src.position + np.array([src.radius * 3, 0.0, 0.0])
        self.bodies.append(clone)
        return clone

    def save_snapshot(self) -> None:
        self._snapshot = [copy.deepcopy(b) for b in self.bodies]
        for b in self._snapshot:
            b.clear_trail()

    def reset_to_snapshot(self) -> None:
        if self._snapshot is None:
            return
        self.bodies = [copy.deepcopy(b) for b in self._snapshot]
        self.sim_time = 0.0
        self.diagnostics.clear()
        self._step_count = 0
        self.wall_elapsed = 0.0
        self._wall_start = wall_time.perf_counter() if self.running else None
        self.log_event("Simulation reset to initial conditions.", "info")

    # ------------------------------------------------------------------
    # Time control
    # ------------------------------------------------------------------
    def play(self) -> None:
        self.running = True
        if self._wall_start is None:
            self._wall_start = wall_time.perf_counter()

    def pause(self) -> None:
        if self.running and self._wall_start is not None:
            self.wall_elapsed += wall_time.perf_counter() - self._wall_start
            self._wall_start = None
        self.running = False

    def toggle(self) -> None:
        if self.running:
            self.pause()
        else:
            self.play()

    def set_speed(self, multiplier: float) -> None:
        self.speed_multiplier = max(0.0, float(multiplier))

    @property
    def sim_time_days(self) -> float:
        return self.sim_time / DAY

    @property
    def real_elapsed(self) -> float:
        extra = 0.0
        if self.running and self._wall_start is not None:
            extra = wall_time.perf_counter() - self._wall_start
        return self.wall_elapsed + extra

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _arrays(self):
        n = len(self.bodies)
        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        acc = np.zeros((n, 3))
        mass = np.zeros(n)
        alive = np.zeros(n, dtype=bool)
        fixed = np.zeros(n, dtype=bool)
        for i, b in enumerate(self.bodies):
            pos[i] = b.position
            vel[i] = b.velocity
            acc[i] = b.acceleration
            mass[i] = b.mass
            alive[i] = b.alive
            fixed[i] = b.fixed
        return pos, vel, acc, mass, alive, fixed

    def _write_back(self, pos, vel, acc) -> None:
        for i, b in enumerate(self.bodies):
            if not b.alive:
                continue
            b.position = pos[i].copy()
            b.velocity = vel[i].copy()
            b.acceleration = acc[i].copy()

    def _accel_fn(self, mass, alive):
        def fn(positions: np.ndarray) -> np.ndarray:
            n_live = int(np.count_nonzero(alive))
            if n_live >= self.use_barnes_hut_threshold:
                return barnes_hut_accelerations(
                    positions,
                    mass,
                    theta=self.barnes_hut_theta,
                    G_const=self.G,
                    softening=self.softening,
                    alive_mask=alive,
                )
            return pairwise_accelerations(
                positions,
                mass,
                G_const=self.G,
                softening=self.softening,
                alive_mask=alive,
            )

        return fn

    def step_once(self, dt: Optional[float] = None) -> None:
        if not self.bodies:
            return
        h = self.dt if dt is None else dt
        pos, vel, acc, mass, alive, fixed = self._arrays()
        accel_fn = self._accel_fn(mass, alive)
        # Ensure acceleration is current before first Verlet half-step
        if self._step_count == 0 or np.allclose(acc, 0):
            acc = accel_fn(pos)
        pos, vel, acc = integrate_step(
            self.integrator, pos, vel, acc, h, accel_fn, fixed_mask=fixed
        )
        self._write_back(pos, vel, acc)
        self.sim_time += h
        self._step_count += 1

        # Collisions
        for ev in resolve_collisions(self.bodies, self.collision_mode, self.sim_time):
            self.log_event(ev.message, "collision")

        # Trails
        if self._step_count % max(1, self.trail_sample_every) == 0:
            for b in self.bodies:
                if b.alive:
                    b.record_trail()

        # Diagnostics (throttle)
        if self._step_count % 5 == 0:
            pos, vel, acc, mass, alive, fixed = self._arrays()
            self.diagnostics.update(
                self.sim_time,
                pos,
                vel,
                mass,
                alive,
                G_const=self.G,
                softening=self.softening,
            )
            if self.diagnostics.energy_drift_warning and self.diagnostics.warning_message:
                # Log once periodically
                if self._step_count % 200 == 0:
                    self.log_event(self.diagnostics.warning_message, "warning")

    def advance(self, real_dt: float) -> None:
        """Advance simulation according to wall-clock frame time and speed."""
        if not self.running:
            return
        # Cap substeps to keep UI responsive
        sim_dt_target = real_dt * self.speed_multiplier * self.dt
        # Interpret speed_multiplier as "how many base dt per real second"
        # More intuitive: speed 1× → advance `dt` seconds of sim per real second
        # speed 100× → 100 * dt per real second
        budget = real_dt * self.speed_multiplier
        # budget is in "multiples of base dt"
        steps = int(min(max(budget, 0), 200))
        # fractional remainder ignored for stability
        if steps == 0 and self.speed_multiplier > 0:
            # For very slow speeds, accumulate via fractional step
            self.step_once(self.dt * min(budget, 1.0))
            return
        for _ in range(steps):
            self.step_once()

    # ------------------------------------------------------------------
    # Events / parent heuristics
    # ------------------------------------------------------------------
    def log_event(self, message: str, category: str = "info") -> None:
        ev = SimEvent(self.sim_time, message, category)
        self.events.append(ev)
        if len(self.events) > 500:
            self.events = self.events[-500:]
        for hook in self._event_hooks:
            hook(ev)

    def clear_events(self) -> None:
        self.events.clear()

    def assign_parents_by_gravity(self) -> None:
        """Heuristic: parent = most massive body that dominates local potential."""
        live = self.live_bodies()
        for b in live:
            best = None
            best_score = 0.0
            for other in live:
                if other.body_id == b.body_id:
                    continue
                if other.mass <= b.mass:
                    continue
                r = b.distance_to(other)
                score = other.mass / max(r, 1.0)
                if score > best_score:
                    best_score = score
                    best = other
            b.parent_id = best.body_id if best else None

    def potential_with(self, a: Body, b: Body) -> float:
        r = a.distance_to(b)
        return -self.G * a.mass * b.mass / np.sqrt(r * r + self.softening**2)

    def kinetic_of(self, body: Body) -> float:
        return body.kinetic_energy

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "sim_time": self.sim_time,
            "bodies": [b.to_dict() for b in self.bodies if b.alive],
            "dt": self.dt,
            "integrator": self.integrator.value,
            "collision_mode": self.collision_mode.value,
            "softening": self.softening,
        }
