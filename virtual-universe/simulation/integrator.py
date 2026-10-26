"""Numerical integrators for the N-body ODE system.

Preferred method: Velocity Verlet (symplectic, good energy behaviour):

    x_{n+1} = x_n + v_n Δt + ½ a_n Δt²
    a_{n+1} = a(x_{n+1})
    v_{n+1} = v_n + ½ (a_n + a_{n+1}) Δt

Also provided: leapfrog (kick–drift–kick) and semi-implicit Euler.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

import numpy as np


class IntegratorKind(str, Enum):
    VELOCITY_VERLET = "velocity_verlet"
    LEAPFROG = "leapfrog"
    SEMI_IMPLICIT_EULER = "semi_implicit_euler"


AccelFn = Callable[[np.ndarray], np.ndarray]


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
    fixed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one Velocity Verlet step. Returns (pos, vel, acc)."""
    pos = positions.copy()
    vel = velocities.copy()
    acc = accelerations.copy()

    movable = np.ones(len(pos), dtype=bool) if fixed_mask is None else ~fixed_mask

    pos[movable] = pos[movable] + vel[movable] * dt + 0.5 * acc[movable] * dt * dt
    new_acc = accel_fn(pos)
    vel[movable] = vel[movable] + 0.5 * (acc[movable] + new_acc[movable]) * dt
    new_acc[~movable] = 0.0
    vel[~movable] = velocities[~movable]
    pos[~movable] = positions[~movable]
    return pos, vel, new_acc


def leapfrog_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
    fixed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kick–drift–kick leapfrog (equivalent accuracy to Velocity Verlet)."""
    pos = positions.copy()
    vel = velocities.copy()
    movable = np.ones(len(pos), dtype=bool) if fixed_mask is None else ~fixed_mask

    vel[movable] = vel[movable] + 0.5 * accelerations[movable] * dt
    pos[movable] = pos[movable] + vel[movable] * dt
    new_acc = accel_fn(pos)
    vel[movable] = vel[movable] + 0.5 * new_acc[movable] * dt
    new_acc[~movable] = 0.0
    vel[~movable] = velocities[~movable]
    pos[~movable] = positions[~movable]
    return pos, vel, new_acc


def semi_implicit_euler_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
    fixed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-implicit (symplectic) Euler: v ← v + a Δt; x ← x + v Δt."""
    pos = positions.copy()
    vel = velocities.copy()
    movable = np.ones(len(pos), dtype=bool) if fixed_mask is None else ~fixed_mask

    # Recompute acceleration at current state for consistency
    acc = accel_fn(pos) if accelerations is None else accelerations
    vel[movable] = vel[movable] + acc[movable] * dt
    pos[movable] = pos[movable] + vel[movable] * dt
    new_acc = accel_fn(pos)
    new_acc[~movable] = 0.0
    vel[~movable] = velocities[~movable]
    pos[~movable] = positions[~movable]
    return pos, vel, new_acc


def step(
    kind: IntegratorKind,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
    fixed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == IntegratorKind.VELOCITY_VERLET:
        return velocity_verlet_step(
            positions, velocities, accelerations, dt, accel_fn, fixed_mask
        )
    if kind == IntegratorKind.LEAPFROG:
        return leapfrog_step(
            positions, velocities, accelerations, dt, accel_fn, fixed_mask
        )
    return semi_implicit_euler_step(
        positions, velocities, accelerations, dt, accel_fn, fixed_mask
    )
