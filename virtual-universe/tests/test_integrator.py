"""Integrator and circular-orbit stability tests."""

import numpy as np
import pytest

from simulation.constants import G, M_SUN, AU
from simulation.integrator import IntegratorKind, step
from simulation.physics import pairwise_accelerations, kinetic_energy_total, potential_energy_total


def test_velocity_verlet_circular_orbit_stable():
    """Earth-mass particle on circular orbit should remain roughly bound over many steps."""
    m_central = M_SUN
    m = 1.0  # test particle
    r0 = AU
    v0 = np.sqrt(G * m_central / r0)
    positions = np.array([[0.0, 0.0, 0.0], [r0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, v0, 0.0]])
    masses = np.array([m_central, m])
    fixed = np.array([True, False])

    def accel_fn(pos):
        return pairwise_accelerations(pos, masses, softening=1e5)

    acc = accel_fn(positions)
    dt = 3600.0  # 1 hour
    for _ in range(500):
        positions, velocities, acc = step(
            IntegratorKind.VELOCITY_VERLET,
            positions,
            velocities,
            acc,
            dt,
            accel_fn,
            fixed_mask=fixed,
        )

    r = np.linalg.norm(positions[1] - positions[0])
    # Stay within 5% of AU after ~500 hours
    assert r == pytest.approx(AU, rel=0.05)


def test_energy_approximately_conserved_verlet():
    positions = np.array([[-1e11, 0.0, 0.0], [1e11, 0.0, 0.0]])
    masses = np.array([1e30, 1e30])
    # Circular-ish relative
    v = np.sqrt(G * 1e30 / (2e11)) / 2
    velocities = np.array([[0.0, -v, 0.0], [0.0, v, 0.0]])

    def accel_fn(pos):
        return pairwise_accelerations(pos, masses, softening=1e6)

    acc = accel_fn(positions)
    e0 = kinetic_energy_total(velocities, masses) + potential_energy_total(
        positions, masses, softening=1e6
    )
    dt = 1000.0
    for _ in range(200):
        positions, velocities, acc = step(
            IntegratorKind.VELOCITY_VERLET, positions, velocities, acc, dt, accel_fn
        )
    e1 = kinetic_energy_total(velocities, masses) + potential_energy_total(
        positions, masses, softening=1e6
    )
    assert abs(e1 - e0) / abs(e0) < 0.02
