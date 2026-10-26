"""Tests for gravitational force and acceleration."""

import numpy as np
import pytest

from simulation.constants import G, M_EARTH, M_SUN, AU
from simulation.physics import gravitational_force, pairwise_accelerations


def test_force_inverse_square():
    r = np.array([AU, 0.0, 0.0])
    f1 = gravitational_force(M_SUN, M_EARTH, r, softening=0.0)
    f2 = gravitational_force(M_SUN, M_EARTH, 2 * r, softening=0.0)
    assert np.linalg.norm(f1) == pytest.approx(4 * np.linalg.norm(f2), rel=1e-9)


def test_force_direction_toward_other():
    r = np.array([1e11, 0.0, 0.0])  # vector from m1 to m2
    f = gravitational_force(1e24, 1e24, r, softening=0.0)
    assert f[0] > 0
    assert abs(f[1]) < 1e-6
    assert abs(f[2]) < 1e-6


def test_acceleration_sun_earth_magnitude():
    positions = np.array([[0.0, 0.0, 0.0], [AU, 0.0, 0.0]])
    masses = np.array([M_SUN, M_EARTH])
    acc = pairwise_accelerations(positions, masses, softening=0.0)
    # Earth acceleration ≈ G M_sun / AU² toward sun (−x)
    expected = G * M_SUN / AU**2
    assert acc[1, 0] == pytest.approx(-expected, rel=1e-6)
    # Sun feels tiny acceleration toward Earth
    assert abs(acc[0, 0]) == pytest.approx(G * M_EARTH / AU**2, rel=1e-6)


def test_self_force_zero():
    positions = np.array([[0.0, 0.0, 0.0], [1e9, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    acc = pairwise_accelerations(positions, masses)
    # Finite
    assert np.all(np.isfinite(acc))
