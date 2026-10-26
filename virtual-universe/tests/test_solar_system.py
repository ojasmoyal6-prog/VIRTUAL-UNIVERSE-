"""Solar system preset and universe step smoke tests."""

import numpy as np
import pytest

from experiments.scenarios import load_solar_system, load_earth_moon
from simulation.universe import Universe


def test_solar_system_loads():
    u = Universe()
    load_solar_system(u, include_moon=True)
    names = {b.name for b in u.live_bodies()}
    for required in ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Moon"]:
        assert required in names


def test_solar_system_advances():
    u = Universe()
    load_solar_system(u, include_moon=False)
    earth = u.get_body_by_name("Earth")
    assert earth is not None
    pos0 = earth.position.copy()
    for _ in range(50):
        u.step_once()
    earth = u.get_body_by_name("Earth")
    assert earth is not None
    # Earth should have moved
    assert np.linalg.norm(earth.position - pos0) > 0


def test_reset_restores_snapshot():
    u = Universe()
    load_earth_moon(u)
    moon = u.get_body_by_name("Moon")
    assert moon is not None
    p0 = moon.position.copy()
    for _ in range(20):
        u.step_once()
    u.reset_to_snapshot()
    moon = u.get_body_by_name("Moon")
    assert moon is not None
    assert moon.position == pytest.approx(p0)
    assert u.sim_time == 0.0
