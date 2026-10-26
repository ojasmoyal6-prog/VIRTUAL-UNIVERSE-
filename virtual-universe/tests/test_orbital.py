"""Orbital analysis helpers."""

import numpy as np
import pytest

from simulation.bodies import Body, BodyType
from simulation.constants import G, M_EARTH, M_SUN, AU, R_EARTH, R_SUN
from simulation.orbital import analyze_orbit, escape_speed


def test_circular_earth_orbit_elements():
    sun = Body("Sun", M_SUN, R_SUN, np.zeros(3), np.zeros(3), BodyType.STAR)
    v = np.sqrt(G * M_SUN / AU)
    earth = Body(
        "Earth",
        M_EARTH,
        R_EARTH,
        np.array([AU, 0, 0]),
        np.array([0, v, 0]),
        BodyType.PLANET,
    )
    orb = analyze_orbit(earth, sun)
    assert orb.bound
    assert orb.eccentricity == pytest.approx(0.0, abs=1e-3)
    assert orb.semi_major_axis == pytest.approx(AU, rel=1e-3)
    assert orb.period == pytest.approx(365.25 * 86400, rel=0.02)


def test_escape_speed_earth_surface():
    vesc = escape_speed(M_EARTH, R_EARTH)
    assert vesc == pytest.approx(11186, rel=0.02)
