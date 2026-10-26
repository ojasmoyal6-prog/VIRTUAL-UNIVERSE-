"""Collision detection and momentum conservation."""

import numpy as np
import pytest

from simulation.bodies import Body, BodyType
from simulation.collisions import CollisionMode, detect_collisions, resolve_collisions


def _ball(name, mass, radius, pos, vel):
    return Body(
        name=name,
        mass=mass,
        radius=radius,
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        body_type=BodyType.CUSTOM,
    )


def test_detect_overlap():
    a = _ball("A", 1e24, 1e6, [0, 0, 0], [0, 0, 0])
    b = _ball("B", 1e24, 1e6, [1.5e6, 0, 0], [0, 0, 0])
    c = _ball("C", 1e24, 1e6, [1e8, 0, 0], [0, 0, 0])
    hits = detect_collisions([a, b, c])
    assert (0, 1) in hits
    assert (0, 2) not in hits


def test_merge_conserves_momentum():
    a = _ball("A", 2.0, 1.0, [0, 0, 0], [1.0, 0, 0])
    b = _ball("B", 1.0, 1.0, [1.5, 0, 0], [-1.0, 0, 0])
    p_before = a.momentum + b.momentum
    events = resolve_collisions([a, b], CollisionMode.MERGE, sim_time=0.0)
    assert events
    live = [x for x in [a, b] if x.alive]
    # After merge, one body replaced in list — resolve mutates list
    # Re-run with list container
    bodies = [
        _ball("A", 2.0, 1.0, [0, 0, 0], [1.0, 0, 0]),
        _ball("B", 1.0, 1.0, [1.5, 0, 0], [-1.0, 0, 0]),
    ]
    p_before = bodies[0].momentum + bodies[1].momentum
    resolve_collisions(bodies, CollisionMode.MERGE, 0.0)
    survivor = next(b for b in bodies if b.alive)
    assert survivor.momentum == pytest.approx(p_before, rel=1e-9)
    assert survivor.mass == pytest.approx(3.0)


def test_destroy_smaller():
    bodies = [
        _ball("Big", 3.0, 2.0, [0, 0, 0], [0, 0, 0]),
        _ball("Small", 1.0, 2.0, [3, 0, 0], [0, 0, 0]),
    ]
    resolve_collisions(bodies, CollisionMode.DESTROY_SMALLER, 0.0)
    assert bodies[0].alive
    assert not bodies[1].alive
