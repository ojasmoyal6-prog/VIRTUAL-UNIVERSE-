"""Collision detection and response.

Detection: two live bodies collide when |r_i - r_j| ≤ (R_i + R_j).

Configurable modes:
    merge            — conserve momentum; combine mass; volume-additive radius
    bounce           — approximate elastic/inelastic bounce along contact normal
    destroy_smaller  — remove the less massive body
    record_only      — log the event, leave dynamics unchanged
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .bodies import Body, BodyType


class CollisionMode(str, Enum):
    MERGE = "merge"
    BOUNCE = "bounce"
    DESTROY_SMALLER = "destroy_smaller"
    RECORD_ONLY = "record_only"


@dataclass
class CollisionEvent:
    time: float
    body_a: str
    body_b: str
    mode: CollisionMode
    message: str


def detect_collisions(bodies: List[Body]) -> List[Tuple[int, int]]:
    """Return index pairs (i, j) with i < j that currently overlap."""
    live = [(i, b) for i, b in enumerate(bodies) if b.alive]
    hits: List[Tuple[int, int]] = []
    for a in range(len(live)):
        i, bi = live[a]
        for b in range(a + 1, len(live)):
            j, bj = live[b]
            min_dist = bi.radius + bj.radius
            if bi.distance_to(bj) <= min_dist:
                hits.append((i, j))
    return hits


def resolve_collisions(
    bodies: List[Body],
    mode: CollisionMode,
    sim_time: float,
    *,
    restitution: float = 0.6,
) -> List[CollisionEvent]:
    """Detect and resolve all current collisions. Mutates ``bodies`` in place."""
    events: List[CollisionEvent] = []
    pairs = detect_collisions(bodies)
    # Process heavier interactions first when merging to reduce chain issues
    pairs_sorted = sorted(
        pairs,
        key=lambda ij: -(bodies[ij[0]].mass + bodies[ij[1]].mass),
    )

    consumed = set()
    for i, j in pairs_sorted:
        if i in consumed or j in consumed:
            continue
        a, b = bodies[i], bodies[j]
        if not a.alive or not b.alive:
            continue

        if mode == CollisionMode.RECORD_ONLY:
            events.append(
                CollisionEvent(
                    sim_time,
                    a.name,
                    b.name,
                    mode,
                    f"Collision detected: {a.name} + {b.name}",
                )
            )
            continue

        if mode == CollisionMode.DESTROY_SMALLER:
            victim, survivor = (a, b) if a.mass <= b.mass else (b, a)
            victim.alive = False
            victim.clear_trail()
            consumed.add(i if victim is a else j)
            events.append(
                CollisionEvent(
                    sim_time,
                    a.name,
                    b.name,
                    mode,
                    f"Collision detected: {victim.name} destroyed by {survivor.name}",
                )
            )
            continue

        if mode == CollisionMode.BOUNCE:
            _bounce(a, b, restitution)
            events.append(
                CollisionEvent(
                    sim_time,
                    a.name,
                    b.name,
                    mode,
                    f"Collision detected: {a.name} bounced off {b.name}",
                )
            )
            continue

        # MERGE — momentum conservation: m_tot v = m_a v_a + m_b v_b
        if a.mass < b.mass:
            a, b = b, a
            i, j = j, i
        merged = _merge(a, b)
        # Replace heavier with merged; kill lighter
        bodies[i] = merged
        bodies[j].alive = False
        bodies[j].clear_trail()
        consumed.add(j)
        events.append(
            CollisionEvent(
                sim_time,
                a.name,
                b.name,
                mode,
                f"Collision detected: {a.name} + {b.name} → {merged.name}",
            )
        )
    return events


def _merge(a: Body, b: Body) -> Body:
    m = a.mass + b.mass
    v = (a.momentum + b.momentum) / m
    # Volume-additive sphere radius (approximation)
    vol = (4.0 / 3.0) * np.pi * (a.radius**3 + b.radius**3)
    radius = float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))
    # Mass-weighted centre of mass position
    pos = (a.mass * a.position + b.mass * b.position) / m
    # Prefer more "dominant" type
    type_rank = {
        BodyType.BLACK_HOLE: 6,
        BodyType.STAR: 5,
        BodyType.PLANET: 4,
        BodyType.DWARF_PLANET: 3,
        BodyType.MOON: 2,
        BodyType.ASTEROID: 1,
        BodyType.CUSTOM: 0,
    }
    btype = a.body_type if type_rank.get(a.body_type, 0) >= type_rank.get(b.body_type, 0) else b.body_type
    color = a.color if a.mass >= b.mass else b.color
    return Body(
        name=f"{a.name}+{b.name}",
        mass=m,
        radius=radius,
        position=pos,
        velocity=v,
        body_type=btype,
        color=color,
        display_scale=max(a.display_scale, b.display_scale),
        parent_id=a.parent_id or b.parent_id,
        trail_maxlen=max(a.trail_maxlen, b.trail_maxlen),
        body_id=a.body_id,
    )


def _bounce(a: Body, b: Body, restitution: float) -> None:
    """1D collision along the contact normal with optional mass-weighted push-out."""
    n = a.position - b.position
    dist = float(np.linalg.norm(n))
    if dist < 1e-12:
        n = np.array([1.0, 0.0, 0.0])
        dist = 1.0
    else:
        n = n / dist

    # Separate overlapping spheres
    overlap = a.radius + b.radius - dist
    if overlap > 0:
        if a.fixed and not b.fixed:
            b.position = b.position - n * overlap
        elif b.fixed and not a.fixed:
            a.position = a.position + n * overlap
        else:
            a.position = a.position + 0.5 * n * overlap
            b.position = b.position - 0.5 * n * overlap

    if a.fixed and b.fixed:
        return

    rv = a.velocity - b.velocity
    vel_along_n = float(np.dot(rv, n))
    if vel_along_n > 0:
        return  # already separating

    inv_a = 0.0 if a.fixed else 1.0 / a.mass
    inv_b = 0.0 if b.fixed else 1.0 / b.mass
    impulse = -(1.0 + restitution) * vel_along_n / (inv_a + inv_b + 1e-30)
    impulse_vec = impulse * n
    if not a.fixed:
        a.velocity = a.velocity + impulse_vec * inv_a
    if not b.fixed:
        b.velocity = b.velocity - impulse_vec * inv_b
