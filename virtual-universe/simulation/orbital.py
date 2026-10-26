"""Orbital element estimates for a secondary about a primary.

Assumptions (documented):
    - Instantaneous two-body problem using relative state
    - Central body much more massive OR mutual reduced-mass treatment
    - Newtonian gravity; no drag, relativity, or oblateness

Vis-viva / specific orbital energy:

    ε = |v_rel|² / 2 − G(M+m)/r

Semi-major axis (bound ellipse, ε < 0):

    a = −G(M+m) / (2ε)

Eccentricity from eccentricity vector:

    e⃗ = (v × h)/μ − r̂
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bodies import Body
from .constants import G, DAY


@dataclass
class OrbitalElements:
    parent_name: str
    distance: float
    speed: float
    specific_energy: float
    semi_major_axis: Optional[float]
    eccentricity: Optional[float]
    periapsis: Optional[float]
    apoapsis: Optional[float]
    period: Optional[float]
    bound: bool
    orbital_velocity_circular: float


def analyze_orbit(
    body: Body,
    parent: Body,
    *,
    G_const: float = G,
) -> OrbitalElements:
    r_vec = body.position - parent.position
    v_vec = body.velocity - parent.velocity
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    mu = G_const * (parent.mass + body.mass)

    specific_energy = 0.5 * v * v - mu / max(r, 1.0)
    v_circ = float(np.sqrt(mu / max(r, 1.0)))

    h_vec = np.cross(r_vec, v_vec)
    h2 = float(np.dot(h_vec, h_vec))
    if r > 0 and mu > 0:
        e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
        ecc = float(np.linalg.norm(e_vec))
    else:
        ecc = None

    bound = specific_energy < 0
    a = peri = apo = period = None
    if bound and specific_energy < 0:
        a = -mu / (2.0 * specific_energy)
        if ecc is not None:
            peri = a * (1.0 - ecc)
            apo = a * (1.0 + ecc)
        # Kepler period T = 2π sqrt(a³ / μ)
        if a > 0:
            period = 2.0 * np.pi * np.sqrt(a**3 / mu)

    return OrbitalElements(
        parent_name=parent.name,
        distance=r,
        speed=v,
        specific_energy=specific_energy,
        semi_major_axis=a,
        eccentricity=ecc,
        periapsis=peri,
        apoapsis=apo,
        period=period,
        bound=bound,
        orbital_velocity_circular=v_circ,
    )


def escape_speed(central_mass: float, distance: float, G_const: float = G) -> float:
    """v_esc = sqrt(2GM / r)"""
    return float(np.sqrt(2.0 * G_const * central_mass / max(distance, 1.0)))


def format_period(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "—"
    days = seconds / DAY
    if days >= 365.25:
        return f"{days / 365.25:.2f} yr"
    if days >= 1:
        return f"{days:.2f} d"
    hours = seconds / 3600.0
    if hours >= 1:
        return f"{hours:.2f} h"
    return f"{seconds:.1f} s"
