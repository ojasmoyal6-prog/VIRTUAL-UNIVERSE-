"""Ready-made scenarios for the Virtual Universe Simulator.

Initial conditions use approximate heliocentric / relative states.
Positions emerge from Newtonian N-body integration — nothing is keyframed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from simulation.bodies import Body, BodyType, circular_orbit_velocity
from simulation.constants import (
    AU,
    DAY,
    G,
    M_EARTH,
    M_MOON,
    M_SUN,
    R_EARTH,
    R_MOON,
    R_SUN,
    YEAR,
)
from simulation.universe import Universe


def _clear_and_name(universe: Universe, name: str, description: str) -> None:
    universe.clear()
    universe.name = name
    universe.description = description


def _place_orbiting(
    name: str,
    mass: float,
    radius: float,
    central: Body,
    distance: float,
    body_type: BodyType,
    color: Tuple[int, int, int],
    display_scale: float = 1.0,
    inclination_deg: float = 0.0,
    phase_deg: float = 0.0,
) -> Body:
    """Create a body on an approximate circular orbit about ``central``."""
    phase = np.radians(phase_deg)
    incl = np.radians(inclination_deg)
    # Position in orbital plane then tilt
    x = distance * np.cos(phase)
    y = distance * np.sin(phase)
    z = 0.0
    # Rotate about x for inclination
    y2 = y * np.cos(incl) - z * np.sin(incl)
    z2 = y * np.sin(incl) + z * np.cos(incl)
    pos = central.position + np.array([x, y2, z2])

    # Tangential direction in plane
    speed = np.sqrt(G * (central.mass + mass) / distance)
    # Perpendicular to radius in xy before tilt
    tx, ty = -np.sin(phase), np.cos(phase)
    vel_orbit = speed * np.array([tx, ty * np.cos(incl), ty * np.sin(incl)])
    vel = central.velocity + vel_orbit

    return Body(
        name=name,
        mass=mass,
        radius=radius,
        position=pos,
        velocity=vel,
        body_type=body_type,
        color=color,
        display_scale=display_scale,
        parent_id=central.body_id,
    )


# Approximate planetary data (semi-major axis, mass, radius, color)
_PLANETS = [
    # name, a(AU), mass(kg), radius(m), color, display_scale, phase
    ("Mercury", 0.387, 3.301e23, 2.440e6, (180, 180, 180), 2.5, 10),
    ("Venus", 0.723, 4.867e24, 6.052e6, (230, 200, 140), 2.2, 40),
    ("Earth", 1.000, M_EARTH, R_EARTH, (70, 130, 230), 2.2, 0),
    ("Mars", 1.524, 6.417e23, 3.390e6, (200, 100, 70), 2.2, 80),
    ("Jupiter", 5.203, 1.898e27, 6.991e7, (210, 170, 120), 1.4, 120),
    ("Saturn", 9.537, 5.683e26, 5.823e7, (230, 210, 160), 1.4, 160),
    ("Uranus", 19.191, 8.681e25, 2.536e7, (160, 220, 230), 1.6, 200),
    ("Neptune", 30.069, 1.024e26, 2.462e7, (70, 100, 220), 1.6, 240),
]


def load_solar_system(universe: Universe, *, include_moon: bool = True) -> None:
    _clear_and_name(
        universe,
        "Solar System",
        "Approximate heliocentric initial conditions for the Sun and eight planets.",
    )
    # Prefer larger timesteps for outer planets visibility; users can change speed
    universe.dt = 6 * 3600  # 6 hours
    sun = Body(
        name="Sun",
        mass=M_SUN,
        radius=R_SUN,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.STAR,
        color=(255, 220, 80),
        display_scale=0.15,
    )
    universe.add_body(sun)

    earth = None
    for name, a_au, mass, radius, color, dscale, phase in _PLANETS:
        body = _place_orbiting(
            name,
            mass,
            radius,
            sun,
            a_au * AU,
            BodyType.PLANET,
            color,
            display_scale=dscale,
            phase_deg=phase,
        )
        universe.add_body(body)
        if name == "Earth":
            earth = body

    if include_moon and earth is not None:
        moon = _place_orbiting(
            "Moon",
            M_MOON,
            R_MOON,
            earth,
            3.844e8,
            BodyType.MOON,
            (200, 200, 210),
            display_scale=3.0,
            phase_deg=45,
        )
        universe.add_body(moon)

    universe.assign_parents_by_gravity()
    universe.save_snapshot()
    universe.log_event("Loaded Solar System preset.", "info")


def load_earth_moon(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Earth–Moon System",
        "Two-body Earth–Moon system with circular-orbit initial conditions.",
    )
    universe.dt = 1800.0
    # Place in CM frame
    d = 3.844e8
    m_e, m_m = M_EARTH, M_MOON
    r_e = d * m_m / (m_e + m_m)
    r_m = d * m_e / (m_e + m_m)
    omega = np.sqrt(G * (m_e + m_m) / d**3)
    earth = Body(
        name="Earth",
        mass=m_e,
        radius=R_EARTH,
        position=np.array([-r_e, 0.0, 0.0]),
        velocity=np.array([0.0, -omega * r_e, 0.0]),
        body_type=BodyType.PLANET,
        color=(70, 130, 230),
        display_scale=1.0,
    )
    moon = Body(
        name="Moon",
        mass=m_m,
        radius=R_MOON,
        position=np.array([r_m, 0.0, 0.0]),
        velocity=np.array([0.0, omega * r_m, 0.0]),
        body_type=BodyType.MOON,
        color=(200, 200, 210),
        display_scale=2.0,
        parent_id=earth.body_id,
    )
    universe.add_body(earth)
    universe.add_body(moon)
    universe.save_snapshot()
    universe.log_event("Loaded Earth–Moon system.", "info")


def load_binary_star(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Binary Star",
        "Equal-mass binary with an optional circumbinary planet.",
    )
    universe.dt = 3600.0
    sep = 2.0 * AU
    m = M_SUN
    r = sep / 2
    omega = np.sqrt(G * (2 * m) / sep**3)
    star_a = Body(
        name="Star A",
        mass=m,
        radius=R_SUN,
        position=np.array([-r, 0.0, 0.0]),
        velocity=np.array([0.0, -omega * r, 0.0]),
        body_type=BodyType.STAR,
        color=(255, 200, 100),
        display_scale=0.2,
    )
    star_b = Body(
        name="Star B",
        mass=m,
        radius=R_SUN * 0.9,
        position=np.array([r, 0.0, 0.0]),
        velocity=np.array([0.0, omega * r, 0.0]),
        body_type=BodyType.STAR,
        color=(255, 160, 120),
        display_scale=0.2,
    )
    universe.add_body(star_a)
    universe.add_body(star_b)
    # Circumbinary planet
    planet = _place_orbiting(
        "Planet P",
        M_EARTH,
        R_EARTH,
        Body(
            name="CM",
            mass=2 * m,
            radius=1.0,
            position=np.zeros(3),
            velocity=np.zeros(3),
            body_type=BodyType.CUSTOM,
        ),
        6.0 * AU,
        BodyType.PLANET,
        (100, 180, 255),
        display_scale=3.0,
    )
    # Fix parent references
    planet.parent_id = None
    universe.add_body(planet)
    universe.assign_parents_by_gravity()
    universe.save_snapshot()
    universe.log_event("Loaded binary star system.", "info")


def load_three_body(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Three-Body Problem",
        "Classic figure-eight choreography (approximate) / chaotic three-body demo.",
    )
    universe.dt = 1000.0
    # Moore figure-eight initial conditions (scaled), equal masses
    # Reference: Chenciner & Montgomery; scaled to astronomical units-ish
    m = M_EARTH * 1e6
    scale = AU * 0.5
    vscale = 1.2e4
    # Standard figure-eight relative IC (normalized), then scaled
    p1 = np.array([0.97000436, -0.24308753, 0.0]) * scale
    p2 = np.array([-0.97000436, 0.24308753, 0.0]) * scale
    p3 = np.zeros(3)
    v3 = np.array([-0.93240737, -0.86473146, 0.0]) * vscale
    v1 = np.array([0.466203685, 0.43236573, 0.0]) * vscale
    v2 = np.array([0.466203685, 0.43236573, 0.0]) * vscale
    # Adjust so v1 and v2 match literature (v1 = -v3/2 etc.)
    v1 = -0.5 * v3
    v2 = -0.5 * v3

    colors = [(255, 120, 120), (120, 255, 140), (120, 160, 255)]
    for i, (p, v, c) in enumerate(zip([p1, p2, p3], [v1, v2, v3], colors), start=1):
        universe.add_body(
            Body(
                name=f"Body {i}",
                mass=m,
                radius=R_EARTH * 5,
                position=p,
                velocity=v,
                body_type=BodyType.CUSTOM,
                color=c,
                display_scale=2.0,
            )
        )
    universe.save_snapshot()
    universe.log_event(
        "Loaded three-body problem. Long-term motion is highly sensitive to ICs.",
        "info",
    )


def load_planetary_collision(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Planetary Collision",
        "Two planets on a collision course near a star.",
    )
    universe.dt = 1800.0
    universe.collision_mode = universe.collision_mode  # keep current / merge default
    from simulation.collisions import CollisionMode

    universe.collision_mode = CollisionMode.MERGE
    sun = Body(
        name="Star",
        mass=M_SUN,
        radius=R_SUN,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.STAR,
        color=(255, 220, 80),
        display_scale=0.2,
    )
    universe.add_body(sun)
    p1 = Body(
        name="Planet A",
        mass=M_EARTH,
        radius=R_EARTH,
        position=np.array([1.0 * AU, 0.2 * AU, 0.0]),
        velocity=np.array([-5e3, 2.5e4, 0.0]),
        body_type=BodyType.PLANET,
        color=(80, 160, 255),
        display_scale=3.0,
    )
    p2 = Body(
        name="Planet B",
        mass=0.8 * M_EARTH,
        radius=0.9 * R_EARTH,
        position=np.array([1.0 * AU, -0.25 * AU, 0.0]),
        velocity=np.array([5e3, 2.2e4, 0.0]),
        body_type=BodyType.PLANET,
        color=(255, 120, 90),
        display_scale=3.0,
    )
    universe.add_body(p1)
    universe.add_body(p2)
    universe.save_snapshot()
    universe.log_event("Loaded planetary collision scenario.", "info")


def load_asteroid_belt(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Asteroid Belt",
        "Sun, Mars, Jupiter, and a belt of asteroids between ~2.2–3.2 AU.",
    )
    universe.dt = 12 * 3600
    sun = Body(
        name="Sun",
        mass=M_SUN,
        radius=R_SUN,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.STAR,
        color=(255, 220, 80),
        display_scale=0.15,
    )
    universe.add_body(sun)
    mars = _place_orbiting(
        "Mars", 6.417e23, 3.390e6, sun, 1.524 * AU, BodyType.PLANET, (200, 100, 70), 2.0, phase_deg=30
    )
    jupiter = _place_orbiting(
        "Jupiter", 1.898e27, 6.991e7, sun, 5.203 * AU, BodyType.PLANET, (210, 170, 120), 1.4, phase_deg=90
    )
    universe.add_body(mars)
    universe.add_body(jupiter)
    rng = np.random.default_rng(42)
    for i in range(60):
        a = rng.uniform(2.2, 3.2) * AU
        phase = rng.uniform(0, 360)
        mass = rng.uniform(1e15, 1e18)
        radius = rng.uniform(1e3, 5e4)
        rock = _place_orbiting(
            f"Asteroid {i+1}",
            mass,
            radius,
            sun,
            a,
            BodyType.ASTEROID,
            (160, 140, 120),
            display_scale=8.0,
            phase_deg=phase,
            inclination_deg=float(rng.uniform(-8, 8)),
        )
        # Slight eccentricity via velocity tweak
        rock.velocity *= rng.uniform(0.92, 1.05)
        universe.add_body(rock)
    universe.save_snapshot()
    universe.log_event("Loaded asteroid belt scenario.", "info")


def load_rogue_planet(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Rogue Planet",
        "A star–planet system visited by an unbound interstellar interloper.",
    )
    universe.dt = 3600.0
    sun = Body(
        name="Star",
        mass=M_SUN,
        radius=R_SUN,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.STAR,
        color=(255, 220, 80),
        display_scale=0.2,
    )
    universe.add_body(sun)
    planet = _place_orbiting(
        "Home Planet", M_EARTH, R_EARTH, sun, AU, BodyType.PLANET, (70, 160, 230), 2.5
    )
    universe.add_body(planet)
    rogue = Body(
        name="Rogue Planet",
        mass=2 * M_EARTH,
        radius=1.2 * R_EARTH,
        position=np.array([-4 * AU, 1.5 * AU, 0.3 * AU]),
        velocity=np.array([3.5e4, -5e3, 0.0]),
        body_type=BodyType.PLANET,
        color=(180, 80, 200),
        display_scale=2.5,
    )
    universe.add_body(rogue)
    universe.save_snapshot()
    universe.log_event("Loaded rogue planet flyby.", "info")


def load_artificial(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Artificial Solar System",
        "Compact educational system with exaggerated sizes for clarity.",
    )
    universe.dt = 1800.0
    star = Body(
        name="Star A",
        mass=1.5 * M_SUN,
        radius=R_SUN,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.STAR,
        color=(255, 180, 60),
        display_scale=0.25,
    )
    universe.add_body(star)
    for i, (a, mass_f, col) in enumerate(
        [
            (0.5, 0.5, (200, 200, 200)),
            (1.0, 1.0, (80, 140, 255)),
            (1.8, 3.0, (255, 140, 80)),
            (3.0, 20.0, (220, 200, 140)),
        ]
    ):
        p = _place_orbiting(
            f"Planet {i+1}",
            mass_f * M_EARTH,
            R_EARTH * (mass_f ** (1 / 3)),
            star,
            a * AU,
            BodyType.PLANET,
            col,
            display_scale=3.0,
            phase_deg=i * 50,
        )
        universe.add_body(p)
    universe.save_snapshot()
    universe.log_event("Loaded artificial solar system.", "info")


def load_black_hole(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Black-Hole Gravity Experiment",
        "WARNING: This scenario uses a simplified Newtonian treatment. "
        "General relativity (event horizons, frame dragging, light bending) "
        "is NOT implemented. The 'black hole' is a compact, massive Newtonian point mass.",
    )
    universe.dt = 600.0
    # 10 solar-mass compact object
    bh = Body(
        name="Black Hole (Newtonian)",
        mass=10 * M_SUN,
        radius=R_SUN * 0.05,  # visual only — not a Schwarzschild radius treatment
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.BLACK_HOLE,
        color=(30, 10, 50),
        display_scale=1.0,
    )
    universe.add_body(bh)
    for i in range(8):
        a = (0.3 + i * 0.25) * AU
        p = _place_orbiting(
            f"Probe {i+1}",
            1e20,
            1e5,
            bh,
            a,
            BodyType.ASTEROID,
            (180, 120, 255),
            display_scale=10.0,
            phase_deg=i * 40,
        )
        universe.add_body(p)
    universe.save_snapshot()
    universe.log_event(
        "Loaded black-hole experiment (Newtonian approximation only).",
        "warning",
    )


def load_random_universe(
    universe: Universe,
    *,
    n_stars: int = 2,
    n_planets: int = 8,
    n_moons: int = 4,
    mass_range: Tuple[float, float] = (0.1 * M_EARTH, 2 * M_SUN),
    distance_range: Tuple[float, float] = (0.5 * AU, 12 * AU),
    velocity_range: Tuple[float, float] = (0.0, 3e4),
    seed: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    _clear_and_name(
        universe,
        "Random Universe",
        f"Procedurally generated system ({n_stars} stars, {n_planets} planets, {n_moons} moons).",
    )
    universe.dt = 3600.0
    stars: List[Body] = []
    for i in range(max(1, n_stars)):
        m = rng.uniform(0.5, 2.0) * M_SUN
        if n_stars == 1:
            pos = np.zeros(3)
            vel = np.zeros(3)
        else:
            pos = rng.uniform(-distance_range[1] * 0.3, distance_range[1] * 0.3, size=3)
            pos[2] *= 0.2
            vel = rng.uniform(-5e3, 5e3, size=3)
        star = Body(
            name=f"Star {i+1}",
            mass=m,
            radius=R_SUN * (m / M_SUN) ** 0.5,
            position=pos,
            velocity=vel,
            body_type=BodyType.STAR,
            color=(255, int(rng.integers(160, 240)), int(rng.integers(60, 140))),
            display_scale=0.2,
        )
        universe.add_body(star)
        stars.append(star)

    planets: List[Body] = []
    for i in range(n_planets):
        host = stars[int(rng.integers(0, len(stars)))]
        dist = rng.uniform(*distance_range)
        mass = rng.uniform(mass_range[0], min(mass_range[1], 300 * M_EARTH))
        planet = _place_orbiting(
            f"Planet {i+1}",
            mass,
            R_EARTH * (mass / M_EARTH) ** (1 / 3),
            host,
            dist,
            BodyType.PLANET,
            (
                int(rng.integers(60, 255)),
                int(rng.integers(60, 255)),
                int(rng.integers(60, 255)),
            ),
            display_scale=2.5,
            phase_deg=float(rng.uniform(0, 360)),
            inclination_deg=float(rng.uniform(-15, 15)),
        )
        # Optional velocity jitter within range
        jitter = rng.uniform(*velocity_range) * 0.05
        planet.velocity += rng.normal(0, jitter, size=3)
        universe.add_body(planet)
        planets.append(planet)

    for i in range(n_moons):
        if not planets:
            break
        host = planets[int(rng.integers(0, len(planets)))]
        dist = rng.uniform(host.radius * 5, max(host.radius * 40, 5e8))
        moon = _place_orbiting(
            f"Moon {i+1}",
            rng.uniform(1e20, M_MOON),
            R_MOON * rng.uniform(0.3, 1.2),
            host,
            dist,
            BodyType.MOON,
            (200, 200, 210),
            display_scale=4.0,
            phase_deg=float(rng.uniform(0, 360)),
        )
        universe.add_body(moon)

    universe.assign_parents_by_gravity()
    universe.save_snapshot()
    universe.log_event("Generated random universe.", "info")


# Experiment-oriented loaders -------------------------------------------------

def load_experiment_velocity(universe: Universe) -> None:
    """Experiment 1: planet around star; user tweaks velocity."""
    _clear_and_name(
        universe,
        "Experiment: Change Planet Velocity",
        "Increase/decrease the planet's velocity to see circular, elliptical, escape, or crash orbits.",
    )
    universe.dt = 1800.0
    star = Body(
        name="Star",
        mass=M_SUN,
        radius=R_SUN,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.STAR,
        color=(255, 220, 80),
        display_scale=0.25,
    )
    universe.add_body(star)
    planet = _place_orbiting(
        "Test Planet", M_EARTH, R_EARTH, star, AU, BodyType.PLANET, (80, 160, 255), 3.0
    )
    universe.add_body(planet)
    universe.save_snapshot()
    universe.log_event(
        "Experiment ready: select Test Planet and modify velocity in the builder.",
        "info",
    )


def load_experiment_star_mass(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Experiment: Change Star Mass",
        "Alter the star's mass and observe how orbital period and shape respond.",
    )
    universe.dt = 1800.0
    star = Body(
        name="Variable Star",
        mass=M_SUN,
        radius=R_SUN,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.STAR,
        color=(255, 200, 100),
        display_scale=0.25,
    )
    universe.add_body(star)
    for i, a in enumerate([0.7, 1.0, 1.5]):
        p = _place_orbiting(
            f"Planet {i+1}",
            M_EARTH,
            R_EARTH,
            star,
            a * AU,
            BodyType.PLANET,
            (100 + i * 40, 140, 255 - i * 40),
            3.0,
            phase_deg=i * 60,
        )
        universe.add_body(p)
    universe.save_snapshot()
    universe.log_event("Experiment ready: edit Variable Star mass.", "info")


def load_experiment_escape(universe: Universe) -> None:
    _clear_and_name(
        universe,
        "Experiment: Escape Velocity",
        "Launch a probe from a planet; compare speed to v_esc = sqrt(2GM/r).",
    )
    universe.dt = 60.0
    planet = Body(
        name="Planet",
        mass=M_EARTH,
        radius=R_EARTH,
        position=np.zeros(3),
        velocity=np.zeros(3),
        body_type=BodyType.PLANET,
        color=(70, 130, 230),
        display_scale=1.0,
        fixed=True,
    )
    universe.add_body(planet)
    # Start just above surface with sub-escape velocity by default
    r0 = R_EARTH * 1.05
    v_esc = np.sqrt(2 * G * M_EARTH / r0)
    probe = Body(
        name="Probe",
        mass=1000.0,
        radius=10.0,
        position=np.array([r0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.85 * v_esc, 0.0]),
        body_type=BodyType.CUSTOM,
        color=(255, 220, 80),
        display_scale=20.0,
        parent_id=planet.body_id,
    )
    universe.add_body(probe)
    universe.save_snapshot()
    universe.log_event(
        f"Escape experiment: surface-ish v_esc ≈ {v_esc/1000:.2f} km/s. "
        f"Probe starts at 0.85 v_esc.",
        "info",
    )


SCENARIO_LOADERS: Dict[str, Callable[[Universe], None]] = {
    "Solar System": load_solar_system,
    "Earth–Moon System": load_earth_moon,
    "Binary Star": load_binary_star,
    "Three-Body Problem": load_three_body,
    "Planetary Collision": load_planetary_collision,
    "Asteroid Belt": load_asteroid_belt,
    "Rogue Planet": load_rogue_planet,
    "Artificial Solar System": load_artificial,
    "Black-Hole Gravity Experiment": load_black_hole,
    "Random Universe": lambda u: load_random_universe(u, seed=None),
    "Experiment: Planet Velocity": load_experiment_velocity,
    "Experiment: Star Mass": load_experiment_star_mass,
    "Experiment: Binary Star": load_binary_star,
    "Experiment: Three-Body": load_three_body,
    "Experiment: Planet Collision": load_planetary_collision,
    "Experiment: Escape Velocity": load_experiment_escape,
}


def list_scenarios() -> List[str]:
    return list(SCENARIO_LOADERS.keys())


def load_scenario(universe: Universe, name: str) -> None:
    loader = SCENARIO_LOADERS.get(name)
    if loader is None:
        raise KeyError(f"Unknown scenario: {name}")
    loader(universe)
