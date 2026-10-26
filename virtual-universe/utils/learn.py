"""Educational Learn Mode explanations."""

from __future__ import annotations

from typing import List, Optional

from simulation.bodies import Body
from simulation.orbital import analyze_orbit, escape_speed
from simulation.universe import Universe


EXPLANATIONS = {
    "orbits": (
        "Orbits occur because sideways motion combines with continuous gravitational "
        "acceleration toward a massive body. The object keeps falling toward the "
        "primary but also keeps missing it."
    ),
    "escape": (
        "Escape velocity is the speed needed to reach infinity with zero residual speed: "
        "v_esc = √(2GM/r). Above this (ignoring other bodies), the trajectory is unbound."
    ),
    "ellipse": (
        "Bound orbits in the two-body Newtonian problem are conic sections. "
        "Ellipses arise when specific orbital energy is negative and eccentricity < 1."
    ),
    "com": (
        "Bodies orbit their common centre of mass. In an equal-mass binary both stars "
        "trace similar paths around the midpoint."
    ),
    "slingshot": (
        "A gravitational slingshot exchanges momentum with a moving planet: in the "
        "planet's frame the flyby is symmetric, but in the heliocentric frame the "
        "probe can gain or lose speed."
    ),
    "three_body": (
        "The three-body problem is generally chaotic: tiny changes in initial "
        "conditions grow exponentially, so long-term prediction becomes impractical."
    ),
}


def learn_messages(universe: Universe, selected: Optional[Body]) -> List[str]:
    msgs: List[str] = []
    name = universe.name.lower()
    if "three-body" in name or "three body" in name:
        msgs.append(EXPLANATIONS["three_body"])
    if "binary" in name:
        msgs.append(EXPLANATIONS["com"])
    if "escape" in name:
        msgs.append(EXPLANATIONS["escape"])
    if "black" in name:
        msgs.append(
            "This black-hole scenario is Newtonian only — no event horizon physics."
        )

    if selected is not None:
        parent = universe.get_body(selected.parent_id) if selected.parent_id else None
        if parent is not None:
            orb = analyze_orbit(selected, parent, G_const=universe.G)
            msgs.append(EXPLANATIONS["orbits"])
            if orb.bound:
                msgs.append(
                    f"{selected.name} is gravitationally bound to {parent.name}. "
                    + EXPLANATIONS["ellipse"]
                )
                if orb.eccentricity is not None and orb.eccentricity > 0.2:
                    msgs.append(
                        f"Eccentricity ≈ {orb.eccentricity:.3f}: the orbit is noticeably elliptical."
                    )
            else:
                msgs.append(
                    f"{selected.name} appears unbound relative to {parent.name} "
                    f"(specific energy ≥ 0)."
                )
            vesc = escape_speed(parent.mass, orb.distance, universe.G)
            msgs.append(
                f"Local escape speed from {parent.name} at this distance ≈ {vesc/1000:.2f} km/s "
                f"(object speed ≈ {orb.speed/1000:.2f} km/s)."
            )
        else:
            msgs.append(
                f"{selected.name} has no assigned parent. Parent assignment is a heuristic "
                f"based on the strongest nearby gravitational influence."
            )
    if not msgs:
        msgs.append(
            "Enable Learn Mode and select an object to see contextual physics explanations."
        )
    return msgs[:5]
