"""CSV and experiment-report export."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from simulation.orbital import analyze_orbit
from simulation.universe import Universe
from utils.formatting import fmt_time_days, sci


def export_csv(universe: Universe, path: Path) -> Path:
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Time_s",
                "Object",
                "X",
                "Y",
                "Z",
                "Velocity_X",
                "Velocity_Y",
                "Velocity_Z",
                "Speed",
                "Acceleration",
                "Distance_from_parent",
            ]
        )
        for b in universe.live_bodies():
            parent = universe.get_body(b.parent_id) if b.parent_id else None
            dist = b.distance_to(parent) if parent else ""
            writer.writerow(
                [
                    universe.sim_time,
                    b.name,
                    b.position[0],
                    b.position[1],
                    b.position[2],
                    b.velocity[0],
                    b.velocity[1],
                    b.velocity[2],
                    b.speed,
                    float(__import__("numpy").linalg.norm(b.acceleration)),
                    dist,
                ]
            )
    return path


def export_experiment_report(universe: Universe, path: Path) -> Path:
    path = Path(path)
    diag = universe.diagnostics.series()
    lines = []
    lines.append("Virtual Universe Simulator — Experiment Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Scenario: {universe.name}")
    lines.append(f"Description: {universe.description}")
    lines.append(f"Simulation duration: {fmt_time_days(universe.sim_time)}")
    lines.append(f"Integrator: {universe.integrator.value}")
    lines.append(f"Timestep dt: {universe.dt} s")
    lines.append(f"Softening ε: {universe.softening} m")
    lines.append(f"Collision mode: {universe.collision_mode.value}")
    lines.append("")
    lines.append("=== Initial / Current Objects ===")
    for b in universe.live_bodies():
        lines.append(
            f"- {b.name} [{b.body_type.value}] mass={sci(b.mass,'kg')} "
            f"pos=({b.position[0]:.6e}, {b.position[1]:.6e}, {b.position[2]:.6e}) "
            f"vel=({b.velocity[0]:.6e}, {b.velocity[1]:.6e}, {b.velocity[2]:.6e})"
        )
    lines.append("")
    lines.append("=== Energy Conservation ===")
    if diag["te"]:
        e0, e1 = diag["te"][0], diag["te"][-1]
        drift = abs(e1 - e0) / abs(e0) * 100 if abs(e0) > 0 else 0.0
        lines.append(f"Initial total energy: {sci(e0, 'J')}")
        lines.append(f"Final total energy:   {sci(e1, 'J')}")
        lines.append(f"Relative drift:       {drift:.4f} %")
        lines.append(
            "Note: Softened Newtonian N-body + finite timestep ⇒ energy is only approximately conserved."
        )
    else:
        lines.append("No diagnostics samples recorded.")
    lines.append("")
    lines.append("=== Important Events ===")
    if not universe.events:
        lines.append("(none)")
    else:
        for ev in universe.events[-50:]:
            lines.append(f"[{fmt_time_days(ev.sim_time)}] ({ev.category}) {ev.message}")
    lines.append("")
    lines.append("=== Orbital Notes ===")
    for b in universe.live_bodies():
        parent = universe.get_body(b.parent_id) if b.parent_id else None
        if parent is None:
            continue
        orb = analyze_orbit(b, parent, G_const=universe.G)
        lines.append(
            f"{b.name} about {parent.name}: bound={orb.bound}, "
            f"e={orb.eccentricity}, T={orb.period}, a={orb.semi_major_axis}"
        )
    lines.append("")
    lines.append(
        "Disclaimer: Educational Newtonian simulator. Not scientifically exact for "
        "relativistic, tidal, or radiative regimes."
    )
    path.write_text("\n".join(lines))
    return path
