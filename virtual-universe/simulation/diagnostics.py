"""Physics diagnostics: energy, momentum, stability warnings."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from .physics import (
    kinetic_energy_total,
    potential_energy_total,
    total_angular_momentum,
    total_momentum,
)


@dataclass
class DiagnosticsSample:
    time: float
    kinetic: float
    potential: float
    total_energy: float
    momentum_mag: float
    angular_momentum_mag: float


@dataclass
class Diagnostics:
    history: Deque[DiagnosticsSample] = field(default_factory=lambda: deque(maxlen=2000))
    initial_energy: Optional[float] = None
    energy_drift_warning: bool = False
    warning_message: str = ""

    def clear(self) -> None:
        self.history.clear()
        self.initial_energy = None
        self.energy_drift_warning = False
        self.warning_message = ""

    def update(
        self,
        time: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        alive_mask: np.ndarray,
        *,
        G_const: float,
        softening: float,
        warn_threshold: float = 0.05,
    ) -> DiagnosticsSample:
        ke = kinetic_energy_total(velocities, masses, alive_mask)
        pe = potential_energy_total(
            positions, masses, G_const=G_const, softening=softening, alive_mask=alive_mask
        )
        te = ke + pe
        p = total_momentum(velocities, masses, alive_mask)
        L = total_angular_momentum(positions, velocities, masses, alive_mask)
        sample = DiagnosticsSample(
            time=time,
            kinetic=ke,
            potential=pe,
            total_energy=te,
            momentum_mag=float(np.linalg.norm(p)),
            angular_momentum_mag=float(np.linalg.norm(L)),
        )
        if self.initial_energy is None and abs(te) > 0:
            self.initial_energy = te
        if self.initial_energy is not None and abs(self.initial_energy) > 0:
            drift = abs(te - self.initial_energy) / abs(self.initial_energy)
            if drift > warn_threshold:
                self.energy_drift_warning = True
                self.warning_message = (
                    f"Numerical instability warning: energy drift "
                    f"{drift * 100:.1f}% from initial value. "
                    f"Try a smaller timestep or fewer close encounters."
                )
        self.history.append(sample)
        return sample

    def series(self) -> dict:
        if not self.history:
            return {
                "t": [],
                "ke": [],
                "pe": [],
                "te": [],
                "p": [],
                "L": [],
            }
        return {
            "t": [s.time for s in self.history],
            "ke": [s.kinetic for s in self.history],
            "pe": [s.potential for s in self.history],
            "te": [s.total_energy for s in self.history],
            "p": [s.momentum_mag for s in self.history],
            "L": [s.angular_momentum_mag for s in self.history],
        }
