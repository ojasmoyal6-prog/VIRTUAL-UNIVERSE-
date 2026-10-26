"""Newtonian gravity force and acceleration calculations.

Primary law (pairwise):

    F_ij = G * m_i * m_j / r_ij²   ·   r̂_ij

With Plummer softening ε (numerical regularisation):

    F_ij = G * m_i * m_j / (r_ij² + ε²)^{3/2}   ·   r⃗_ij

Acceleration on body i:

    a_i = Σ_{j≠i} G * m_j * r⃗_ij / (r_ij² + ε²)^{3/2}

All operations are NumPy-vectorised for N-body performance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .constants import G, DEFAULT_SOFTENING


def pairwise_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    G_const: float = G,
    softening: float = DEFAULT_SOFTENING,
    alive_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute gravitational accelerations for all bodies.

    Parameters
    ----------
    positions : (N, 3) array
    masses : (N,) array
    softening : Plummer ε in metres
    alive_mask : optional boolean mask; dead bodies neither feel nor exert force

    Returns
    -------
    accelerations : (N, 3) array
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if alive_mask is None:
        alive_mask = np.ones(n, dtype=bool)

    # Displacement tensors: r[i,j] = r_j - r_i  → force on i pulls toward j
    # We want a_i ← Σ_j G m_j (r_j - r_i) / |r|³_soft
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N,N,3)
    dist2 = np.sum(diff * diff, axis=2)  # (N,N)
    soft2 = softening * softening
    # Avoid 0^{-1.5} on the diagonal when softening is zero (self pairs).
    np.fill_diagonal(dist2, np.inf)
    inv_r3 = (dist2 + soft2) ** (-1.5)
    inv_r3[~np.isfinite(inv_r3)] = 0.0

    # Zero self-interaction and dead contributors
    np.fill_diagonal(inv_r3, 0.0)
    if not np.all(alive_mask):
        inv_r3[~alive_mask, :] = 0.0
        inv_r3[:, ~alive_mask] = 0.0

    # a_i = Σ_j G m_j * diff_ij * inv_r3_ij
    weighted = inv_r3 * masses[np.newaxis, :]  # (N,N)
    acc = G_const * np.einsum("ij,ijk->ik", weighted, diff)
    acc[~alive_mask] = 0.0
    return acc


def gravitational_force(
    m1: float,
    m2: float,
    r_vec: np.ndarray,
    *,
    G_const: float = G,
    softening: float = DEFAULT_SOFTENING,
) -> np.ndarray:
    """Force on mass-1 due to mass-2 (vector pointing toward mass-2)."""
    r2 = float(np.dot(r_vec, r_vec))
    soft2 = softening * softening
    factor = G_const * m1 * m2 / ((r2 + soft2) ** 1.5)
    return factor * r_vec


def potential_energy_total(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    G_const: float = G,
    softening: float = DEFAULT_SOFTENING,
    alive_mask: Optional[np.ndarray] = None,
) -> float:
    """Total gravitational potential energy (softened).

    U = -½ Σ_i Σ_{j≠i} G m_i m_j / sqrt(r_ij² + ε²)
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0
    if alive_mask is None:
        alive_mask = np.ones(n, dtype=bool)

    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + softening * softening)
    # Upper triangle only to avoid double count; then no 1/2 needed if we sum i<j
    u = 0.0
    for i in range(n):
        if not alive_mask[i]:
            continue
        for j in range(i + 1, n):
            if not alive_mask[j]:
                continue
            u -= G_const * masses[i] * masses[j] / dist[i, j]
    return float(u)


def kinetic_energy_total(
    velocities: np.ndarray,
    masses: np.ndarray,
    alive_mask: Optional[np.ndarray] = None,
) -> float:
    if alive_mask is None:
        alive_mask = np.ones(len(masses), dtype=bool)
    ke = 0.5 * masses * np.sum(velocities * velocities, axis=1)
    return float(np.sum(ke[alive_mask]))


def total_momentum(
    velocities: np.ndarray,
    masses: np.ndarray,
    alive_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if alive_mask is None:
        alive_mask = np.ones(len(masses), dtype=bool)
    p = (masses[:, np.newaxis] * velocities)
    return np.sum(p[alive_mask], axis=0)


def total_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    alive_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """L = Σ m (r × v) about the origin."""
    if alive_mask is None:
        alive_mask = np.ones(len(masses), dtype=bool)
    L = np.cross(positions, velocities) * masses[:, np.newaxis]
    return np.sum(L[alive_mask], axis=0)


def barnes_hut_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    theta: float = 0.5,
    G_const: float = G,
    softening: float = DEFAULT_SOFTENING,
    alive_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optional O(N log N) approximation via a simple Barnes–Hut octree.

    Used automatically when N is large. Documented approximation:
    a distant cluster is treated as a single monopole if s/d < θ.
    """
    # For moderate N the direct method is faster and exact (within softening).
    # This implementation builds a lightweight tree for educational transparency.
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    if alive_mask is None:
        alive_mask = np.ones(n, dtype=bool)

    # Fall back to direct method below a practical threshold — tree overhead dominates.
    if n < 64:
        return pairwise_accelerations(
            positions, masses, G_const=G_const, softening=softening, alive_mask=alive_mask
        )

    return _barnes_hut_impl(
        positions, masses, theta, G_const, softening, alive_mask
    )


class _Node:
    __slots__ = ("center", "half", "mass", "com", "body_idx", "children")

    def __init__(self, center: np.ndarray, half: float):
        self.center = center
        self.half = half
        self.mass = 0.0
        self.com = np.zeros(3)
        self.body_idx: Optional[int] = None
        self.children: Optional[list] = None


def _barnes_hut_impl(
    positions: np.ndarray,
    masses: np.ndarray,
    theta: float,
    G_const: float,
    softening: float,
    alive_mask: np.ndarray,
) -> np.ndarray:
    alive_idx = np.where(alive_mask)[0]
    if len(alive_idx) == 0:
        return np.zeros_like(positions)

    pos = positions[alive_idx]
    mas = masses[alive_idx]
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo)) + 1.0
    root = _Node(center, max(half, 1.0))

    for local_i, _ in enumerate(alive_idx):
        _insert(root, local_i, pos, mas)

    acc_alive = np.zeros_like(pos)
    soft2 = softening * softening
    for local_i in range(len(alive_idx)):
        acc_alive[local_i] = _accel_from_node(
            root, local_i, pos, mas, theta, G_const, soft2
        )

    acc = np.zeros_like(positions)
    acc[alive_idx] = acc_alive
    return acc


def _octant(node: _Node, p: np.ndarray) -> int:
    o = 0
    if p[0] >= node.center[0]:
        o |= 1
    if p[1] >= node.center[1]:
        o |= 2
    if p[2] >= node.center[2]:
        o |= 4
    return o


def _child_center(node: _Node, octant: int) -> np.ndarray:
    offset = node.half * 0.5
    c = node.center.copy()
    c[0] += offset if (octant & 1) else -offset
    c[1] += offset if (octant & 2) else -offset
    c[2] += offset if (octant & 4) else -offset
    return c


def _insert(node: _Node, idx: int, pos: np.ndarray, mas: np.ndarray) -> None:
    if node.mass == 0.0 and node.body_idx is None and node.children is None:
        node.body_idx = idx
        node.mass = float(mas[idx])
        node.com = pos[idx].copy()
        return

    if node.children is None:
        # Subdivide existing leaf
        existing = node.body_idx
        node.children = [None] * 8
        node.body_idx = None
        if existing is not None:
            o = _octant(node, pos[existing])
            child = _Node(_child_center(node, o), node.half * 0.5)
            node.children[o] = child
            _insert(child, existing, pos, mas)

    o = _octant(node, pos[idx])
    if node.children[o] is None:
        node.children[o] = _Node(_child_center(node, o), node.half * 0.5)
    _insert(node.children[o], idx, pos, mas)

    # Update multipole (monopole)
    total_mass = node.mass + float(mas[idx])
    if total_mass > 0:
        node.com = (node.com * node.mass + pos[idx] * float(mas[idx])) / total_mass
    node.mass = total_mass


def _accel_from_node(
    node: _Node,
    idx: int,
    pos: np.ndarray,
    mas: np.ndarray,
    theta: float,
    G_const: float,
    soft2: float,
) -> np.ndarray:
    if node.mass == 0.0:
        return np.zeros(3)
    if node.children is None:
        if node.body_idx is None or node.body_idx == idx:
            return np.zeros(3)
        r = node.com - pos[idx]
        r2 = float(np.dot(r, r))
        return G_const * node.mass * r / ((r2 + soft2) ** 1.5)

    r = node.com - pos[idx]
    dist = float(np.linalg.norm(r)) + 1e-30
    size = node.half * 2.0
    if size / dist < theta:
        r2 = float(np.dot(r, r))
        return G_const * node.mass * r / ((r2 + soft2) ** 1.5)

    acc = np.zeros(3)
    for child in node.children:  # type: ignore[union-attr]
        if child is not None:
            acc += _accel_from_node(child, idx, pos, mas, theta, G_const, soft2)
    return acc
