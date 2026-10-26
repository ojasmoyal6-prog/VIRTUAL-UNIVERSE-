"""Physics documentation strings shown in the About / Physics panel."""

PHYSICS_ABOUT = """
VIRTUAL UNIVERSE SIMULATOR — PHYSICS NOTES
==========================================

Newton's law of universal gravitation
-------------------------------------
For two point masses:

    F = G · (m₁ m₂) / r²

Direction: along the line joining the masses (attractive).

Acceleration of body i:

    aᵢ = Σⱼ≠ᵢ G · mⱼ · (rⱼ − rᵢ) / (|rⱼ − rᵢ|² + ε²)^{3/2}

Softening length ε is a numerical regularisation that prevents singular
forces during close encounters. It is an approximation, not a physical force law.


Numerical integration
---------------------
Default method: Velocity Verlet (symplectic):

    xₙ₊₁ = xₙ + vₙ Δt + ½ aₙ Δt²
    aₙ₊₁ = a(xₙ₊₁)
    vₙ₊₁ = vₙ + ½ (aₙ + aₙ₊₁) Δt

Alternatives: leapfrog (kick–drift–kick) and semi-implicit Euler.


Conservation laws
-----------------
In the exact continuous Newtonian N-body problem (no softening, infinite precision):
  • Total momentum is conserved
  • Total angular momentum is conserved
  • Total mechanical energy E = K + U is conserved

In this simulator:
  • Finite Δt, softening, and collisions break exact conservation
  • The Energy monitor shows how well the integrator preserves E
  • Large energy drift ⇒ reduce timestep or avoid extreme close encounters


Orbital mechanics (two-body estimates)
--------------------------------------
Specific orbital energy:

    ε = |v_rel|² / 2 − μ / r     where μ = G(M + m)

Bound ellipse if ε < 0. Semi-major axis a = −μ / (2ε).
Period T = 2π √(a³ / μ). Eccentricity from the eccentricity vector.


Barnes–Hut (large N)
--------------------
When N is large, distant groups of bodies may be approximated as a single
monopole if (box size) / (distance) < θ. This is an O(N log N) approximation.


Limitations (important)
-----------------------
  • Newtonian gravity only — no general relativity
  • No tides, oblateness, radiation pressure, or gas dynamics
  • "Black hole" scenarios are compact Newtonian masses
  • Display sizes are exaggerated for visibility
  • Long integrations can become unstable (chaos + numerical error)

This is an educational computational-physics tool, not a mission-grade ephemeris.
"""


EQUATIONS_SHORT = [
    "F = G m₁ m₂ / r²",
    "a = F / m",
    "Velocity Verlet: x ← x + vΔt + ½aΔt²; v ← v + ½(a + a')Δt",
    "v_esc = √(2GM / r)",
    "v_circ = √(GM / r)",
    "E = Σ ½mv² − Σ_{i<j} G mᵢ mⱼ / r_{ij}",
]
