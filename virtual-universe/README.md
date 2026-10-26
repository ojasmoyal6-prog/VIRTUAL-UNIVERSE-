# Virtual Universe Simulator 🌌

Educational Newtonian **N-body** simulator for building miniature universes, exploring gravity and orbital mechanics, and watching real numerical physics unfold interactively.

Positions are **never keyframed**. Every star, planet, moon, and asteroid moves because Newton's law of gravitation is integrated forward in time.

---

## Overview

Virtual Universe Simulator is a Python computational-physics application that combines:

- a NumPy-accelerated gravity engine (Velocity Verlet / leapfrog)
- interactive 2D + 3D visualisation (pygame-ce)
- a scientific dashboard (object inspector, energy graphs, event log)
- presets, an experiment lab, and a random-universe generator

It is designed to feel like a mix of a NASA-style visualisation, a physics laboratory, and a playful sandbox — while remaining transparent about approximations and limitations.

---

## Features

- **Real physics engine** — Newtonian gravity with Plummer softening; Velocity Verlet by default
- **Celestial bodies** — stars, planets, moons, asteroids, dwarf planets, custom bodies, Newtonian “black hole” demos
- **Interactive view** — zoom, pan, 3D rotate, select, focus/follow, labels, trails, velocity & gravity vectors, grid, starfield
- **2D + 3D modes** — top-down orthographic and rotatable 3D projection
- **Solar System preset** — Sun + 8 planets + Moon, with reset
- **Universe Builder** — add / delete / duplicate / edit mass, radius, position, velocity, type
- **Experiment Lab** — velocity tweaks, star-mass changes, binary stars, three-body chaos, collisions, escape velocity
- **Orbital analysis** — period, eccentricity, periapsis/apoapsis, specific energy (two-body estimates)
- **Energy & momentum monitor** — live diagnostics + drift warnings
- **Time control** — play / pause / step / reset / speeds from 0.1× to 1,000,000×
- **Collisions** — merge (momentum-conserving), bounce, destroy smaller, record only
- **Event log** — collisions, resets, user edits, instability warnings
- **10+ scenarios** including asteroid belt, rogue planet, random universe
- **CSV + experiment report export**
- **Learn Mode** — concise explanations of orbits, escape speed, ellipses, COM, three-body chaos
- **About / Physics panel** — equations, integrator notes, limitations

---

## Physics implemented

### Law of gravity

\[
\mathbf{F}_{ij} = G \frac{m_i m_j}{r_{ij}^{2}}\,\hat{\mathbf{r}}_{ij}
\]

With numerical softening \(\varepsilon\):

\[
\mathbf{a}_i = \sum_{j\neq i} G\, m_j\, \frac{\mathbf{r}_j - \mathbf{r}_i}{\bigl(|\mathbf{r}_j-\mathbf{r}_i|^{2}+\varepsilon^{2}\bigr)^{3/2}}
\]

### Integration (Velocity Verlet)

\[
\begin{aligned}
\mathbf{x}_{n+1} &= \mathbf{x}_n + \mathbf{v}_n\Delta t + \tfrac12\mathbf{a}_n\Delta t^{2} \\
\mathbf{a}_{n+1} &= \mathbf{a}(\mathbf{x}_{n+1}) \\
\mathbf{v}_{n+1} &= \mathbf{v}_n + \tfrac12(\mathbf{a}_n+\mathbf{a}_{n+1})\Delta t
\end{aligned}
\]

### Escape / circular speeds

\[
v_{\mathrm{esc}}=\sqrt{\frac{2GM}{r}},\qquad v_{\mathrm{circ}}=\sqrt{\frac{GM}{r}}
\]

### Large-N option

For \(N \ge 80\), a **Barnes–Hut** monopole tree (\(\theta\)-criterion) may approximate distant forces. Documented as an approximation in the Physics panel.

---

## Mathematical equations (summary)

| Quantity | Formula |
|----------|---------|
| Force | \(F = G m_1 m_2 / r^2\) |
| Softened accel. | \(a \propto r / (r^2+\varepsilon^2)^{3/2}\) |
| Kinetic energy | \(K = \sum \tfrac12 m\|\mathbf{v}\|^2\) |
| Potential energy | \(U = -\sum_{i<j} G m_i m_j / \sqrt{r_{ij}^2+\varepsilon^2}\) |
| Specific orbital energy | \(\varepsilon = \|\mathbf{v}\|^2/2 - \mu/r\) |
| Period (ellipse) | \(T = 2\pi\sqrt{a^3/\mu}\) |

---

## Screenshots

Launch the app and try **Solar System**, **Three-Body Problem**, and **Binary Star**. The deep-space UI emphasises the universe view, with inspector / builder / graphs on the right.

*(Run locally to capture screenshots for your fork.)*

---

## Demo instructions

1. Install dependencies (see below).
2. Run `python main.py`.
3. Press **Solar Sys** — planets begin orbiting from integrated physics.
4. Click **Earth**, enable **Learn Mode**, try **v × 1.2** and watch the orbit change.
5. Load **Three-Body Problem** — observe sensitive, hard-to-predict motion.
6. Open **Graphs** — watch total energy drift as a numerical-quality indicator.
7. Use **Builder** to invent a custom star–planet system.

---

## Installation

```bash
cd virtual-universe
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** This project uses [`pygame-ce`](https://pyga.me/) (community edition), which provides binary wheels for modern Python versions including 3.14.

---

## Usage

```bash
python main.py
```

### Controls

| Input | Action |
|-------|--------|
| Left drag (empty space) | Pan |
| Scroll | Zoom |
| Right drag (3D mode) | Rotate camera |
| Click body | Select |
| Space | Play / Pause |
| `R` | Reset scenario |
| `F` | Fit view |
| `1` / `2` | 2D / 3D |
| `Tab` | Cycle side panels |
| Esc | Quit |

### Simulation speeds

`0.1×`, `1×`, `10×`, `100×`, `1 000×`, `10 000×`, `1 000 000×`

The header distinguishes **simulation time** (days / years) from **real elapsed time**.

---

## Experiments

| Experiment | What to try |
|------------|-------------|
| Change planet velocity | Select planet → `v × 0.8` / `v × 1.2` → stable / elliptical / escape / crash |
| Change star mass | `Mass × 2` / `× 0.5` on the star |
| Binary star | Load **Binary Star** scenario |
| Three-body | Load **Three-Body Problem** |
| Planet collision | Load **Planetary Collision** (merge mode) |
| Escape velocity | Load **Experiment: Escape Velocity**; edit probe speed vs \(v_{\mathrm{esc}}\) |

Scenario list also includes Asteroid Belt, Rogue Planet, Artificial Solar System, Black-Hole Gravity Experiment (Newtonian only), and Random Universe.

---

## Architecture

```
virtual-universe/
├── main.py                 # Entry point
├── simulation/             # Physics engine
│   ├── bodies.py
│   ├── physics.py          # Forces, Barnes–Hut
│   ├── integrator.py       # Verlet / leapfrog / Euler
│   ├── collisions.py
│   ├── orbital.py
│   ├── diagnostics.py
│   └── universe.py
├── visualization/          # Camera + renderer
├── experiments/            # Presets & scenarios
├── ui/                     # App shell, widgets, docs
├── utils/                  # Export, formatting, learn mode
├── tests/
├── requirements.txt
└── README.md
```

---

## Numerical methods

- **Default:** Velocity Verlet (symplectic, good long-term energy behaviour)
- **Optional:** Leapfrog, semi-implicit Euler
- **Softening:** Plummer \(\varepsilon\) (default \(10^6\) m) avoids \(1/r^2\) blow-ups
- **Collisions:** geometric sphere overlap; merge uses momentum conservation and volume-additive radius
- **Performance:** fully vectorised pairwise forces; Barnes–Hut for larger \(N\)

---

## Known limitations

- Newtonian only — **no general relativity**
- Display radii are exaggerated for visibility
- Softening and finite \(\Delta t\) break exact energy conservation
- “Black hole” demos are compact Newtonian masses (stated explicitly in-app)
- Chaotic systems (three-body) diverge under tiny IC / round-off changes
- Not a mission-grade ephemeris

**This project was built as an educational computational-physics simulator.** It does not claim scientific exactness.

---

## Future improvements

- Adaptive timesteps / individual block timesteps
- GPU / sparse force evaluation for thousands of bodies
- Proper relativistic options (post-Newtonian)
- Saved scenario JSON library with sharing
- Embedded matplotlib window for publication-quality plots

---

## Scientific references

- Newton, *Philosophiæ Naturalis Principia Mathematica* (1687) — inverse-square gravity
- Verlet, L. (1967). Computer experiments on classical fluids. *Phys. Rev.*
- Barnes, J. & Hut, P. (1986). A hierarchical \(O(N\log N)\) force-calculation algorithm. *Nature*
- Murray & Dermott, *Solar System Dynamics* — two-body orbital elements
- Chenciner & Montgomery — figure-eight three-body choreography (inspiration for the three-body preset)

---

## Example experiments

1. **Solar System at 10 000×** — watch inner planets lap outer ones from pure integration.  
2. **Boost Earth’s speed 20%** — eccentricity grows; Learn Mode explains the ellipse.  
3. **Double the Sun’s mass** — orbital periods shorten (\(\propto 1/\sqrt{M}\)).  
4. **Three-body** — reset repeatedly; tiny numeric differences yield divergent paths.  
5. **Escape probe** — raise velocity past \(v_{\mathrm{esc}}\) and confirm unbound energy.

---

## Testing

```bash
source .venv/bin/activate
pytest -q
```

Coverage includes force scaling, circular-orbit stability, energy drift bounds, collisions / momentum, solar-system init, and orbital elements.

---

## License

MIT — see [LICENSE](LICENSE).
