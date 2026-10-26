"""Physical constants and unit conventions used by the simulator.

Units (SI throughout unless noted):
    mass     — kilograms (kg)
    length   — metres (m)
    time     — seconds (s)
    velocity — metres per second (m/s)

Astronomical convenience scales are defined for UI display only.
"""

# Newton's gravitational constant (CODATA)
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Softening length ε prevents 1/r² singularities when bodies nearly coincide.
# Forces use F ∝ 1 / (r² + ε²). Documented approximation — not physical.
DEFAULT_SOFTENING = 1.0e6  # m

# Solar-system reference values (approximate)
M_SUN = 1.98847e30  # kg
R_SUN = 6.957e8  # m

M_EARTH = 5.9722e24  # kg
R_EARTH = 6.371e6  # m

M_MOON = 7.342e22  # kg
R_MOON = 1.737e6  # m

AU = 1.495978707e11  # m
DAY = 86400.0  # s
YEAR = 365.25 * DAY  # s

# Display helpers
KM = 1.0e3
SPEED_OF_LIGHT = 2.99792458e8  # m/s — for educational escape/BH context only
