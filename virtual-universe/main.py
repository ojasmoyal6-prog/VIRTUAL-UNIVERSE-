#!/usr/bin/env python3
"""Virtual Universe Simulator — entry point.

Educational Newtonian N-body simulator with interactive 2D/3D visualisation.
Positions and orbits emerge from numerical integration — they are never keyframed.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path when launched as a script
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.app import main


if __name__ == "__main__":
    main()
