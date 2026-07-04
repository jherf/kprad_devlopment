#!/usr/bin/env python
# test_pellet.py
"""
Minimal, self-contained test/demo for the Pellet injector.

Run it interactively:

    ipython -i test_pellet.py       # drops you into a shell with `p`, `hist`, ...

or as a script:

    python test_pellet.py

It does NOT need the ODE solver, the atomic tables, or a config file. The
plasma is a fixed dict {Te, ne}; the pellet's own inventory is advanced with a
tiny hand-rolled Euler loop that plays the role the real solver's aux block
plays in a full run. This isolates the Pellet physics:

  * particle bookkeeping   (Torr-L -> 1e20 molecules, D2 -> 2 H atoms)
  * geometry              (radius from inventory, solid-density mixing)
  * instantaneous ablation (Parks NGS rate, scalings)
  * total ablated amount   (conservation: what leaves the pellet = what is
                            delivered to the plasma)

Everything prints as it goes; assertions guard the numbers.
"""

import numpy as np

from main.util.injectors import Pellet
from main.util.injectors import (
    _W_MOL,
    _N_AVOGADRO,
    _PARKS_LAMBDA_A,
    _PARKS_LAMBDA_B,
)
from main.util.constants import _RHO_SOLID

from main.util.injectors import Pellet
from main.util.injectors import (
    _W_MOL,
    _N_AVOGADRO,
    _PARKS_LAMBDA_A,
    _PARKS_LAMBDA_B,
)
from main.util.constants import _RHO_SOLID
