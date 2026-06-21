"""factory_design -- a simulation engine for agent-factory design.

You describe a factory you are about to build (a `FactoryDesign`), simulate it,
and read back a verdict: will it converge, will it game its own evals, will it
stall, will it thrash, and -- if it shares infrastructure with other factories --
will it crash in sync with them. When the verdict is a failure mode, the engine
names the one boundary lever that fixes it and re-simulates to show the fix lands.

The simulation runs on the validated dynamical organs in `src/` (a finite
population of learning agents, a transfer operator, early-warning signals, a
Kuramoto coupling, a PID controller). Those organs are exact; this package is the
design-facing surface over them. Nothing here measures a real factory -- for that,
build the design and point `factory_probe` at the running system.

Public surface
--------------
    from factory_design import FactoryDesign, simulate
    report = simulate(FactoryDesign(explore_rate=0.001))
    print(report)                      # headline verdict + lenses, with limits

    from factory_design import diagnose, recommend, verify
    for cond in diagnose(report):
        print(recommend(design, cond)) # the lever, as a diff against your design
        print(verify(design, cond))    # before -> after, re-simulated
"""
from __future__ import annotations

import os
import sys

# The dynamical organs live in src/ and are placed on the path the same way
# factory_probe does it, so an editable install keeps them resolvable.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from .design import FactoryDesign, KNOBS  # noqa: E402
from .report import FactoryReport, Lens  # noqa: E402
from .simulate import simulate  # noqa: E402
from .diagnose import diagnose, Condition  # noqa: E402
from .levers import recommend, Lever, LEVERS  # noqa: E402
from .verify import verify  # noqa: E402

__all__ = [
    "FactoryDesign", "KNOBS",
    "FactoryReport", "Lens",
    "simulate",
    "diagnose", "Condition",
    "recommend", "Lever", "LEVERS",
    "verify",
]
