"""factory_probe -- measure structural properties of a running software factory.

A factory here is a population of producer agents that emit actions each round,
are scored by a verifier through a (possibly delayed) reward channel, and are
steered by a priced governance loop. `factory_probe` instruments such a factory
and measures six properties from its behaviour alone:

  versioning    near-invariant regions of the behavioural distribution and their
                robustness (transfer-operator spectrum / timescale separation)
  pathology     stable-failure / overfitting / learning-death / thrash fingerprints
  catastrophe   early-warning signals (variance, lag-1 autocorrelation) before a fold
  governance    cascade-ratio stability of a governance loop over the loops it steers
  entrainment   synchronisation of coupled loops and the coupling threshold
  opacity floor the verified value a bounded, legible reader forgoes vs free search

The measurements run against any object that implements the capability interfaces
in `factory_probe.interfaces`. A reference `MockSubstrate` (`factory_probe.mock`)
implements every capability on top of controllable dynamics with known ground
truth, so the whole package runs end-to-end and each measurement can be checked
against a planted answer. An adapter for a live agent-to-agent substrate
implements the same interfaces against real traces.
"""
from __future__ import annotations
import os as _os
import sys as _sys

_SRC = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "src"))
if not _os.path.isdir(_SRC):
    raise ImportError(
        "factory_probe needs the measurement modules in ../src. Install editable "
        "from the repository root (pip install -e .) or run from the repo root.")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

from .records import Decision, Reward, RoundObs                       # noqa: E402
from .instrumentation import TraceStore, ReturnQueue                  # noqa: E402
from .scorer import ConsequenceScorer                                 # noqa: E402
from .interfaces import (Substrate, SteppableFactory, LearnerGame,    # noqa: E402
                         DividendTask, CoupledLoops)

__all__ = ["Decision", "Reward", "RoundObs", "TraceStore", "ReturnQueue",
           "ConsequenceScorer", "Substrate", "SteppableFactory", "LearnerGame",
           "DividendTask", "CoupledLoops"]
