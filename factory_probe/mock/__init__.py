"""Reference substrate with controllable dynamics and known ground truth.

Every capability is implemented on top of validated dynamics, so the measurement
tracks can be checked against a planted answer before they are pointed at a live
factory.
"""
from __future__ import annotations

from ..interfaces import Substrate
from .population import MockPopulation, REGIMES
from .dss_game import MockDSSGame
from .coupled import MockCoupledLoops
from .dividend_task import MockDividendTask
from .evaluator import planted_evaluator


class MockSubstrate(Substrate):
    name = "mock"

    def capabilities(self) -> set:
        return {"steppable", "learner_game", "dividend_task", "coupled_loops"}

    def default_dividend_specs(self):
        return [dict(d=12, order=o, kind="nk") for o in (1, 2, 3, 4)]

    def steppable(self, **config) -> MockPopulation:
        return MockPopulation().reset(**config)

    def learner_game(self, **config) -> MockDSSGame:
        return MockDSSGame()

    def dividend_task(self, **config) -> MockDividendTask:
        return MockDividendTask(**config)

    def coupled_loops(self, **config) -> MockCoupledLoops:
        return MockCoupledLoops(**config)


__all__ = ["MockSubstrate", "MockPopulation", "MockDSSGame", "MockCoupledLoops",
           "MockDividendTask", "planted_evaluator", "REGIMES"]
