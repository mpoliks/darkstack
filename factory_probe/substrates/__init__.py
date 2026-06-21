"""Substrates beyond the reference mock.

`tabular` is a real-data substrate for the opacity track: the legibility floor is
measured on actual datasets with held-out scoring, off the enumerable cube.
"""
from .tabular import TabularSubstrate, TabularDividendTask

__all__ = ["TabularSubstrate", "TabularDividendTask"]
