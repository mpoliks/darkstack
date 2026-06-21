"""Run measurement tracks against a substrate and collect a Report.

A track runs only if the substrate provides the capability it needs, so the same
call works against the full mock and against a partial live adapter.
"""
from __future__ import annotations

from typing import Optional

from .report import Report
from .tracks import TRACKS


def run(substrate, tracks: Optional[list] = None, config: Optional[dict] = None) -> Report:
    """Run the named tracks (default: all the substrate supports). `config` maps a
    track name to a kwargs dict forwarded to its measure()."""
    config = config or {}
    names = tracks or list(TRACKS)
    rep = Report()
    for name in names:
        if name not in TRACKS:
            raise KeyError(f"unknown track {name!r}; known: {sorted(TRACKS)}")
        fn, capability = TRACKS[name]
        if not substrate.supports(capability):
            continue
        rep.add(fn(substrate, **config.get(name, {})))
    return rep


def runnable_tracks(substrate) -> list:
    return [n for n, (_, cap) in TRACKS.items() if substrate.supports(cap)]
