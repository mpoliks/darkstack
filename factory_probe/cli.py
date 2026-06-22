"""Command line: run the measurements against a substrate and print the report.

  python -m factory_probe run                 # all tracks on the reference substrate
  python -m factory_probe run --track opacity --track versioning
  python -m factory_probe run --json out.json # also write the report as JSON
  python -m factory_probe tracks              # list tracks and required capabilities
"""
from __future__ import annotations

import argparse
import json
import sys

from .tracks import TRACKS
from .experiment import run, runnable_tracks


def _make_substrate(name: str):
    if name == "mock":
        from .mock import MockSubstrate
        return MockSubstrate()
    if name == "tabular":
        from .substrates.tabular import TabularSubstrate
        return TabularSubstrate()
    raise SystemExit(f"unknown substrate {name!r} (built-in: mock, tabular). Point at a live "
                     f"substrate by importing your adapter and calling factory_probe.experiment.run.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="factory_probe",
        description="Measure structural properties of a running software factory "
                    "(the runtime counterpart to factory-design, which simulates the "
                    "same properties at design time).")
    sp = ap.add_subparsers(dest="cmd", required=True)

    rp = sp.add_parser("run", help="run measurement tracks")
    rp.add_argument("--substrate", default="mock", choices=["mock", "tabular"],
                    help="which built-in substrate to measure; default mock. To measure a "
                         "live factory, use an adapter from Python (see factory_probe/README.md).")
    rp.add_argument("--track", action="append", dest="tracks",
                    help="run only this track (repeatable)")
    rp.add_argument("--json", help="write the full report to this JSON path")

    sp.add_parser("tracks", help="list tracks and required capabilities")

    gp = sp.add_parser("steer", help="recommend governance interventions per diagnosed condition")
    gp.add_argument("--verify", action="store_true",
                    help="close the loop: apply each lever on the reference substrate and re-measure")

    args = ap.parse_args(argv)

    if args.cmd == "tracks":
        for name, (_, cap) in TRACKS.items():
            print(f"  {name:12s} requires capability: {cap}")
        return 0

    if args.cmd == "steer":
        from .steering import RECOMMENDATIONS, run_all
        print("recommended interventions by diagnosed condition:\n")
        for cond, iv in RECOMMENDATIONS.items():
            print(f"  {cond:16s} [{iv.lever:16s}] {iv.action} {iv.knob} -- {iv.rationale}")
        if args.verify:
            print("\nclosed-loop verification on the reference substrate:")
            unresolved = 0
            for r in run_all():
                tag = "resolved" if r["resolved"] else "NOT resolved"
                unresolved += not r["resolved"]
                print(f"  {r['condition']:16s} {r['metric']}: {r['before']} -> {r['after']}  [{tag}]")
            if unresolved:
                print(f"\n{unresolved} condition(s) did not resolve")
                return 1
        return 0

    substrate = _make_substrate(args.substrate)
    runnable = runnable_tracks(substrate)
    chosen = args.tracks or runnable
    skipped = [t for t in chosen if t in TRACKS and TRACKS[t][1] not in substrate.capabilities()]
    if skipped:
        print(f"(substrate '{substrate.name}' lacks capabilities for: {skipped})\n", file=sys.stderr)

    rep = run(substrate, tracks=args.tracks)
    print(rep.table())
    print()
    print(rep.falsification_table())
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep.to_dict(), fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
