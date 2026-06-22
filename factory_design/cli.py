"""Command line for the factory-design engine.

    factory-design demo                 a recognisable failure, then the verified fix
    factory-design sim <preset>         simulate a preset (or the default healthy design)
    factory-design fix <preset>         diagnose and verify the fix for every condition found
    factory-design sweep <preset> <knob> <v1,v2,...>   find the safe operating region
    factory-design compare <a> <b>      two designs, side by side
    factory-design reference            the steering result (why a factory needs to explore)
    factory-design presets              list the named designs
    factory-design knobs                list every design knob and what it stands for
    factory-design selftest             the falsification table (classifier + levers)

Override any knob with --set, e.g.  factory-design sim healthy --set explore_rate=0.0005
"""
from __future__ import annotations

import argparse
import sys

from .design import FactoryDesign, KNOBS
from .presets import PRESETS, get
from .simulate import simulate
from .diagnose import diagnose
from .verify import verify
from .levers import recommend
from .sweep import sweep, compare
from . import reference


def _coerce(s: str):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _build_design(name: str | None, sets: list[str]) -> FactoryDesign:
    try:
        design = get(name) if name else FactoryDesign()
    except KeyError as e:
        raise SystemExit(str(e))
    changes = {}
    for kv in sets or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects knob=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in KNOBS:
            raise SystemExit(f"unknown knob {k!r}; see `factory-design knobs`")
        changes[k] = _coerce(v)
    try:
        return design.with_(**changes) if changes else design
    except (ValueError, KeyError) as e:           # out-of-range or unknown knob
        raise SystemExit(f"invalid design: {e}")


# --- subcommands ------------------------------------------------------------
def cmd_sim(a):
    design = _build_design(a.preset, a.set)
    print(simulate(design, seeds=a.seeds, quick=a.quick, early_warning=not a.quick))


def cmd_fix(a):
    design = _build_design(a.preset, a.set)
    rep = simulate(design, seeds=a.seeds, quick=a.quick, early_warning=False)
    print(f"VERDICT: {rep.verdict}\n")
    conds = diagnose(rep)
    if not conds:
        print("no conditions to fix -- the design is healthy.")
        return
    for c in conds:
        print(f"- {c}")
        rec = recommend(design, c.name)
        print(f"    {rec}")
        v = verify(design, c.name, seeds=a.seeds, quick=a.quick)
        print(f"    verified: {v.metric} {v.before:.2f} -> {v.after:.2f}  "
              f"[{'RESOLVED' if v.resolved else 'NOT RESOLVED'}]\n")


def cmd_sweep(a):
    design = _build_design(a.preset, a.set)
    values = [_coerce(v) for v in a.values.split(",")]
    print(sweep(design, a.knob, values, seeds=a.seeds, quick=True))


def cmd_compare(a):
    print(compare(_build_design(a.a, a.set), _build_design(a.b, []), seeds=a.seeds, quick=True))


def cmd_reference(a):
    print(reference.summary(seeds=a.seeds))


def cmd_presets(a):
    from .presets import EXPECTED
    print("named designs:")
    for name in PRESETS:
        print(f"  {name:22s} -> {EXPECTED.get(name, '?')}")


def cmd_knobs(a):
    print("design knobs (knob: what it does | maps to):")
    for k, (gloss, maps_to) in KNOBS.items():
        print(f"  {k}\n      {gloss}\n      maps to: {maps_to}")
    print("\nnot modeled (set these directly in your harness):")
    print("  retry / escalation depth -- retries before human, fallback to a stronger model")
    print("  tool-call success rate, context-window budget, rate limits")


def cmd_selftest(a):
    from .selftest import run as run_selftest
    ok = run_selftest(quick=a.quick)
    sys.exit(0 if ok else 1)


def cmd_demo(a):
    design = get("gameable_judge")
    print("A factory you might build: a fast pipeline scored by a cheap, gameable judge.\n")
    rep = simulate(design, seeds=a.seeds, quick=a.quick, early_warning=False)
    print(rep)
    print()
    conds = diagnose(rep)
    for c in conds:
        rec = recommend(design, c.name)
        v = verify(design, c.name, seeds=a.seeds, quick=a.quick)
        print(f"FIX for {c.name}: set {rec.knob} {rec.before!r} -> {rec.after!r}")
        print(f"  {rec.rationale}")
        print(f"  verified by re-simulation: {v.metric} {v.before:.2f} -> {v.after:.2f}  "
              f"[{'RESOLVED' if v.resolved else 'NOT RESOLVED'}]")


def build_parser() -> argparse.ArgumentParser:
    # shared options, accepted either before or after the subcommand. SUPPRESS
    # defaults so a value given before the subcommand is not clobbered by the
    # subparser's default; the real defaults are applied in main().
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seeds", type=int, default=argparse.SUPPRESS,
                        help="ensemble size (median over this many runs); default 12")
    common.add_argument("--quick", action="store_true", default=argparse.SUPPRESS,
                        help="shorter runs / fewer seeds")

    p = argparse.ArgumentParser(prog="factory-design", parents=[common],
                                description="Simulate a software-factory design before you build it.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("demo", parents=[common], help="a recognisable failure, then the verified fix")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("sim", parents=[common], help="simulate a design")
    sp.add_argument("preset", nargs="?", help="a named preset (default: healthy)")
    sp.add_argument("--set", action="append", help="override a knob: knob=value")
    sp.set_defaults(func=cmd_sim)

    sp = sub.add_parser("fix", parents=[common], help="diagnose and verify the fix for each condition")
    sp.add_argument("preset", nargs="?")
    sp.add_argument("--set", action="append")
    sp.set_defaults(func=cmd_fix)

    sp = sub.add_parser("sweep", parents=[common], help="sweep one knob, find the safe operating region")
    sp.add_argument("preset", nargs="?")
    sp.add_argument("knob")
    sp.add_argument("values", help="comma-separated, e.g. 0.001,0.01,0.06,0.2")
    sp.add_argument("--set", action="append")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("compare", parents=[common], help="compare two designs")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.add_argument("--set", action="append")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("reference", parents=[common],
                        help="the steering result (why a factory needs to explore)")
    sp.set_defaults(func=cmd_reference)

    sp = sub.add_parser("presets", parents=[common], help="list named designs")
    sp.set_defaults(func=cmd_presets)

    sp = sub.add_parser("knobs", parents=[common], help="list design knobs and what they stand for")
    sp.set_defaults(func=cmd_knobs)

    sp = sub.add_parser("selftest", parents=[common], help="run the falsification table")
    sp.set_defaults(func=cmd_selftest)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # resolve the SUPPRESS-default shared options (work before or after the subcommand)
    args.seeds = getattr(args, "seeds", 12)
    args.quick = getattr(args, "quick", False)
    try:
        args.func(args)
    except (ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
