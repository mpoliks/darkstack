"""The falsification table: claims the engine makes about itself, each checkable.

Two batteries:
  - the classifier round-trip: every pathology preset must classify as its own
    failure mode (the confusion matrix must be diagonal);
  - the lever battery: every recommended lever must resolve its condition when
    re-simulated.

Printed as a pass/fail table. A red row is a finding, not a crash -- it tells you
the engine's own claim did not hold on the reference dynamics.
"""
from __future__ import annotations

from .simulate import _run, simulate
from . import pathology
from .presets import PRESETS, EXPECTED, PATHOLOGY_PRESETS
from .verify import verify

# condition -> the preset that exhibits it
_LEVER_CASES = [
    ("no_exploration_floor", "learning_death"),
    ("gameable_judge", "overfitting"),
    ("stuck_failing", "stable_failure"),
    ("never_settles", "thrash"),
    ("repricing_too_fast", "iatrogenic_thrash"),
    ("monoculture", "correlated_crash"),
]


def _classifier_battery(seeds: int, n: int, tail: int) -> list[tuple[str, bool, str]]:
    rows = []
    for name in PATHOLOGY_PRESETS:
        fps = [pathology.fingerprint(r["mean_pos"], r["variety"], r["norm_sat"], r["metric_sat"], tail)
               for r in (_run(PRESETS[name], s, n) for s in range(seeds))]
        labels = [pathology.classify(fp).label for fp in fps]
        want = EXPECTED[name]
        hits = sum(1 for lab in labels if lab == want)
        ok = hits == len(labels)
        rows.append((f"classify {name} -> {want}", ok, f"{hits}/{len(labels)} seeds"))
    return rows


def _lever_battery(seeds: int, quick: bool) -> list[tuple[str, bool, str]]:
    rows = []
    for preset, cond in _LEVER_CASES:
        v = verify(PRESETS[preset], cond, seeds=seeds, quick=quick)
        rows.append((f"lever for {cond} on {preset}", v.resolved,
                     f"{v.metric} {v.before:.2f}->{v.after:.2f}"))
    return rows


def run(quick: bool = False) -> bool:
    seeds = 6 if quick else 10
    n = 8000 if quick else 15000
    tail = 2500 if quick else 4000
    rows = _classifier_battery(seeds, n, tail) + _lever_battery(6 if quick else 8, quick)
    width = max(len(r[0]) for r in rows)
    print("FALSIFICATION TABLE (each row is a claim the engine makes about itself)")
    print("-" * (width + 22))
    npass = 0
    for label, ok, detail in rows:
        mark = "PASS" if ok else "FAIL"
        npass += ok
        print(f"  {label:<{width}}  {mark}  {detail}")
    print("-" * (width + 22))
    print(f"  {npass}/{len(rows)} claims hold")
    return npass == len(rows)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
