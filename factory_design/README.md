# factory_design

Describe a software factory you intend to build, simulate it, and get a verdict for how it
will fail. The engine runs on the exact dynamical models in [../src](../src) and reports a
verdict with the numbers behind it. See the [repo README](../README.md) for the overview;
this is the API map.

## Quickstart

```python
from factory_design import FactoryDesign, simulate, diagnose, recommend, verify

design = FactoryDesign(judge_fidelity=0.2)     # a cheap, gameable judge
report = simulate(design)
print(report)                                  # verdict + lenses, each with its limits

for cond in diagnose(report):
    print(recommend(design, cond.name))        # the one change, as a diff against `design`
    print(verify(design, cond.name))           # before -> after, re-simulated
```

## The public API

| name | what it does |
|---|---|
| `FactoryDesign` | the design, in plain settings (`factory-design knobs` lists them) |
| `simulate(design, seeds=, quick=)` | run the design -> a `FactoryReport` (verdict + lenses) |
| `diagnose(report)` | the report -> the list of conditions worth fixing |
| `recommend(design, condition)` | the one change for a condition, as a diff |
| `verify(design, condition)` | apply the change, re-simulate, report before/after |
| `sweep` / `compare` | sweep a setting for the safe range; two designs side by side |
| `reference` | the steering result (a property of the game, not a design) |

## Modules

```
design.py      FactoryDesign + KNOBS (the settings and their translation to dynamics)
simulate.py    simulate() and the five lenses, seed-averaged
pathology.py   the failure-mode classifier (scores + ambiguity), falsification-gated
diagnose.py    report -> conditions
levers.py      condition -> the one change to make
verify.py      apply the change, re-simulate, before/after
sweep.py       sweep one setting; compare two designs
reference.py   the steering result
presets.py     the named designs
selftest.py    the falsification table
cli.py         the factory-design command
```

## Falsification

```bash
factory-design selftest      # one row per claim: presets classify as themselves; every fix resolves
python -m pytest -q          # the suite
```

Every verdict is a median over a seed ensemble, reported with its spread. No number here
measures a real factory; for that, point [factory_probe](../factory_probe) at the running
system.
