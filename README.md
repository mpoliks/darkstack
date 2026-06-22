# darkstack

A simulation engine for software-factory design. You describe a factory you intend to build,
simulate it, and get a verdict for how it will fail before you commit to the design.

A software factory here is a population of agents that writes, reviews, and ships work on its
own, scored by evals and priced by a control loop. Once the population is learning against
a configuration, changing that configuration is slow and costly, so the design is worth
testing first. `simulate(design)` returns a one-word verdict with the numbers behind it:
whether the factory converges on the goal, games its evals, stalls, never settles, or
(when it shares a base model with peers) crashes in sync with them. On a failure verdict it
names the single configuration change that resolves the condition and re-simulates to
confirm the change works.

The engine runs on the exact dynamical models in [src/](src) and does not re-derive them.
It is the design-time counterpart to [factory_probe/](factory_probe), which measures the
same properties on a running factory.

## Example

```
$ factory-design demo            # add --quick for a faster, lower-fidelity run
```

```
A factory you might build: a fast pipeline scored by a cheap, gameable judge.

  VERDICT: OVERFITTING
  the population scores well on the metric while missing the true goal -- it is gaming the eval.

  [pathology]  overfitting  [overfitting=0.75, stable_failure=0.15, thrash=0.05, healthy=0.04]
      fingerprint={settledness: 0.84, variety: 1.5, distance_from_goal: 1.0, metric_minus_truth: 0.86}
      maps to: the post-mortem you would otherwise write by hand after the build
  ...

FIX for overfitting: set judge_fidelity 0.2 -> 0.8
  make the metric track the true goal (a gameable judge is the overfitting)
  verified by re-simulation: true-goal satisfaction 0.00 -> 0.86  [RESOLVED]
```

The factory scores 0.86 on its metric and 0.0 on the real goal: it found a way to please a
cheap judge. A judge that tracks the goal raises the real-goal score to 0.86. The engine
applied that change and checked it before printing it.

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

This puts a `factory-design` command on the path. `factory-design --help` lists the rest.
Add `--quick` to any command for shorter runs with fewer seeds.

## Usage

The defaults describe a healthy factory. Change one or two settings to express a design in
question, then simulate, diagnose, and verify the fix.

```python
from factory_design import FactoryDesign, simulate, diagnose, recommend, verify

design = FactoryDesign(explore_rate=0.0005)    # agents almost never try anything new
report = simulate(design)
print(report.verdict)                          # 'learning_death'

for cond in diagnose(report):
    print(recommend(design, cond.name))        # set explorer_floor: 0.0 -> 0.06
    print(verify(design, cond.name))           # effective variety 0.01 -> 1.04  [OK]
```

`recommend` returns the change as a diff against the supplied design, not a generic tip.
`verify` applies the change, simulates again, and reports the before and after of the
relevant number, so a suggestion ships only when it resolves the condition. Every reported
number is a median over a seed ensemble with its spread, and the simulation is seed-fixed.

## Failure modes

The verdict is one of these. Each is a distinct mechanism with its own fix.

| verdict | meaning | fix |
|---|---|---|
| `healthy` | converges on the goal and keeps exploring | none |
| `stable_failure` | settles into a pattern that misses the goal and will not leave it | enable the governance controller to price the failure |
| `overfitting` | scores the metric while missing the real goal | a judge that tracks the goal, or faster eval sampling |
| `learning_death` | exploration collapses; stuck on one approach, cannot adapt | reserve a protected exploration share |
| `thrash` | never commits; finds approaches and drops them | lower exploration so the core can hold a result |

Two further conditions come from outside the population:

- `iatrogenic_thrash`: the governance loop reprices faster than the factory settles and
  drives the oscillation it is meant to damp. Fix: reprice 3–10x slower than the factory
  settles.
- `correlated_crash`: peers share a base model or control plane and move in lockstep, the
  mechanism behind machine-time cascades such as the 2010 flash crash. Fix: spread
  vendors, models, and schedules across the peer group.

## Lenses

A simulation reports the lenses relevant to the design, not always all of them.
`pathology`, `versions`, and `governance` always appear; the rest appear only when the
design exercises them. Each lens names the real thing it stands for and the limit of what
it claims.

- **pathology** (always): the verdict, as scores over the five modes plus a fingerprint. A
  run between two modes is flagged `ambiguous` and names both.
- **versions** (always): how many distinct stable operating modes the factory settles into,
  and how durable each is. Determines whether a release pins to a config or must be tracked
  by what the factory does.
- **governance** (always): whether the repricing loop is stable or oscillating, read from
  the measured price swings; reads "no controller" when the controller is off. Sets the
  cadence of the eval-to-policy loop.
- **early_warning** (on a full run, i.e. without `--quick`): whether rising variance and
  autocorrelation flag an approaching tipping point, tested against a flat baseline and a
  phase-randomized surrogate. These are the two signals to monitor in production.
- **ecology** (when the design has peers): whether peers sharing infrastructure synchronize
  into a correlated crash. A vendor-and-model diversity decision.
- **steering** (when the design reserves no exploration): a note on why that ceilings what
  the factory can reach (`factory-design reference`).

### Reading a report

The fingerprint has four axes, each in [0,1]: **settledness** (how firmly the factory sits
in one behaviour; 0 = never commits), **variety** (how many distinct approaches stay in
play; collapses toward 0 at learning death), **distance_from_goal** (how far the realised
behaviour sits from what the spec wants), and **metric_minus_truth** (how much it scores
the metric above the true goal -- the gaming signal). Other scalars: **stickiness** (how
hard a single stable mode is to dislodge), **cascade_ratio** (repricing period over inner
settling time; below 3:1 the loop oscillates), and **synchronisation** (peer lockstep, 0-1;
above the onset threshold is correlated-crash exposure).

## Presets

The named designs are recognizable cases. Simulate one, read the verdict, apply the fix.

```
$ factory-design presets
  healthy               -> healthy
  gameable_judge        -> overfitting
  no_exploration_floor  -> learning_death
  stuck_failing         -> stable_failure
  repricing_too_fast    -> stable_failure   (governance lens reports iatrogenic_thrash)
  never_settles         -> thrash
  monoculture           -> healthy          (ecology lens reports correlated_crash)
```

The bare `presets` listing prints only the `name -> verdict` rows; the parentheticals above
name what each preset's other lenses report, which you see by running
`factory-design sim <preset>`.

Override any setting from the command line:

```bash
factory-design sim no_exploration_floor                   # the failure
factory-design fix no_exploration_floor                   # the verified fix
factory-design sim --set explore_rate=0.0005              # a custom case
factory-design sweep healthy explore_rate 0.001,0.06,0.3  # the safe range
factory-design compare healthy never_settles              # two designs side by side
```

## Settings

`factory-design knobs` lists every setting with a gloss and the harness thing it stands
for. The main ones:

- `explore_rate`, `explorer_floor`: how much the population tries new approaches, and a
  protected minimum the scoring cannot starve.
- `eval_period`, `judge_fidelity`: how often the judge re-scores, and how faithfully its
  metric tracks the real goal.
- `controller`, `repricing_period`: the governance loop (off / gentle / aggressive) and how
  often it reprices.
- `peer_factories`, `shared_dependency`, `dependency_diversity`: how many other factories
  share infrastructure, how tightly, and how varied their stacks are.

Some real settings have no model here, and the tool reports that rather than faking it:
retry and escalation depth, tool-call success rate, context-window budget, and rate limits.
Set those in the harness directly.

## Scope and limits

The simulation runs on a small set of exact dynamical models in [src/](src): a finite
population of learning agents under selection, a transfer operator that reads stable modes
from what the population does, early-warning signals, a coupled-oscillator model for the
ecology, and a PID controller for governance. The models are exact, so the proved results
hold: the steering value and its collapse, the synchronization onset, the incompressibility
count. The verdicts and fixes that come off the dynamics are observed, seed-averaged, and
reported with their spread and their limits.

No number here measures a real factory. The mechanisms match the ones a real factory has,
which is enough to choose a design before building. Once the factory is running, point
[factory_probe/](factory_probe) at it: that package measures the same properties on a live
agent-to-agent system through the interfaces it exposes, with a 7/7 falsification table.

## Steering reference

Reserve some non-greedy exploration, or the factory can only reach what you designed in.
That is the one result that sits apart, because it is a property of the game, not of any one
design: a population that reserves some exploration can be steered to a better outcome for
the architect than the value committed to at design time, while a purely retentive one
cannot exceed it (and with only two options the gap is gone). `factory-design reference`
shows it on the canonical Deng–Schneider–Sivan game, with the committed value computed and
the published optimum cited.

## Layout

```
factory_design/      the engine: describe a design, simulate it, diagnose and fix it
  design.py          FactoryDesign -- the settings, in plain terms, and their translation
  simulate.py        simulate(design) -> a report, composing the src/ models
  pathology.py       the failure-mode classifier (scores + ambiguity), falsification-gated
  diagnose.py        read a report -> the conditions worth fixing
  levers.py          for each condition, the one change to make
  verify.py          apply the change, re-simulate, report before and after
  sweep.py           sweep a setting for the safe range; compare two designs
  reference.py       the steering result (a property of the game)
  presets.py         the named designs
  cli.py             the factory-design command
  selftest.py        the falsification table (classifier round-trip + lever battery)
src/                 the exact dynamical models the engine runs on
factory_probe/       measure the same properties on a live, running factory
figures/             worked examples, and the checks that the models are exact
```

## Tests

```bash
factory-design selftest          # the falsification table: classifier round-trip + every fix
python -m pytest -q              # the full suite (engine + probe)
python run_factory.py            # the dynamical models, rendered as figures
```

`factory-design selftest` prints one row per claim the engine makes about itself: every
preset classifies as its own failure mode, and every recommended fix resolves its condition
on re-simulation. A red row is a finding, not a crash.
