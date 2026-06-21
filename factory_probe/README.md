# factory_probe

Measure structural properties of a running agent factory — a population of
producer agents scored through a (possibly delayed) reward channel under a priced
governance loop. Six properties are measured from behaviour alone:

| track | property | confirms when | falsifies when |
|---|---|---|---|
| versioning | near-invariant behavioural versions | one relaxation mode an order of magnitude slower than the rest (τ₂/τ₃ large, stable across box counts) | smooth relaxation continuum, or the gap swings with the estimator |
| pathology | which convergence pathology a regime is in | each regime's fingerprint classifies to its own label | regimes collapse to one indistinguishable fingerprint |
| catastrophe | early-warning signals before a fold | variance/autocorrelation rise pre-fold and stay flat in a no-fold null | no pre-fold rise, or the null rises too (not specific) |
| governance | cascade-ratio stability of a governance loop | price instability falls as the repricing period crosses the cascade band | instability flat or rising with the cascade ratio |
| entrainment | synchronisation threshold + condensate | onset ≈ Kc, diversity raises it, a one-shot halt re-locks, decoupling holds | onset independent of diversity, or a one-shot halt breaks the lock for good |
| opacity | the verified value a legible reader forgoes | the free-vs-legible floor is ~0 when separable and rises with interaction order | floor flat across orders, or positive on a separable task |

Each track returns a `Finding`: the numbers measured, whether the expected signal
is present, and the criteria that would confirm or falsify it — so the same object
reads as a result against known ground truth and as a falsification test against a
live factory.

## Install

The measurement modules live in `../src`, so install editable from the repo root
(or run from the repo root):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"     # or: pip install -r requirements.txt
```

An editable install exposes `factory_probe` (and the `factory-probe` console
script) to any environment while keeping the `src` modules resolvable. A plain
file-copy install is not supported because of the sibling `src` layout.

## Run

```bash
python -m factory_probe tracks          # list tracks and required capabilities
python -m factory_probe run             # run every track on the reference substrate
python -m factory_probe run --track opacity --track versioning
python -m factory_probe run --json report.json
python -m factory_probe._smoke          # end-to-end check against planted ground truth
pytest tests/                           # unit + per-track tests with negative controls
```

## How it is put together

```
factory_probe/
  records.py          Decision / Reward / RoundObs -- the only objects tracks read
  instrumentation.py  TraceStore (behaviour + control series, reward joins, lags),
                      ReturnQueue (delayed, jittered reward channel)
  interfaces.py       capability interfaces a substrate implements:
                      SteppableFactory, LearnerGame, DividendTask, CoupledLoops,
                      bundled by Substrate
  scorer.py           ConsequenceScorer -- out-of-loop realized-consequence grading
  tracks/             one measure(substrate, **config) -> Finding per property
  report.py           Finding + Report (table, falsification table, JSON)
  experiment.py       run(substrate, tracks) -> Report
  mock/               reference substrate: all four capabilities on controllable
                      dynamics with known ground truth
  adapters/skeleton.py  template for a live agent-to-agent substrate
```

A track reads a substrate only through a capability interface and the trace
records, never through substrate internals, so a track that passes on the mock
runs unchanged against a live factory.

## Testing a live factory

Subclass `factory_probe.adapters.skeleton.LiveSubstrate` and fill the methods that
pull from your factory's real traffic: map each sampled producer decision to a
`Decision` (with the decision-time propensity it disclosed), each verifier score to
a `Reward`, the per-metric verdict stream to the behavioural series, and the
applied governance price / injected intent to the control series. Declare the
capabilities your instrumentation can serve; `run` executes only those tracks.
`skeleton.py` documents the mapping for each capability.

Two parts need real-factory work beyond instrumentation: estimating the committed
value V and steerable optimum U\* off a non-enumerable game (the Stackelberg
track), and supplying a bounded legible model class for the task domain (the
opacity track — sparse rule lists or depth-capped programs for code).

## Requirements

Python 3.12, numpy, scipy, scikit-learn (transitively, via the measurement
modules in `../src`). `pytest` for the tests.
