# factory_probe

Measure structural properties of a running software factory — a population of
producer agents scored through a (possibly delayed) reward channel under a priced
governance loop. It is the runtime counterpart to [factory-design](../README.md),
which simulates the same properties at design time, before you build. Seven
properties are measured — six from behaviour alone, plus the committed-value gap
from a committed game:

| track | property | confirms when | falsifies when |
|---|---|---|---|
| versioning | near-invariant behavioural versions | one relaxation mode an order of magnitude slower than the rest (τ₂/τ₃ large, stable across box counts) | smooth relaxation continuum, or the gap swings with the estimator |
| pathology | which convergence pathology a regime is in | each regime's fingerprint classifies to its own label | regimes collapse to one indistinguishable fingerprint |
| catastrophe | early-warning signals before a fold | variance/autocorrelation rise pre-fold and stay flat in a no-fold null | no pre-fold rise, or the null rises too (not specific) |
| governance | cascade-ratio stability of a governance loop | price instability falls as the repricing period crosses the cascade band | instability flat or rising with the cascade ratio |
| entrainment | synchronisation threshold + condensate | onset ≈ Kc, diversity raises it, a one-shot halt re-locks, decoupling holds | onset independent of diversity, or a one-shot halt breaks the lock for good |
| opacity | the verified value a legible reader forgoes | the free-vs-legible floor is ~0 when separable and rises with interaction order | floor flat across orders, or positive on a separable task |
| stackelberg | the committed-value gap (V vs U\*) | a mean-based frontier is steered above V while a no-swap core stays at or below it, the gap needs three or more actions, and disclosure shrinks it | the frontier is not steerable above V, or the core is extractable above it |

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
  adapters/openfang.py  concrete adapter for OpenFang (reads native dashboard metrics)
  adapters/openclaw.py  concrete adapter for OpenClaw (reads its control-plane SQLite DB)
  adapters/hermes.py    concrete adapter for Hermes (reads its /api/sessions roster)
  adapters/_derived.py  derives behavioural keys from tokens/status/tools when a
                        framework reports no native ones
  legible.py          a reusable bounded legible reader for the opacity track
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

Three concrete, worked adapters ship against real frameworks, each reading a
running instance's own run history and replaying it as the behavioural stream:

- `adapters/openfang.py` — OpenFang (github.com/RightNow-AI/openfang): reads
  `GET /api/workflows/{id}/runs` and per-Hand dashboard metrics. OpenFang reports
  the behavioural keys natively, so they map straight through.
- `adapters/openclaw.py` — OpenClaw (github.com/openclaw/openclaw): reads the
  control-plane SQLite database (`cron_run_logs`), one row per scheduled run.
- `adapters/hermes.py` — Hermes (github.com/NousResearch/hermes-agent): reads the
  `GET /api/sessions` roster, one session per round.

All three serve the observational tracks (versioning, catastrophe with supplied
epochs, and the pathology classifier on the observed trace). On the OpenClaw and
message-less Hermes paths the run records carry no tool-name signal, so the
fingerprint's variety / learning-death dimension is unavailable there (versioning
and catastrophe are unaffected). They differ in what is real versus derived, and
the code says which: OpenFang exposes native metrics;
OpenClaw and Hermes report only tokens, durations, statuses, and tool names, so
their behavioural keys are *derived* proxies (`adapters/_derived.py`), their
decision-time propensity is a flagged uniform stub (neither stores a log-prob),
their reward is the binary completion status, and neither has a runtime price
channel, so the governance track needs the controllable OpenFang path. Each
mapping is tested against recorded API/database responses (the `test_*_adapter.py`
files); point the client or source at a live instance to run against one.

To run against your own factory, instantiate an adapter and pass it to
`experiment.run` (the CLI's `--substrate` covers only the built-in `mock` and
`tabular`):

```python
from factory_probe.adapters.openfang import OpenFangSubstrate, HttpOpenFangClient
from factory_probe.experiment import run

client = HttpOpenFangClient("https://your-openfang-instance", token="...")
report = run(OpenFangSubstrate(client, workflow_id="wf_123"),
             tracks=["versioning", "catastrophe", "pathology"])
print(report.table())
print(report.falsification_table())
```

The OpenClaw and Hermes adapters follow the same shape:
`run(OpenClawSubstrate(SqliteOpenClawSource("~/.openclaw/state/openclaw.sqlite")), ...)`
and `run(HermesSubstrate(HttpHermesClient("https://your-hermes")), ...)`.

## How the two tools line up

The design-time engine and the runtime probe measure the same properties; a verdict
or condition in one maps to a track in the other:

| factory-design (design time) | factory_probe (runtime track) |
|---|---|
| `versions` lens | versioning |
| `overfitting` / `learning_death` / `thrash` / `stable_failure` verdict | pathology |
| `early_warning` lens | catastrophe |
| `iatrogenic_thrash` condition | governance |
| `correlated_crash` condition | entrainment (condensate) |
| `steering` note | stackelberg (V vs U\*) |

Two tracks need a little more than wiring traffic, and the package now supplies the
hard part of each:

- Stackelberg: wire only `run()` (commit a leader move against a follower configured
  mean-based or no-swap-regret, return the running extracted value). V and U\* are
  read off `run()` — the value reachable against a no-swap follower estimates V,
  against a mean-based follower estimates U\*. These are reachable-value estimates on
  the sampled action set, not exact optima on the full action space.
- Opacity: supply `sample` / `featurize` / `verifier_score` and the floors come from
  the provided bounded legible reader (`factory_probe.legible.FeatureDividendTask`:
  an additive model at order 1, a depth-capped model above it;
  `substrates/tabular.py` is the reference, with a real nonzero floor on interacting
  datasets and ~0 on additive ones). The part that stays yours is a non-vector legible
  form for code — a depth-capped program or a sparse rule list — implemented under the
  same one-call-per-candidate meter.

## Requirements

Python 3.12, numpy, scipy, scikit-learn (transitively, via the measurement
modules in `../src`). `pytest` for the tests.
