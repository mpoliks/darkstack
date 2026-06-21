# The Dark Stack — toy model and figures

A small, self-contained simulation that makes the core claims of *The Dark Stack:
Toward the Full Automation of Software* measurable, with the eight figures it generates.
There is **one object**, a toy dark factory whose organs are the exact models the paper
cites, and every figure is a view of it. You run it with

```bash
python run_factory.py
```

which builds one `DarkFactory`, prints its five headline lenses, and renders all five
core figures from that single object. Because the organs are exact, the theorems are
exact: V = 0 and U\* = ½ (Deng–Schneider–Sivan), Kc = 2γ (Kuramoto), and the
incompressibility counting bound all hold. Only the versions and the pathologies emerge
from the dynamics: the proved results are proved, and the rest is observed.

## Layout

```
run_factory.py       the capstone: one DarkFactory -> all five views
src/
  darkfactory.py     the object: five lenses + the full data each figure plots
  factory.py         the population dynamics organ (stochastic replicator)
  learners.py        Hedge, EXP3, and the Blum–Mansour no-swap-regret reduction
  transfer_operator.py   Ulam estimate of the transfer operator + spectral tools
  ews.py             early-warning signals (variance, lag-1 autocorrelation)
  kuramoto.py        coupled-oscillator synchronisation
  control.py         PID-priced governance and cascade timing
  style.py           shared figure styling
figures/             one thin, plot-only view per figure (data comes from DarkFactory)
  audit_layout.py    layout checker (no text or legend collisions)
out/                 the numbers each figure computed
```

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_factory.py          # one factory -> the summary + all five figures

# or a single figure, or the object alone:
python figures/fig2_stackelberg.py
python src/darkfactory.py       # just the five-lens summary

# model self-tests (regret bounds, metastability, EWS, Kc, cascade)
for m in learners transfer_operator ews kuramoto; do python src/$m.py; done

# check that no figure collides text with a legend
python figures/audit_layout.py
```

## The five figures

1. **Opacity–cost frontier.** Legible (low-dimensional) assemblies are exponentially
   rare, and the premium for staying legible grows as the best designs get more complex.
2. **The Stackelberg gap (V vs U\*).** On the Deng–Schneider–Sivan game: a mean-based
   frontier is steered to U\*, a no-swap-regret core is held to V.
3. **Versioning by the transfer operator.** Versions as near-invariant modes of
   operation; the spectral gap grades how robust each one is.
4. **Pathologies and catastrophe.** Four distinct fingerprints, critical slowing down
   before a fold, and overfitting as aliasing.
5. **Governance timing and the anything factory.** The cascade ratio that tames
   iatrogenic thrash, Kuramoto entrainment, and the phase-locked condensate.

Results are seed-fixed and reproduce bit-for-bit; the figure data sits in `out/`.

## Three robustness studies

Separate from the five-figure sequence, these test whether the results survive outside
the exact settings the figures use.

- **Robustness** (`figures/robustness.py` → `figR_robustness.png`): each result holds
  across a region of parameter space, and the operating points the figures use sit
  inside that region. The opacity premium over (d,k), the Stackelberg gap over learning
  rate, the two-version split over (M,μ), the four pathologies tiling (μ,c), the
  early-warning trend over ramp rate and noise, and the Kuramoto onset tracking the
  analytic Kc = 2γ.
- **Dynamics invariance** (`figures/dynamics_invariance.py` → `figD_dynamics.png`): the
  Stackelberg gap holds across several mean-based learners: Hedge, FTPL, and EXP3,
  alongside multiplicative weights. The metastable versions also survive a change of
  selection functional, from exponential to linear fitness. They disappear only under a
  different class of dynamics: best-response (logit) gives one version, because
  metastability needs imitation.
- **Closed-form checks** (`figures/analytic_anchor.py` → `figA_anchor.png`): the two
  emergent results fall on the laws their mechanisms predict. Metastable escape is
  memoryless (dwell times are exponential), the transfer-operator spectral gap sets the
  escape timescale (mean dwell = 2τ₂), and as the spec approaches the fold both
  early-warning signals obey the AR(1) law σ² ∝ 1/(1−α²).

## The opacity dividend — a measured law

(`python figures/opacity_dividend.py` → `figO_dividend.png`; toolkit in `src/dividend.py`)

The toy's headline cost, the opacity premium, was priced by part-count, so it reported a
positive cost even for a separable task whose optimum a one-variable reader recovers
exactly. The dividend reprices opacity by interaction order. A reader who can hold K-way
interactions builds the best degree-≤K model of the value landscape and acts on its
argmax; the dividend `D_K` is the value that reader leaves unreached. Three results
follow.

- The **forced-opacity order** `K*` (the first K with `D_K = 0`) equals the landscape's
  true interaction order `r`, stable across d = 8 to 14.
- The **budget-invariant floor** `Φ*` (the gap a legible reader keeps after a free
  searcher has closed its own) is zero for a separable task and positive for an
  interacting one.
- The dividend sends the linear control's part-count premium of ~0.7 to a true zero.

Acting near-optimally is a weaker condition than fitting well, which is what separates
the price of opacity from the price of performance.

A second figure (`python figures/dividend_depth.py` → `figP_dividend.png`) grounds the
dividend in Kauffman's NK model and shows it as a law. The order-1 forced floor rises
with the interaction order `K` — zero at `K = 0` (separable), climbing past 2σ by
`K = 4` — because the value migrates up the interaction orders as the task tangles, its
Walsh spectrum spreading from order 1 into orders 2–5. The same figure shows that acting
and fitting are distinct quantities (the variance a reader captures leaves about half its
decision regret unexplained), and that the floor is recoverable from a verifier's
queries — the precondition for taking the law onto a real, non-enumerable task.
