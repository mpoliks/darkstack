# The Dark Stack — toy model and figures

A small, self-contained simulation that makes the core claims of *The Dark Stack:
Toward the Full Automation of Software* measurable, and the five figures it
generates. There is **one object** — a toy dark factory whose organs are the exact
models the paper cites — and the five figures are *views* of it. You run it with

```bash
python run_factory.py
```

which instantiates one `DarkFactory`, prints its five headline lenses, and renders
all five figures from that single object. The organs are exact, so the theorems
are exact: V = 0 / U\* = ½ (Deng–Schneider–Sivan), Kc = 2γ (Kuramoto), and the
incompressibility counting bound all hold; only the versions and pathologies are
emergent, which is as it should be.

## Layout

```
run_factory.py       the capstone: one DarkFactory -> all five views
src/
  darkfactory.py     THE object: five lenses + the full data each figure plots
  learners.py        Hedge, EXP3, and the Blum–Mansour no-swap-regret reduction
  factory.py         the population dynamics organ (stochastic replicator)
  transfer_operator.py   Ulam estimate of the transfer operator + spectral tools
  ews.py             early-warning signals (variance, lag-1 autocorrelation)
  kuramoto.py        coupled-oscillator synchronisation
  control.py         PID-priced governance and cascade timing
  style.py           shared figure styling
figures/             one thin, plot-only view per figure (data comes from DarkFactory)
  audit_layout.py    layout checker (no text/legend collisions)
out/                 the numbers each figure computed
```

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_factory.py          # one factory -> the summary + all five figures

# or a single figure / the object alone:
python figures/fig2_stackelberg.py
python src/darkfactory.py       # just the five-lens summary

# model self-tests (regret bounds, metastability, EWS, Kc, cascade)
for m in learners transfer_operator ews kuramoto; do python src/$m.py; done

# check no figure has a text/legend collision
python figures/audit_layout.py
```

## The five figures

1. **Opacity–cost frontier** — legible (low-dimensional) assemblies are exponentially
   rare and carry a premium that grows with structural complexity.
2. **The Stackelberg gap (V vs U\*)** — on the Deng–Schneider–Sivan game: a mean-based
   frontier is steered to U\*, a no-swap-regret core is held to V.
3. **Versioning by the transfer operator** — versions as near-invariant behavioural
   modes; the spectral gap grades robustness.
4. **Pathologies & catastrophe** — distinct behavioural fingerprints, critical slowing
   down before a fold, and overfitting as aliasing.
5. **Governance timing & the anything factory** — cascade-ratio (iatrogenic thrash),
   Kuramoto entrainment, and the phase-locked condensate.

Results are seed-fixed and reproduce bit-for-bit; the figure data is in `out/`.
