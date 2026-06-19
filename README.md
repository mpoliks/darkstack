# The Dark Stack — toy model and figures

A small, self-contained simulation that makes the core claims of *The Dark Stack:
Toward the Full Automation of Software* measurable, and the five figures it
generates. One toy "dark factory" — a finite population of no-regret /
no-swap-regret learners searching a space of assemblies under priced feedback —
is read through five lenses, one per figure.

## Layout

```
src/                 the toy model (each module has a self-test under __main__)
  learners.py        Hedge, EXP3, and the Blum–Mansour no-swap-regret reduction
  factory.py         the population dynamics (stochastic replicator)
  transfer_operator.py   Ulam estimate of the transfer operator + spectral tools
  ews.py             early-warning signals (variance, lag-1 autocorrelation)
  kuramoto.py        coupled-oscillator synchronisation
  control.py         PID-priced governance and cascade timing
  style.py           shared figure styling
figures/             one script per figure; running it writes the PNG + out/figN.json
  audit_layout.py    layout checker (no text/legend collisions)
out/                 the numbers each figure computed
```

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# regenerate every figure (writes figures/*.png and out/*.json)
for f in 1_opacity 2_stackelberg 3_versioning 4_pathologies 5_anything_factory; do
    python figures/fig${f}.py
done

# run the model self-tests (regret bounds, metastability, EWS, Kc, cascade)
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
