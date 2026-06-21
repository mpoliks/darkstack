"""Entrainment track: synchronisation threshold and the condensate.

Sweeps coupling at two dependency-diversity levels and locates the synchronisation
onset; diversity should raise it. If the substrate can stage interventions, it
also drives a locked population and tests a one-shot desynchronisation (should
re-lock under strong coupling) against structural decoupling (should hold). The
intervention sub-signals are optional: a substrate that cannot stage them reports
them unavailable and the verdict rests on the onset and diversity signals.
"""
from __future__ import annotations

import numpy as np

from ..report import Finding


def _onset(Ks, rs, thr=0.3):
    rs = np.asarray(rs)
    hit = np.where(rs > thr)[0]
    return float(Ks[hit[0]]) if len(hit) else float("nan")


def measure(substrate, N: int = 500, diversities=(0.35, 0.85), tol: float = 0.6,
            seed: int = 7, **kw) -> Finding:
    loops = substrate.coupled_loops(N=N, T=50)
    Ks = np.linspace(0.0, 3.0, 16)
    by_div = {}
    for g in diversities:
        rs = [loops.order_parameter(coupling=float(K), diversity=g, seed=seed) for K in Ks]
        by_div[g] = dict(Kc=loops.critical_coupling(g), onset=_onset(Ks, rs))
    g_lo, g_hi = min(diversities), max(diversities)
    onset_tracks = abs(by_div[g_lo]["onset"] - by_div[g_lo]["Kc"]) < tol
    diversity_raises = by_div[g_hi]["onset"] > by_div[g_lo]["onset"]

    measured = dict(onset_low_div=round(by_div[g_lo]["onset"], 2),
                    Kc_low_div=round(by_div[g_lo]["Kc"], 2),
                    onset_high_div=round(by_div[g_hi]["onset"], 2),
                    Kc_high_div=round(by_div[g_hi]["Kc"], 2))

    # optional condensate sub-signals
    try:
        lock = float(loops.condensate(coupling=2.2, diversity=0.3, intervention="none", seed=3)[-50:].mean())
        halt = float(loops.condensate(coupling=2.2, diversity=0.3, intervention="halt", seed=3)[-50:].mean())
        decouple = float(loops.condensate(coupling=2.2, diversity=0.3, intervention="decouple", seed=3)[-50:].mean())
        condensate_available = True
        halt_relocks = halt > 0.5
        decouple_holds = decouple < 0.2
        measured.update(r_locked=round(lock, 2), r_after_halt=round(halt, 2),
                        r_after_decouple=round(decouple, 2))
    except NotImplementedError:
        condensate_available = False
        halt_relocks = decouple_holds = True            # not required for the verdict
        measured["condensate"] = "unavailable (substrate stages no interventions)"

    confirms = bool(onset_tracks and diversity_raises and halt_relocks and decouple_holds)
    cond_txt = (f"; halt re-locks (r={measured['r_after_halt']}), decouple holds "
                f"(r={measured['r_after_decouple']})") if condensate_available else "; condensate n/a"
    return Finding(
        track="entrainment", property="synchronisation threshold + condensate",
        capability="coupled_loops", measured=measured, confirms=confirms,
        summary=f"onset tracks Kc, diversity raises it ({measured['onset_low_div']}->"
                f"{measured['onset_high_div']}){cond_txt}",
        confirm_criterion="onset ~ Kc=2*diversity, higher diversity raises onset, and (if "
                          "interventions are available) a one-shot halt re-locks while "
                          "structural decoupling collapses the condensate",
        falsify_criterion="onset independent of diversity, or a one-shot halt permanently "
                          "breaks the lock (then a circuit breaker would suffice)",
    )
