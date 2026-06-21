"""
The opacity dividend: opacity priced in interaction-order space, not part-count space.

A value landscape f lives on the Boolean cube {0,1}^d. A "reader" of order K can hold
K-way interactions and no more; its best model of f is the degree-<=K parity (Walsh)
truncation f_K, and it ACTS by choosing argmax f_K.

    D_K(f) = (f* - f[argmax f_K]) / sigma_f      # value an order-K reader cannot reach
    K*(f)  = min{ K : D_K = 0 }                   # forced-opacity order
    R2_K   = fraction of variance the order-K model captures   (fitting, not acting)

The dividend prices ACTING (regret of a low-order reader's argmax); R2 prices FITTING
(approximation error). They come apart -- which is the separation of opacity from
performance. K* equals the landscape's true interaction order r.

The budget-invariant floor Phi*: a free searcher and a legibility-restricted searcher
get the same metered evaluation budget B. The free searcher's gap to f* vanishes as
B grows (search for performance); the legible searcher plateaus at Phi* (efficiency
available only illegibly -- a fact about the landscape, not a compute bill).
"""
import numpy as np
from itertools import combinations


# ---- Walsh / parity machinery (exact on the enumerated cube) ----------------
def fwht(a):
    """In-place fast Walsh-Hadamard transform (natural order). Self-inverse up to n."""
    a = np.asarray(a, float).copy()
    n = len(a); h = 1
    while h < n:
        a = a.reshape(n // (2 * h), 2 * h)
        x = a[:, :h].copy(); y = a[:, h:].copy()
        a[:, :h] = x + y
        a[:, h:] = x - y
        a = a.ravel()
        h *= 2
    return a


def popcounts(d):
    pc = np.zeros(1 << d, dtype=int)
    idx = np.arange(1 << d)
    for i in range(d):
        pc += (idx >> i) & 1
    return pc


def walsh_coeffs(f):
    """f-hat(S) = (1/n) sum_x f(x) chi_S(x), chi_S(x) = (-1)^{<S,x>}."""
    return fwht(f) / len(f)


def f_truncate(fhat, pc, K):
    """f_K(x) = sum_{|S|<=K} f-hat(S) chi_S(x): the best degree-<=K model of f (L2-optimal)."""
    return fwht(np.where(pc <= K, fhat, 0.0))


def dividend(f, pc, Kmax=None, tol=1e-9):
    """Return (D, R2, Kstar): the opacity dividend and variance-captured at each order.

    Ties in argmax f_K are broken in the reader's FAVOUR (best true value among the
    model-optimal vertices), so D_K is the regret a reader keeps even when it resolves
    its own indifference optimally -- a lower bound on the dividend, and the conservative
    reading of "opacity is forced". (Equal to the single-argmax value when f_K's optimum
    is unique, e.g. on dense landscapes.)"""
    n = len(f); d = int(round(np.log2(n)))
    Kmax = d if Kmax is None else Kmax
    fhat = walsh_coeffs(f); coef2 = fhat ** 2
    var = coef2[pc >= 1].sum(); sd = f.std()
    D, R2 = {}, {}
    for K in range(0, Kmax + 1):
        fk = f_truncate(fhat, pc, K)
        thr = fk.max() - tol * (1.0 + np.abs(fk).max())     # model-optimal set (tie-robust)
        D[K] = float((f.max() - f[fk >= thr].max()) / sd)
        R2[K] = float(coef2[(pc >= 1) & (pc <= K)].sum() / var) if var > 0 else 1.0
    Kstar = next((K for K in range(1, Kmax + 1) if D[K] <= 1e-9), None)
    return D, R2, Kstar


# ---- landscapes of a known interaction order --------------------------------
def planted_order(d, r, rng, dense=True, terms=4):
    """f whose highest non-vanishing Walsh order is exactly r.

    dense=True (default) gives EVERY order-<=r term a random coefficient, so the optimum
    is generically unique and free of tied maximizers -- the family on which K*=r is a
    clean test. dense=False plants only `terms` coefficients per order (sparse), which
    leaves free bits and tied model-optima; that family is tie-degenerate and should not
    be used to measure K*."""
    n = 1 << d; pc = popcounts(d); fhat = np.zeros(n)
    for j in range(1, r + 1):
        masks = np.where(pc == j)[0]
        if dense:
            fhat[masks] = rng.standard_normal(len(masks))
        else:
            sel = rng.choice(masks, size=min(terms, len(masks)), replace=False)
            fhat[sel] = rng.standard_normal(len(sel))
    return fwht(fhat), pc


# ---- legible vs free search under a metered budget (the Phi* experiment) -----
def parity_features(d, K):
    """Design matrix Phi[x, S] = (-1)^{<S,x>} for all S with |S| <= K (columns)."""
    n = 1 << d; idx = np.arange(n)
    subsets = [()] + [c for k in range(1, K + 1) for c in combinations(range(d), k)]
    cols = []
    for S in subsets:
        if not S:
            cols.append(np.ones(n))
        else:
            par = np.zeros(n, int)
            for i in S:
                par ^= (idx >> i) & 1
            cols.append(1.0 - 2.0 * par)        # (-1)^par
    return np.column_stack(cols)


def budget_curves(f, d, Ks, budgets, rng, reps=24):
    """Gap-to-optimum vs metered budget B for a FREE searcher (best-of-B random draws)
    and LEGIBLE order-K searchers (fit a degree-<=K model from B samples, act on its
    argmax). Returns dict: 'free' -> array, and K -> array, all gaps in sigma units."""
    n = len(f); sd = f.std(); fmax = f.max()
    feats = {K: parity_features(d, K) for K in Ks}
    out = {"free": np.zeros(len(budgets))}
    for K in Ks:
        out[K] = np.zeros(len(budgets))
    for bi, B in enumerate(budgets):
        gfree = np.zeros(reps); gleg = {K: np.zeros(reps) for K in Ks}
        for t in range(reps):
            samp = rng.choice(n, size=min(B, n), replace=False)   # metered, no double-counting
            gfree[t] = (fmax - f[samp].max()) / sd
            for K in Ks:
                Phi = feats[K]
                w, *_ = np.linalg.lstsq(Phi[samp], f[samp], rcond=None)
                pred = Phi @ w
                gleg[K][t] = (fmax - f[np.argmax(pred)]) / sd
        out["free"][bi] = gfree.mean()
        for K in Ks:
            out[K][bi] = gleg[K].mean()
    return out


def phi_floor(f, d, K, rng, B_big=None, reps=40):
    """Budget-asymptotic floor for an order-K legible searcher: the gap that survives a
    large budget. Equals D_K in the limit; estimated here at a large finite B."""
    n = len(f); B_big = B_big or min(n, 1 << min(d + 2, 16))
    return float(budget_curves(f, d, [K], [B_big], rng, reps=reps)[K][0])
