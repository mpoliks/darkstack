"""
Transfer-operator versioning from a behavioural trajectory.

A factory version is a *near-invariant
region of its input-output distribution*, that the transfer operator
(Perron-Frobenius) is "exactly the type of measurement tooling required to
identify a version," and that the "spectral gap" grades version robustness: a
wide gap => durable, well-defined version; a small gap => versions bleeding into
each other / transitional.

We implement the standard data-driven Perron-Frobenius estimator (Ulam's
method): partition observed behaviour into boxes, count transitions, normalise
to a row-stochastic matrix P. Eigenvalues of P near 1 correspond to
almost-invariant (metastable) sets; the gap between the cluster of near-1
eigenvalues and the rest is the spectral gap.

References
----------
Ulam 1960 (Problems in Modern Mathematics);
Dellnitz & Junge 1999, "On the approximation of complicated dynamical behavior";
Froyland & Dellnitz; Schuette et al. (metastability & eigenvalue clusters);
Deuflhard & Weber 2005 (PCCA+).
"""
from __future__ import annotations

import numpy as np


def ulam_operator(states: np.ndarray, n_boxes: int, box_edges=None, lag: int = 1):
    """Estimate the Perron-Frobenius operator from a 1-D behavioural trajectory.

    Parameters
    ----------
    states : (T,) array of a scalar behavioural observable (e.g. the population's
             mean output, or its spec-satisfaction). Versions are read off
             *behaviour*, not internal configuration -- this is that observable.
    n_boxes : number of partition cells.
    lag : transition lag tau.

    Returns
    -------
    P : (n_boxes, n_boxes) row-stochastic transition matrix (empty rows -> uniform)
    edges : box edges used
    occupancy : count of points per box
    """
    states = np.asarray(states, dtype=float)
    if box_edges is None:
        lo, hi = np.nanmin(states), np.nanmax(states)
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, n_boxes + 1)
    else:
        edges = np.asarray(box_edges, dtype=float)
        n_boxes = len(edges) - 1
    # assign each state to a box
    idx = np.clip(np.digitize(states, edges[1:-1]), 0, n_boxes - 1)
    C = np.zeros((n_boxes, n_boxes))
    src, dst = idx[:-lag], idx[lag:]
    np.add.at(C, (src, dst), 1.0)
    occupancy = C.sum(axis=1)
    P = np.zeros_like(C)
    nz = occupancy > 0
    P[nz] = C[nz] / occupancy[nz, None]
    P[~nz] = 1.0 / n_boxes               # unvisited boxes -> uniform (harmless)
    return P, edges, occupancy


def reversibilize(P: np.ndarray):
    """Return the additive reversibilization (P + P_tr)/2, where P_tr is the
    time-reversed chain P_tr[i,j] = pi_j P[j,i] / pi_i and pi is the stationary
    distribution. Froyland/Schuette: the clean eigenvalue<->metastable-set
    correspondence is a theorem for reversible (self-adjoint) operators; for
    non-reversible data we symmetrise so the spectrum is real and the spectral
    gap is well defined. Returns (P_rev, pi)."""
    vals, vecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(np.real(vecs[:, idx]))
    pi = pi / pi.sum() if pi.sum() > 0 else np.full(P.shape[0], 1.0 / P.shape[0])
    pi_safe = np.where(pi > 0, pi, 1e-12)
    P_tr = (pi[None, :] * P.T) / pi_safe[:, None]      # detailed-balance transpose
    return 0.5 * (P + P_tr), pi


def coherence_ratio(P: np.ndarray, member_mask: np.ndarray, pi: np.ndarray | None = None) -> float:
    """rho(A) = P(stay in A | in A) = sum_{i,j in A} pi_i P_ij / sum_{i in A} pi_i.
    rho ~ 1 => a near-invariant 'version' (Froyland 2005); the implied period of
    behavioural consistency is 1/(1-rho)."""
    if pi is None:
        _, pi = reversibilize(P)
    A = np.asarray(member_mask, bool)
    num = pi[A][:, None] * P[np.ix_(A, A)]
    den = pi[A].sum()
    return float(num.sum() / den) if den > 0 else float("nan")


def spectrum(P: np.ndarray):
    """Real eigenvalues of P sorted descending, with eigenvectors.

    For a row-stochastic P, the leading eigenvalue is 1 (stationary). We take
    real parts (metastable systems are near-reversible -> near-real spectrum).
    """
    vals, vecs = np.linalg.eig(P.T)          # left eigenvectors (densities)
    vals = np.real(vals)
    order = np.argsort(vals)[::-1]
    return vals[order], np.real(vecs[:, order])


def spectral_gap(P: np.ndarray, n_top: int = 1) -> float:
    """Gap between the n_top-th and (n_top+1)-th eigenvalues.

    n_top=1 gives the classic mixing gap 1 - |lambda_2| (robustness of a single
    version: how strongly the dynamics resist leaving the dominant attractor).
    n_top=m gives the separation of an m-cluster of metastable versions.
    """
    vals, _ = spectrum(P)
    if len(vals) <= n_top:
        return float("nan")
    return float(vals[n_top - 1] - vals[n_top])


def n_metastable(P: np.ndarray, threshold: float = 0.9) -> int:
    """Number of eigenvalues above `threshold`: the count of near-invariant
    sets (versions) currently coexisting."""
    vals, _ = spectrum(P)
    return int(np.sum(vals >= threshold))


def almost_invariant_sets(P: np.ndarray, m: int = 2):
    """Partition boxes into m almost-invariant sets via the sign / k-means of
    the leading m eigenvectors (a lightweight PCCA+ surrogate)."""
    vals, vecs = spectrum(P)
    V = vecs[:, :m]
    if m == 2:
        # sign of the second (sub-dominant) eigenvector splits the two basins
        labels = (V[:, 1] > 0).astype(int)
    else:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=m, n_init=10, random_state=0).fit_predict(V)
    return labels, vals[:m]


def coherence_timescale(lam2: float, lag: int = 1) -> float:
    """Implied metastability lifetime t = -lag / ln(lambda_2): the duration of a
    version / slowest-loop period."""
    lam2 = min(max(lam2, 1e-6), 1 - 1e-9)
    return -lag / np.log(lam2)


if __name__ == "__main__":
    # Sanity check: a two-well metastable process (telegraph + noise). Two basins
    # at -1 and +1 with rare hops should produce exactly 2 eigenvalues near 1 and
    # a sign-structured 2nd eigenvector separating the basins.
    rng = np.random.default_rng(0)
    T = 200_000
    x = np.empty(T)
    hop = 0.001                               # P(basin flip) per step
    basin = -1                                # hidden metastable state in {-1,+1}
    for t in range(T):
        if rng.random() < hop:
            basin = -basin                    # rare basin hop
        x[t] = basin + 0.30 * rng.standard_normal()   # observed = basin + noise
    P, edges, occ = ulam_operator(x, n_boxes=40)
    vals, vecs = spectrum(P)
    print("top 5 eigenvalues:", np.round(vals[:5], 4))
    print("mixing spectral gap (1-lam2):", round(1 - vals[1], 4))
    print("cluster gap (lam2-lam3):", round(spectral_gap(P, n_top=2), 4))
    print("n metastable (>0.9):", n_metastable(P, 0.9))
    print("coherence timescale from lam2:", round(coherence_timescale(vals[1]), 1), "steps")
    labels, lv = almost_invariant_sets(P, m=2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    print("basin A box-centre mean:", round(centers[labels == 0].mean(), 2),
          " basin B box-centre mean:", round(centers[labels == 1].mean(), 2))
