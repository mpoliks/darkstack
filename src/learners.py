"""
Learning primitives: mean-based no-regret and no-swap-regret learners.

Two load-bearing learner classes:

  * mean-based no-regret learners  -- the surplus-generating frontier. Their
    action probabilities are a (smooth) function of CUMULATIVE reward only.
    Hedge / multiplicative-weights (full information) and EXP3 (bandit) are the
    canonical instances. Deng-Schneider-Sivan (NeurIPS 2019) prove these can be
    steered BEYOND the Stackelberg value V toward U*.

  * no-swap-regret learners -- the surplus-retaining core. Built
    (Blum & Mansour 2007) as N copies of a no-regret learner glued
    together with a per-round stationary-distribution fixed point. They converge
    to the correlated equilibrium of the committed game and then stop; against
    them the optimizer can extract at most V.

Conventions: everything is phrased in terms of REWARDS in [0, 1] that the
learner MAXIMISES. Losses are 1 - reward where a loss form is needed.

References
----------
Freund & Schapire 1997 (Hedge); Auer et al. 2002 (EXP3);
Blum & Mansour 2007, "From External to Internal Regret" (JMLR);
Deng, Schneider, Sivan 2019, "Strategizing against No-Regret Learners".
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Mean-based no-regret learners
# --------------------------------------------------------------------------- #
class Hedge:
    """Multiplicative-weights / Hedge (full-information, mean-based).

    p_i(t) ∝ exp(eta * G_i(t)) where G_i is the cumulative reward of action i.
    This is *exactly* a function of cumulative reward, hence mean-based: an
    action whose cumulative reward falls eta-far below the leader gets
    exponentially suppressed probability.
    """

    def __init__(self, n_actions: int, eta: float | None = None, horizon: int | None = None,
                 rng: np.random.Generator | None = None):
        self.K = int(n_actions)
        # Near-optimal learning rate sqrt(8 ln K / T) if horizon known, else a default.
        if eta is None:
            T = horizon if horizon is not None else 10_000
            eta = np.sqrt(8.0 * np.log(self.K) / max(T, 1))
        self.eta = float(eta)
        self.G = np.zeros(self.K)              # cumulative reward per action
        self.rng = rng or np.random.default_rng()

    def distribution(self) -> np.ndarray:
        z = self.eta * (self.G - self.G.max())  # shift for numerical stability
        w = np.exp(z)
        return w / w.sum()

    def sample(self) -> int:
        return int(self.rng.choice(self.K, p=self.distribution()))

    def update(self, reward_vector: np.ndarray) -> None:
        """Full-information update: observe reward of EVERY action."""
        self.G += np.asarray(reward_vector, dtype=float)

    def mean_based_gap(self) -> float:
        """Largest probability assigned to an action that is >0 below the best,
        as a fraction; a diagnostic of the mean-based property."""
        p = self.distribution()
        best = self.G.max()
        suboptimal = self.G < best - 1e-12
        return float(p[suboptimal].max()) if suboptimal.any() else 0.0


class EXP3:
    """EXP3 (bandit, mean-based). Observes reward only for the played action;
    uses importance weighting to build an unbiased cumulative-reward estimate."""

    def __init__(self, n_actions: int, gamma: float = 0.07, eta: float | None = None,
                 rng: np.random.Generator | None = None):
        self.K = int(n_actions)
        self.gamma = float(gamma)
        self.eta = float(eta) if eta is not None else gamma / self.K
        self.S = np.zeros(self.K)              # cumulative importance-weighted reward
        self.rng = rng or np.random.default_rng()

    def distribution(self) -> np.ndarray:
        z = self.eta * (self.S - self.S.max())
        w = np.exp(z)
        p = w / w.sum()
        return (1 - self.gamma) * p + self.gamma / self.K   # forced exploration

    def sample(self) -> int:
        return int(self.rng.choice(self.K, p=self.distribution()))

    def update(self, action: int, reward: float) -> None:
        p = self.distribution()
        self.S[action] += reward / p[action]   # unbiased estimator


# --------------------------------------------------------------------------- #
#  No-swap-regret learner (Blum-Mansour reduction)
# --------------------------------------------------------------------------- #
def stationary_distribution(Q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Stationary distribution p of a row-stochastic matrix Q: p = p Q.

    This is the per-round fixed point: the array of experts' recommendations is
    read as a transition matrix and its stationary distribution is found. We solve it as the left eigenvector for eigenvalue 1,
    with a tiny uniform regularisation to guarantee irreducibility/aperiodicity.
    """
    K = Q.shape[0]
    Qr = (1 - eps) * Q + eps / K                      # ergodic regularisation
    vals, vecs = np.linalg.eig(Qr.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    p = np.real(vecs[:, idx])
    p = np.abs(p)
    s = p.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return p / s


class BlumMansourSwap:
    """No-swap-regret learner via the Blum-Mansour reduction.

    Maintains K internal Hedge experts, one per action i. Expert i answers:
    "on the rounds where the overall policy plays action i, what should it have
    played instead?" Each round:
      1. read the experts' distributions q^i as rows of a transition matrix Q,
      2. play p = stationary distribution of Q (the fixed point),
      3. observe the full reward vector r,
      4. feed expert i the reward vector scaled by p_i (its jurisdiction share).
    Swap regret is O(sqrt(T K log K)); the play converges to the correlated
    equilibrium of the committed game.
    """

    def __init__(self, n_actions: int, eta: float | None = None, horizon: int | None = None,
                 rng: np.random.Generator | None = None):
        self.K = int(n_actions)
        self.rng = rng or np.random.default_rng()
        self.experts = [Hedge(self.K, eta=eta, horizon=horizon, rng=self.rng) for _ in range(self.K)]
        self._p = np.full(self.K, 1.0 / self.K)

    def distribution(self) -> np.ndarray:
        Q = np.vstack([e.distribution() for e in self.experts])  # row i = q^i
        self._p = stationary_distribution(Q)
        return self._p

    def sample(self) -> int:
        return int(self.rng.choice(self.K, p=self.distribution()))

    def update(self, reward_vector: np.ndarray) -> None:
        r = np.asarray(reward_vector, dtype=float)
        p = self._p
        for i, e in enumerate(self.experts):
            e.update(p[i] * r)                  # jurisdiction-scaled reward


# --------------------------------------------------------------------------- #
#  Self-tests: verify the regret guarantees hold.
# --------------------------------------------------------------------------- #
def _external_regret(rewards_played: np.ndarray, reward_history: np.ndarray) -> float:
    """(best fixed action in hindsight - algorithm) / T."""
    T = reward_history.shape[0]
    best = reward_history.sum(axis=0).max()
    return (best - rewards_played.sum()) / T


def _swap_regret(actions: np.ndarray, reward_history: np.ndarray, K: int) -> float:
    """max over swap functions phi of the gain from applying phi, per round."""
    T = reward_history.shape[0]
    played = reward_history[np.arange(T), actions].sum()
    best_swapped = 0.0
    for i in range(K):
        mask = actions == i
        if not mask.any():
            continue
        # best single action to swap i -> j
        best_swapped += reward_history[mask].sum(axis=0).max()
    return (best_swapped - played) / T


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, T = 5, 20_000
    # Adversarial stochastic rewards with a shifting best action.
    base = rng.uniform(0.2, 0.8, size=K)
    R = np.clip(base[None, :] + 0.15 * rng.standard_normal((T, K)), 0, 1)
    R[T // 2:, 0] += 0.3  # regime shift: action 0 becomes great late
    R = np.clip(R, 0, 1)

    for name, make in [
        ("Hedge", lambda: Hedge(K, horizon=T, rng=rng)),
        ("BlumMansourSwap", lambda: BlumMansourSwap(K, horizon=T, rng=rng)),
    ]:
        alg = make()
        acts = np.empty(T, dtype=int)
        got = np.empty(T)
        for t in range(T):
            a = alg.sample()
            acts[t] = a
            got[t] = R[t, a]
            alg.update(R[t])
        er = _external_regret(got, R)
        sr = _swap_regret(acts, R, K)
        print(f"{name:18s}  external_regret/T={er:+.4f}   swap_regret/T={sr:+.4f}")

    h = Hedge(K, horizon=T, rng=rng)
    for t in range(2000):
        h.update(R[t])
    print(f"Hedge mean-based gap after 2000 rounds: {h.mean_based_gap():.4f}")

    # ------------------------------------------------------------------ #
    # Adaptive adversary that targets INTERNAL/SWAP regret. Each round the
    # adversary rewards the cyclic successor of the algorithm's current modal
    # action (a rock-paper-scissors forcing): this is exactly the structure
    # under which plain Hedge incurs linear swap regret while the Blum-Mansour
    # reduction does not. This is the empirical separation between the two classes.
    # ------------------------------------------------------------------ #
    print("\nAdaptive (cyclic) adversary -- swap regret separation:")
    Kc = 3
    for name, make in [
        ("Hedge", lambda: Hedge(Kc, eta=0.5, rng=np.random.default_rng(1))),
        ("BlumMansourSwap", lambda: BlumMansourSwap(Kc, eta=0.5, rng=np.random.default_rng(1))),
    ]:
        alg = make()
        Tc = 20_000
        acts = np.empty(Tc, dtype=int)
        hist = np.empty((Tc, Kc))
        for t in range(Tc):
            p = alg.distribution()
            modal = int(np.argmax(p))
            r = np.zeros(Kc)
            r[(modal + 1) % Kc] = 1.0           # reward the successor of modal action
            acts[t] = alg.sample()
            hist[t] = r
            alg.update(r)
        sr = _swap_regret(acts, hist, Kc)
        print(f"  {name:18s} swap_regret/T = {sr:+.4f}")
