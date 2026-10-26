"""Experiment 2: two Gaussian classes in R^10 with equicorrelated covariance.

Sigma = (1 - rho) I + rho 11^T, mu_0 = 0, mu_1 = c e_1, with c chosen so that the
Mahalanobis distance between the means is 2 (Bayes accuracy Phi(1) for every rho).
"""
import json

import numpy as np
from joblib import Parallel, delayed

from common import LAMBDAS, RESULTS, em_base, staged
from em_metric import EMNearestCentroid

P = 10
RHOS = [0.0, 0.5, 0.9]
NS = [10, 30, 100, 300]
N_TEST = 4000
REPS = 30
T_EVAL = [1, 10, 50, 200]


def sample(rho, n, rng):
    S = (1 - rho) * np.eye(P) + rho * np.ones((P, P))
    mu1 = np.zeros(P)
    mu1[0] = 2.0 / np.sqrt(np.linalg.inv(S)[0, 0])
    L = np.linalg.cholesky(S)
    X0 = rng.normal(size=(n, P)) @ L.T
    X1 = rng.normal(size=(n, P)) @ L.T + mu1
    return np.vstack([X0, X1]), np.r_[np.zeros(n, int), np.ones(n, int)]


def run(rho, n, rep):
    rng = np.random.default_rng(10_000 * rep + int(100 * rho) + n)
    Xtr, ytr = sample(rho, n, rng)
    Xte, yte = sample(rho, N_TEST, rng)
    nc, ab, members = [], [], []
    for lam in LAMBDAS:
        nc.append(float((EMNearestCentroid(lam).fit(Xtr, ytr).predict(Xte) == yte).mean()))
        Pr, m = staged(em_base(lam), Xtr, ytr, Xte)
        ab.append([float((Pr[t - 1] == yte).mean()) for t in T_EVAL])
        members.append(m)
    return {"rho": rho, "n": n, "rep": rep, "nc": nc, "ab": ab, "members": members}


if __name__ == "__main__":
    jobs = [(r, n, k) for r in RHOS for n in NS for k in range(REPS)]
    runs = Parallel(n_jobs=2)(delayed(run)(*j) for j in jobs)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "synth.json").write_text(json.dumps(
        {"lambdas": [float(v) for v in LAMBDAS], "T_eval": T_EVAL, "runs": runs}))
