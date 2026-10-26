"""Experiment 1: UCI data sets, repeated stratified 5-fold CV with nested selection of lambda and T.

    python exp_main.py [names] [repeats]
"""
import json
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from common import LAMBDAS, RESULTS, T_GRID, em_base, scale, staged, stump
from datasets import NAMES, load
from em_metric import EMNearestCentroid

INF = LAMBDAS.index(np.inf)
ONE = LAMBDAS.index(1.0)


def first_argmax(a):
    return np.unravel_index(np.argmax(a), a.shape)


def run_fold(X, y, tr, te, fold):
    t0 = time.time()
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    acc_em = np.zeros((len(LAMBDAS), len(T_GRID)))
    acc_st = np.zeros(len(T_GRID))
    acc_nc = np.zeros(len(LAMBDAS))
    inner = StratifiedKFold(3, shuffle=True, random_state=fold)
    for itr, iva in inner.split(Xtr, ytr):
        A, B = scale(Xtr[itr], Xtr[iva])
        for i, lam in enumerate(LAMBDAS):
            P, _ = staged(em_base(lam), A, ytr[itr], B)
            acc_em[i] += [(P[T - 1] == ytr[iva]).mean() for T in T_GRID]
            acc_nc[i] += (EMNearestCentroid(lam).fit(A, ytr[itr]).predict(B) == ytr[iva]).mean()
        P, _ = staged(stump(), A, ytr[itr], B)
        acc_st += [(P[T - 1] == ytr[iva]).mean() for T in T_GRID]

    i_ab, j_ab = first_argmax(acc_em)
    j_euc = int(np.argmax(acc_em[INF]))
    j_one = int(np.argmax(acc_em[ONE]))
    j_st = int(np.argmax(acc_st))
    i_nc = int(np.argmax(acc_nc))

    A, B = scale(Xtr, Xte)
    P_em, m_em, nc = [], [], []
    for lam in LAMBDAS:
        P, m = staged(em_base(lam), A, ytr, B)
        P_em.append(P)
        m_em.append(m)
        nc.append(EMNearestCentroid(lam).fit(A, ytr).predict(B))
    P_st, m_st = staged(stump(), A, ytr, B)

    preds = {
        "NC-Euc": nc[INF],
        "NC-EM1": nc[ONE],
        "NC-EM*": nc[i_nc],
        "AB-Euc": P_em[INF][T_GRID[j_euc] - 1],
        "AB-EM1": P_em[ONE][T_GRID[j_one] - 1],
        "AB-EM*": P_em[i_ab][T_GRID[j_ab] - 1],
        "AB-Stump": P_st[T_GRID[j_st] - 1],
    }

    def curve(P):
        return (P == yte).mean(axis=1).round(5).tolist()

    return {
        "fold": fold,
        "acc": {k: float((v == yte).mean()) for k, v in preds.items()},
        "f1": {k: float(f1_score(yte, v, average="macro")) for k, v in preds.items()},
        "lam_ab": float(LAMBDAS[i_ab]), "T_ab": T_GRID[j_ab],
        "lam_nc": float(LAMBDAS[i_nc]),
        "T_euc": T_GRID[j_euc], "T_one": T_GRID[j_one], "T_st": T_GRID[j_st],
        "members_em": m_em, "members_st": m_st,
        "sens_nc": [float((p == yte).mean()) for p in nc],
        "sens_ab": [[float((P[T - 1] == yte).mean()) for T in T_GRID] for P in P_em],
        "curves": {"AB-Euc": curve(P_em[INF]), "AB-EM1": curve(P_em[ONE]),
                   "AB-EM*": curve(P_em[i_ab]), "AB-Stump": curve(P_st)},
        "sec": time.time() - t0,
    }


def main(name, n_repeats=5, n_jobs=2):
    X, y = load(name)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=2026)
    folds = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(X, y, tr, te, k) for k, (tr, te) in enumerate(cv.split(X, y)))
    out = {"dataset": name, "n": int(X.shape[0]), "p": int(X.shape[1]),
           "K": int(len(np.unique(y))), "lambdas": [float(v) for v in LAMBDAS],
           "T_grid": T_GRID, "folds": folds}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"main_{name}.json").write_text(json.dumps(out))
    acc = {k: round(float(np.mean([f["acc"][k] for f in folds])), 4) for k in folds[0]["acc"]}
    print(name, acc, flush=True)


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else NAMES
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for n in names:
        main(n, repeats)
