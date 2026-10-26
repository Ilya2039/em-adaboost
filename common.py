from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from em_metric import EMNearestCentroid

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LAMBDAS = [1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0, np.inf]
T_MAX = 200
T_GRID = [10, 25, 50, 100, 200]


def staged(est, Xtr, ytr, Xte, T=T_MAX):
    """Predictions of the ensembles of size 1..T and the number of members actually fitted.

    After early stopping the last prediction is repeated. If the first base
    classifier is already worse than chance, the base classifier alone is used (m=0).
    """
    ab = AdaBoostClassifier(est, n_estimators=T, random_state=0)
    try:
        ab.fit(Xtr, ytr)
    except ValueError:
        p = est.fit(Xtr, ytr).predict(Xte)
        return np.repeat(p[None, :], T, axis=0), 0
    P = np.array(list(ab.staged_predict(Xte)))
    m = len(P)
    if m < T:
        P = np.vstack([P, np.repeat(P[-1:], T - m, axis=0)])
    return P, m


def em_base(lam):
    return EMNearestCentroid(lam=lam)


def stump():
    return DecisionTreeClassifier(max_depth=1)


def scale(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)
