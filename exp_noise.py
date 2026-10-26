"""Experiment 3: label noise and outliers in the training part of each outer fold.

Labels: a fraction eta of training objects gets a random wrong label.
Outliers: a fraction rho of training objects is shifted by 10 * sd_j * N(0, 1) per feature.
The test part is never corrupted.
"""
import json
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from common import RESULTS
from datasets import load
from exp_main import run_fold

CONDITIONS = [("clean", 0.0), ("label", 0.1), ("label", 0.2), ("label", 0.3),
              ("outlier", 0.05), ("outlier", 0.1)]


def corrupt(X, y, tr, kind, level, seed):
    rng = np.random.default_rng(seed)
    X, y = X.copy(), y.copy()
    if level == 0:
        return X, y
    idx = rng.choice(tr, size=int(round(level * len(tr))), replace=False)
    if kind == "label":
        classes = np.unique(y)
        for i in idx:
            y[i] = rng.choice(classes[classes != y[i]])
    else:
        sd = X[tr].std(axis=0)
        X[idx] += 10 * sd * rng.normal(size=(len(idx), X.shape[1]))
    return X, y


def job(X, y, tr, te, k, kind, level):
    Xc, yc = corrupt(X, y, tr, kind, level, seed=1000 + k)
    r = run_fold(Xc, yc, tr, te, k)
    return {"kind": kind, "level": level, "fold": k, "acc": r["acc"],
            "lam_ab": r["lam_ab"], "lam_nc": r["lam_nc"], "members_em": r["members_em"]}


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["wdbc", "pima", "seeds", "glass"]
    out = {}
    for name in names:
        X, y = load(name)
        cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7).split(X, y))
        out[name] = Parallel(n_jobs=2)(
            delayed(job)(X, y, tr, te, k, kind, lev)
            for kind, lev in CONDITIONS for k, (tr, te) in enumerate(cv))
        print(name, "done", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "noise.json").write_text(json.dumps(out))
