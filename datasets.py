import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

DATA = Path(os.environ.get("EM_DATA", Path(__file__).parent / "data"))


def _csv(name):
    d = pd.read_csv(DATA / "Datasets" / f"{name}.csv", header=None)
    return d.iloc[:, :-1].to_numpy(float), pd.factorize(d.iloc[:, -1])[0]


def _segment():
    parts = [arff.loadarff(DATA / "weka" / "data" / f"segment-{s}.arff")[0]
             for s in ("challenge", "test")]
    d = pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True)
    return d.iloc[:, :-1].to_numpy(float), pd.factorize(d.iloc[:, -1])[0]


LOADERS = {
    "sonar": lambda: _csv("sonar"),
    "ionosphere": lambda: _csv("ionosphere"),
    "wdbc": lambda: load_breast_cancer(return_X_y=True),
    "pima": lambda: _csv("pima-indians-diabetes"),
    "banknote": lambda: _csv("banknote_authentication"),
    "glass": lambda: _csv("glass"),
    "wine": lambda: load_wine(return_X_y=True),
    "seeds": lambda: _csv("wheat-seeds"),
    "thyroid": lambda: _csv("new-thyroid"),
    "segment": _segment,
    "digits": lambda: load_digits(return_X_y=True),
}

NAMES = list(LOADERS)


def load(name):
    X, y = LOADERS[name]()
    return np.asarray(X, float), np.asarray(y)


if __name__ == "__main__":
    for n in NAMES:
        X, y = load(n)
        print(f"{n:11s} n={X.shape[0]:5d} p={X.shape[1]:3d} K={len(np.unique(y))}")
