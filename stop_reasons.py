"""Why SAMME stops with the EM nearest centroid base classifier (full standardized data).

Reason codes: T - all 200 rounds, zero - zero weighted error, thr - error >= 1 - 1/K.
"""
import json

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from common import LAMBDAS, RESULTS
from datasets import NAMES, load
from em_metric import EMNearestCentroid

if __name__ == "__main__":
    out = {}
    for name in NAMES:
        X, y = load(name)
        X = StandardScaler().fit_transform(X)
        row = []
        for lam in LAMBDAS:
            ab = AdaBoostClassifier(EMNearestCentroid(lam), n_estimators=200, random_state=0).fit(X, y)
            m = len(ab.estimators_)
            if m == 200:
                reason = "T"
            elif ab.estimator_errors_[m - 1] == 0:
                reason = "zero"
            else:
                reason = "thr"
            row.append((m, reason, float(ab.estimator_errors_[0])))
        out[name] = row
        print(name, " ".join(f"{m}{r[0]}" for m, r, _ in row), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stop_reasons.json").write_text(json.dumps(out))
