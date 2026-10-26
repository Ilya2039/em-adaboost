import numpy as np
from scipy.linalg import cholesky, solve_triangular
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data


def em_distance(x, y, S, lam):
    u = np.asarray(x, float) - np.asarray(y, float)
    if np.isinf(lam):
        return float(np.linalg.norm(u))
    A = S + lam * np.eye(len(u))
    return float(np.sqrt(u @ np.linalg.solve(A, u)))


def weighted_mean_cov(X, w):
    w = np.asarray(w, float)
    w = w / w.sum()
    mu = w @ X
    Z = X - mu
    return mu, (Z * w[:, None]).T @ Z


class EMNearestCentroid(ClassifierMixin, BaseEstimator):
    """Nearest class centroid under the regularized Euclid-Mahalanobis distance.

    lam=np.inf gives the Euclidean nearest centroid.
    """

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(self, X, y)
        if not self.lam > 0:
            raise ValueError("lam must be positive")
        self.classes_ = unique_labels(y)
        p = X.shape[1]
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.means_ = np.zeros((len(self.classes_), p))
        self.chol_ = []
        for k, c in enumerate(self.classes_):
            m = y == c
            wk = w[m] if w[m].sum() > 0 else np.ones(m.sum())
            mu, S = weighted_mean_cov(X[m], wk)
            self.means_[k] = mu
            self.chol_.append(None if np.isinf(self.lam)
                              else cholesky(S + self.lam * np.eye(p), lower=True))
        return self

    def squared_distances(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        D = np.empty((X.shape[0], len(self.classes_)))
        for k, L in enumerate(self.chol_):
            U = (X - self.means_[k]).T
            V = U if L is None else solve_triangular(L, U, lower=True)
            D[:, k] = np.einsum("ij,ij->j", V, V)
        return D

    def predict(self, X):
        return self.classes_[np.argmin(self.squared_distances(X), axis=1)]
