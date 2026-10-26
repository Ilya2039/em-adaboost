import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.datasets import load_wine
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from em_metric import EMNearestCentroid, em_distance, weighted_mean_cov

rng = np.random.default_rng(0)


def random_psd(p, rank):
    B = rng.normal(size=(p, rank))
    return B @ B.T


def wine():
    X, y = load_wine(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def test_matches_scipy():
    p = 6
    S = random_psd(p, 3)
    for lam in [1e-3, 0.5, 1.0, 10.0]:
        VI = np.linalg.inv(S + lam * np.eye(p))
        for _ in range(50):
            x, y = rng.normal(size=p), rng.normal(size=p)
            assert np.isclose(em_distance(x, y, S, lam), mahalanobis(x, y, VI))


def test_metric_axioms_with_singular_covariance():
    p = 5
    S = random_psd(p, 2)
    worst = -np.inf
    for lam in [1e-4, 1e-2, 1.0, 100.0]:
        for _ in range(3000):
            x, y, z = rng.normal(size=(3, p)) * rng.exponential(3, size=(3, 1))
            dxy = em_distance(x, y, S, lam)
            assert dxy >= 0
            assert np.isclose(dxy, em_distance(y, x, S, lam))
            worst = max(worst, dxy - em_distance(x, z, S, lam) - em_distance(z, y, S, lam))
        assert em_distance(x, x, S, lam) == 0
    assert worst <= 1e-10


def test_squared_distance_breaks_triangle_inequality():
    S = random_psd(5, 2)
    x, y = np.zeros(5), np.ones(5)
    z = (x + y) / 2
    d2 = lambda a, b: em_distance(a, b, S, 1.0) ** 2
    assert d2(x, y) > d2(x, z) + d2(z, y)


def test_spectral_form():
    S = random_psd(4, 4)
    lam = 0.3
    s, V = np.linalg.eigh(S)
    u = rng.normal(size=4)
    assert np.isclose(em_distance(u, 0 * u, S, lam) ** 2, np.sum((V.T @ u) ** 2 / (s + lam)))


def test_limits():
    S = random_psd(4, 4)
    u = rng.normal(size=4)
    assert np.isclose(1e9 * em_distance(u, 0 * u, S, 1e9) ** 2, u @ u, rtol=1e-6)
    assert np.isclose(em_distance(u, 0 * u, 1e-12 * S, 4.0), np.linalg.norm(u) / 2)


def test_weights_equal_replication():
    X = rng.normal(size=(7, 3))
    reps = np.array([1, 3, 2, 1, 4, 1, 2])
    mu, S = weighted_mean_cov(X, reps)
    Xr = np.repeat(X, reps, axis=0)
    assert np.allclose(mu, Xr.mean(0))
    assert np.allclose(S, np.cov(Xr.T, bias=True))


def test_large_lambda_is_euclidean():
    X, y = wine()
    a = EMNearestCentroid(lam=1e8).fit(X, y).predict(X)
    b = EMNearestCentroid(lam=np.inf).fit(X, y).predict(X)
    assert (a == b).all()


def test_adaboost():
    X, y = wine()
    ab = AdaBoostClassifier(EMNearestCentroid(lam=1.0), n_estimators=20, random_state=0).fit(X, y)
    assert ab.score(X, y) > 0.9
