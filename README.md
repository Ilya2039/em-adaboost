# Regularized Euclid–Mahalanobis metric in AdaBoost

Code and results for the paper by M. V. Khachumov and I. M. Yarusskiy, "Regularized Euclid–Mahalanobis metric as a base classifier for AdaBoost" (in Russian, submitted).

The distance

$$d_\lambda(x, y) = \sqrt{(x - y)^T (S + \lambda E)^{-1} (x - y)}, \qquad \lambda > 0,$$

is used in a nearest class centroid classifier with per-class covariance matrices `S`. Means and covariances are estimated with sample weights, so the classifier can be boosted with SAMME (`sklearn.ensemble.AdaBoostClassifier`). `lam=np.inf` gives the Euclidean nearest centroid.

```python
from sklearn.ensemble import AdaBoostClassifier
from em_metric import EMNearestCentroid

model = AdaBoostClassifier(EMNearestCentroid(lam=0.3), n_estimators=50)
```

## Setup

```
pip install -r requirements.txt
./get_data.sh
```

`get_data.sh` clones [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) into `data/` and unpacks the Weka archive from it (for `segment`). `wdbc`, `wine` and `digits` come with scikit-learn. Another data location can be set with the `EM_DATA` environment variable.

## Reproducing the paper

```
python -m pytest test_em_metric.py
python exp_main.py          # 11 UCI data sets, 5x5 CV, nested selection of lambda and T  (~20 min)
python exp_synth.py         # synthetic Gaussian data                                    (~2 min)
python exp_noise.py         # label noise and outliers                                   (~6 min)
python stop_reasons.py      # why SAMME stops early
python analyze.py           # tables -> results/summary.md, figures -> figures/
```

Times are for two CPU cores. All random seeds are fixed. `results/` and `figures/` contain the outputs used in the paper, so `analyze.py` can be run without repeating the experiments.

| File | Contents |
|---|---|
| `em_metric.py` | the distance and the weighted nearest centroid classifier |
| `common.py` | λ and T grids, staged SAMME predictions |
| `datasets.py` | data loaders |
| `exp_main.py`, `exp_synth.py`, `exp_noise.py`, `stop_reasons.py` | experiments |
| `analyze.py` | tables, Friedman/Wilcoxon and corrected resampled t-tests, figures |

## License

MIT
