import numpy as np
import pandas as pd

from _typing import ArrayLike
from .cluster import ClusterData


__all__ = ['sample_weight', 'ClassifyByStyle']


def sample_weight(y: ArrayLike) -> ArrayLike:
    y = np.array(y, dtype=int).flatten()
    n_class = len(set(y))
    ret = np.zeros_like(y)
    weight = 1 / n_class / np.bincount(y)
    for i in range(n_class):
        ret = np.where(y == i, weight[i], ret)
    return ret


class ClassifyByStyle(object):

    def __init__(self, clf, n: int, seed: int = 0) -> None:
        self.__n     = n
        self.__fitted = False
        self.__clfs = [clf(random_state=seed) for _ in range(n)]
    
    @property
    def n(self):
        return self.__n

    def fit(self, X: pd.DataFrame, Y: pd.Series, style: ClusterData) -> None:
        self.__fitted = True
        assert len(X) == len(style.y) and len(Y) == len(style.y)
        # divide training samples by market styles.
        x_by_style = [X[style.y == s] for s in range(style.n_clusters)]
        y_by_style = [Y[style.y == s] for s in range(style.n_clusters)]
        # training models of corresponding styles.
        for clf, x_s, y_s in zip(self.__clfs, x_by_style, y_by_style):
            clf.fit(x_s, y_s, sample_weight(y_s))

    def predict(self, X: pd.DataFrame, style: ClusterData) -> ArrayLike:
        if not self.__fitted:
            raise RuntimeError('MarketStyleAnalyzer has not been fitted.')
        assert len(X) == len(style.y)
        # test by each day.
        y_pred = [self.__clfs[s].predict([x]) for x, s in zip(X.values, style.y)]
        y_pred = np.concatenate(y_pred)
        return y_pred