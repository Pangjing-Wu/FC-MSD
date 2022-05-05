from functools import partial
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from _typing import ArrayLike, PdData_T
from .cluster import ClusterData
from .classifier import sample_weight


__all__ = ['MarketStyleAnalyzer', 'GridSearchMSA']


class MarketStyleAnalyzer(object):
    """ Market style analyzer.
    
    Args:
        offset: int, offset of rolling windows, default: `0`. 
            Because the length of discarded data is different
            among each stock, the output need be offset to align 
            data with each other. 
            For example: given two processed series `A` and `B`,
            `A = [t25, t26, ..., t100]`,
            `B = [t60, t61, ..., t100]`,
            to align `A` with `B`, `t25` to `t59` should be 
            discarded by setting `offset = 60`.
    """
    def __init__(self, clf, cls, n_cluster: int, tau: int,
                 y_col: str, offset: int = 0) -> None:
        self.__clf = clf
        self.__cls = cls(n_clusters=n_cluster)
        self.__tau = tau
        self.__y_col  = y_col
        self.__offset = max(offset, tau)
        self.__fitted  = False

    def fit_predict(self, X: pd.DataFrame):
        self.__fitted = True
        X, Y = self.__process_feature(X)
        # calculate features' importance.
        importance = list()
        for x, y in zip(X, Y):
            self.__clf.fit(x, y, sample_weight(y))
            importance.append(self.__clf.feature_importances_)
        importance = np.array(importance)
        # cluster market style based on features' importance.
        cluster = self.__cls.fit_predict(importance)
        return ClusterData(importance, cluster)

    def predict(self, X: pd.DataFrame) -> ClusterData:
        if not self.__fitted:
            raise RuntimeError('MarketStyleAnalyzer has not been fitted.')
        X, Y = self.__process_feature(X)
        # calculate features' importance.
        importance = list()
        for x, y in zip(X, Y):
            self.__clf.fit(x, y, sample_weight(y))
            importance.append(self.__clf.feature_importances_)
        importance = np.array(importance)
        # cluster market style based on features' importance.
        cluster = self.__cls.predict(importance)
        return ClusterData(importance, cluster)

    def __process_feature(self, df:pd.DataFrame) -> Tuple[ArrayLike, ArrayLike]:
        # split features and lable.
        X = df.drop(self.__y_col, axis=1)
        Y = df.loc[:, self.__y_col]
        # rolling generation.
        X = np.array([x for x in self.__rolling(X, self.__tau)])
        Y = np.array([y for y in self.__rolling(Y, self.__tau)])
        # postprocessing.
        Y = np.where(Y < 0, 0, 1) # binarization.
        return X, Y

    def __rolling(self, data: PdData_T, window: Optional[int] = None,
                  step: Optional[int] = None) -> Generator[PdData_T, None, None]:
        """ Rolling window generator.
        
        Return: generator of PdData. Each step returns:
            data[offset: offset + window]
            data[offset + 1: offset + window + 1]
            ...
        """
        window  = max(window, 1) if window else 1
        step    = max(step, 1) if step else 1
        windows = [d for d in data.rolling(window, step)]
        for w in windows[self.__offset:]:
            yield w


class GridSearchMSA(object):
    """ Gride search of MarketStyleAnalyzer.
    Args:
        msa: callable, partial wrappered MarketStyleAnalyzer with
            unspecified args: `clf`, `cls`, `n_cluster`, and `tau`.
        clf: callable, partial wrappered classifier with unspecified
            args: `random_state`.
        cls:  callable, partial wrappered cluster with unspecified
            args: `n_clusters` and `random_state`.
        n_clusters: List[int], grid search param of cluster number.
        taus: List[int], grid search param of period length.
        repeat: int, repeat times of parameter search, default: `10`.
    """
    def __init__(self, msa, clf, cls, n_clusters: List[int], 
                 taus: List[int]) -> None:
        self.__msa = msa
        self.__clf = clf
        self.__cls = cls
        self.__metric = silhouette_score
        self.__taus   = taus
        self.__n_clusters = n_clusters

    @property
    def taus(self) -> List[int]:
        return self.__taus

    @property
    def n_clusters(self) -> List[int]:
        return self.__n_clusters

    @property
    def scores(self) -> ArrayLike:
        return self.__scores

    @property
    def best_score(self) -> float:
        return np.max(self.__scores)

    @property
    def best_parameters(self) -> Dict[str, Union[int, float]]:
        return dict(n=self.__best_n, tau=self.__best_tau)

    @property
    def search_results(self) -> pd.DataFrame:
        ret = dict(n=[], tau=[], score=[])
        for n, socre in zip(self.__n_clusters, self.__scores):
            for tau, s in zip(self.__taus, socre):
                ret['n'].append(n)
                ret['tau'].append(tau)
                ret['score'].append(s)
        return pd.DataFrame(ret)
    
    def fit(self, X: pd.DataFrame, repeat: int = 10, seed: int = 0):
        self.__scores = list()
        # grid search.
        for tau in self.__taus:
            scores = list()
            for n in self.__n_clusters:
                score = list()
                # repeat on each random seed.
                for i in range(repeat):
                    clf = self.__clf(random_state=i)
                    cls = partial(self.__cls, random_state=i)
                    msa = self.__msa(clf=clf, cls=cls, n_cluster=n, tau=tau)
                    cluster = msa.fit_predict(X)
                    score.append(self.__metric(cluster.X, cluster.y))
                scores.append(np.mean(score))
            self.__scores.append(scores)
        # get best parameters.
        self.__scores   = np.array(self.__scores).T
        self.__best_tau = self.__taus[np.argmax(self.__scores) % len(self.__taus)]
        self.__best_n   = self.__n_clusters[np.argmax(self.__scores) // len(self.__taus)]
        # get best model.
        n, tau = self.best_parameters['n'], self.best_parameters['tau']
        clf = self.__clf(random_state=seed)
        cls = partial(self.__cls, random_state=seed)
        msa = self.__msa(clf=clf, cls=cls, n_cluster=n, tau=tau)
        msa.fit_predict(X)
        return msa