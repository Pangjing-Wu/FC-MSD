import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

from _typing import ArrayLike, ModelInput_T


__all__ = ['HierarchyCluster']


class ClusterData(object):

    def __init__(self, feature: ArrayLike, cluster: ArrayLike) -> None:
        self.__feature = feature
        self.__cluster = cluster
        self.__center  = self.__cal_center()
        self.__radius  = self.__cal_radius()
    
    @property
    def X(self) -> ArrayLike:
        return self.__feature

    @property
    def y(self) -> ArrayLike:
        return self.__cluster

    @property
    def n_clusters(self) -> int:
        return max(self.__cluster) + 1

    @property
    def center(self) -> ArrayLike:
        return self.__center

    @property
    def radius(self) -> ArrayLike:
        return self.__radius

    def __len__(self):
        return self.__cluster.__len__()

    def __str__(self):
        return f'feature:\n{self.__feature}\ncluster:\n{self.__cluster}'

    def __cal_center(self) -> ArrayLike:
        feature_in_cls = [self.__feature[self.__cluster == i] for i in range(self.n_clusters)]
        center = list()
        for feature in feature_in_cls:
            c = feature.mean(axis=0) if len(feature) else np.zeros(self.__feature.shape[1])
            center.append(c)
        return np.array(center)
        
    def __cal_radius(self) -> ArrayLike:
        feature_in_cls = [self.__feature[self.__cluster == i] for i in range(self.n_clusters)]
        radius = list()
        for i, feature in enumerate(feature_in_cls):
            r = [np.linalg.norm(x - self.__center[i]) for x in feature]
            r = max(r) if len(r) else 0
            radius.append(r)
        return np.array(radius)


class HierarchyCluster(object):

    def __init__(self, n_clusters: int, affinity: str = 'euclidean',
                 linkage: str = 'ward', n_neighbors: int = 1,
                 random_state: int = 0) -> None:
        self.__cls = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            linkage=linkage
            )
        self.__clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=affinity)
        self.__fitted = False

    def fit(self, X: ModelInput_T) -> ClusterData:
        self.__fitted = True
        y = self.__cls.fit_predict(X)
        self.__clf.fit(X, y)

    def predict(self, X: ModelInput_T) -> ClusterData:
        if not self.__fitted:
            raise RuntimeError('Hierarchy cluster has not been fitted.')
        return self.__clf.predict(X)

    def fit_predict(self, X: ModelInput_T) -> ClusterData:
        self.__fitted = True
        y = self.__cls.fit_predict(X)
        self.__clf.fit(X, y)
        return y