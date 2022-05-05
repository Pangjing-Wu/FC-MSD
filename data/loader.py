import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from _typing import ArrayLike, PdData_T


__all__ = ['DataLoader']


class DataLoader(object):
    """Load indicators and sentiments dataset.

    Dataset structure: [dataset_dir] / [indicators | sentiments] / [stock].csv

    Args:
        data_dir: str, direction of indicators and sentiments csv files.
        stock: str, stock code, such as '0001.HK'.
        lexicon: str, sentiment lexicon.
        y_col: str, column name of labels, default: `None`.
        y_offset: int, label offset, ignored when `y_col = None`, default: `0`.
    """
    def __init__(self, data_dir: str, stock:str, lexicon:str, 
                 y_col: str, y_offset: int = 0) -> None:
        indicator = pd.read_csv(os.path.join(data_dir, 'indicators', '%s.csv' % stock),
                                index_col='date', parse_dates=True)
        sentiment = pd.read_csv(os.path.join(data_dir, 'sentiments', lexicon, '%s.csv' % stock), 
                                index_col='date', parse_dates=True)
        self.__data  = pd.merge(indicator, sentiment, how='left', left_index=True, right_index=True)
        self.__data  = self.__data_preprocess(self.__data)
        self.__y_col = y_col
        self.__y_offset = max(y_offset, 0)
        self.__X, self.__y = self.__build()

    @property
    def X(self) -> pd.DataFrame:
        return self.__X.copy()

    @property
    def y(self) -> Union[pd.Series, None]:
        return self.__y.copy()

    def get_X(self, split: float = 1, norm: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get features.
        
        Args:
            split: float, train-test split, between (0,1], default: 1.
            norm: bool, normalize features by z-score of training set.

        Returns:
            tuple of pd.DataFrame of training features and test features.
        """
        self.__check_split(split)
        X = self.X
        i = int(X.shape[0] * split)
        X_train, X_test = X.iloc[:i], X.iloc[i:]
        if norm:
            normer  = StandardScaler().fit(X_train).transform
            X_train = self.__pd_deassign(normer(X_train), X_train)
            X_test  = self.__pd_deassign(normer(X_test), X_test) if len(X_test) else X_test
        return X_train, X_test
    
    def get_y(self, split: float = 1, binarize: bool = False) -> Tuple[pd.Series, pd.Series]:
        """Get labels.
        
        Args:
            split: float, train-test split, between (0,1], default: 1.
            binarize: bool, binarize labels, i.e., `y = 0 if y <= 0 else 1`.

        Returns:
            tuple of pd.Series of training labels and test labels.
        """
        self.__check_split(split)
        i = int(self.y.shape[0] * split)
        y_train, y_test = self.y.iloc[:i], self.y.iloc[i:]
        if binarize:
            y_train = self.__pd_deassign(np.where(y_train.values < 0, 0, 1), y_train)
            y_test  = self.__pd_deassign(np.where(y_test.values < 0, 0, 1), y_test)
        return y_train, y_test

    def __build(self):
        X = self.__data.iloc[:self.__data.shape[0] - self.__y_offset]
        y = self.__data[self.__y_col].iloc[self.__y_offset:]
        y.index = X.index # align date index after offset.
        return X, y

    def __data_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(['open', 'high', 'low', 'adj close'], axis=1)
        data = data.fillna(0)
        return data

    def __check_split(self, split: float) -> float:
        if split <= 0 or split > 1:
            raise ValueError(f"argument 'split' must in range (0, 1], but got {split}.")

    def __pd_deassign(self, value: ArrayLike, proto: PdData_T) -> PdData_T:
        """ Deassign value for pandas DataFrame and Series.

        Args:
            value: array, deassign value.
            proto: pd.DataFrame or pd.Series, deassign prototype.

        Returns:
            pd.DataFrame or pd.Series, deassign variable.
        """
        if type(proto) == pd.DataFrame:
            ret = pd.DataFrame(data=value, columns=proto.columns, index=proto.index)
        elif type(proto) == pd.Series:
            ret = pd.Series(data=value, index=proto.index)
        else:
            raise TypeError('unknown prototype.')
        return ret