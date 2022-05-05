from typing import Generator, Union

import pandas as pd
from numpy.typing import ArrayLike


__all__ = ['ArrayLike', 'PdData_T', 'Generator_T', 'ModelInput_T']


ArrayLike    = ArrayLike
PdData_T     = Union[pd.DataFrame, pd.Series]
ModelInput_T = Union[pd.DataFrame, ArrayLike]