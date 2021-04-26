from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import (
    EfficientFCParameters,
    MinimalFCParameters,
    ComprehensiveFCParameters,
)
from tsfresh import (
    extract_features,
    select_features,
    extract_relevant_features,
)
import re
import os
from datetime import datetime
from joblib import dump


class BuildRollingWindows(BaseEstimator, TransformerMixin):
    """
    Takes in a Pandas Dataframe, and creates a rolled version of it building multiple successive windows of data.
    Each window will contain data up to and including a final index. Shift target y ahead by one time step to honour temporal dependence.
    Wraps `tsfresh` `roll_time_series`, all arguments passed through
    """

    def __init__(
        self,
        column_id,
        column_sort=None,
        column_kind=None,
        rolling_direction=1,
        max_timeshift=None,
        min_timeshift=0,
        chunksize=None,
        n_jobs=4,
        show_warnings=False,
        disable_progressbar=False,
        distributor=None,
    ):

        self.column_id = column_id
        self.column_sort = column_sort
        self.column_kind = column_kind
        self.rolling_direction = rolling_direction
        self.max_timeshift = max_timeshift
        self.min_timeshift = min_timeshift
        self.chunksize = chunksize
        self.n_jobs = n_jobs
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.distributor = distributor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        windows = roll_time_series(
            X,
            self.column_id,
            self.column_sort,
            self.column_kind,
            self.rolling_direction,
            self.max_timeshift,
            self.min_timeshift,
            self.chunksize,
            self.n_jobs,
            self.show_warnings,
            self.disable_progressbar,
            self.distributor,
        )

        return windows


class AddEma(BaseEstimator, TransformerMixin):
    """
    Takes in a time series Pandas dataframe, adds exponential moving averages for each column in input dataframe for each amount of days..
    Exponential moving average parametrized in terms of smoothing = 2/(days + 1).
    """

    def __init__(self, periods=[12, 25, 50, 100], except_cols=["id"]):
        self.periods = periods
        self.except_cols = except_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_out = X.copy()
        for d in self.periods:
            for c in set(X.columns).difference(set(self.except_cols)):
                X_out[f"{c}_ema_{d}_period"] = X[c].ewm(span=d).mean()

        return X_out


class ExtractTSFeatures(BaseEstimator, TransformerMixin):
    """
    Quick wrapper to allow for properly aligned time series X,y to use `tsfresh` feature augmentation.
    Expects Pandas DataFrames
    ASSUMES X,y sorted from oldest to newest already!
    """

    def __init__(
        self,
        column_id,
        column_sort,
        chunk_size=100,
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0,
    ):
        """
        Parameters
        ----------
        column_id: str
            Which column to identify different groups of time series. If only one present, add a dummy column of 1's and pass the name here
        chunk_size: int
            Chunk size for distributing calculations to processes in multi processing
        default_fc_parameters: tsfresh.setting
            A dictionary setting parameters for features desired in Tsfresh feature calculator.
            Built in options include MinimalFCParameters(), EfficientFCParameters()
        """
        self.column_id = column_id
        self.column_sort = column_sort
        self.chunk_size = chunk_size
        self.default_fc_parameters = default_fc_parameters
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feats = extract_features(
            X,
            column_id=self.column_id,
            column_sort=self.column_sort,
            chunksize=self.chunk_size,
            default_fc_parameters=self.default_fc_parameters,
            n_jobs=self.n_jobs,
        )

        # Rename columns to allow use with LightGBM, doesn't like "-", "."
        feats = feats.rename(
            columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x)
        )

        # Grab the datetime index out of tuple multi index that tsfresh uses
        return feats.set_index(feats.index.map(lambda x: x[1]), drop=True)


class SelectTSFeatures(BaseEstimator, TransformerMixin):
    """
    Quick wrapper to allow for properly aligned time series X,y to use `tsfresh` feature augmentation.
    Expects Pandas DataFrames
    ASSUMES X,y sorted from oldest to newest already and aligned on index!
    """

    def __init__(self, n_jobs):
        print(" Selecting Relevant Features.....")
        self.n_jobs = n_jobs
        return None

    def fit(self, X, y):
        self.selected_features = select_features(
            X, y, n_jobs=self.n_jobs
        ).columns
        print(
            f"{len(self.selected_features)} features found significant out of {X.shape[1]} possible."
        )
        return self

    def transform(self, X, y=None):
        return X.loc[
            :, list(set(self.selected_features).intersection(set(X.columns)))
        ]


class RemoveNACols(BaseEstimator, TransformerMixin):
    """
    Transformer to remove columns with NA's in X.
    """

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.na_cols = X.isna().any()
        return self

    def transform(self, X, y=None):
        return X.loc[:, ~self.na_cols]


class RemoveZeroCols(BaseEstimator, TransformerMixin):
    """
    Transformer to remove columns with all 0's in X.
    """

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.non_zero_cols = (X != 0).any(axis=0)
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.non_zero_cols]


def snapshot_model(model, path):
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    dump(model, os.path.join(path, f"{date_time}.joblib"))