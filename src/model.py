import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer


def average_rides_last_4_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Averages rides from t-7 days, t-24 days, t-21 days, and t-28 days ago
    """
    df['average_rides_last_4_weeks'] = 0.25 * (
        df[f'rides_previous_{7*24}_hour'] + 
        df[f'rides_previous_{2*7*24}_hour'] + 
        df[f'rides_previous_{3*7*24}_hour'] +
        df[f'rides_previous_{4*7*24}_hour']
    )

    return df

class TemporalFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_ = X.copy()

        # Generate numeric columns from datetime
        X_['hour'] = X_['pickup_hour'].dt.hour
        X_['day_of_week'] = X_['pickup_hour'].dt.day_of_week
        X_['day_of_year'] = X_['pickup_hour'].dt.day_of_week

        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparms) -> Pipeline:

    # sklearn transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False)
    
    # sklearn transform
    add_temporal_features = TemporalFeatureEngineering()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparms)
    )