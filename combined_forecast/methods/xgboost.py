"""
XGBoost regression model
"""
# imports:
import pandas as pd
from typing import List, Dict, Any
from xgboost import XGBRegressor

def run_xgboost(
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        features: List[str],
        params: Dict[str, Any]
    ):
    """
    This function runs the XGBoost regression model.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column.
        features (List[str]): The feature columns.
        params (Dict[str, Any]): The parameters for the model.
            Expected keys include:
                - 'n_estimators': Number of boosting rounds (default: 100).
                - 'max_depth': Maximum depth of a tree (default: 3).
                - 'learning_rate': Boosting learning rate (default: 0.1).
                - 'random_state': Random seed (default: 42).
                - 'objective': Learning objective (default: 'reg:squarederror').

    Returns:
        pd.DataFrame: The testing data with the predictions.
    """
    # Set default parameters if not provided
    if 'n_estimators' not in params:
        params['n_estimators'] = 100
    if 'max_depth' not in params:
        params['max_depth'] = 3
    if 'learning_rate' not in params:
        params['learning_rate'] = 0.1
    if 'random_state' not in params:
        params['random_state'] = 42
    if 'objective' not in params:
        params['objective'] = 'reg:squarederror'

    # Create and fit the model
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=params['random_state'],
        objective=params['objective']
    )
    model.fit(train[features], train[target])
    test[target] = model.predict(test[features])

    # Return the testing data with the predictions
    return test
