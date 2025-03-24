"""
Lasso regression model
"""
# imports:
import pandas as pd
from typing import List, Dict, Any
from sklearn.linear_model import Lasso

def run_lasso(
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        features: List[str],
        params: Dict[str, Any]
    ):
    """
    This function runs the lasso regression model.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column.
        features (List[str]): The feature columns.
        params (Dict[str, Any]): The parameters for the model. Must contain 'alpha', 'normalize', and 'random_state'. 'alpha' is the regularization strength, 'normalize' is whether to normalize the data, and 'random_state' is the random seed. 'alpha' defaults to 1.0, 'normalize' defaults to False, and 'random_state' defaults to 42 if not provided.

    Returns:
        pd.DataFrame: The testing data with the predictions.
    """
    # Check if the parameters are provided, and set defaults if not
    if 'alpha' not in params:
        params['alpha'] = 1.0
    if 'normalize' not in params:
        params['normalize'] = False
    if 'random_state' not in params:
        params['random_state'] = 42

    # Create and fit the model
    model = Lasso(alpha=params['alpha'], normalize=params['normalize'], random_state=params['random_state'])
    model.fit(train[features], train[target])
    test[target] = model.predict(test[features])

    # Return the testing data with the predictions
    return test