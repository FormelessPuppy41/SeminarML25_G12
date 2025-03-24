"""
Elastic Net Regression
"""
# imports:
import pandas as pd
from typing import List, Dict, Any
from sklearn.linear_model import ElasticNet

def run_elastic_net(
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        features: List[str],
        params: Dict[str, Any]
    ):
    """
    This function runs the elastic net regression model.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column.
        features (List[str]): The feature columns.
        params (Dict[str, Any]): The parameters for the model. Must contain 'alpha', 'l1_ratio', 'normalize', and 'random_state'. 'alpha' is the regularization strength, 'l1_ratio' is the L1 ratio, 'normalize' is whether to normalize the data, and 'random_state' is the random seed. 'alpha' defaults to 1.0, 'l1_ratio' defaults to 0.5, 'normalize' defaults to False, and 'random_state' defaults to 42 if not provided.

    Returns:
        pd.DataFrame: The testing data with the predictions
    """
    # Check if the parameters are provided, and set defaults if not
    if 'alpha' not in params:
        params['alpha'] = 1.0
    if 'l1_ratio' not in params:
        params['l1_ratio'] = 0.5
    if 'normalize' not in params:
        params['normalize'] = False
    if 'random_state' not in params:
        params['random_state'] = 42

    # Create and fit the model
    model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], normalize=params['normalize'], random_state=params['random_state'])
    model.fit(train[features], train[target])
    test[target] = model.predict(test[features])

    # Return the testing data with the predictions
    return test
