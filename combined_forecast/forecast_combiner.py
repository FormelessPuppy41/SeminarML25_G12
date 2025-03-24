"""
Forecast combiner module
"""

# imports:
import pandas as pd
from typing import List, Dict, Any
from combined_forecast.methods import run_adaptive_elastic_net, run_elastic_net, run_lasso, run_ridge

class ForecastCombiner:
    """
    Forecast combiner class
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, features: List[str]):
        self.train = train
        self.test = test
        self.target = target
        self.features = features

    def combine_ridge(self, params: Dict[str, Any]):
        """
        This function combines the ridge regression model.

        Args:
            params (Dict[str, Any]): The parameters for the model. Must contain 'alpha', 'normalize', and 'random_state'. 'alpha' is the regularization strength, 'normalize' is whether to normalize the data, and 'random_state' is the random seed. 'alpha' defaults to 1.0, 'normalize' defaults to False, and 'random_state' defaults to 42 if not provided.

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        return run_ridge(self.train, self.test, self.target, self.features, params)

    def combine_lasso(self, params: Dict[str, Any]):
        """
        This function combines the lasso regression model.

        Args:
            params (Dict[str, Any]): The parameters for the model. Must contain 'alpha', 'normalize', and 'random_state'. 'alpha' is the regularization strength, 'normalize' is whether to normalize the data, and 'random_state' is the random seed. 'alpha' defaults to 1.0, 'normalize' defaults to False, and 'random_state' defaults to 42 if not provided.

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        return run_lasso(self.train, self.test, self.target, self.features, params)
    
    def combine_elastic_net(self, params: Dict[str, Any]):
        """
        This function combines the elastic net regression model.

        Args:
            params (Dict[str, Any]): The parameters for the model. Must contain 'alpha', 'l1_ratio', 'normalize', and 'random_state'. 'alpha' is the regularization strength, 'l1_ratio' is the L1 ratio, 'normalize' is whether to normalize the data, and 'random_state' is the random seed. 'alpha' defaults to 1.0, 'l1_ratio' defaults to 0.5, 'normalize' defaults to False, and 'random_state' defaults to 42 if not provided.

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        return run_elastic_net(self.train, self.test, self.target, self.features, params)
    
    def combine_adaptive_elastic_net(self, params: Dict[str, Any]):
        """
        This function combines the adaptive elastic net regression model.

        Args:
            params (Dict[str, Any]): The parameters for the model. Must contain 'alpha', 'l1_ratio', 'normalize', 'random_state', 'cv', 'n_jobs', and 'verbose'. 'alpha' is the regularization strength, 'l1_ratio' is the L1 ratio, 'normalize' is whether to normalize the data, 'random_state' is the random seed, 'cv' is the number of cross-validation folds, 'n_jobs' is the number of jobs to run in parallel, and 'verbose' is the verbosity level. 'alpha' defaults to 1.0, 'l1_ratio' defaults to 0.5, 'normalize' defaults to False, 'random_state' defaults to 42, 'cv' defaults to 5, 'n_jobs' defaults to -1, and 'verbose' defaults to 0 if not provided.

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        return run_adaptive_elastic_net(self.train, self.test, self.target, self.features, params)