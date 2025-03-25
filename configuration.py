from dataclasses import dataclass


@dataclass
class ModelParameters:
    """
    Class to store model parameters. 

    Attributes:
        - ridge_params (Dict[str, Any]): Parameters for Ridge regression.
            - alpha: regularization strength 
        - lasso_params (Dict[str, Any]): Parameters for Lasso regression.
            - alpha: regularization strength 
        - elastic_net_params (Dict[str, Any]): Parameters for Elastic Net regression.
            - alpha: regularization strength 
            - l1_ratio: L1 ratio 
        - adaptive_elastic_net_params (Dict[str, Any]): Parameters for Adaptive Elastic Net
            - elasticnet__alpha: regularization strength 
            - elasticnet__l1_ratio: L1 ratio 
            - cv: number of cross-validation folds 
            - n_jobs: number of parallel jobs 
            - verbose: verbosity level 
    """
    ridge_params = {
        'alpha': 0.5
    }

    lasso_params = {
        'alpha': 0.5
    }

    elastic_net_params = {
        'alpha': 0.5,
        'l1_ratio': 0.5
    }

    adaptive_elastic_net_params = {
        'elasticnet__alpha': [0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.0, 0.5, 1.0],
        'cv': 5,
        'n_jobs': -1,
        'verbose': 1
    }
