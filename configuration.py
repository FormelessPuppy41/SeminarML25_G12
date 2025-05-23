from dataclasses import dataclass

import numpy as np


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
        'alpha_grid': np.logspace(-3, 3, 7), 
        'l1_ratio_grid': [0.0],# Only one value: ridge behavior
    }

    lasso_params = {
        'alpha_grid': np.logspace(-3, 3, 7),
        'l1_ratio_grid': [1.0],# Only one value: lasso behavior
    }

    elastic_net_params = {
        'alpha_grid': np.logspace(-3, 3, 7),
        'l1_ratio_grid': [1e-6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # [0.5]
    }

    adaptive_elastic_net_params = {
        'alpha_grid': np.logspace(-3, 3, 7), 
        'l1_ratio_grid': [1e-6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma_grid': [1.0]
    }


    xgboost_params = {
        'n_estimators_grid': [75, 100, 125],
        'max_depth_grid': [3, 5, 7],
        'learning_rate_grid': [0.01, 0.1, 0.2],
        'cv': 5,
        'random_state': 42,
        'objective': 'reg:squarederror'
    }


@dataclass
class ModelSettings:
    """
    Class to store model settings.

    Attributes:
        - target (str): The target column.
        - features (List[str]): The feature columns.
        - forecast_horizon (int): The forecast horizon.
        - rolling_window_days (int): The rolling window size.
        - datetime_col (str): The datetime column.
        - freq (str): The frequency of the data.
    """
    target = 'HR'
    features = ["A1", "A2", "A3", "A4", "A5", "A6"]#, "A7"] # "A7" is used in the price data.
    forecast_horizon = 96 # 96 for 15min, 24 for 1H for hr vs price resp.
    rolling_window_days = 165
    datetime_col = 'datetime'
    freq = '15min' #'15min' or '1H' based on hr vs price resp.
    fit_intercept = False # False for hr data, True for price data
    standard_scaler_with_mean = False # False for hr data, True for price data


@dataclass
class FileNames:
    @dataclass
    class InputFiles:
        """
        Class to store input file names.

        Attributes:
            - data_file (str): The data file.
        """
        params_file = 'params.csv'
        solar_combined_data = 'SolarDataCombined.csv'
        combined_forecasts = 'combined_forecasts.csv'
        flag_matrix = 'flag_matrix.csv'
        real_error_data = 'error_model_combined_forecasts.csv'
        real_error_data2 = 'error_model_combined_forecasts2.csv'
        data_different_group = 'data_ander_groepje.csv'

    @dataclass
    class OutputFiles:
        """
        Class to store output file names.

        Attributes:
            - processed_data (str): The processed data file ready to be used for forecasting.
            - sample_complete_Ty (str): The sample complete Ty file.
        """
        pass

    @dataclass
    class ModelResultFiles:
        """
        Class to store model result file names.

        Attributes:
            - simple_average_forecast (str): The Simple Average forecast file.
            - ridge_forecast (str): The Ridge forecast file.
            - lasso_forecast (str): The Lasso forecast file.
            - elastic_net_forecast (str): The Elastic Net forecast file.
            - adaptive_elastic_net_forecast (str): The Adaptive Elastic Net forecast file.
            - xgboost_forecast (str): The XGBoost forecast file.
        """
        simple_average_forecast = 'simple_average_forecast.csv'
        ridge_forecast = 'ridge_forecast.csv'
        lasso_forecast = 'lasso_forecast.csv'
        elastic_net_forecast = 'elastic_net_forecast.csv'
        tune_elastic_net_forecast = 'tune_elastic_net_forecast.csv'
        adaptive_elastic_net_forecast = 'adaptive_elastic_net_forecast.csv'
        xgboost_forecast = 'xgboost_forecast.csv'

    @dataclass
    class HZ50Files:
        """
        Class to store 50Hertz file names.

        Attributes:
            - solar (str): The combined solar data file.
            - wind (str): The combined wind data file.
        """
        solar = 'Solar_Combined.csv'
        wind = 'Windenergie_Combined.csv'


    input_files = InputFiles()
    output_files = OutputFiles()
    model_result_files = ModelResultFiles()
    hz50_files = HZ50Files()
