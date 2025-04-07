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
        'alpha': 0.5, 
        'alpha_grid': [0.1, 1.0, 10.0],
    }

    lasso_params = {
        'alpha': 0.5,
        'alpha_grid': [0.1, 1.0, 10.0]
    }

    elastic_net_params = {
        'alpha': 0.5,
        'l1_ratio': 0.5,
        'alpha_grid': [0.1, 1.0, 10.0],
    }

    adaptive_elastic_net_params = {
        'elasticnet__alpha': [0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.0, 0.5, 1.0],
        'cv': 5,
        'n_jobs': -1,
        'verbose': 1
    }

    xgboost_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
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
    target = 'generation solar'
    features = ["A1", "A2", "A3", "A4", "A5", "A6"]
    forecast_horizon = 24
    rolling_window_days = 165
    datetime_col = 'datetime'
    freq = '15min'


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
            - ridge_forecast (str): The Ridge forecast file.
            - lasso_forecast (str): The Lasso forecast file.
            - elastic_net_forecast (str): The Elastic Net forecast file.
            - adaptive_elastic_net_forecast (str): The Adaptive Elastic Net forecast file.
        """
        ridge_forecast = 'ridge_forecast.csv'
        lasso_forecast = 'lasso_forecast.csv'
        elastic_net_forecast = 'elastic_net_forecast.csv'
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
