"""
Data loader for the datasets
"""
import pandas as pd
from configuration import FileNames

file_names = FileNames()


class DataLoader: 
    """
    Data loader for the datasets. Used to load input, output, 50hertz and model results data.
    """
    def __init__(self):
        self.path = 'data/data_files'

    def load_input_data(self, file_name: str):
        """
        This function loads the input data from a CSV file.

        Args:
            file_name (str): The file name for the input data.

        Returns:
            pd.DataFrame: The input data.
        """
        path = f'{self.path}/input_files/{file_name}'
        return self._load_data(path)
    
    def load_output_data(self, file_name: str):
        """
        This function loads the output data from a CSV file.

        Args:
            file_name (str): The file name for the output data.

        Returns:
            pd.DataFrame: The output data.
        """
        path = f'{self.path}/output_files/{file_name}'
        return self._load_data(path)
    
    def load_model_results(self, file_name: str):
        """
        This function loads the model results from a CSV file.

        Args:
            file_name (str): The file name for the model results.

        Returns:
            pd.DataFrame: The model results.
        """
        path = f'{self.path}/model_results/{file_name}'
        return self._load_data(path)
    
    def load_50hertz_data(self, file_name: str):
        """
        This function loads the 50Hertz data from a CSV file.

        Args:
            file_name (str): The file name for the 50Hertz data.

        Returns:
            pd.DataFrame: The 50Hertz data.
        """
        path = f'{self.path}/50hertz_files/{file_name}'
        return self._load_data(path)

    def load_kaggle_data(self, file_name: str = 'energy.csv'):
        """
        This function loads the Kaggle data from a CSV file.

        Args:
            file_name (str): The file name for the Kaggle data. Either 'energy.csv' or 'weather.csv'. 

        Returns:
            pd.DataFrame: The Kaggle data.
        """
        # check if the file name is valid
        if file_name not in [file_names.kaggle_files.energy_data_file, file_names.kaggle_files.weather_data_file]:
            raise ValueError(f'File name not valid. Use either {file_names.kaggle_files.energy_data_file} or {file_names.kaggle_files.weather_data_file}')
       
        path = f'{self.path}/kaggle_data/{file_name}'
        return self._load_data(path)
    
    def _load_data(self, path: str):
        """
        This function loads the data from a CSV file.

        Args:
            path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The data.
        """
        # check the file type to use either csv or xlsx
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.xlsx'):
            return pd.read_excel(path)
        else:
            raise ValueError('File type not supported')