"""
This file contains the functions to process the data.
"""

import pandas as pd
import numpy as np


def load_data():
    """
    Load the data from the given path.
    """
    data = pd.read_csv("data/data_files/data.csv")
    return data

def write_preprocessed_data(data: pd.DataFrame):
    """
    Write the preprocessed data to a file.

    Args:
        data (pd.DataFrame): The preprocessed data.
    """
    data.to_csv("data/data_files/preprocessed_data.csv")

def preprocess_data(data: pd.DataFrame):
    """
    Preprocess the data.

    Args:
        data (pd.DataFrame): The data to be preprocessed.

    Returns:
        None
    """
    raise NotImplementedError("This function is not implemented yet.")
    return data


def explore_data(data:pd.DataFrame, df_name: str = None):
    """
    Explore the data.

    Args:
        data (pd.DataFrame): The data to be explored.
        df_name (str): The name of the dataframe. Default is None.

    """
    print("\nData Exploration:")
    if df_name is not None:
        print(f"Dataframe Name: {df_name}")
    print("\nData Shape:")
    print(data.shape)
    print("\nData Info:")
    print(data.info())
    print("\nData Description:")
    print(data.describe())
    print("\nData Head:")
    print(data.head())
    print('\n\n Exploration: Finished! \n\n')


if __name__ == "main":
    data = load_data()
    data = preprocess_data(data)
    write_preprocessed_data(data)