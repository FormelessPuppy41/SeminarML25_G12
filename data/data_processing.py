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


if __name__ == "main":
    data = load_data()
    data = preprocess_data(data)
    write_preprocessed_data(data)