import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def split_data(data: pd.DataFrame, parameters: Dict)-> Tuple:
    """
    Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    
    """
    params = parameters
    
    X = data[params["features"]]
    y = data[params["target"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=params["train_size"], test_size=params["test_size"], random_state=params["random_state"]
    )
    
    return X_train, X_test, y_train, y_test
	

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor