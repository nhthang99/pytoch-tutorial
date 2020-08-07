import pandas as pd
import torch

def load_dataset(data_path):
    """Load dataset from csv file

    Args:
        data_path (str): Path to csv file

    Returns:
        tuple(tensor, tensor): training set
    """
    df = pd.read_csv(data_path, usecols=["x", "y"])

    X_train = torch.tensor(df["x"].values.reshape(-1, 1))
    y_train = torch.tensor(df["y"].values.reshape(-1, 1))

    return X_train, y_train