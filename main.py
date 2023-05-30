# Import necessary libraries
import pandas as pd
import numpy as np
import typing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from flytekit.types.file import PythonPickledFile
import pathlib

from sidetrek.dataset import load_dataset, build_dataset
from sidetrek.types.dataset import SidetrekDataset, SidetrekIterDataPipe, SidetrekMapDataPipe


@dataclass_json
@dataclass
class Hyperparameters(object):
    test_size: float = 0.25
    random_state: int = 42


hp = Hyperparameters()


def collect_data(ds: SidetrekDataset) -> typing.Tuple[np.ndarray, pd.Series]:
    # Load the data
    data = load_dataset(ds=ds, data_type="csv")
    # Create dataframe
    columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud']
    data_dict = {}
    for i,v in enumerate(data):
        data_dict[i]=v
    df = pd.DataFrame.from_dict(data_dict, columns=columns, orient="index")
    df.drop(index=df.index[0], axis=0, inplace=True)
    # Define X and Y
    X = df.drop(columns=["fraud"])
    y = df["fraud"]
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return (X, y)


def split_data(
    hp: Hyperparameters, X: np.ndarray, y: pd.Series) -> typing.Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    # Split the data into training and test sets
    return train_test_split(X, y, test_size = hp.test_size, random_state = hp.random_state)


def train_model(
    X_train: np.ndarray, y_train: pd.Series
) -> RandomForestClassifier:
    model = RandomForestClassifier()
    # Fit the model
    return model.fit(X_train, y_train)

