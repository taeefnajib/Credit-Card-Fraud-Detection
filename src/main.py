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
from sidetrek.dataset import load_dataset
from sidetrek.types.dataset import SidetrekDataset


@dataclass_json
@dataclass
class Hyperparameters(object):
    test_size: float = 0.25
    random_state: int = 42


hp = Hyperparameters()


def load_data(ds: SidetrekDataset) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Load the dataset
    csv_data = load_dataset(ds, data_type="csv")

    # Create df
    csv_data_dict = {}
    for i, row in enumerate(csv_data):
        csv_data_dict[i] = row

    df = pd.DataFrame.from_dict(csv_data_dict, orient="index")

    # Add columns
    df.columns = df.iloc[0]
    df.drop(index=df.index[0], axis=0, inplace=True)
    print(df.head(5))

    # Separate the data into 
    y = df.loc[:,"fraud"].to_numpy(dtype=float)
    X = df.drop(["fraud"], axis=1)

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return (X, y)


def split_data(
    feature: np.ndarray, target: np.ndarray, test_size: float, random_state: int
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Split the data into training and test sets
    return train_test_split(feature, target, test_size = test_size, random_state = random_state)


def train_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    model = RandomForestClassifier()
    # Fit the model
    return model.fit(X_train, y_train)


def run_wf(hp: Hyperparameters, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = split_data(
        feature=X, target=y, test_size=hp.test_size, random_state=hp.random_state
    )
    model = train_model(X_train=X_train, y_train=y_train)
    return model


if __name__=="__main__":
    run_wf(file_name = hp.file_name, test_size = hp.test_size, random_state= hp.random_state)