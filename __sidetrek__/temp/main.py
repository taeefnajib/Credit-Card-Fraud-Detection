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

from sidetrek.dataset import build_dataset, load_dataset
from sidetrek.types.dataset import SidetrekDataset, SidetrekIterDataPipe, SidetrekMapDataPipe


@dataclass_json
@dataclass
class Hyperparameters(object):
    test_size: float = 0.25
    random_state: int = 42


hp = Hyperparameters()


def collect_data(ds: SidetrekDataset) -> typing.Tuple[pd.DataFrame, pd.Series]:
    # Load the data
    data = load_dataset(ds=ds, data_type="csv")
    # Create dataframe
    columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud']
    data_list = []
    for item in data:
        data_list.append(item)
    df = pd.DataFrame(data_list, columns=columns)
    df.drop(index=df.index[0], axis=0, inplace=True)
    # Define X and Y
    X = df.drop(columns=["fraud"])
    y = df["fraud"]
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(data=X, columns=columns[:-1])
    return (X, y)


def split_data(
    hp: Hyperparameters, X: pd.DataFrame, y: pd.Series) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Split the data into training and test sets
    return train_test_split(X, y, test_size = hp.test_size, random_state = hp.random_state)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    model = RandomForestClassifier()
    # Fit the model
    return model.fit(X_train, y_train)


# model_new = train_model(X_train=X_train, y_train= y_train, model=model)


# def predict(model: PythonPickledFile, X_test, y_test) -> np.ndarray:
#     # Make predictions for Random Forest on the test set
#     y_pred = model.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     return y_pred


# def run_wf(hp: Hyperparameters) -> RandomForestClassifier:
#     X, y = collect_data(hp.file_name)
#     X_train, X_test, y_train, y_test = split_data(
#         feature=X, target=y, test_size=hp.test_size, random_state=hp.random_state
#     )
#     model = train_model(X_train=X_train, y_train=y_train)
#     return model

# if __name__=="__main__":
#     train_model(X_train = hp.file_name, test_size = hp.test_size, random_state= hp.random_state)
