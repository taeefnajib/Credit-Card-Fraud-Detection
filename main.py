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


@dataclass_json
@dataclass
class Hyperparameters(object):
    file_name: str = "card_transdata.csv"
    test_size: float = 0.25
    random_state: int = 42


hp = Hyperparameters()


def collect_data(filename: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Load the data
    df = pd.read_csv(filename)
    # Define X and Y
    X = df.drop(["fraud"], axis=1)
    y = df["fraud"]
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return (X, y)


def split_data(
    feature: np.ndarray, target: np.ndarray, test_size: float, random_state: int
) -> typing.Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    # Split the data into training and test sets
    return train_test_split(feature, target, test_size = test_size, random_state = random_state)


def train_model(
    X_train: np.ndarray, y_train: pd.Series
) -> RandomForestClassifier:
    model = RandomForestClassifier()
    # Fit the model
    return model.fit(X_train, y_train)


# model_new = train_model(X_train=X_train, y_train= y_train, model=model)


def predict(model: PythonPickledFile, X_test, y_test) -> np.ndarray:
    # Make predictions for Random Forest on the test set
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return y_pred


def run_wf(file_name: str, test_size: float, random_state: int) -> np.ndarray:
    X, y = collect_data(file_name)
    X_train, X_test, y_train, y_test = split_data(
        feature=X, target=y, test_size=test_size, random_state=random_state
    )
    y_pred = predict(
        train_model(X_train=X_train, y_train=y_train), X_test=X_test, y_test=y_test
    )
    print("Model training complete. Prediction is returned")
    return y_pred

if __name__=="__main__":
    run_wf(file_name = hp.file_name, test_size = hp.test_size, random_state= hp.random_state)
