from pathlib import Path

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "preprocessing" / "data_preprocessing"

X_TRAIN_PATH = DATA_DIR / "X_train_scaled.json"
Y_TRAIN_PATH = DATA_DIR / "y_train.json"

TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
EXPERIMENT_NAME = "smsl-experiment"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow.sklearn.autolog()

    X_train = pd.read_json(X_TRAIN_PATH)
    y_train = pd.read_json(Y_TRAIN_PATH, typ="series")

    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)

if __name__ == "__main__":
    main()