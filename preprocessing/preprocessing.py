import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.3
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data_preprocessing"))

X_TRAIN_PATH = OUTPUT_DIR / "X_train_scaled.json"
X_TEST_PATH = OUTPUT_DIR / "X_test_scaled.json"
Y_TRAIN_PATH = OUTPUT_DIR / "y_train.json"
Y_TEST_PATH = OUTPUT_DIR / "y_test.json"

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_json(X_TRAIN_PATH, orient="records", indent=2)
    X_test_scaled.to_json(X_TEST_PATH, orient="records", indent=2)
    y_train.to_json(Y_TRAIN_PATH, orient="records", indent=2)
    y_test.to_json(Y_TEST_PATH, orient="records", indent=2)