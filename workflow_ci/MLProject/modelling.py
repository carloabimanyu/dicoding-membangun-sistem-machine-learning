import argparse
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "preprocessing" / "data_preprocessing"

X_TRAIN_PATH = DATA_DIR / "X_train_scaled.json"
Y_TRAIN_PATH = DATA_DIR / "y_train.json"

ARTIFACT_DIR = BASE_DIR.parent / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "model"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.sklearn.autolog()

    X_train = pd.read_json(X_TRAIN_PATH)
    y_train = pd.read_json(Y_TRAIN_PATH, typ="series")

    with mlflow.start_run() as run:
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)
        ARTIFACT_DIR.mkdir(parents=True)
        mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=str(ARTIFACT_DIR))
        mlflow.sklearn.save_model(
            model,
            MODEL_DIR,
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
        )
        (ARTIFACT_DIR / "run_id.txt").write_text(run.info.run_id + "\n")


if __name__ == "__main__":
    main()
