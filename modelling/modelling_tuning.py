import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils import estimator_html_repr

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "preprocessing" / "data_preprocessing"

X_TRAIN_PATH = DATA_DIR / "X_train_scaled.json"
Y_TRAIN_PATH = DATA_DIR / "y_train.json"
X_TEST_PATH = DATA_DIR / "X_test_scaled.json"
Y_TEST_PATH = DATA_DIR / "y_test.json"

TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
EXPERIMENT_NAME = "smsl-experiment"

PARAM_GRID = {"C": [0.01, 0.1, 1, 10]}
CV = 5

CV_METRIC_KEYS = [
    "mean_fit_time",
    "std_fit_time",
    "mean_score_time",
    "std_score_time",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]


def classification_metrics(model, X, y, prefix):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        f"{prefix}_accuracy_score": accuracy_score(y, y_pred),
        f"{prefix}_precision_score": precision_score(y, y_pred, average="weighted"),
        f"{prefix}_recall_score": recall_score(y, y_pred, average="weighted"),
        f"{prefix}_f1_score": f1_score(y, y_pred, average="weighted"),
        f"{prefix}_log_loss": log_loss(y, y_proba),
        f"{prefix}_roc_auc": roc_auc_score(y, y_proba[:, 1]),
        f"{prefix}_score": model.score(X, y),
    }


def log_plots(model, X, y, prefix):
    plots = {
        "confusion_matrix": ConfusionMatrixDisplay.from_estimator,
        "roc_curve": RocCurveDisplay.from_estimator,
        "precision_recall_curve": PrecisionRecallDisplay.from_estimator,
    }
    for name, plot in plots.items():
        display = plot(model, X, y)
        mlflow.log_figure(display.figure_, f"{prefix}_{name}.png")
        plt.close(display.figure_)


def log_coefficients_plot(model, feature_names):
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 9))
    coefs.plot.barh(ax=ax)
    ax.set_title("Logistic Regression coefficients")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_coefficients.png")
    plt.close(fig)


def log_child_runs(search):
    results = search.cv_results_
    base_params = search.estimator.get_params()
    for i, candidate in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
            mlflow.log_params({**base_params, **candidate})
            mlflow.log_metrics({key: float(results[key][i]) for key in CV_METRIC_KEYS})
            mlflow.set_tags(estimator_tags(search.estimator))


def estimator_tags(estimator):
    cls = type(estimator)
    return {
        "estimator_name": cls.__name__,
        "estimator_class": f"{cls.__module__}.{cls.__qualname__}",
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train = pd.read_json(X_TRAIN_PATH)
    y_train = pd.read_json(Y_TRAIN_PATH, typ="series")
    X_test = pd.read_json(X_TEST_PATH)
    y_test = pd.read_json(Y_TEST_PATH, typ="series")

    search = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        param_grid=PARAM_GRID,
        cv=CV,
    )

    with mlflow.start_run(run_name="logreg_grid_search"):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        mlflow.log_params(search.get_params(deep=False))
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.set_tags(estimator_tags(search))
        mlflow.log_input(
            mlflow.data.from_pandas(X_train.assign(target=y_train), targets="target"),
            context="train",
        )

        mlflow.log_metric("best_cv_score", search.best_score_)
        mlflow.log_metrics(classification_metrics(search, X_train, y_train, "training"))
        mlflow.log_metrics(classification_metrics(search, X_test, y_test, "test"))

        mlflow.log_text(estimator_html_repr(search), "estimator.html")
        log_plots(search, X_train, y_train, "training")
        with tempfile.TemporaryDirectory() as tmp:
            cv_results_path = Path(tmp) / "cv_results.csv"
            pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)
            mlflow.log_artifact(cv_results_path)

        report = classification_report(y_test, search.predict(X_test), output_dict=True)
        mlflow.log_text(json.dumps(report, indent=2), "test_classification_report.json")

        log_coefficients_plot(best_model, X_train.columns)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        log_child_runs(search)


if __name__ == "__main__":
    main()
