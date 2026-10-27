from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from preprocess_lapse import (
    CATEGORY_COLUMNS,
    NUMERIC_COLUMNS,
    make_preprocessor,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    train = pd.read_csv(OUTPUTS / "lapse_train.csv")

    X_train = train[NUMERIC_COLUMNS + CATEGORY_COLUMNS]
    y_train = train["lapsed"]

    assert not X_train.isna().any().any()

    # Preprocessing is fitted separately inside each training fold.
    logistic_pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    forest_pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ])

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid={
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        scoring="f1",
        cv=5,
        n_jobs=1,
        verbose=1,
    )

    forest_search = GridSearchCV(
        estimator=forest_pipeline,
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        scoring="f1",
        cv=5,
        n_jobs=1,
        verbose=1,
    )

    print("TUNING LOGISTIC REGRESSION")
    logistic_search.fit(X_train, y_train)

    print("\nTUNING RANDOM FOREST")
    forest_search.fit(X_train, y_train)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # GridSearchCV refits the best pipeline on all training rows.
    joblib.dump(
        logistic_search.best_estimator_,
        OUTPUTS / "logistic_tuned.joblib",
    )

    joblib.dump(
        forest_search.best_estimator_,
        OUTPUTS / "forest_tuned.joblib",
    )

    # Save every parameter combination and its CV results.
    pd.DataFrame(logistic_search.cv_results_).to_csv(
        OUTPUTS / "logistic_cv_results.csv",
        index=False,
    )

    pd.DataFrame(forest_search.cv_results_).to_csv(
        OUTPUTS / "forest_cv_results.csv",
        index=False,
    )

    summary = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "best_parameters": str(logistic_search.best_params_),
            "best_cv_f1": logistic_search.best_score_,
        },
        {
            "model": "Random Forest",
            "best_parameters": str(forest_search.best_params_),
            "best_cv_f1": forest_search.best_score_,
        },
    ])

    summary.to_csv(OUTPUTS / "model_tuning_summary.csv", index=False)

    print("\nTUNING RESULTS")
    print(summary.to_string(index=False))

    print("\nBoth models were tuned using training data only.")
    print("The test dataset was not loaded or used.")
    print("TUNING COMPLETED")


if __name__ == "__main__":
    main()