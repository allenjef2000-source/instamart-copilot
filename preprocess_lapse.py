from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

NUMERIC_COLUMNS = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure",
]

CATEGORY_COLUMNS = [
    "city",
    "gender",
    "favourite_category",
]


def make_preprocessor():
    # Return a fresh preprocessor for later use inside model pipelines.
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            (
                "category",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                CATEGORY_COLUMNS,
            ),
        ]
    )


def main():
    train = pd.read_csv(OUTPUTS / "lapse_train.csv")
    test = pd.read_csv(OUTPUTS / "lapse_test.csv")

    feature_columns = NUMERIC_COLUMNS + CATEGORY_COLUMNS

    # Customer ID and the answer column are excluded from features.
    X_train = train[feature_columns]
    X_test = test[feature_columns]
    y_test = test["lapsed"]

    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    preprocessor = make_preprocessor()

    # Learn scaling and category rules from training rows only.
    transformed_train = preprocessor.fit_transform(X_train)

    # Apply those same rules to test rows without fitting again.
    transformed_test = preprocessor.transform(X_test)

    scaler = preprocessor.named_transformers_["numeric"]
    frequency_index = NUMERIC_COLUMNS.index("frequency")
    learned_mean = scaler.mean_[frequency_index]

    cohort_frequency = pd.concat(
        [train["frequency"], test["frequency"]],
        ignore_index=True,
    ).mean()

    assert np.isclose(learned_mean, train["frequency"].mean())

    print("PREPROCESSING SUMMARY")
    print(f"Training matrix shape: {transformed_train.shape}")
    print(f"Test matrix shape: {transformed_test.shape}")
    print(f"Scaler frequency mean: {learned_mean:.4f}")
    print(f"Full-cohort frequency mean: {cohort_frequency:.4f}")
    print(
        "The means differ because the scaler learns from "
        "the training subset, not the full cohort."
    )

    # Baseline: predict not lapsed for every test customer.
    baseline_predictions = np.zeros(len(y_test), dtype=int)

    accuracy = accuracy_score(y_test, baseline_predictions)
    f1 = f1_score(
        y_test,
        baseline_predictions,
        pos_label=1,
        zero_division=0,
    )

    print("\nBASELINE RESULTS")
    print("Prediction for every customer: not lapsed")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Lapsed-class F1: {f1:.2f}")

    assert np.isclose(accuracy, 143 / 232)
    assert f1 == 0

    print("\nPREPROCESSING CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()