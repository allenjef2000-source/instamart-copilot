from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support

from preprocess_lapse import CATEGORY_COLUMNS, NUMERIC_COLUMNS

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def get_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    train = pd.read_csv(OUTPUTS / "lapse_train.csv")
    test = pd.read_csv(OUTPUTS / "lapse_test.csv")

    columns = NUMERIC_COLUMNS + CATEGORY_COLUMNS
    X_test = test[columns]
    y_test = test["lapsed"]

    # Keep every original training row.
    lapsed_rows = train.loc[train["lapsed"].eq(1)]
    not_lapsed_count = int(train["lapsed"].eq(0).sum())

    duplicates_needed = not_lapsed_count - len(lapsed_rows)
    assert duplicates_needed == 216

    # Add only duplicates of lapsed training customers.
    extra_rows = lapsed_rows.sample(
        n=duplicates_needed,
        replace=True,
        random_state=42,
    )

    balanced_train = pd.concat(
        [train, extra_rows],
        ignore_index=True,
    )

    counts = balanced_train["lapsed"].value_counts()

    assert counts.loc[0] == 570
    assert counts.loc[1] == 570
    assert set(balanced_train["customer_id"]).isdisjoint(
        test["customer_id"]
    )

    # Use the previously selected Logistic Regression settings.
    original_model = joblib.load(
        OUTPUTS / "logistic_tuned.joblib"
    )

    # Clone creates an unfitted pipeline with the same settings.
    balanced_model = clone(original_model)
    balanced_model.fit(
        balanced_train[columns],
        balanced_train["lapsed"],
    )

    original_metrics = get_metrics(original_model, X_test, y_test)
    balanced_metrics = get_metrics(balanced_model, X_test, y_test)

    comparison = pd.DataFrame(
        [original_metrics, balanced_metrics],
        index=["Original training", "Balanced training"],
    )
    comparison.index.name = "experiment"

    changes = comparison.loc["Balanced training"] - comparison.loc[
        "Original training"
    ]
    biggest_change = changes.abs().idxmax()

    print("BALANCED TRAINING CLASS COUNTS")
    print(counts.sort_index().to_string())

    print(f"\nTraining duplicates added: {duplicates_needed}")
    print(f"Balanced training rows: {len(balanced_train)}")
    print(f"Unchanged test rows: {len(test)}")

    print("\nLOGISTIC REGRESSION COMPARISON")
    print(comparison.round(4).to_string())

    print("\nMETRIC CHANGES")
    print(changes.round(4).to_string())
    print(f"\nLargest absolute change: {biggest_change}")

    comparison.to_csv(OUTPUTS / "class_balance_results.csv")

    print("\nSaved: outputs/class_balance_results.csv")
    print("CLASS-BALANCE EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
    