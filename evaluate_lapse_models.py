from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from preprocess_lapse import CATEGORY_COLUMNS, NUMERIC_COLUMNS

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    test = pd.read_csv(OUTPUTS / "lapse_test.csv")

    X_test = test[NUMERIC_COLUMNS + CATEGORY_COLUMNS]
    y_test = test["lapsed"]

    models = {
        "Logistic Regression": joblib.load(
            OUTPUTS / "logistic_tuned.joblib"
        ),
        "Random Forest": joblib.load(
            OUTPUTS / "forest_tuned.joblib"
        ),
    }

    results = []

    for name, model in models.items():
        # Predict only: do not fit anything on the test data.
        predictions = model.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrix.ravel()

        print(f"\n{name.upper()}")
        print("Confusion matrix: rows = actual, columns = predicted")

        labelled_matrix = pd.DataFrame(
            matrix,
            index=["Actual not lapsed", "Actual lapsed"],
            columns=["Predicted not lapsed", "Predicted lapsed"],
        )

        print(labelled_matrix.to_string())

        print(f"\nLapsed-class precision: {precision:.4f}")
        print(f"Lapsed-class recall: {recall:.4f}")
        print(f"Lapsed-class F1: {f1:.4f}")

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUTS / "model_test_results.csv", index=False)

    print("\nTEST RESULTS SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nBaseline lapsed-class F1: 0.0000")
    print("Saved: outputs/model_test_results.csv")
    print("TEST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()