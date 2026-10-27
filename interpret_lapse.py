from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess_lapse import CATEGORY_COLUMNS, NUMERIC_COLUMNS

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    forest = joblib.load(OUTPUTS / "forest_tuned.joblib")
    logistic = joblib.load(OUTPUTS / "logistic_tuned.joblib")

    # Random Forest feature importance.
    forest_names = forest.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance = pd.DataFrame({
        "feature": forest_names,
        "importance": forest.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)

    print("RANDOM FOREST — TOP FIVE FEATURES")
    print(importance.head(5).to_string(index=False))

    importance.to_csv(
        OUTPUTS / "forest_feature_importance.csv",
        index=False,
    )

    # Logistic Regression coefficients.
    logistic_names = logistic.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    coefficients = pd.Series(
        logistic.named_steps["model"].coef_[0],
        index=logistic_names,
    )

    print("\nLOGISTIC REGRESSION — THREE COEFFICIENTS")

    for feature in ["recency", "frequency", "monetary"]:
        coefficient = coefficients[f"numeric__{feature}"]
        sign = "positive" if coefficient > 0 else (
            "negative" if coefficient < 0 else "zero"
        )

        print(f"{feature}: {coefficient:.4f} ({sign})")

    coefficients.rename("coefficient").to_csv(
        OUTPUTS / "logistic_coefficients.csv",
    )

    # Save the selected pipeline, including its preprocessing.
    test = pd.read_csv(OUTPUTS / "lapse_test.csv")
    X_test = test[NUMERIC_COLUMNS + CATEGORY_COLUMNS]

    original_predictions = forest.predict(X_test)

    model_path = OUTPUTS / "lapse_model.joblib"
    joblib.dump(forest, model_path)

    reloaded_model = joblib.load(model_path)
    reloaded_predictions = reloaded_model.predict(X_test)

    assert np.array_equal(
        original_predictions,
        reloaded_predictions,
    ), "Reloaded predictions differ"

    print("\nSelected model: Random Forest")
    print("Saved: outputs/lapse_model.joblib")
    print("Reloaded predictions match exactly.")
    print("MODEL PERSISTENCE CHECKPOINT PASSED")


if __name__ == "__main__":
    main()