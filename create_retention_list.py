from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess_lapse import CATEGORY_COLUMNS, NUMERIC_COLUMNS

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    test = pd.read_csv(OUTPUTS / "lapse_test.csv")
    segments = pd.read_csv(OUTPUTS / "customer_segments.csv")
    model = joblib.load(OUTPUTS / "lapse_model.joblib")

    X_test = test[NUMERIC_COLUMNS + CATEGORY_COLUMNS]

    # Find the probability column for lapsed = 1.
    positive_column = np.flatnonzero(model.classes_ == 1).item()
    probabilities = model.predict_proba(X_test)[:, positive_column]

    predictions = test[["customer_id", "lapsed"]].copy()
    predictions["lapse_probability"] = probabilities

    # Read Premium flags safely whether stored as text or booleans.
    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    predictions = predictions.merge(
        segments[["customer_id", "is_premium"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    premium = predictions.loc[
        predictions["is_premium"].eq(True)
    ].copy()

    retention = (
        premium
        .sort_values(
            ["lapse_probability", "customer_id"],
            ascending=[False, True],
        )
        .head(10)
        .rename(columns={"lapsed": "actually_lapsed"})
    )

    retention = retention[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]

    assert len(retention) == min(10, len(premium))
    assert retention["lapse_probability"].between(0, 1).all()
    assert retention["lapse_probability"].is_monotonic_decreasing
    assert retention["customer_id"].isin(test["customer_id"]).all()

    retention.to_csv(
        OUTPUTS / "retention_list.csv",
        index=False,
    )

    print("PREMIUM CUSTOMER RETENTION LIST")
    print(retention.round(4).to_string(index=False))

    print(f"\nPremium customers in test set: {len(premium)}")
    print(f"Customers listed: {len(retention)}")
    print(
        "Actually lapsed among listed customers: "
        f"{int(retention['actually_lapsed'].sum())}"
    )

    print(
        "\nLIMITATION: The Premium flag is a post-hoc whole-window "
        "label that includes September behaviour, so this retention "
        "list is illustrative and not a leakage-free evaluation."
    )

    print("\nSaved: outputs/retention_list.csv")
    print("RETENTION LIST CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()