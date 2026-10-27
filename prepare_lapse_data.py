from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"


def main():
    orders = pd.read_csv(CLEAN / "orders.csv")
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    cutoff = pd.Timestamp("2025-09-01")
    september_end = pd.Timestamp("2025-10-01")

    # Features use only orders before September.
    history = orders.loc[orders["order_ts"] < cutoff].copy()
    assert history["order_ts"].lt(cutoff).all()

    completed = history.loc[history["status"].eq("completed")].copy()

    # One row per customer with a completed order before the cutoff.
    features = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )

    features["recency"] = (
        cutoff - features["last_order_ts"]
    ).dt.days

    features = features.drop(columns="last_order_ts")

    # Cancellation rate uses all final-status orders before September.
    final_orders = history.loc[
        history["status"].isin(["completed", "cancelled"])
    ].copy()

    final_orders["is_cancelled"] = final_orders["status"].eq("cancelled")

    cancellation_rate = final_orders.groupby("customer_id")[
        "is_cancelled"
    ].mean()

    features["cancellation_rate"] = cancellation_rate

    # Favourite category uses completed orders before September.
    completed = completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        validate="many_to_one",
    )

    # If categories tie, choose the alphabetically first category.
    favourite_category = completed.groupby("customer_id")["category"].agg(
        lambda values: values.mode().iloc[0]
    )

    features["favourite_category"] = favourite_category

    features = features.reset_index().merge(
        customers[
            ["customer_id", "city", "age", "gender", "signup_date"]
        ],
        on="customer_id",
        validate="one_to_one",
    )

    features["tenure"] = (
        cutoff - features["signup_date"]
    ).dt.days

    features = features.drop(columns="signup_date")

    # September is used only to create the target label.
    september_orders = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(cutoff)
        & orders["order_ts"].lt(september_end)
    ]

    september_customers = set(september_orders["customer_id"])

    features["lapsed"] = (
        ~features["customer_id"].isin(september_customers)
    ).astype(int)

    # Whole-window customer segment labels are deliberately excluded.
    assert "segment_id" not in features.columns
    assert "segment_name" not in features.columns
    assert "is_premium" not in features.columns
    assert not features.isna().any().any(), "Missing values remain"

    # Split before fitting any encoder, scaler or model.
    train, test = train_test_split(
        features,
        test_size=0.2,
        stratify=features["lapsed"],
        random_state=42,
    )

    assert len(features) == 1156, "Expected 1156 customers"
    assert features["lapsed"].sum() == 443, "Expected 443 lapsed customers"
    assert len(train) == 924, "Expected 924 training rows"
    assert len(test) == 232, "Expected 232 test rows"
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    features.to_csv(OUTPUTS / "lapse_features.csv", index=False)
    train.to_csv(OUTPUTS / "lapse_train.csv", index=False)
    test.to_csv(OUTPUTS / "lapse_test.csv", index=False)

    print("LAPSE DATA SUMMARY")
    print(f"Eligible customers: {len(features)}")
    print(f"Lapsed customers: {features['lapsed'].sum()}")
    print(f"Lapsed percentage: {features['lapsed'].mean() * 100:.2f}%")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    print("\nTraining class counts:")
    print(train["lapsed"].value_counts().sort_index().to_string())

    print("\nTest class counts:")
    print(test["lapsed"].value_counts().sort_index().to_string())

    print("\nMissing values: 0")
    print("No September orders were used as features.")
    print("Customer segment labels were not used as features.")
    print("No encoder, scaler or model has been fitted yet.")

    print("\nSaved: outputs/lapse_features.csv")
    print("Saved: outputs/lapse_train.csv")
    print("Saved: outputs/lapse_test.csv")
    print("LAPSE DATA CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()