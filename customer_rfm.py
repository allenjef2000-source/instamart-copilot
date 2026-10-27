from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def main():
    orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    # Customer behaviour uses completed orders only.
    completed = orders.loc[orders["status"].eq("completed")].copy()

    snapshot = pd.Timestamp("2025-10-01 00:00:00")

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )

    # .dt.days gives the number of whole days.
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

    rfm = rfm[["recency", "frequency", "monetary"]].reset_index()

    # Check against the assignment's reference values.
    assert len(rfm) == 1175, "Expected 1175 customers"
    assert rfm["recency"].median() == 19, "Incorrect median recency"
    assert rfm["recency"].min() == 0, "Incorrect minimum recency"
    assert rfm["recency"].max() == 180, "Incorrect maximum recency"
    assert round(rfm["frequency"].mean(), 2) == 9.46, \
        "Incorrect mean frequency"
    assert rfm["frequency"].max() == 33, "Incorrect maximum frequency"
    assert round(rfm["monetary"].mean(), 2) == 2552.06, \
        "Incorrect mean monetary value"
    assert abs(rfm["monetary"].max() - 14766.55) < 0.01, \
        "Incorrect maximum monetary value"

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(OUTPUTS / "customer_rfm.csv", index=False)

    print("RFM — FIRST FIVE CUSTOMERS")
    print(rfm.head().to_string(index=False))

    print("\nRFM SUMMARY")
    print(f"Customers: {len(rfm)}")
    print(f"Median recency: {rfm['recency'].median():.0f} days")
    print(f"Minimum recency: {rfm['recency'].min()} days")
    print(f"Maximum recency: {rfm['recency'].max()} days")
    print(f"Mean frequency: {rfm['frequency'].mean():.2f}")
    print(f"Maximum frequency: {rfm['frequency'].max()}")
    print(f"Mean monetary: INR {rfm['monetary'].mean():,.2f}")
    print(f"Maximum monetary: INR {rfm['monetary'].max():,.2f}")
    print("\nSaved: outputs/customer_rfm.csv")
    print("RFM CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()