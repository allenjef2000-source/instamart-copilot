from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False)

# 1. Convert order timestamps.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    format="%Y-%m-%d %H:%M:%S",
    errors="raise",
)

# 2. Remove exact duplicate rows.
raw_count = len(orders)
orders = orders.drop_duplicates().copy()
exact_removed = raw_count - len(orders)

# 3. Keep the latest record for each order ID.
before_latest = len(orders)

orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

earlier_removed = before_latest - len(orders)

# Look up product prices and categories without adding extra columns.
prices = products.set_index("product_id")["price_inr"]
categories = products.set_index("product_id")["category"]

assert orders["product_id"].isin(prices.index).all()

# 4. Repair missing or non-numeric order amounts.
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

missing_amount = orders["order_amount"].isna()
amounts_repaired = int(missing_amount.sum())

calculated_amount = (
    orders["product_id"].map(prices)
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders.loc[missing_amount, "order_amount"] = calculated_amount[missing_amount]

# 5. Replace negative delivery times with missing values.
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

negative_delivery = orders["delivery_minutes"].lt(0)
negative_repaired = int(negative_delivery.sum())

orders.loc[negative_delivery, "delivery_minutes"] = float("nan")

completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")

# Calculate category medians using completed orders only.
completed_deliveries = orders.loc[completed, ["delivery_minutes"]].copy()
completed_deliveries["category"] = (
    orders.loc[completed, "product_id"].map(categories)
)

category_medians = completed_deliveries.groupby("category")[
    "delivery_minutes"
].transform("median")

missing_delivery = completed & orders["delivery_minutes"].isna()
deliveries_filled = int(missing_delivery.sum())

orders.loc[missing_delivery, "delivery_minutes"] = category_medians.reindex(
    orders.index[missing_delivery]
)

# Cancelled orders were never delivered.
orders.loc[cancelled, "delivery_minutes"] = float("nan")

# 6. Validate the results.
revenue = orders.loc[completed, "order_amount"].sum()

assert len(orders) == 12309
assert orders["order_id"].is_unique
assert completed.sum() == 11110
assert cancelled.sum() == 1199
assert orders["status"].eq("pending").sum() == 0
assert orders["order_amount"].isna().sum() == 0
assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0
assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199
assert abs(revenue - 2998668.80) < 0.01

# 7. Save the clean tables.
orders.to_csv(
    CLEAN / "orders.csv",
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
)

# The supplied product table needs no cleaning.
products.to_csv(CLEAN / "products.csv", index=False)

print("ORDER CLEANING SUMMARY")
print(f"Raw rows: {raw_count}")
print(f"Exact duplicate rows removed: {exact_removed}")
print(f"Earlier order records removed: {earlier_removed}")
print(f"Order amounts repaired: {amounts_repaired}")
print(f"Negative delivery times marked missing: {negative_repaired}")
print(f"Completed delivery times filled: {deliveries_filled}")
print(f"Clean order rows: {len(orders)}")
print(f"Completed orders: {int(completed.sum())}")
print(f"Cancelled orders: {int(cancelled.sum())}")
print(f"Completed revenue: INR {revenue:,.2f}")
print("\nSaved: data/clean/orders.csv")
print("Saved: data/clean/products.csv")
print("ALL CHECKPOINTS PASSED")