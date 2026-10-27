from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"


def main():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as connection:
        customers = pd.read_sql_query("SELECT * FROM customers", connection)
        products = pd.read_sql_query("SELECT * FROM products", connection)
        orders = pd.read_sql_query("SELECT * FROM orders", connection)

    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")

    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }

    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }

    assert len(customers) == 1200, "Incorrect customer count"
    assert customers["customer_id"].is_unique, "Duplicate customer IDs"
    assert customers["city"].value_counts().to_dict() == expected_cities, \
        "City counts do not match"
    assert customers["gender"].value_counts().to_dict() == expected_genders, \
        "Gender counts do not match"
    assert customers["age"].notna().all(), "Missing customer ages"
    assert customers["age"].between(18, 60).all(), "Unexpected customer ages"

    assert len(products) == 40, "Incorrect product count"
    assert len(orders) == 12309, "Incorrect order count"
    assert orders["order_id"].is_unique, "Duplicate order IDs"
    assert completed.sum() == 11110, "Incorrect completed order count"
    assert cancelled.sum() == 1199, "Incorrect cancelled order count"
    assert orders["status"].eq("pending").sum() == 0, "Pending orders remain"
    assert orders["order_amount"].notna().all(), "Missing order amounts"
    assert orders.loc[completed, "delivery_minutes"].notna().all(), \
        "Completed orders have missing delivery times"
    assert orders.loc[cancelled, "delivery_minutes"].isna().all(), \
        "Cancelled orders have delivery times"

    revenue = orders.loc[completed, "order_amount"].sum()
    assert abs(revenue - 2998668.80) < 0.01, \
        f"Incorrect completed revenue: {revenue}"

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()