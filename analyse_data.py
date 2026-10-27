from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"


def main():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as connection:
        orders = pd.read_sql_query("SELECT * FROM orders", connection)
        customers = pd.read_sql_query("SELECT * FROM customers", connection)
        products = pd.read_sql_query("SELECT * FROM products", connection)

        sql_total = connection.execute(
            """
            SELECT SUM(order_amount)
            FROM orders
            WHERE status = 'completed'
            """
        ).fetchone()[0]

    completed = orders.loc[orders["status"].eq("completed")].copy()

    data = (
        completed
        .merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            validate="many_to_one",
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            validate="many_to_one",
        )
    )

    # 1. Rebuild Q2 using a pandas pivot table.
    pivot = pd.pivot_table(
        data,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total",
    )

    print("REVENUE BY CITY AND CATEGORY")
    print(pivot.round(2).to_string())

    pandas_total = pivot.loc["Total", "Total"]
    assert abs(pandas_total - sql_total) < 0.01, \
        "Pandas and SQL revenue totals do not match"

    print(f"\nPandas total: INR {pandas_total:,.2f}")
    print(f"SQL total: INR {sql_total:,.2f}")
    print("PANDAS CROSS-CHECK PASSED")

    # 2. Named aggregation: city revenue, average and order count.
    city_summary = data.groupby("city").agg(
        total_revenue=("order_amount", "sum"),
        average_order_amount=("order_amount", "mean"),
        completed_orders=("order_id", "count"),
    )

    print("\nCITY SUMMARY")
    print(city_summary.round(2).to_string())

    # 3. Distribution statistics and IQR outliers.
    amounts = data["order_amount"]

    q1 = amounts.quantile(0.25)
    q2 = amounts.quantile(0.50)
    q3 = amounts.quantile(0.75)

    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    statistics = pd.Series({
        "Mean": amounts.mean(),
        "Median / Q2": q2,
        "Skewness": amounts.skew(),
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower fence": lower_fence,
        "Upper fence": upper_fence,
        "Outliers below lower fence": int((amounts < lower_fence).sum()),
        "Outliers above upper fence": int((amounts > upper_fence).sum()),
    })

    print("\nORDER AMOUNT STATISTICS")
    print(statistics.round(2).to_string())

    # 4. Correlations between the required numeric columns.
    columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]

    correlations = data[columns].corr()

    print("\nCORRELATION MATRIX")
    print(correlations.round(4).to_string())

    # Compare distinct pairs only, excluding self-correlations.
    pairs = [
        (first, second, correlations.loc[first, second])
        for index, first in enumerate(columns)
        for second in columns[index + 1:]
    ]

    first, second, value = max(pairs, key=lambda pair: abs(pair[2]))

    print(f"\nStrongest absolute correlation: {first} with {second}")
    print(f"Correlation value: {value:.4f}")
    print("A correlation of this size does not establish causation.")


if __name__ == "__main__":
    main()