from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"


def main():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    sql_text = SQL_PATH.read_text(encoding="utf-8")

    # These SELECT queries are separated by semicolons.
    queries = [
        statement.strip()
        for statement in sql_text.split(";")
        if statement.strip()
    ]

    assert len(queries) == 7, "queries.sql should contain Q1 through Q7"

    with sqlite3.connect(DB_PATH) as connection:
        q1 = pd.read_sql_query(queries[0], connection)
        q2 = pd.read_sql_query(queries[1], connection)
        q3 = pd.read_sql_query(queries[2], connection)
        q4 = pd.read_sql_query(queries[3], connection)
        q5 = pd.read_sql_query(queries[4], connection)
        q6 = pd.read_sql_query(queries[5], connection)
        q7 = pd.read_sql_query(queries[6], connection)

    assert len(q1) == 30, "Q1: expected 30 customers"
    print("Q1 CHECKPOINT PASSED")

    print("\nQ2 RESULTS")
    print(q2.to_string(index=False))

    total = q2["revenue"].sum()
    largest = q2.iloc[0]

    assert len(q2) == 30, "Q2: expected 30 city/category combinations"
    assert abs(total - 2998668.80) < 0.01, "Q2: incorrect total revenue"
    assert largest["city"] == "Bengaluru", "Q2: incorrect largest city"
    assert largest["category"] == "Electronics", \
        "Q2: incorrect largest category"
    assert abs(largest["revenue"] - 233909.45) < 0.01, \
        "Q2: incorrect largest revenue"

    print(f"\nCity/category combinations: {len(q2)}")
    print(f"Total revenue: INR {total:,.2f}")
    print(
        f"Largest combination: {largest['city']} / "
        f"{largest['category']} — INR {largest['revenue']:,.2f}"
    )
    print("Q2 CHECKPOINT PASSED")

    print("\nQ3 RESULTS")
    
    print(q3.to_string(index=False))

    assert len(q3) == 11, "Q3: expected 11 products"

    print(f"\nProducts above average catalogue price: {len(q3)}")
    print("Q3 CHECKPOINT PASSED")

    print("\nQ4 RESULTS")
    print(q4.to_string(index=False))

    assert len(q4) == 25, "Q4: expected 25 customers"

    print(f"\nCustomers with no completed order: {len(q4)}")
    print("Q4 CHECKPOINT PASSED")

    print("\nQ5 RESULTS — FIRST 10 CUSTOMERS")
    print(q5.head(10).to_string(index=False))

    assert len(q5) == 550, "Q5: expected 550 customers"

    average = q5["average_customer_spend"].iloc[0]
    assert round(average, 2) == 2552.06, "Q5: incorrect average spending"

    print(f"\nAverage customer spending: INR {average:,.2f}")
    print(f"Customers above average: {len(q5)}")
    print("Q5 CHECKPOINT PASSED")

    print("\nQ6 — TOP CUSTOMER IN EACH CITY")

    top_customers = q6.loc[q6["spend_rank"].eq(1)]
    print(
        top_customers[
            ["city", "customer_id", "total_spend"]
        ].to_string(index=False)
    )

    expected_top = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }

    assert len(q6) == 1175, "Q6: expected 1175 customers"
    assert len(top_customers) == 6, "Q6: expected six city leaders"

    for city, (customer_id, spend) in expected_top.items():
        row = top_customers.loc[top_customers["city"].eq(city)]
        assert len(row) == 1, f"Q6: incorrect leader count for {city}"
        assert row.iloc[0]["customer_id"] == customer_id, \
            f"Q6: incorrect top customer for {city}"
        assert abs(row.iloc[0]["total_spend"] - spend) < 0.01, \
            f"Q6: incorrect top spending for {city}"

    different_ranks = (
        q6["spend_rank"] != q6["spend_row_number"]
    ).sum()

    print(f"\nRows where RANK differs from ROW_NUMBER: {different_ranks}")
    assert different_ranks == 8, "Q6: expected eight differing rows"

    print("Q6 CHECKPOINT PASSED")

    print("\nQ7 — FIRST FIVE DAYS")
    print(q7.head().to_string(index=False))

    print("\nQ7 — LAST FIVE DAYS")
    print(q7.tail().to_string(index=False))

    assert len(q7) == 183, "Q7: expected 183 daily rows"

    final_total = q7["cumulative_revenue"].iloc[-1]
    assert abs(final_total - 2998668.80) < 0.01, \
        "Q7: incorrect final cumulative revenue"

    print(f"\nDaily rows: {len(q7)}")
    print(f"Final cumulative revenue: INR {final_total:,.2f}")
    print("Q7 CHECKPOINT PASSED")
    
if __name__ == "__main__":
    main()