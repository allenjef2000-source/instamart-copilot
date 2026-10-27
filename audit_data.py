"""Audit generated Instamart CSV data for schema and data-quality issues."""

import pandas as pd
import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


SCHEMAS = {
    "customers.csv": ["customer_id", "city", "signup_date", "age", "gender"],
    "products.csv": ["product_id", "product_name", "category", "price_inr"],
    "orders.csv": [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_ts",
        "status",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    ],
    "tickets.csv": ["ticket_id", "customer_id", "order_id", "text", "true_category"],
    "policy_eval.csv": ["question", "expected_doc", "expected_fact"],
    "agent_scenarios.csv": [
        "scenario_id",
        "customer_id",
        "message",
        "expected_outcome",
        "expected_refund_inr",
    ],
}


def read_csv(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != expected_header:
                print(
                    f"ERROR {path.name}: expected header {expected_header}, "
                    f"got {reader.fieldnames}"
                )
                return []
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        print(f"ERROR {path.name}: could not read CSV: {exc}")
        return []


def count_bad(rows: list[dict[str, str]], predicate) -> int:
    return sum(predicate(row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check generated CSV schemas, references, and data quality."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="directory containing generated CSV files (default: data/raw)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    has_errors = False
    tables: dict[str, list[dict[str, str]]] = {}
    print(f"Auditing CSV files in {data_dir}")

    for filename, header in SCHEMAS.items():
        path = data_dir / filename
        if not path.is_file():
            print(f"ERROR {filename}: file not found")
            has_errors = True
            tables[filename] = []
            continue
        tables[filename] = read_csv(path, header)
        if not tables[filename]:
            # Empty files are invalid for these generated datasets.
            print(f"ERROR {filename}: no data rows or invalid header")
            has_errors = True
        print(f"  {filename}: {len(tables[filename]):,} rows")

    customers = tables["customers.csv"]
    products = tables["products.csv"]
    orders = tables["orders.csv"]
    tickets = tables["tickets.csv"]
    policy_eval = tables["policy_eval.csv"]
    scenarios = tables["agent_scenarios.csv"]

    def unique_ids(rows: list[dict[str, str]], field: str, label: str) -> set[str]:
        nonempty = [row[field] for row in rows if row[field]]
        counts = Counter(nonempty)
        duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
        missing_count = len(rows) - len(nonempty)
        if missing_count or duplicate_count:
            print(
                f"ERROR {label}: {missing_count} missing {field} values, "
                f"{duplicate_count} duplicate IDs"
            )
            nonlocal has_errors
            has_errors = True
        return set(nonempty)

    customer_ids = unique_ids(customers, "customer_id", "customers.csv")
    product_ids = unique_ids(products, "product_id", "products.csv")
    unique_ids(tickets, "ticket_id", "tickets.csv")
    unique_ids(scenarios, "scenario_id", "agent_scenarios.csv")

    order_id_counts = Counter(row["order_id"] for row in orders if row["order_id"])
    duplicate_order_rows = sum(count - 1 for count in order_id_counts.values() if count > 1)
    missing_order_ids = sum(not row["order_id"] for row in orders)
    if missing_order_ids:
        print(f"ERROR orders.csv: {missing_order_ids} missing order_id values")
        has_errors = True
    if duplicate_order_rows:
        print(
            f"WARN orders.csv: {duplicate_order_rows} rows share an order ID "
            "(duplicates are planted in this dataset)"
        )

    checks = [
        (
            "customers with missing or out-of-range ages",
            count_bad(
                customers,
                lambda row: not row["age"]
                or not row["age"].lstrip("+-").isdigit()
                or not 1 <= int(row["age"]) <= 120,
            ),
        ),
        (
            "customers with blank or unrecognized cities",
            count_bad(
                customers,
                lambda row: row["city"].strip().casefold()
                not in {"chennai", "mumbai", "bengaluru", "delhi", "hyderabad", "pune"},
            ),
        ),
        (
            "orders with invalid or pending statuses",
            count_bad(orders, lambda row: row["status"] not in {"completed", "cancelled"}),
        ),
        (
            "orders with invalid quantities",
            count_bad(orders, lambda row: not row["quantity"].isdigit() or int(row["quantity"]) <= 0),
        ),
        (
            "orders with unparseable amounts",
            count_bad(
                orders,
                lambda row: _not_positive_number(row["order_amount"]),
            ),
        ),
        (
            "completed orders with missing or invalid delivery times",
            count_bad(
                orders,
                lambda row: row["status"] == "completed"
                and (
                    not row["delivery_minutes"].isdigit()
                    or int(row["delivery_minutes"]) <= 0
                ),
            ),
        ),
        (
            "tickets with blank text or unrecognized categories",
            count_bad(
                tickets,
                lambda row: not row["text"].strip()
                or row["true_category"] not in {"billing", "delivery", "technical", "fraud", "none"},
            ),
        ),
    ]
    for description, count in checks:
        if count:
            print(f"WARN {description}: {count}")

    invalid_timestamps = count_bad(
        orders,
        lambda row: not _valid_timestamp(row["order_ts"]),
    )
    if invalid_timestamps:
        print(f"WARN orders with invalid timestamps: {invalid_timestamps}")

    bad_products = count_bad(
        products,
        lambda row: not _positive_number(row["price_inr"]),
    )
    if bad_products:
        print(f"ERROR products with invalid prices: {bad_products}")
        has_errors = True

    customer_references = sum(row["customer_id"] not in customer_ids for row in orders)
    product_references = sum(row["product_id"] not in product_ids for row in orders)
    ticket_customer_references = sum(row["customer_id"] not in customer_ids for row in tickets)
    ticket_order_references = sum(
        bool(row["order_id"]) and row["order_id"] not in order_id_counts for row in tickets
    )
    scenario_customer_references = sum(
        row["customer_id"] not in customer_ids for row in scenarios
    )
    bad_references = (
        customer_references
        + product_references
        + ticket_customer_references
        + ticket_order_references
        + scenario_customer_references
    )
    if bad_references:
        print(
            "ERROR broken references: "
            f"{customer_references} order customers, {product_references} order products, "
            f"{ticket_customer_references} ticket customers, {ticket_order_references} "
            "ticket orders, "
            f"{scenario_customer_references} scenario customers"
        )
        has_errors = True

    policies_dir = data_dir.parent / "policies"
    missing_policy_docs = sum(
        row["expected_doc"] != "NONE"
        and not (policies_dir / row["expected_doc"]).is_file()
        for row in policy_eval
    )
    if missing_policy_docs:
        print(f"ERROR policy evaluation rows with missing documents: {missing_policy_docs}")
        has_errors = True

    print("Result:", "FAILED" if has_errors else "PASSED (quality warnings may be intentional)")
    return 1 if has_errors else 0


def _positive_number(value: str) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _not_positive_number(value: str) -> bool:
    return not _positive_number(value)


def _valid_timestamp(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return True
    except ValueError:
        return False

def assignment_audit():
    raw = Path(__file__).resolve().parent / "data" / "raw"

    # Preserve text such as "N/A" so we can count it explicitly.
    tables = {
        name: pd.read_csv(raw / f"{name}.csv", keep_default_na=False)
        for name in ["customers", "products", "orders"]
    }

    def is_blank(column):
        return column.astype(str).str.strip().eq("")

    for name, df in tables.items():
        missing = df.apply(lambda column: is_blank(column).sum())

        print(f"\n{name.upper()}: {df.shape[0]} rows, {df.shape[1]} columns")

        summary = pd.DataFrame({
            "data_type": df.dtypes.astype(str),
            "missing_count": missing,
            "missing_percent": (missing / len(df) * 100).round(2),
        })

        print(summary.to_string())

    customers = tables["customers"]
    orders = tables["orders"]

    ages = pd.to_numeric(customers["age"], errors="coerce")
    amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

    genders = customers.loc[
        ~is_blank(customers["gender"]), "gender"
    ]

    checks = [
        ("Raw order rows", len(orders), 13084),
        ("Exact duplicate rows", orders.duplicated().sum(), 119),
        ("Pending rows", orders["status"].eq("pending").sum(), 656),
        ("Distinct order IDs", orders["order_id"].nunique(), 12309),
        ("Non-numeric amounts", amounts.isna().sum(), 120),
        ("Negative delivery times", delivery.lt(0).sum(), 106),
        ("Blank delivery times",
         is_blank(orders["delivery_minutes"]).sum(), 2541),
        ("Blank ages", is_blank(customers["age"]).sum(), 61),
        ("Ages outside 10–100",
         (ages.notna() & ~ages.between(10, 100)).sum(), 6),
        ("Raw city spellings", customers["city"].nunique(), 24),
        ("Non-blank gender spellings", genders.nunique(), 10),
        ("Blank genders", is_blank(customers["gender"]).sum(), 30),
    ]

    results = pd.DataFrame(
        checks, columns=["Check", "Actual", "Expected"]
    )
    results["Matches"] = results["Actual"].eq(results["Expected"])

    print("\nASSIGNMENT AUDIT CHECKPOINTS")
    print(results.to_string(index=False))
if __name__ == "__main__":
    exit_code = main()

    if exit_code == 0:
        assignment_audit()

    sys.exit(exit_code)
