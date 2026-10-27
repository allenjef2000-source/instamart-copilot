"""Build a relational SQLite database from the cleaned CSV files."""

import argparse
import csv
import math
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "clean"
DEFAULT_DATABASE = ROOT / "data" / "instamart.db"

SCHEMAS = {
    "customers.csv": [
        "customer_id",
        "city",
        "signup_date",
        "age",
        "gender",
    ],
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
}


def load_csv(data_dir: Path, filename: str) -> list[dict[str, str]]:
    path = data_dir / filename
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            expected_columns = SCHEMAS[filename]
            if reader.fieldnames != expected_columns:
                raise ValueError(
                    f"{path} has columns {reader.fieldnames}; "
                    f"expected {expected_columns}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"{path}, line {line_number}: too many CSV fields")
        if any(value is None for value in row.values()):
            raise ValueError(f"{path}, line {line_number}: too few CSV fields")
    return rows


def required_text(row: dict[str, str], field: str, filename: str) -> str:
    value = row[field].strip()
    if not value:
        raise ValueError(f"{filename}: {field} must not be blank")
    return value


def parse_integer(value: str, field: str, filename: str) -> int:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{filename}: {field} must be an integer, got {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"{filename}: {field} must be an integer, got {value!r}")
    return int(number)


def parse_number(value: str, field: str, filename: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{filename}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{filename}: {field} must be finite, got {value!r}")
    return number


def parse_iso_date(value: str, field: str, filename: str) -> str:
    value = value.strip()
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{filename}: {field} must be YYYY-MM-DD, got {value!r}") from exc
    return value


def parse_iso_datetime(value: str, field: str, filename: str) -> str:
    value = value.strip()
    try:
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            f"{filename}: {field} must be YYYY-MM-DD HH:MM:SS, got {value!r}"
        ) from exc
    return value


def prepare_customers(rows: list[dict[str, str]]) -> list[tuple[Any, ...]]:
    prepared = []
    for row in rows:
        age_value = row["age"].strip()
        age = parse_number(age_value, "age", "customers.csv") if age_value else None
        prepared.append(
            (
                required_text(row, "customer_id", "customers.csv"),
                required_text(row, "city", "customers.csv"),
                parse_iso_date(row["signup_date"], "signup_date", "customers.csv"),
                age,
                required_text(row, "gender", "customers.csv").lower(),
            )
        )
    return prepared


def prepare_products(rows: list[dict[str, str]]) -> list[tuple[Any, ...]]:
    return [
        (
            required_text(row, "product_id", "products.csv"),
            required_text(row, "product_name", "products.csv"),
            required_text(row, "category", "products.csv"),
            parse_number(row["price_inr"], "price_inr", "products.csv"),
        )
        for row in rows
    ]


def prepare_orders(rows: list[dict[str, str]]) -> list[tuple[Any, ...]]:
    prepared = []
    for row in rows:
        quantity = parse_integer(row["quantity"], "quantity", "orders.csv")
        discount = parse_integer(row["discount_pct"], "discount_pct", "orders.csv")
        amount = parse_number(row["order_amount"], "order_amount", "orders.csv")
        delivery_value = row["delivery_minutes"].strip()
        delivery = (
            parse_number(delivery_value, "delivery_minutes", "orders.csv")
            if delivery_value
            else None
        )
        prepared.append(
            (
                required_text(row, "order_id", "orders.csv"),
                required_text(row, "customer_id", "orders.csv"),
                required_text(row, "product_id", "orders.csv"),
                quantity,
                parse_iso_datetime(row["order_ts"], "order_ts", "orders.csv"),
                required_text(row, "status", "orders.csv").lower(),
                discount,
                amount,
                delivery,
            )
        )
    return prepared


def create_tables(connection: sqlite3.Connection) -> None:
    statements = [
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            city TEXT NOT NULL,
            signup_date TEXT NOT NULL,
            age REAL CHECK (age IS NULL OR age BETWEEN 10 AND 100),
            gender TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            price_inr REAL NOT NULL CHECK (price_inr > 0)
        )
        """,
        """
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            product_id TEXT NOT NULL REFERENCES products(product_id),
            quantity INTEGER NOT NULL CHECK (quantity > 0),
            order_ts TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled')),
            discount_pct INTEGER NOT NULL CHECK (discount_pct BETWEEN 0 AND 100),
            order_amount REAL NOT NULL CHECK (order_amount > 0),
            delivery_minutes REAL CHECK (delivery_minutes IS NULL OR delivery_minutes >= 0)
        )
        """,
        "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
        "CREATE INDEX idx_orders_product_id ON orders(product_id)",
        "CREATE INDEX idx_orders_order_ts ON orders(order_ts)",
    ]
    for statement in statements:
        connection.execute(statement)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an SQLite database from cleaned customer, product, and order CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"directory containing the cleaned CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite database output path (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing database file",
    )
    args = parser.parse_args()

    for filename in SCHEMAS:
        if not (args.data_dir / filename).is_file():
            parser.error(f"required cleaned CSV not found: {args.data_dir / filename}")
    if args.database.exists() and not args.replace:
        parser.error(
            f"database already exists: {args.database}; pass --replace to rebuild it"
        )

    try:
        customers = prepare_customers(load_csv(args.data_dir, "customers.csv"))
        products = prepare_products(load_csv(args.data_dir, "products.csv"))
        orders = prepare_orders(load_csv(args.data_dir, "orders.csv"))
    except ValueError as exc:
        parser.error(str(exc))

    args.database.parent.mkdir(parents=True, exist_ok=True)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(args.database)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        if args.replace:
            for table in ("orders", "products", "customers"):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
        create_tables(connection)
        connection.executemany(
            """
            INSERT INTO customers (customer_id, city, signup_date, age, gender)
            VALUES (?, ?, ?, ?, ?)
            """,
            customers,
        )
        connection.executemany(
            """
            INSERT INTO products (product_id, product_name, category, price_inr)
            VALUES (?, ?, ?, ?)
            """,
            products,
        )
        connection.executemany(
            """
            INSERT INTO orders (
                order_id, customer_id, product_id, quantity, order_ts, status,
                discount_pct, order_amount, delivery_minutes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            orders,
        )
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise ValueError(f"Database has foreign-key violations: {foreign_key_errors}")
        connection.commit()
    except (OSError, sqlite3.Error, ValueError) as exc:
        if connection is not None:
            connection.rollback()
        parser.error(f"could not create database {args.database}: {exc}")
    finally:
        if connection is not None:
            connection.close()

    print(f"Created database: {args.database}")
    print(f"Customers: {len(customers):,}")
    print(f"Products: {len(products):,}")
    print(f"Orders: {len(orders):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
