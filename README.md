# Overview

# Reproduce

Run these commands in Windows PowerShell from the
Instamart-copilot project folder. Python 3.11 or newer is required.

1. Create the virtual environment:
   `python -m venv .venv`

2. Install the required packages:
   `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

3. Generate the raw data:
   `.\.venv\Scripts\python.exe generate_data.py`

4. Audit the raw data:
   `.\.venv\Scripts\python.exe audit_data.py`

5. Clean the customers:
   `.\.venv\Scripts\python.exe clean_customers.py`

6. Clean the orders and save the products:
   `.\.venv\Scripts\python.exe clean_orders.py`

7. Create or rebuild the database:
   `.\.venv\Scripts\python.exe create_database.py --replace`

8. Validate the cleaned database:
   `.\.venv\Scripts\python.exe validate_data.py`

9. Run all seven SQL queries:
   `.\.venv\Scripts\python.exe run_queries.py`

10. Run the pandas cross-check and exploratory analysis:
    `.\.venv\Scripts\python.exe analyse_data.py`

    
# Part A

## Raw data audit

The raw tables contain 1,200 customers with 5 columns, 40 products
with 4 columns, and 13,084 order records with 9 columns.
Run audit_data.py for column data types and missing-value percentages.

| Audit check | Count |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank ages | 61 |
| Ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Non-blank gender spellings | 10 |
| Blank genders | 30 |

## Cleaning rules and results

Customer cleaning:
- Standardised cities by trimming spaces and applying title case:
  476 values changed.
- Standardised gender to male, female or unknown:
  950 values changed.
- Marked 6 impossible ages as missing.
- Filled 67 missing ages using each customer's city median.
- Converted signup_date to datetime.
- Retained all 1,200 customers, with no missing ages.

Final city counts: Pune 227, Delhi 210, Bengaluru 197,
Chennai 191, Hyderabad 191 and Mumbai 184.

Final gender counts: female 610, male 560 and unknown 30.

Order cleaning, performed in this order:
1. Converted order_ts to datetime.
2. Removed 119 exact duplicate rows.
3. Kept the latest timestamp for each order ID, removing 656 earlier records.
4. Recalculated 112 missing amounts using
   round(price_inr × quantity × (1 − discount_pct / 100), 2).
5. Marked 105 negative delivery times as missing.
6. Filled 774 missing completed-order delivery times using the
   median delivery time of completed orders in the same product category.
7. Left delivery times missing for all 1,199 cancelled orders.

Repair counts are measured after deduplication, so they differ
from counts in the raw audit.

The product table required no changes.

Cleaned CSV files are saved in data/clean/.
The SQLite database data/instamart.db contains customers,
products and orders.

Final orders: 12,309, comprising 11,110 completed and
1,199 cancelled orders, with no pending orders.
There are no missing order amounts or missing delivery times
among completed orders.

Completed revenue: INR 2,998,668.80.
Running validate_data.py prints ALL CHECKPOINTS PASSED.

## SQL results

All seven queries are saved in sql/queries.sql and executed
using pandas.read_sql_query in run_queries.py.

| Query | Result |
|---|---|
| Q1: Qualifying customer spending above INR 8,000 | 30 customers |
| Q2: Revenue by city and category | 30 combinations; total INR 2,998,668.80 |
| Q2: Largest combination | Bengaluru / Electronics: INR 233,909.45 |
| Q3: Products above average catalogue price | 11 products |
| Q4: Customers with no completed order | 25 customers |
| Q5: Customers above average completed spending | 550 customers; average INR 2,552.06 |
| Q6: Rows where RANK differs from ROW_NUMBER | 8 |
| Q7: Daily revenue | 183 rows; final cumulative revenue INR 2,998,668.80 |

RANK gives tied customers the same rank, whereas ROW_NUMBER
assigns each row a unique number, so their values can differ.

Top completed-spend customer in each city:

| City | Customer | Total spending (INR) |
|---|---|---:|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

## Pandas cross-check and city analysis

A pandas pivot table with margins reproduced the city/category
revenue analysis. Its grand total matched SQL: INR 2,998,668.80.

Named aggregation produced:

| City | Revenue (INR) | Mean order amount (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

## Completed-order amount statistics

| Statistic | Value |
|---|---:|
| Mean (INR) | 269.91 |
| Median / Q2 (INR) | 135.00 |
| Skewness | 4.61 |
| Q1 (INR) | 64.80 |
| Q3 (INR) | 297.00 |
| IQR (INR) | 232.20 |
| Lower fence (INR) | -283.50 |
| Upper fence (INR) | 645.30 |
| Outliers below lower fence | 0 |
| Outliers above upper fence | 897 |

## Exploratory findings

This synthetic dataset describes customers, products and grocery
delivery orders across six cities. Completed-order revenue totals
INR 2,998,668.80.

Order amounts are strongly right-skewed: the mean is INR 269.91,
compared with a median of INR 135.00, and skewness is 4.61.
A relatively small number of expensive orders raises the average.

The raw data contains duplicates, inconsistent spellings, impossible
ages, invalid amounts and missing or negative delivery times.
After cleaning, 897 completed orders exceed the upper IQR fence
of INR 645.30. These are statistical outliers, not automatically
incorrect records.

The strongest absolute correlation among the four required numeric
variables is between order_amount and discount_pct, at -0.0951.
Order amount and quantity have a correlation of +0.0873.
These relationships are weak, and correlation of this size does
not establish causation.

# Part B

# Part C

# Part D

# Limitations and Next Steps
