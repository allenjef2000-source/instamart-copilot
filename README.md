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

11. Calculate customer recency, frequency and monetary values:
    `.\.venv\Scripts\python.exe customer_rfm.py`

12. Compare K-means customer groupings:
    `.\.venv\Scripts\python.exe cluster_customers.py`

13. Profile and name customer segments:
    `.\.venv\Scripts\python.exe profile_segments.py`

14. Prepare the lapse prediction dataset and train/test split:
    `.\.venv\Scripts\python.exe prepare_lapse_data.py`

15. Check preprocessing and baseline performance:
    `.\.venv\Scripts\python.exe preprocess_lapse.py`

16. Tune Logistic Regression and Random Forest:
    `.\.venv\Scripts\python.exe train_lapse_models.py`

17. Evaluate both tuned models on the test set:
    `.\.venv\Scripts\python.exe evaluate_lapse_models.py`

18. Run the training class-balance experiment:
    `.\.venv\Scripts\python.exe balance_lapse.py`

19. Interpret model features and save the selected model:
    `.\.venv\Scripts\python.exe interpret_lapse.py`

20. Create the illustrative Premium customer retention list:
    `.\.venv\Scripts\python.exe create_retention_list.py`

    
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

## RFM and customer segmentation

RFM uses completed orders, with snapshot 2025-10-01 00:00:00.

| Measure | Result |
|---|---:|
| Customers with completed orders | 1,175 |
| Median recency | 19 days |
| Minimum / maximum recency | 0 / 180 days |
| Mean / maximum frequency | 9.46 / 33 |
| Mean monetary value (INR) | 2,552.06 |
| Maximum monetary value (INR) | 14,766.55 |

The three RFM features were standardised before K-means.
Every run used n_init=10 and random_state=42.

| K | Inertia | Silhouette |
|---|---:|---:|
| 2 | 1820.7606 | 0.4650 |
| 3 | 1268.2346 | 0.4452 |
| 4 | 743.0711 | 0.5255 |
| 5 | 617.3666 | 0.4805 |
| 6 | 503.1706 | 0.4819 |
| 7 | 408.4605 | 0.4163 |
| 8 | 366.5035 | 0.4123 |

K=4 was selected because it has the highest silhouette score,
a substantial inertia improvement over K=3, and useful business groups.
Adding clusters allows tighter groups, so inertia decreases as K grows.
Therefore, lower inertia alone cannot determine the best K.

| ID | Segment | Customers | Mean recency | Mean frequency | Mean monetary (INR) | Action |
|---|---|---:|---:|---:|---:|---|
| 0 | Occasional Customers | 436 | 28.86 | 3.77 | 1,057.62 | Relevant recommendations |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | Personalised service and retention benefits |
| 2 | Frequent Customers | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty rewards |
| 3 | Inactive Customers | 182 | 109.04 | 2.20 | 685.99 | Reactivation campaign |

Cluster IDs are labels with no ranking.
Premium is the segment with the highest mean monetary value.

PCA explained variance ratios were 0.6343 and 0.2180,
totalling 0.8523.

Required segment output: outputs/customer_segments.csv.

## Lapse dataset and leakage prevention

The cohort contains customers with a completed order before
2025-09-01. A customer is labelled lapsed=1 if they have no
completed order during September 2025.

There are 1,156 eligible customers, including 443 lapsed customers
(38.32%). The stratified split used test_size=0.2 and random_state=42.

| Split | Not lapsed | Lapsed | Total |
|---|---:|---:|---:|
| Training | 570 | 354 | 924 |
| Test | 143 | 89 | 232 |

Features include recency, frequency, monetary value, average discount,
cancellation rate, average delivery time, favourite category, city,
age, gender and tenure.

No orders from 2025-09-01 onwards entered the features.
Whole-window segment labels were not used as model features.
No encoder, scaler or resampler was fitted using test rows.
Preprocessing was inside each model's cross-validation pipeline.

Numeric features were standardised. Nominal features were one-hot
encoded with drop="first". No feature values were missing.

The training scaler's frequency mean was 8.1147, compared with
8.0285 for the full cohort, because the scaler learned only from
the training subset.

Always predicting not lapsed gave test accuracy 61.64% and
lapsed-class F1 of 0.

## Model tuning and test evaluation

Both searches used five-fold cross-validation with scoring="f1".

Logistic Regression used max_iter=1000 and searched
C in {0.01, 0.1, 1, 10, 100}.
Best C: 100. Best cross-validated F1: 0.650553.

Random Forest used n_estimators=100 and random_state=42.
It searched max_depth in {3, 5, 8} and min_samples_leaf in
{5, 10, 20}, giving 45 cross-validation fits.
Best max_depth: 8. Best min_samples_leaf: 10.
Best cross-validated F1: 0.681053.

Test evaluation occurred after tuning.

| Model | Accuracy | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7284 | 0.6444 | 0.6517 | 0.6480 |
| Random Forest | 0.7371 | 0.6591 | 0.6517 | 0.6554 |

Confusion matrices use actual classes as rows and predicted classes
as columns, with class order [not lapsed, lapsed].

Logistic Regression: [[111, 32], [31, 58]].
Random Forest: [[113, 30], [31, 58]].

Random Forest was selected because its F1 and precision were
slightly higher, with the same recall as Logistic Regression.
Both identified 58 of 89 lapsed customers, but Random Forest
incorrectly targeted two fewer continuing customers.
Missing lapsing customers risks losing future business, while
incorrect targeting wastes retention resources.
Its test F1 of 0.6554 exceeds the required 0.55.

## Class-balance experiment

All original training rows were retained. An additional 216 lapsed
training rows were sampled with replacement using random_state=42.
The balanced training set had 570 rows per class.
The test set remained unchanged.

The best Logistic Regression settings were refitted on balanced data.

| Training | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original | 0.6444 | 0.6517 | 0.6480 |
| Balanced | 0.5891 | 0.8539 | 0.6972 |

Recall changed most, increasing by 20.22 percentage points.
The model identified 76 rather than 58 lapsed customers, while
false positives increased from 32 to 53.
This helps the retention team reach more lapsing customers but
also increases spending on customers who would continue ordering.

Balanced Logistic Regression achieved higher measured F1.
Random Forest remains the previously selected model; this experiment
is reported separately rather than used for repeated test-set selection.

## Interpretation and persistence

Random Forest's five most important features:

| Feature | Importance | Business reading |
|---|---:|---|
| Frequency | 0.279589 | Order frequency contributes most to predictions |
| Monetary | 0.271176 | Total spending is another major predictor |
| Recency | 0.115149 | Time since the last order helps distinguish lapse risk |
| Average delivery time | 0.061286 | Delivery experience contributes to predictions |
| Cancellation rate | 0.052325 | Cancellation behaviour provides predictive information |

Importance does not indicate the direction of an association
or establish causation.

Logistic Regression coefficients for standardised numeric features,
holding other features constant:
- Recency: +0.0396; longer since the last order is associated
  with higher predicted lapse risk.
- Frequency: -1.6982; more orders are associated with lower risk.
- Monetary: +0.2846; higher spending is associated with higher risk
  after accounting for the other features.

The selected Random Forest pipeline was saved to
outputs/lapse_model.joblib. Reloading it reproduced the original
test predictions exactly.

## Illustrative retention list

Among 40 Premium test customers, the 10 with the highest predicted
lapse probabilities were saved to outputs/retention_list.csv.
One of the 10 actually lapsed.

The Premium flag is a post-hoc whole-window label that includes
September behaviour, so this retention list is illustrative and
not a leakage-free evaluation.


# Part C

# Part D

# Limitations and Next Steps

The Premium flag is a post-hoc whole-window label that includes
September behaviour, so the retention list is illustrative and
not a leakage-free evaluation. A future operational retention list
should define Premium status using only pre-cutoff information.

The data and policies are synthetic. Model results do not establish
real-world performance or causal relationships.