-- Q1: Find customers with qualifying spend above INR 8000.

SELECT
    customer_id,
    SUM(order_amount) AS qualifying_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC, customer_id;

-- Q2: Completed revenue by city and product category.

SELECT
    c.city,
    p.category,
    SUM(o.order_amount) AS revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC, c.city, p.category;

-- Q3: Products priced above the average catalogue price.

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC, product_id;

-- Q4: Customers with no completed order.

SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Customers whose total completed spending exceeds
-- the average total among customers with completed orders.

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS average_customer_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    cs.total_spend,
    av.average_customer_spend
FROM customer_spend AS cs
CROSS JOIN average_spend AS av
WHERE cs.total_spend > av.average_customer_spend
ORDER BY cs.total_spend DESC, cs.customer_id;

-- Q6: Rank customers by completed spending within each city.

WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS spend_dense_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_row_number;

-- Q7: Daily completed revenue and cumulative revenue.

WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    SUM(revenue) OVER (
        ORDER BY day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;