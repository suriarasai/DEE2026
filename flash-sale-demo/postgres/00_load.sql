-- =====================================================================
-- Postgres: build the RELATIONAL SCHEMA and load the shared dataset.
-- Run by start.cmd and seed.cmd, which copy this file 
-- and the data into the container first.
-- Safe to re-run: it drops and rebuilds everything.
-- =====================================================================
\set ON_ERROR_STOP on

DROP TABLE IF EXISTS fact_sales, dim_date, dim_product, dim_user, stg_orders, stg_items, stg_profiles CASCADE;

-- ---------- Dimensions: one row per thing, descriptive columns ----------
CREATE TABLE dim_user (
    user_id      int PRIMARY KEY,
    name         text NOT NULL,
    country      text NOT NULL,
    loyalty_tier text NOT NULL,
    signup_date  date NOT NULL,
    referred_by  int,                              -- Act 3: the graph hides in this column
    profile      jsonb NOT NULL DEFAULT '{}'       -- Act 2: the document hides in this column
);

CREATE TABLE dim_product (
    product_id  int PRIMARY KEY,
    name        text NOT NULL,
    category    text NOT NULL,
    brand       text NOT NULL,
    list_price  numeric(10,2) NOT NULL
);

CREATE TABLE dim_date (
    date_key       date PRIMARY KEY,
    year           int  NOT NULL,
    month          int  NOT NULL,
    month_name     text NOT NULL,
    day_name       text NOT NULL,
    is_flash_sale  boolean NOT NULL
);

-- ---------- Fact: one row per order line, numbers plus keys ----------
CREATE TABLE fact_sales (
    order_id    int NOT NULL,
    line_no     int NOT NULL,
    date_key    date NOT NULL REFERENCES dim_date,
    user_id     int  NOT NULL REFERENCES dim_user,
    product_id  int  NOT NULL REFERENCES dim_product,
    order_ts    timestamp NOT NULL,
    status      text NOT NULL,
    channel     text NOT NULL,
    quantity    int  NOT NULL,
    unit_price  numeric(10,2) NOT NULL,
    revenue     numeric(12,2) NOT NULL,
    PRIMARY KEY (order_id, line_no)
);
-- NOTE: there is deliberately NO index on fact_sales.date_key. Act 1 adds it live.

-- ---------- Load the shared CSV files ----------
\copy dim_user (user_id, name, country, loyalty_tier, signup_date, referred_by) FROM '/dataset/users.csv' CSV HEADER
\copy dim_product FROM '/dataset/products.csv' CSV HEADER

CREATE TABLE stg_profiles (user_id int, profile jsonb);
\copy stg_profiles FROM '/dataset/user_profiles.csv' CSV HEADER
UPDATE dim_user u SET profile = s.profile FROM stg_profiles s WHERE s.user_id = u.user_id;

INSERT INTO dim_date
SELECT d::date, extract(year FROM d)::int, extract(month FROM d)::int,
       trim(to_char(d, 'Month')), trim(to_char(d, 'Day')), d::date = DATE '2025-11-11'
FROM generate_series(DATE '2025-10-01', DATE '2026-09-30', interval '1 day') AS d;

CREATE TABLE stg_orders (order_id int, user_id int, order_date date, order_ts timestamp, status text, channel text);
CREATE TABLE stg_items  (order_id int, line_no int, product_id int, quantity int, unit_price numeric(10,2));
\copy stg_orders FROM '/dataset/orders.csv' CSV HEADER
\copy stg_items  FROM '/dataset/order_items.csv' CSV HEADER

INSERT INTO fact_sales
SELECT o.order_id, i.line_no, o.order_date, o.user_id, i.product_id, o.order_ts, o.status, o.channel,
       i.quantity, i.unit_price, i.quantity * i.unit_price
FROM stg_orders o JOIN stg_items i USING (order_id)
ORDER BY o.order_id, i.line_no;

DROP TABLE stg_orders, stg_items, stg_profiles;
VACUUM ANALYZE;

SELECT (SELECT count(*) FROM dim_user)                  AS users,
       (SELECT count(*) FROM dim_product)               AS products,
       (SELECT count(DISTINCT order_id) FROM fact_sales) AS orders,
       (SELECT count(*) FROM fact_sales)                AS order_lines,
       (SELECT sum(revenue) FROM fact_sales)            AS total_revenue;
