-- =====================================================================
-- ACT 1 - RELATIONAL SCHEMA, in its natural home (Postgres)
-- Open a shell:  double-click shell-postgres.cmd
--   (or: podman exec -it flashsale-postgres psql -P pager=off -U postgres -d flashsale)
-- Then paste one block at a time.
-- =====================================================================

-- 1.1  THE QUESTION: on the 11.11 flash sale, which category + loyalty tier
--      combinations earned the most?
--      Shape of every relational query: fact in the middle, join out to dimensions,
--      filter, GROUP BY, aggregate.
SELECT p.category, u.loyalty_tier, sum(f.revenue) AS revenue
FROM fact_sales f
JOIN dim_product p USING (product_id)
JOIN dim_user    u USING (user_id)
WHERE f.date_key = DATE '2025-11-11'
GROUP BY p.category, u.loyalty_tier
ORDER BY revenue DESC
LIMIT 10;

-- 1.2  THE DELIBERATE BUG: ask the room why this reads the whole fact table.
--      Look for "Seq Scan on fact_sales" and "Rows Removed by Filter".
EXPLAIN (ANALYZE, COSTS OFF)
SELECT p.category, u.loyalty_tier, sum(f.revenue) AS revenue
FROM fact_sales f
JOIN dim_product p USING (product_id)
JOIN dim_user    u USING (user_id)
WHERE f.date_key = DATE '2025-11-11'
GROUP BY p.category, u.loyalty_tier;

-- 1.3  THE FIX, live. Then re-run 1.2 and look for "idx_fact_date".
CREATE INDEX idx_fact_date ON fact_sales (date_key);

-- 1.4  SAME SHAPE, DIFFERENT SLICE: revenue per month, one column per tier.
--      The point: a new business question is a new GROUP BY, not a new data model.
SELECT d.year, d.month, d.month_name,
       sum(f.revenue) FILTER (WHERE u.loyalty_tier = 'bronze')   AS bronze,
       sum(f.revenue) FILTER (WHERE u.loyalty_tier = 'silver')   AS silver,
       sum(f.revenue) FILTER (WHERE u.loyalty_tier = 'gold')     AS gold,
       sum(f.revenue) FILTER (WHERE u.loyalty_tier = 'platinum') AS platinum
FROM fact_sales f
JOIN dim_date d USING (date_key)
JOIN dim_user u USING (user_id)
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;

-- 1.5  THE NUANCE: marketing renames a category.
--      In a relational schema the name lives in ONE place, so this touches a handful of
--      rows and every past sale reports under the new name immediately.
--      (Wrapped in a transaction and rolled back so the demo data stays the same.)
BEGIN;
UPDATE dim_product SET category = 'Home & Garden' WHERE category = 'Home & Living';
SELECT p.category, sum(f.revenue) AS revenue
FROM fact_sales f JOIN dim_product p USING (product_id)
WHERE p.category LIKE 'Home%' GROUP BY p.category;
ROLLBACK;
