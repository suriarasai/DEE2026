-- =====================================================================
-- ACT 2 - DOCUMENT questions, asked of Postgres (the twist)
-- Shell:  double-click shell-postgres.cmd
--   (or: podman exec -it flashsale-postgres psql -P pager=off -U postgres -d flashsale)
-- =====================================================================

-- 2.1  THE QUESTION: show order 13611 the way the app wants it - one object with
--      its line items and the buyer. In a relational schema the order is scattered over
--      three fact rows and two dimensions, so we have to REBUILD the document.
SELECT jsonb_pretty(jsonb_build_object(
         'order_id', f.order_id,
         'order_ts', f.order_ts,
         'status',   f.status,
         'buyer',    jsonb_build_object('name', u.name, 'tier', u.loyalty_tier),
         'items',    jsonb_agg(jsonb_build_object(
                         'product',    p.name,
                         'category',   p.category,
                         'quantity',   f.quantity,
                         'unit_price', f.unit_price) ORDER BY f.line_no)
       )) AS order_document
FROM fact_sales f
JOIN dim_user    u USING (user_id)
JOIN dim_product p USING (product_id)
WHERE f.order_id = 13611
GROUP BY f.order_id, f.order_ts, f.status, u.user_id;

-- 2.2  Postgres CAN store documents: three users, three different shapes, no ALTER TABLE.
SELECT user_id, name, jsonb_pretty(profile) AS profile
FROM dim_user WHERE user_id IN (2, 8, 12) ORDER BY user_id;

-- 2.3  ...and query inside them. "How many users have an iOS device?"
--      @> means "contains this fragment".
SELECT count(*) AS ios_users
FROM dim_user WHERE profile @> '{"devices": [{"type": "ios"}]}';

-- 2.4  THE DELIBERATE BUG: marketing wants an early-access perk on every platinum
--      profile. This reports "UPDATE 75"... ask the room whether it worked.
UPDATE dim_user
SET profile = jsonb_set(profile, '{perks,early_access}', 'true')
WHERE loyalty_tier = 'platinum';

SELECT count(*) AS users_with_perk
FROM dim_user WHERE profile @> '{"perks": {"early_access": true}}';
-- 0 rows have it. jsonb_set only creates the LAST key in the path; because "perks"
-- did not exist, it silently changed nothing.

-- 2.5  THE FIX: build the parent object yourself. It works, but compare how much
--      you have to spell out against the one-line $set in mongo/act2_document.js.
UPDATE dim_user
SET profile = jsonb_set(profile, '{perks}',
                        coalesce(profile -> 'perks', '{}') || '{"early_access": true}')
WHERE loyalty_tier = 'platinum';

SELECT count(*) AS users_with_perk
FROM dim_user WHERE profile @> '{"perks": {"early_access": true}}';

-- 2.6  OPTIONAL: documents can be indexed too. Run the EXPLAIN, create the index,
--      run the EXPLAIN again and watch "Seq Scan" become "Bitmap Index Scan".
EXPLAIN (ANALYZE, COSTS OFF)
SELECT user_id, name FROM dim_user WHERE profile @> '{"marketing": {"campaign": "1111-early-access"}}';

CREATE INDEX idx_user_profile ON dim_user USING gin (profile jsonb_path_ops);
