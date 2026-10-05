-- =====================================================================
-- ACT 3 - GRAPH questions, asked of Postgres (the twist)
-- The graph is the referred_by column on dim_user: each user points at who brought them in.
-- Shell:  double-click shell-postgres.cmd
--   (or: podman exec -it flashsale-postgres psql -P pager=off -U postgres -d flashsale)
-- =====================================================================

-- 3.1  THE QUESTION: who did Chloe Zhang (user 303) bring in, three levels deep?
--      SQL has no "follow this link N times", so we use a recursive query:
--      start with her direct referrals, then keep joining the result back to the table.
WITH RECURSIVE chain AS (
    SELECT user_id, name, 1 AS depth
    FROM dim_user WHERE referred_by = 303
  UNION ALL
    SELECT u.user_id, u.name, c.depth + 1
    FROM dim_user u JOIN chain c ON u.referred_by = c.user_id
    WHERE c.depth < 3
)
SELECT depth, count(*) AS people, string_agg(name, ', ' ORDER BY user_id) AS who
FROM chain GROUP BY depth ORDER BY depth;

-- 3.2  THE DELIBERATE BUG: the same query for user 1998 with no depth limit.
--      The data has a loop (1998 -> 1999 -> 2000 -> 1998), so this never finishes.
--      The timeout is your safety net: it gives up after 3 seconds.
SET statement_timeout = '3s';
WITH RECURSIVE chain AS (
    SELECT user_id, name, 1 AS depth
    FROM dim_user WHERE referred_by = 1998
  UNION ALL
    SELECT u.user_id, u.name, c.depth + 1
    FROM dim_user u JOIN chain c ON u.referred_by = c.user_id
)
SELECT * FROM chain;
RESET statement_timeout;

-- 3.3  THE FIX: tell Postgres to track where it has been (CYCLE, Postgres 14+).
WITH RECURSIVE chain AS (
    SELECT user_id, name, 1 AS depth
    FROM dim_user WHERE referred_by = 1998
  UNION ALL
    SELECT u.user_id, u.name, c.depth + 1
    FROM dim_user u JOIN chain c ON u.referred_by = c.user_id
) CYCLE user_id SET is_cycle USING path
SELECT user_id, name, depth FROM chain WHERE NOT is_cycle ORDER BY depth;

-- 3.4  THE HARDER QUESTION: how are Rohan Santoso (1884) and Max Wilson (1045) connected?
--      We have to build the edge list in both directions, walk every route while
--      remembering the path so far, and keep the shortest one that arrives.
WITH RECURSIVE edges AS (
    SELECT referred_by AS a, user_id AS b FROM dim_user WHERE referred_by IS NOT NULL
  UNION ALL
    SELECT user_id, referred_by FROM dim_user WHERE referred_by IS NOT NULL
), walk AS (
    SELECT 1884 AS node, ARRAY[1884] AS path
  UNION ALL
    SELECT e.b, w.path || e.b
    FROM walk w JOIN edges e ON e.a = w.node
    WHERE e.b <> ALL (w.path)             -- never revisit a node
      AND cardinality(w.path) <= 10       -- and give up after 10 hops
)
SELECT cardinality(path) - 1 AS hops,
       (SELECT string_agg(u.name, ' - ' ORDER BY t.ord)
        FROM unnest(path) WITH ORDINALITY AS t(id, ord)
        JOIN dim_user u ON u.user_id = t.id) AS route
FROM walk WHERE node = 1045
ORDER BY hops LIMIT 1;

-- 3.5  THE ROOM'S CHALLENGE (keep this hidden until someone tries):
--      "Recommend products to user 303: what have people within two referral hops
--      of her bought that she has not?" Compare with the 6 lines in neo4j/act3_graph.cypher.
WITH RECURSIVE edges AS (
    SELECT referred_by AS a, user_id AS b FROM dim_user WHERE referred_by IS NOT NULL
  UNION ALL
    SELECT user_id, referred_by FROM dim_user WHERE referred_by IS NOT NULL
), network AS (
    SELECT 303 AS node, ARRAY[303] AS path, 0 AS depth
  UNION ALL
    SELECT e.b, n.path || e.b, n.depth + 1
    FROM network n JOIN edges e ON e.a = n.node
    WHERE e.b <> ALL (n.path) AND n.depth < 2
)
SELECT p.name, p.category, count(DISTINCT f.user_id) AS friends_who_bought
FROM network n
JOIN fact_sales  f ON f.user_id = n.node
JOIN dim_product p USING (product_id)
WHERE n.depth > 0
  AND NOT EXISTS (SELECT 1 FROM fact_sales mine
                  WHERE mine.user_id = 303 AND mine.product_id = f.product_id)
GROUP BY p.product_id, p.name, p.category
ORDER BY friends_who_bought DESC, p.name
LIMIT 5;
