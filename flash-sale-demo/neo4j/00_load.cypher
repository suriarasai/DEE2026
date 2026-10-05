// =====================================================================
// Neo4j: load the shared dataset as a GRAPH.
// Run by start.cmd and seed.cmd, which copy this file and the data into the container first.
// Safe to re-run: it deletes everything first.
//
// Same users, products and orders as Postgres and MongoDB, but shaped the way a
// graph wants them: things are nodes, and every connection is stored as a relationship.
//
//   (User)-[:REFERRED]->(User)
//   (User)-[:PLACED]->(Order)-[:CONTAINS {quantity, unit_price}]->(Product)
// =====================================================================

MATCH (n) DETACH DELETE n;

CREATE CONSTRAINT user_id    IF NOT EXISTS FOR (u:User)    REQUIRE u.user_id    IS UNIQUE;
CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE;
CREATE CONSTRAINT order_id   IF NOT EXISTS FOR (o:Order)   REQUIRE o.order_id   IS UNIQUE;
CREATE INDEX order_date      IF NOT EXISTS FOR (o:Order)   ON (o.order_date);

// ---------- Nodes ----------
LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
CREATE (:User {user_id: toInteger(row.user_id), name: row.name, country: row.country,
               loyalty_tier: row.loyalty_tier, signup_date: date(row.signup_date)});

LOAD CSV WITH HEADERS FROM 'file:///products.csv' AS row
CREATE (:Product {product_id: toInteger(row.product_id), name: row.name, category: row.category,
                  brand: row.brand, list_price: toFloat(row.list_price)});

// ---------- Relationships ----------
// The referred_by column becomes a first-class REFERRED relationship.
LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
WITH row WHERE row.referred_by IS NOT NULL AND row.referred_by <> ''
MATCH (referrer:User {user_id: toInteger(row.referred_by)})
MATCH (referred:User {user_id: toInteger(row.user_id)})
CREATE (referrer)-[:REFERRED]->(referred);

LOAD CSV WITH HEADERS FROM 'file:///orders.csv' AS row
MATCH (u:User {user_id: toInteger(row.user_id)})
CREATE (u)-[:PLACED]->(:Order {order_id: toInteger(row.order_id), order_date: date(row.order_date),
                               status: row.status, channel: row.channel});

LOAD CSV WITH HEADERS FROM 'file:///order_items.csv' AS row
MATCH (o:Order {order_id: toInteger(row.order_id)})
MATCH (p:Product {product_id: toInteger(row.product_id)})
CREATE (o)-[:CONTAINS {line_no: toInteger(row.line_no), quantity: toInteger(row.quantity),
                       unit_price: toFloat(row.unit_price)}]->(p);

// NOTE: user_profiles.csv is deliberately NOT loaded. Act 2 shows why.

// ---------- Check: these numbers must match Postgres and MongoDB ----------
MATCH (u:User) WITH count(u) AS users
MATCH (p:Product) WITH users, count(p) AS products
MATCH (o:Order)-[c:CONTAINS]->(:Product)
RETURN users, products, count(DISTINCT o) AS orders, count(c) AS order_lines,
       round(sum(c.quantity * c.unit_price), 2) AS total_revenue;
