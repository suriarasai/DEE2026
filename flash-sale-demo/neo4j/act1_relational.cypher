// =====================================================================
// ACT 1 - RELATIONAL SCHEMA question, asked of Neo4j (the twist)
// Shell:    double-click shell-neo4j.cmd
//   (or: podman exec -it flashsale-neo4j cypher-shell)
// Browser:  http://localhost:7474  (choose "No authentication")
// Paste one block at a time. Every statement ends with a semicolon.
// =====================================================================

// 1.1  THE SAME QUESTION: flash-sale revenue by category and loyalty tier.
//      It reads nicely: walk from users to their orders to the products.
//      But there is no fact table. Every aggregate walks relationships one by one,
//      which is fine for 56,000 order lines and painful for 5 billion.
MATCH (u:User)-[:PLACED]->(o:Order)-[c:CONTAINS]->(p:Product)
WHERE o.order_date = date('2025-11-11')
RETURN p.category AS category, u.loyalty_tier AS tier,
       round(sum(c.quantity * c.unit_price), 2) AS revenue
ORDER BY revenue DESC
LIMIT 10;

// 1.2  THE NUANCE: the category rename. Like the relational schema, a graph stores each
//      product once, so only these nodes would change.
MATCH (p:Product {category: 'Home & Living'})
RETURN count(p) AS products_to_rename;
