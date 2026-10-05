// =====================================================================
// ACT 2 - DOCUMENT questions, asked of Neo4j (the twist)
// Shell:  double-click shell-neo4j.cmd
//   (or: podman exec -it flashsale-neo4j cypher-shell)
// =====================================================================

// 2.1  THE QUESTION: show order 13611 the way the app wants it.
//      Like Postgres, the graph has the order in pieces (one Order node, three
//      CONTAINS relationships, three Product nodes), so we collect them back together.
MATCH (u:User)-[:PLACED]->(o:Order {order_id: 13611})-[c:CONTAINS]->(p:Product)
WITH u, o, c, p ORDER BY c.line_no
WITH u, o, collect({product: p.name, category: p.category,
                    quantity: c.quantity, unit_price: c.unit_price}) AS items
RETURN {order_id: o.order_id, order_date: toString(o.order_date), status: o.status,
        buyer: {name: u.name, tier: u.loyalty_tier}, items: items} AS order_document;

// 2.2  Flat flexibility works: any node can gain a property with no migration.
MATCH (u:User {loyalty_tier: 'platinum'})
SET u.perk_early_access = true
RETURN count(u) AS users_updated;

// 2.3  EXPECTED ERROR - run this one last. Try to store a nested profile.
//      A property can hold a value or a list of values, never an object inside an object.
//      That is why the profiles were not loaded here: in a graph you would either
//      flatten them, store them as an opaque string, or turn devices and addresses
//      into their own nodes.
MATCH (u:User {user_id: 303})
SET u.profile = {preferences: {newsletter: false, categories: ['Beauty']}};
