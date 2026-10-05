// =====================================================================
// ACT 3 - GRAPH questions, in their natural home (Neo4j)
// Shell:    double-click shell-neo4j.cmd
//   (or: podman exec -it flashsale-neo4j cypher-shell)
// Browser:  http://localhost:7474  (choose "No authentication")
// =====================================================================

// 3.1  THE QUESTION: who did Chloe Zhang (user 303) bring in, three levels deep?
//      "*1..3" means "follow this relationship one to three times".
MATCH path = (:User {user_id: 303})-[:REFERRED*1..3]->(r:User)
WITH length(path) AS depth, r ORDER BY r.user_id
RETURN depth, count(r) AS people, collect(r.name) AS who
ORDER BY depth;

//      THE PICTURE: paste this one into the Browser instead of the shell.
MATCH path = (:User {user_id: 303})-[:REFERRED*1..3]->(:User)
RETURN path;

// 3.2  THE LOOP: user 1998 with no depth limit. No hang and no special syntax,
//      because a path may not reuse a relationship.
MATCH path = (:User {user_id: 1998})-[:REFERRED*]->(r:User)
RETURN r.user_id AS user_id, r.name AS name, length(path) AS depth
ORDER BY depth;

// 3.3  THE HARDER QUESTION: how are Rohan Santoso (1884) and Max Wilson (1045) connected?
//      No arrow on the pattern, so the path may run up or down the referral chain.
MATCH (a:User {user_id: 1884}), (b:User {user_id: 1045})
MATCH path = shortestPath((a)-[:REFERRED*..15]-(b))
RETURN length(path) AS hops, [n IN nodes(path) | n.name] AS route;

//      THE PICTURE: paste this one into the Browser.
MATCH (a:User {user_id: 1884}), (b:User {user_id: 1045})
MATCH path = shortestPath((a)-[:REFERRED*..15]-(b))
RETURN path;

// 3.4  THE PAYOFF: recommend products to user 303. What have people within two
//      referral hops of her bought that she has not?
//      This mixes the referral graph with the purchase graph in one pattern.
MATCH (me:User {user_id: 303})-[:REFERRED*1..2]-(friend:User)
MATCH (friend)-[:PLACED]->(:Order)-[:CONTAINS]->(p:Product)
WHERE NOT EXISTS { (me)-[:PLACED]->(:Order)-[:CONTAINS]->(p) }
RETURN p.name AS name, p.category AS category, count(DISTINCT friend) AS friends_who_bought
ORDER BY friends_who_bought DESC, name
LIMIT 5;
