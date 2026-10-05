// =====================================================================
// MongoDB: load the shared dataset as DOCUMENTS.
// Run by start.cmd and seed.cmd, which copy this file and the data into the container first.
// Safe to re-run: it drops the database first.
//
// Same users, products and orders as Postgres and Neo4j, but shaped the way a
// document store wants them: each order carries its line items inside it.
// =====================================================================
const fs = require('fs');

function load(name) {
  // One JSON document per line. EJSON turns {"$date": "..."} into real dates.
  return fs.readFileSync(`/dataset/${name}.jsonl`, 'utf8')
    .trim().split('\n').map(line => EJSON.parse(line));
}

db.dropDatabase();

db.users.insertMany(load('users'));
db.products.insertMany(load('products'));
const orders = load('orders');
for (let i = 0; i < orders.length; i += 5000) {
  db.orders.insertMany(orders.slice(i, i + 5000));
}

db.users.createIndex({ referred_by: 1 });   // Act 3 follows this field
db.orders.createIndex({ user_id: 1 });
// NOTE: there is deliberately NO index on orders.order_ts. Act 1 adds it live.

const totals = db.orders.aggregate([
  { $unwind: '$items' },
  { $group: { _id: null, order_lines: { $sum: 1 },
              total_revenue: { $sum: { $multiply: ['$items.quantity', '$items.unit_price'] } } } }
]).toArray()[0];

printjson({
  users: db.users.countDocuments({}),
  products: db.products.countDocuments({}),
  orders: db.orders.countDocuments({}),
  order_lines: totals.order_lines,
  total_revenue: Math.round(totals.total_revenue * 100) / 100
});
