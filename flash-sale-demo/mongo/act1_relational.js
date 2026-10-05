// =====================================================================
// ACT 1 - RELATIONAL SCHEMA question, asked of MongoDB (the twist)
// Shell:  double-click shell-mongo.cmd
//   (or: podman exec -it flashsale-mongo mongosh flashsale)
// Then paste one block at a time.
// =====================================================================

// 1.1  THE SAME QUESTION: flash-sale revenue by category and loyalty tier.
//      Category is copied into every order line, so that part is free.
//      Loyalty tier lives on the user, so we need a $lookup (MongoDB's join),
//      and the line items are an array, so we need $unwind to get one row per line.
db.orders.aggregate([
  { $match: { order_ts: { $gte: ISODate('2025-11-11'), $lt: ISODate('2025-11-12') } } },
  { $lookup: { from: 'users', localField: 'user_id', foreignField: '_id', as: 'buyer' } },
  { $unwind: '$buyer' },
  { $unwind: '$items' },
  { $group: {
      _id: { category: '$items.category', tier: '$buyer.loyalty_tier' },
      revenue: { $sum: { $multiply: ['$items.quantity', '$items.unit_price'] } } } },
  { $project: { _id: 0, category: '$_id.category', tier: '$_id.tier', revenue: { $round: ['$revenue', 2] } } },
  { $sort: { revenue: -1 } },
  { $limit: 10 }
])

// 1.2  SAME BUG, DIFFERENT DATABASE: look for stage "COLLSCAN" and totalDocsExamined: 20000.
db.orders.find({ order_ts: { $gte: ISODate('2025-11-11'), $lt: ISODate('2025-11-12') } })
  .explain('executionStats').executionStats

// 1.3  THE FIX. Re-run 1.2: totalDocsExamined drops to 3000.
db.orders.createIndex({ order_ts: 1 })

// 1.4  THE NUANCE: marketing renames a category.
//      We copied the category into every order line to avoid a join.
//      How many orders now hold a copy that needs rewriting?
db.orders.countDocuments({ 'items.category': 'Home & Living' })

//      The rename has to visit every one of them, and reach inside each array...
db.orders.updateMany(
  { 'items.category': 'Home & Living' },
  { $set: { 'items.$[line].category': 'Home & Garden' } },
  { arrayFilters: [{ 'line.category': 'Home & Living' }] }
)
//      ...and the product catalogue as well. Miss one and the reports disagree.
db.products.updateMany({ category: 'Home & Living' }, { $set: { category: 'Home & Garden' } })

//      Put it back so the other acts still match Postgres and Neo4j.
db.orders.updateMany(
  { 'items.category': 'Home & Garden' },
  { $set: { 'items.$[line].category': 'Home & Living' } },
  { arrayFilters: [{ 'line.category': 'Home & Garden' }] }
)
db.products.updateMany({ category: 'Home & Garden' }, { $set: { category: 'Home & Living' } })
