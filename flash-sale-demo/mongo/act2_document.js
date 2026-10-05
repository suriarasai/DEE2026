// =====================================================================
// ACT 2 - DOCUMENT questions, in their natural home (MongoDB)
// Shell:  double-click shell-mongo.cmd
//   (or: podman exec -it flashsale-mongo mongosh flashsale)
// =====================================================================

// 2.1  THE QUESTION: show order 13611 the way the app wants it.
//      The order was stored as one object, so reading it is one lookup by key.
db.orders.findOne({ _id: 13611 })

// 2.2  Three users, three different shapes, no migration.
db.users.find({ _id: { $in: [2, 8, 12] } }, { name: 1, profile: 1 })

// 2.3  Query inside the document: how many users have an iOS device?
//      Dot notation walks into nested objects and arrays.
db.users.countDocuments({ 'profile.devices.type': 'ios' })

// 2.4  Marketing wants an early-access perk on every platinum profile.
//      One line. The "perks" object does not exist yet; $set creates it on the way.
db.users.updateMany({ loyalty_tier: 'platinum' }, { $set: { 'profile.perks.early_access': true } })
db.users.countDocuments({ 'profile.perks.early_access': true })

// 2.5  THE NUANCE: flexibility has a price. Nothing enforced a schema, so how many
//      different profile shapes does the application now have to cope with?
db.users.aggregate([
  { $project: { shape: { $map: { input: { $objectToArray: '$profile' }, in: '$$this.k' } } } },
  { $group: { _id: '$shape' } },
  { $count: 'distinct_profile_shapes' }
])

//      The five most common ones:
db.users.aggregate([
  { $project: { shape: { $map: { input: { $objectToArray: '$profile' }, in: '$$this.k' } } } },
  { $group: { _id: '$shape', users: { $sum: 1 } } },
  { $sort: { users: -1 } },
  { $limit: 5 }
])
