// =====================================================================
// ACT 3 - GRAPH questions, asked of MongoDB (the twist)
// The graph is the referred_by field on each user.
// Shell:  double-click shell-mongo.cmd
//   (or: podman exec -it flashsale-mongo mongosh flashsale)
// =====================================================================

// 3.1  THE QUESTION: who did Chloe Zhang (user 303) bring in, three levels deep?
//      $graphLookup follows a field repeatedly. maxDepth counts from 0, so 2 means three levels.
db.users.aggregate([
  { $match: { _id: 303 } },
  { $graphLookup: {
      from: 'users',
      startWith: '$_id',
      connectFromField: '_id',
      connectToField: 'referred_by',
      as: 'chain',
      maxDepth: 2,
      depthField: 'depth' } },
  { $unwind: '$chain' },
  { $sort: { 'chain._id': 1 } },
  { $group: { _id: { $toInt: { $add: ['$chain.depth', 1] } }, people: { $sum: 1 }, who: { $push: '$chain.name' } } },
  { $sort: { _id: 1 } }
])

// 3.2  THE LOOP: user 1998 with no depth limit.
//      Unlike the SQL version this does not hang; $graphLookup remembers where it has been.
db.users.aggregate([
  { $match: { _id: 1998 } },
  { $graphLookup: {
      from: 'users',
      startWith: '$_id',
      connectFromField: '_id',
      connectToField: 'referred_by',
      as: 'chain',
      depthField: 'depth' } },
  { $unwind: '$chain' },
  { $sort: { 'chain.depth': 1 } },
  { $project: { _id: 0, user_id: '$chain._id', name: '$chain.name', depth: { $toInt: { $add: ['$chain.depth', 1] } } } }
])

// 3.3  THE HARDER QUESTION: how are users 1884 and 1045 connected?
//      There is no shortest-path operator. The closest we get is each person's
//      upline; finding where the two meet is left to application code.
db.users.aggregate([
  { $match: { _id: { $in: [1884, 1045] } } },
  { $graphLookup: {
      from: 'users',
      startWith: '$referred_by',
      connectFromField: 'referred_by',
      connectToField: '_id',
      as: 'upline',
      depthField: 'depth' } },
  { $project: { name: 1, upline: { $map: {
      input: { $sortArray: { input: '$upline', sortBy: { depth: 1 } } },
      in: '$$this.name' } } } }
])
