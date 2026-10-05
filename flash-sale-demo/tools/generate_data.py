#!/usr/bin/env python3
"""Generate the shared flash-sale dataset used by all three databases.

The repo already ships with the generated files in ../data, so you only need
this script if you want to change the size or shape of the data.

    python3 tools/generate_data.py

Output (all describing the SAME users, products and orders):
    data/users.csv, user_profiles.csv, products.csv, orders.csv, order_items.csv
        -> loaded by Postgres and Neo4j
    data/users.jsonl, products.jsonl, orders.jsonl
        -> loaded by MongoDB (same records, nested the way a document store wants)

If you change the seed or the sizes, the IDs used in the act files (demo user,
demo order, shortest-path pair) and the expected results in the README change too.
"""
import csv, json, random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 42
N_USERS, N_PRODUCTS, N_ORDERS = 2000, 150, 20000
FLASH_DAY, FLASH_ORDERS = date(2025, 11, 11), 3000
FIRST_DAY, LAST_DAY = date(2025, 10, 1), date(2026, 9, 30)
CYCLE = (1998, 1999, 2000)          # deliberately broken referral loop

rnd = random.Random(SEED)
out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)

FIRST = """Aisha Wei Arjun Mei Daniel Siti Kenji Priya Lucas Nurul Hiro Ananya Ethan Li Rahul
Sofia Jun Farah Noah Yuki Ravi Chloe Ming Amir Emma Kai Devi Oliver Hana Vikram Grace Tao Zara
Liam Sakura Imran Mia Jia Rohan Isla Bao Leila Max Aiko Sanjay Ella Hui Omar Ruby Ken""".split()
LAST = """Tan Lim Lee Ng Wong Sharma Patel Suzuki Tanaka Rahman Abdullah Santoso Wijaya Smith Jones
Brown Taylor Muller Schmidt Kumar Singh Chen Wang Zhang Goh Chua Ong Teo Nair Iyer Sato Ito
Hassan Ismail Putri Pratama Wilson Clark Hall Weber Koh Yeo Chan Ho Gupta Reddy Kato Mori""".split()
COUNTRIES = [("SG", 30), ("MY", 15), ("ID", 15), ("IN", 10), ("AU", 8), ("JP", 6), ("US", 6), ("GB", 5), ("DE", 5)]
TIERS = [("bronze", 55), ("silver", 28), ("gold", 13), ("platinum", 4)]
CATEGORIES = {  # category: (price low, price high, product nouns)
    "Electronics":   (49, 1299, ["Earbuds", "Phone", "Tablet", "Monitor", "Speaker", "Smartwatch", "Router", "Keyboard"]),
    "Fashion":       (15, 220,  ["Sneakers", "Jacket", "T-Shirt", "Jeans", "Dress", "Backpack", "Sunglasses"]),
    "Home & Living": (12, 480,  ["Lamp", "Air Fryer", "Kettle", "Bedsheet Set", "Vacuum", "Desk", "Cookware Set"]),
    "Beauty":        (8, 150,   ["Serum", "Moisturiser", "Sunscreen", "Lipstick", "Shampoo", "Perfume"]),
    "Sports":        (10, 350,  ["Yoga Mat", "Dumbbells", "Running Shorts", "Water Bottle", "Racket", "Bike Helmet"]),
    "Toys":          (9, 180,   ["Building Set", "Puzzle", "RC Car", "Plush Bear", "Board Game"]),
    "Groceries":     (3, 45,    ["Coffee Beans", "Olive Oil", "Granola", "Green Tea", "Dark Chocolate", "Jasmine Rice"]),
    "Books":         (9, 60,    ["Novel", "Cookbook", "Travel Guide", "Tech Handbook", "Biography"]),
}
BRANDS = "Orion Lumo Kite Nimbus Terra Pico Vanta Sora Maple Zephyr Koru Atlas".split()


def weighted(pairs):
    return rnd.choices([p[0] for p in pairs], weights=[p[1] for p in pairs])[0]


# ---------------------------------------------------------------- users
signup_span = (date(2025, 9, 30) - date(2024, 1, 1)).days
users = []
for uid in range(1, N_USERS + 1):
    referred_by = rnd.randint(1, uid - 1) if uid > 5 and rnd.random() < 0.70 else None
    profile = {}  # marketing keeps adding features, so every profile has a different shape
    if rnd.random() < 0.70:
        profile["preferences"] = {"newsletter": rnd.random() < 0.5,
                                  "categories": rnd.sample(list(CATEGORIES), rnd.randint(1, 3))}
    if rnd.random() < 0.40:
        profile["devices"] = [{"type": rnd.choice(["ios", "android", "web"]),
                               "last_seen": str(FIRST_DAY + timedelta(days=rnd.randint(0, 364)))}
                              for _ in range(rnd.randint(1, 2))]
    if rnd.random() < 0.25:
        profile["marketing"] = {"campaign": rnd.choice(["1111-early-access", "refer-a-friend", "app-install"]),
                                "ab_bucket": rnd.choice("AB")}
    if rnd.random() < 0.35:
        profile["addresses"] = [{"label": lab, "postcode": str(rnd.randint(10000, 99999))}
                                for lab in rnd.sample(["home", "office", "parents"], rnd.randint(1, 2))]
    users.append({
        "user_id": uid,
        "name": f"{rnd.choice(FIRST)} {rnd.choice(LAST)}",
        "country": weighted(COUNTRIES),
        "loyalty_tier": weighted(TIERS),
        "signup_date": date(2024, 1, 1) + timedelta(days=(uid - 1) * signup_span // (N_USERS - 1)),
        "referred_by": referred_by,
        "profile": profile,
    })
# The data-quality bug for Act 3: 1998 referred 1999, who referred 2000, who "referred" 1998.
a, b, c = CYCLE
users[a - 1]["referred_by"], users[b - 1]["referred_by"], users[c - 1]["referred_by"] = c, a, b

# ---------------------------------------------------------------- products
products = []
for pid in range(1, N_PRODUCTS + 1):
    cat = list(CATEGORIES)[(pid - 1) % len(CATEGORIES)]
    lo, hi, nouns = CATEGORIES[cat]
    brand = rnd.choice(BRANDS)
    products.append({
        "product_id": pid,
        "name": f"{brand} {rnd.choice(nouns)} {rnd.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rnd.randint(10, 99)}",
        "category": cat,
        "brand": brand,
        "list_price": round(rnd.uniform(lo, hi), 2),
    })

# ---------------------------------------------------------------- orders
other_days = [FIRST_DAY + timedelta(days=d) for d in range((LAST_DAY - FIRST_DAY).days + 1)]
other_days.remove(FLASH_DAY)
midnight = datetime.min.time()
stamps = [datetime.combine(FLASH_DAY, midnight) + timedelta(seconds=rnd.randint(0, 86399))
          for _ in range(FLASH_ORDERS)]
stamps += [datetime.combine(rnd.choice(other_days), midnight) + timedelta(seconds=rnd.randint(0, 86399))
           for _ in range(N_ORDERS - FLASH_ORDERS)]
stamps.sort()

user_order = list(range(1, N_USERS + 1)); rnd.shuffle(user_order)
user_w = [1 / (rank ** 0.6) for rank in range(1, N_USERS + 1)]       # a few heavy buyers
prod_order = list(range(1, N_PRODUCTS + 1)); rnd.shuffle(prod_order)
prod_w = [1 / (rank ** 0.5) for rank in range(1, N_PRODUCTS + 1)]    # a few best-sellers

orders = []
for i, ts in enumerate(stamps):
    flash = ts.date() == FLASH_DAY
    n_lines = rnd.choice([1, 2, 2, 3, 3, 4, 5])
    picked = list(dict.fromkeys(rnd.choices(prod_order, weights=prod_w, k=n_lines)))  # distinct products
    items = []
    for line_no, pid in enumerate(picked, start=1):
        p = products[pid - 1]
        discount = rnd.uniform(0.30, 0.50) if flash else rnd.choice([0, 0, 0, 0.05, 0.10])
        items.append({"line_no": line_no, "product_id": pid, "name": p["name"], "category": p["category"],
                      "quantity": rnd.choice([1, 1, 1, 2, 2, 3]),
                      "unit_price": round(p["list_price"] * (1 - discount), 2)})
    orders.append({
        "order_id": 10001 + i,
        "user_id": rnd.choices(user_order, weights=user_w)[0],
        "order_ts": ts,
        "status": rnd.choices(["delivered", "refunded", "cancelled"], weights=[90, 6, 4])[0],
        "channel": rnd.choices(["ios", "android", "web"], weights=[40, 35, 25])[0],
        "items": items,
    })


# ---------------------------------------------------------------- write CSV (Postgres + Neo4j)
def write_csv(name, header, rows):
    with open(out / name, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


write_csv("users.csv", ["user_id", "name", "country", "loyalty_tier", "signup_date", "referred_by"],
          [[u["user_id"], u["name"], u["country"], u["loyalty_tier"], u["signup_date"], u["referred_by"] or ""]
           for u in users])
write_csv("user_profiles.csv", ["user_id", "profile"],
          [[u["user_id"], json.dumps(u["profile"])] for u in users])
write_csv("products.csv", ["product_id", "name", "category", "brand", "list_price"],
          [[p["product_id"], p["name"], p["category"], p["brand"], f'{p["list_price"]:.2f}'] for p in products])
write_csv("orders.csv", ["order_id", "user_id", "order_date", "order_ts", "status", "channel"],
          [[o["order_id"], o["user_id"], o["order_ts"].date(), o["order_ts"].strftime("%Y-%m-%d %H:%M:%S"),
            o["status"], o["channel"]] for o in orders])
write_csv("order_items.csv", ["order_id", "line_no", "product_id", "quantity", "unit_price"],
          [[o["order_id"], it["line_no"], it["product_id"], it["quantity"], f'{it["unit_price"]:.2f}']
           for o in orders for it in o["items"]])


# ---------------------------------------------------------------- write JSON Lines (MongoDB)
def write_jsonl(name, docs):
    with open(out / name, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


def iso(d):  # MongoDB Extended JSON date
    return {"$date": d.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(d, datetime) else f"{d}T00:00:00Z"}


mongo_users = []
for u in users:
    doc = {"_id": u["user_id"], "name": u["name"], "country": u["country"],
           "loyalty_tier": u["loyalty_tier"], "signup_date": iso(u["signup_date"])}
    if u["referred_by"]:
        doc["referred_by"] = u["referred_by"]       # the field simply does not exist for organic sign-ups
    doc["profile"] = u["profile"]
    mongo_users.append(doc)
write_jsonl("users.jsonl", mongo_users)
write_jsonl("products.jsonl",
            [{"_id": p["product_id"], "name": p["name"], "category": p["category"], "brand": p["brand"],
              "list_price": p["list_price"]} for p in products])
write_jsonl("orders.jsonl",
            [{"_id": o["order_id"], "user_id": o["user_id"], "order_ts": iso(o["order_ts"]),
              "status": o["status"], "channel": o["channel"], "items": o["items"]} for o in orders])

print(f"users={len(users)} products={len(products)} orders={len(orders)} "
      f"order_lines={sum(len(o['items']) for o in orders)}")
