#!/usr/bin/env python3
"""Generate the extra, multimodal data for the AI decision demo.

It reads the flash-sale dataset copied into ../data (users, products, orders) and adds
what a real post-mortem would also have on the table:

    data_extra/product_costs.csv        unit cost per product (finance system)
    data_extra/reviews.csv              free-text customer reviews, 4 languages
    data_extra/support_calls/*.txt      call-centre transcripts
    data_extra/images/*.png             competitor flyers + ops dashboard screenshot
    data_extra/docs/*.pdf               logistics contract excerpt + finance note

    python3 tools/generate_extra.py            (run from the ai-decision-demo folder)

The orders themselves are NOT changed, so the three-database demo still matches.
Everything is synthetic. Competitor and logistics company names are invented.
"""
import csv, random, sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "data_extra"
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "docs").mkdir(parents=True, exist_ok=True)
(OUT / "support_calls").mkdir(parents=True, exist_ok=True)

rnd = random.Random(1111)
FLASH = "2025-11-11"
BAD_SKU = 113                       # Kite Smartwatch E59: a bad batch went out in the flash sale

users = pd.read_csv(DATA / "users.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")
items = pd.read_csv(DATA / "order_items.csv")

# ------------------------------------------------------------- product costs
# Cost as a share of list price. Electronics and Groceries are thin-margin;
# Beauty and Fashion are fat-margin. A 40% discount therefore hurts them very differently.
COST_SHARE = {"Electronics": 0.72, "Groceries": 0.70, "Home & Living": 0.58, "Books": 0.55,
              "Toys": 0.52, "Sports": 0.50, "Fashion": 0.38, "Beauty": 0.30}
with open(OUT / "product_costs.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["product_id", "unit_cost_sgd", "supplier_lead_time_days"])
    for p in products.itertuples():
        share = COST_SHARE[p.category] + rnd.uniform(-0.04, 0.04)
        w.writerow([p.product_id, round(p.list_price * share, 2), rnd.choice([7, 10, 14, 21, 28])])

# ------------------------------------------------------------- reviews
lines = (items.merge(orders, on="order_id").merge(products, on="product_id")
         .merge(users.rename(columns={"name": "user_name"}), on="user_id"))
lines = lines[lines.status != "cancelled"]

def noun(name):                      # "Kite Smartwatch E59" -> "smartwatch"
    return " ".join(name.split()[1:-1]).lower()

T = {
 "en": {
  "late": ["Ordered on 11.11 and it took {d} days to arrive. Product is fine but the wait was ridiculous.",
           "Delivery was very late ({d} days). No tracking updates at all.",
           "Great price but the shipping delay spoiled it. {d} days for a {n}.",
           "Arrived after {d} days and the box was dented. Not ordering on sale day again.",
           "Customer service told me the warehouse was overloaded. Took {d} days.",
           "The {n} is good, the delivery was not. {d} days."],
  "defect": ["Watch gets really hot while charging. Returned it.",
             "Battery drains in half a day and the back gets warm. Worried it is unsafe.",
             "Second unit that overheats on the charger. Something is wrong with this batch.",
             "Strap is nice but the battery swelled after two weeks. Requested a refund.",
             "Too hot to wear after charging. Do not buy this one."],
  "discover": ["First time trying {b} thanks to the 11.11 deal. Already bought a second one at full price.",
               "Would never have tried this {n} without the sale. Now it is part of my routine.",
               "Bought it on discount, loved it, reordered at normal price last week."],
  "deal": ["Unbeatable price on 11.11. Very happy.",
           "Got the {n} for almost half price. Quality as expected.",
           "Good deal on sale day, smooth checkout.",
           "Half price {n}, cannot complain."],
  "wait": ["Good deal, but I would not pay full price for this.",
           "I only buy on sale days now, the normal prices feel too high.",
           "Nice {n}. Waiting for the next sale to buy another."],
  "pos": ["Good quality {n}, works as described.", "Fast delivery and well packed.",
          "Value for money. Would recommend.", "Exactly what I needed.",
          "Second time buying this {n}. Still happy.", "Arrived in {d} days, no issues."],
  "neu": ["Okay {n}, a bit overpriced.", "Colour is different from the photos.",
          "Does the job. Nothing special.", "Packaging could be better."],
  "neg": ["Stopped working after a month. Refund was quick though.",
          "Not as described. Returned it.", "Poor quality {n} for this price."],
 },
 "ms": {"late": ["Barang sampai lewat, {d} hari. Kecewa dengan penghantaran."],
        "defect": ["Jam ini jadi panas semasa dicas. Saya pulangkan."],
        "deal": ["Harga sangat murah masa 11.11, puas hati."],
        "pos": ["Kualiti bagus, harga berbaloi."]},
 "id": {"late": ["Pengiriman sangat lambat, {d} hari baru sampai."],
        "defect": ["Jam tangan ini panas sekali waktu dicas. Saya kembalikan."],
        "deal": ["Harganya murah sekali waktu 11.11."],
        "pos": ["Produk bagus, akan beli lagi."]},
 "zh": {"late": ["物流太慢了，等了{d}天才到。"],
        "defect": ["手表充电时很烫，电池也不耐用，已经退货。"],
        "deal": ["双十一价格很划算。"],
        "pos": ["质量很好，下次还会买。"]},
}
RATING = {"late": (1, 3), "defect": (1, 2), "discover": (5, 5), "deal": (4, 5), "wait": (4, 4),
          "pos": (4, 5), "neu": (3, 3), "neg": (1, 2)}

def language(country):
    r = rnd.random()
    if country == "MY" and r < 0.35: return "ms"
    if country == "ID" and r < 0.35: return "id"
    if country == "SG" and r < 0.12: return "zh"
    return "en"

def pick_kind(row, flash):
    bad_batch = row.product_id == BAD_SKU and row.order_date <= "2025-12-31" and row.order_date >= FLASH
    if bad_batch and rnd.random() < 0.55: return "defect"
    r = rnd.random()
    if flash:
        late_rate = 0.48 if row.category in ("Electronics", "Home & Living") else 0.33
        if r < late_rate: return "late"
        r = rnd.random()
        if row.category == "Beauty" and r < 0.35: return "discover"
        if r < 0.55: return "deal"
        if r < 0.75: return "wait"
        return "pos"
    if row.status == "refunded" and r < 0.7: return "neg"
    if r < 0.05: return "late"
    if r < 0.75: return "pos"
    if r < 0.90: return "neu"
    return "neg"

reviews, rid = [], 500001
for oid, grp in lines.groupby("order_id", sort=True):
    row = grp.iloc[rnd.randrange(len(grp))]
    flash = row.order_date == FLASH
    has_bad = (grp.product_id == BAD_SKU).any() and FLASH <= row.order_date <= "2025-12-31"
    rate = 0.60 if has_bad else (0.25 if flash else 0.08)
    if rnd.random() > rate:
        continue
    if has_bad:
        row = grp[grp.product_id == BAD_SKU].iloc[0]
    kind = pick_kind(row, flash)
    lang = language(row.country)
    if kind not in T[lang]:
        lang = "en"
    pool = T[lang][kind]
    delay = rnd.randint(7, 12) if kind == "late" and flash else (rnd.randint(5, 8) if kind == "late" else rnd.randint(2, 4))
    text = rnd.choice(pool).format(d=delay, n=noun(row["name"]), b=row.brand)
    lo, hi = RATING[kind]
    odate = date.fromisoformat(row.order_date)
    reviews.append([rid, int(row.order_id), int(row.product_id), int(row.user_id),
                    (odate + timedelta(days=delay + rnd.randint(0, 5))).isoformat(),
                    rnd.randint(lo, hi), lang, text])
    rid += 1

# Planted data-quality problems, for the "trust but verify" moment.
for r in rnd.sample(reviews, 15):                       # 15 exact duplicates (export ran twice)
    reviews.append(list(r))
for k in range(5):                                      # 5 orphans: order ids that do not exist
    reviews.append([rid, 40001 + k * 17, 89, rnd.randint(1, 2000), "2025-11-20", 1, "en",
                    "Monitor never arrived. Where is my order?"]); rid += 1
for r in rnd.sample(reviews, 4):                        # 4 ratings outside 1..5 (form bug)
    r[5] = rnd.choice([0, 6, 10])
rnd.shuffle(reviews)

with open(OUT / "reviews.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["review_id", "order_id", "product_id", "user_id", "review_date", "rating", "language", "text"])
    w.writerows(reviews)
print(f"reviews.csv: {len(reviews)} rows")

# ------------------------------------------------------------- support call transcripts
CALLS = {
"call_01_smartwatch_overheating.txt": """CALL TRANSCRIPT  |  2025-11-19 10:42  |  Agent: Mei (CS-14)  |  Customer: Max Lim, user 1769
Orders: 12381, 12523 (both 11.11)  |  Product: Kite Smartwatch E59

AGENT: Thanks for calling, how can I help?
CUSTOMER: I bought two of the Kite watches on 11.11, one for my wife. Both get very hot on the charger.
          Last night the back was too hot to touch. I'm scared to leave it charging overnight.
AGENT: I'm sorry. Is there any swelling or smell?
CUSTOMER: My wife's one, the back cover is lifting a little.
AGENT: Please stop using both. I'm logging this as a product safety incident and arranging a courier pick-up.
       Full refund for both.
CUSTOMER: Is this a known problem?
AGENT: [internal note] Third overheating call on E59 this morning. All from the 11.11 stock. Escalated to category manager.
""",
"call_02_late_delivery_13611.txt": """CALL TRANSCRIPT  |  2025-11-18 16:05  |  Agent: Arjun (CS-09)  |  Customer: Chloe Zhang, user 303
Order: 13611 (11.11, 13:20)  |  Items: green tea, coffee beans, desk

CUSTOMER: My order from the 11.11 sale still says "processing". It's been a week.
AGENT: I can see it. The warehouse had a backlog after the sale, it was dispatched yesterday.
CUSTOMER: The desk was the reason I ordered. I needed it for my new flat on the weekend.
AGENT: I understand. I'll add a 5-dollar voucher to your account for the delay.
CUSTOMER: Five dollars... fine. Honestly, next time I'll just order from ShopNova, they do next-day delivery.
AGENT: [internal note] Voucher LATE-5 applied. 6th "ShopNova next-day" mention on my calls this week.
""",
"call_03_platinum_no_early_access.txt": """CALL TRANSCRIPT  |  2025-11-12 09:15  |  Agent: Siti (CS-03)  |  Customer: platinum member (name withheld)

CUSTOMER: I'm a platinum member. By the time I logged in at 8am everything good was sold out.
          The monitors and the earbuds were gone by midnight.
AGENT: The sale was open to all members from midnight.
CUSTOMER: So what is platinum for? Kedai Prime gives their top members early access the day before.
AGENT: I'll pass that feedback to the loyalty team.
AGENT: [internal note] Loyalty-tier complaint. Customer mentioned Kedai Prime early access.
""",
"call_04_waiting_for_sales.txt": """CALL TRANSCRIPT  |  2025-12-02 14:30  |  Agent: Mei (CS-14)  |  Customer: gold member (name withheld)

CUSTOMER: Quick question, when is your next big sale? 12.12?
AGENT: We haven't announced anything yet.
CUSTOMER: OK. I have three things in my cart, I'll wait. After 11.11 the normal prices feel expensive.
AGENT: [internal note] Customer deferring full-price purchases until next sale.
""",
"call_05_ops_voice_memo.txt": """VOICE MEMO (transcribed)  |  2025-11-14  |  From: Daniel Tan, Fulfilment Lead  |  To: Commercial team

Quick update on the backlog. Swiftlane only gave us standard capacity on the eleventh because we didn't
book surge with five working days' notice. That's in Schedule B of the contract, I've attached it.
We dispatched 1,200 on the eleventh and 1,200 on the twelfth, and cleared the rest on the thirteenth.
On-time delivery dropped to the mid-40s on the thirteenth and fourteenth.
If we do 12.12, I need the decision and the expected order volume by the fifth of December so I can book
surge. Even with surge the ceiling is 2,000 a day. If marketing expects 3,000 orders in one day again,
we will be late again. Spread it over three days and we're fine on standard capacity.
""",
"call_06_refund_delay_malay.txt": """CALL TRANSCRIPT  |  2025-11-25 11:20  |  Agent: Nurul (CS-11, Kuala Lumpur)  |  Customer: Oliver Lim, user 840, MY
Order: 12136 (11.11, cancelled)

CUSTOMER: Saya batalkan order 11.11 minggu lepas, tapi refund belum masuk lagi.
          (I cancelled my 11.11 order last week but the refund hasn't come in yet.)
AGENT: Maaf encik. Refund untuk jualan 11.11 mengambil masa 7 hingga 10 hari bekerja sebab jumlah yang tinggi.
       (Sorry sir. Refunds for the 11.11 sale take 7 to 10 working days because of the high volume.)
CUSTOMER: Sepuluh hari? Itu terlalu lama.  (Ten days? That's too long.)
AGENT: [internal note] Refund backlog complaint. Payment gateway queue still 900+ items.
""",
}
for fname, text in CALLS.items():
    (OUT / "support_calls" / fname).write_text(text, encoding="utf-8")
print("support_calls: ", len(CALLS))
