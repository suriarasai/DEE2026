#!/usr/bin/env python3
"""Reference answers for the AI decision demo. Presenter's safety net, not for the audience.

    python3 answer_key/answer_key.py            (from the ai-decision-demo folder)

Prints the numbers the AI should arrive at, so you can check it live.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
EXTRA = HERE / "data_extra"
FLASH, BAD_SKU = pd.Timestamp("2025-11-11"), 113
GATEWAY, FULFIL, MARKETING, VOUCHERS = 0.022, 4.20, 85_000, 1_630 * 5

o = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
i = pd.read_csv(DATA / "order_items.csv")
p = pd.read_csv(DATA / "products.csv")
u = pd.read_csv(DATA / "users.csv", parse_dates=["signup_date"])
c = pd.read_csv(EXTRA / "product_costs.csv")
r = pd.read_csv(EXTRA / "reviews.csv", parse_dates=["review_date"])

f = i.merge(o, on="order_id").merge(p, on="product_id").merge(c, on="product_id")
f["revenue"] = f.quantity * f.unit_price
f["cogs"] = f.quantity * f.unit_cost_sgd
f["discount"] = 1 - f.unit_price / f.list_price
fmt = lambda x: f"{x:,.0f}"
H = lambda t: print(f"\n## {t}\n")

H("1. Sanity checks (should match the three-database demo)")
print("Total revenue, all statuses:", f"{f.revenue.sum():,.2f}", "(README says 16,255,674.43)")
day = f[f.order_date == FLASH]
print("11.11 orders:", day.order_id.nunique(), "| lines:", len(day), "| revenue all statuses:", fmt(day.revenue.sum()))
print("Average discount on 11.11:", f"{day.discount.mean():.1%}", "| other days:", f"{f[f.order_date != FLASH].discount.mean():.1%}")
typical = f[f.order_date != FLASH].groupby("order_date").revenue.sum()
print("Typical day revenue (median):", fmt(typical.median()), "| 11.11 is", f"{day.revenue.sum() / typical.median():.0f}x")

H("2. Did 11.11 make money? (delivered orders only; refunds/cancellations earn nothing)")
d = day[day.status == "delivered"].copy()
d.loc[d.product_id == BAD_SKU, "revenue"] = 0          # recalled and refunded, stock written off
orders_by_cat = d.groupby("category").order_id.nunique()
cat = d.groupby("category").agg(revenue=("revenue", "sum"), cogs=("cogs", "sum"))
cat["gross_margin"] = cat.revenue - cat.cogs
cat["gateway"] = cat.revenue * GATEWAY
# fulfilment is per order; allocate each order's S$4.20 across its lines by revenue share
d["fulfil"] = FULFIL * d.revenue / d.groupby("order_id").revenue.transform("sum").replace(0, 1)
cat["fulfilment"] = d.groupby("category").fulfil.sum()
cat["contribution"] = cat.gross_margin - cat.gateway - cat.fulfilment
cat["margin_%"] = (cat.contribution / cat.revenue * 100).round(1)
cat = cat.sort_values("contribution")
print(cat.round(0).to_string())
tot = cat.contribution.sum()
print("\nContribution before campaign costs:", fmt(tot))
print("Less marketing 85,000 and late vouchers", fmt(VOUCHERS), "=> 11.11 result:", fmt(tot - MARKETING - VOUCHERS))
e59 = day[(day.product_id == BAD_SKU) & (day.status == "delivered")]
print("Kite Smartwatch E59 on 11.11: units", int(e59.quantity.sum()), "| revenue refunded", fmt(e59.revenue.sum()), "| stock written off", fmt(e59.cogs.sum()))

H("3. Break-even discount by category (price after discount, less 2.2% gateway, = unit cost)")
cs = f.drop_duplicates("product_id").groupby("category").apply(lambda g: (g.unit_cost_sgd / g.list_price).mean(), include_groups=False)
be = (1 - cs / (1 - GATEWAY)).rename("break_even_discount")
print((be * 100).round(1).sort_values().astype(str).add("%").to_string())
print("\nSensitivity: contribution margin % of list price at each discount level")
print(pd.DataFrame({f"{k}% off": ((1 - k / 100) * (1 - GATEWAY) - cs) * 100 for k in (10, 20, 30, 40, 50)}).round(1).to_string())

H("4. Did 11.11 create customers? (the marketing claim)")
print("Sign-ups after 30 Sep 2025:", int((u.signup_date > "2025-09-30").sum()), "-> no new users at all; the claim cannot be supported by this data")
buyers = set(day.order_id.map(o.set_index("order_id").user_id))
def spend(lo, hi):
    w = f[(f.order_date >= lo) & (f.order_date < hi) & (f.order_date != FLASH)]
    return w.groupby(w.user_id.isin(buyers)).revenue.sum() / pd.Series({True: len(buyers), False: u.user_id.nunique() - len(buyers)})
pre, post = spend("2025-10-01", "2025-11-11"), spend("2025-11-12", "2026-02-10")
print("Flash buyers:", len(buyers))
print(f"Avg spend per user, 41 days before: flash buyers {pre[True]:,.0f} vs others {pre[False]:,.0f}")
print(f"Avg spend per user, 90 days after:  flash buyers {post[True]:,.0f} vs others {post[False]:,.0f}")
by_tier = day.merge(u[["user_id", "loyalty_tier"]], on="user_id").groupby("loyalty_tier").order_id.nunique()
print("11.11 orders by tier:", by_tier.to_dict(), "| tier mix of all users:", u.loyalty_tier.value_counts(normalize=True).round(2).to_dict())

H("5. What the customers said (reviews.csv)")
print("Rows:", len(r), "| exact duplicates:", int(r.duplicated().sum()),
      "| orphan order_ids:", int((~r.order_id.isin(o.order_id)).sum()),
      "| ratings outside 1-5:", int((~r.rating.between(1, 5)).sum()))
rc = r.drop_duplicates()
rc = rc[rc.order_id.isin(o.order_id) & rc.rating.between(1, 5)]
rc = rc.merge(o[["order_id", "order_date"]], on="order_id").merge(p[["product_id", "category"]], on="product_id")
rc["flash"] = rc.order_date == FLASH
late = rc.text.str.contains(r"days|late|lambat|lewat|慢|delivery", case=False) & (rc.rating <= 3)
rc["late"] = late
print("Clean reviews:", len(rc))
print(rc.groupby("flash").agg(reviews=("rating", "size"), avg_rating=("rating", "mean"), late_share=("late", "mean")).round(2).to_string())
print("\nLate-delivery share of 11.11 reviews by category:")
print(rc[rc.flash].groupby("category").late.mean().mul(100).round(0).sort_values(ascending=False).to_string())
w = rc[rc.product_id == BAD_SKU]
hot = w.text.str.contains(r"hot|overheat|swell|warm|panas|烫", case=False)
batch = w.order_date.between("2025-11-11", "2025-12-31")
print(f"\nKite Smartwatch E59: {len(w)} reviews, {int(hot.sum())} mention heat/battery. "
      f"Orders 11 Nov - 31 Dec: {int(hot[batch].sum())} of {int(batch.sum())}; all other dates: {int(hot[~batch].sum())} of {int((~batch).sum())}")
print("Languages:", rc.language.value_counts().to_dict())
print("'Only buy on sale' / 'would not pay full price' reviews on 11.11:",
      int(rc[rc.flash].text.str.contains("full price|sale days|next sale", case=False).sum()))

H("6. Basket check: is Electronics a loss leader that pulls in high-margin items?")
od = day[day.status == "delivered"]
has = od.groupby("order_id").category.agg(set)
elec = has[has.apply(lambda x: "Electronics" in x)]
print(f"Delivered 11.11 orders with Electronics: {len(elec)}; of those also containing Beauty or Fashion: "
      f"{elec.apply(lambda x: bool(x & {'Beauty', 'Fashion'})).mean():.0%}; "
      f"same share among orders without Electronics: {has[~has.index.isin(elec.index)].apply(lambda x: bool(x & {'Beauty', 'Fashion'})).mean():.0%}")

H("7. 12.12 options, same customers and basket as 11.11 (E59 batch excluded everywhere)")
base = od[od.product_id != BAD_SKU].copy()
TARGET = {"Electronics": .15, "Groceries": .15, "Home & Living": .25, "Books": .30, "Toys": .30,
          "Sports": .30, "Fashion": .40, "Beauty": .40}
VOLUME = {"Electronics": .5, "Groceries": .5, "Home & Living": .8}          # ASSUMPTION: shallower discount, fewer units
def scenario(targeted, days):
    x = base.copy()
    if targeted:
        x["unit_price"] = x.list_price * (1 - x.category.map(TARGET))
        x["w"] = x.category.map(VOLUME).fillna(1.0)
    else:
        x["w"] = 1.0
    contrib = (x.w * x.quantity * (x.unit_price * (1 - GATEWAY) - x.unit_cost_sgd)).sum()
    revenue = (x.w * x.quantity * x.unit_price).sum()
    n_orders = (x.groupby("order_id").w.max()).sum()
    per_day = n_orders / days
    fulfil = n_orders * FULFIL
    surge = max(0, min(per_day, 2000) - 1200) * (6.00 - FULFIL) * days if per_day > 1200 else 0
    late = max(0, per_day - 2000) * days * 5          # carried orders get a S$5 voucher
    return dict(orders=round(n_orders), revenue=round(revenue), contribution=round(contrib - fulfil - surge - late - MARKETING))
opts = {"A Repeat 11.11 as-is (1 day, surge booked)": scenario(False, 1),
        "B Targeted discounts (1 day)": scenario(True, 1),
        "C Same discounts, spread over 3 days": scenario(False, 3),
        "D Targeted + 3 days + platinum early access": scenario(True, 3)}
print(pd.DataFrame(opts).T.to_string())
print("Volume assumption for B and D:", VOLUME, "(others unchanged). Change it and re-run: B/D stay ahead because the units removed were sold below cost.")

H("8. Weighted scoring (deck slides 20-23). Scores are the presenter's judgement, shown so the room can argue with them")
W = {"Profit": 9, "Customer experience": 8, "Operational risk": 7, "Competitive position": 6, "Revenue": 5, "Loyalty fairness": 4}
S = pd.DataFrame({"A": [1, 2, 3, 6, 9, 3], "B": [7, 4, 5, 5, 6, 4], "C": [2, 8, 8, 6, 9, 3], "D": [8, 8, 8, 5, 6, 7]}, index=list(W))
print(S.assign(weight=pd.Series(W)).to_string())
print("\nWeighted totals:", (S.mul(pd.Series(W), axis=0).sum()).to_dict())
print("Sensitivity - winner as the Profit weight changes (other weights as above):",
      {k: S.mul(pd.Series({**W, "Profit": k}), axis=0).sum().idxmax() for k in (1, 3, 5, 7, 9, 10)})
print("Sensitivity - same sweep for a 'GMV first' manager (Revenue weight 10):",
      {k: S.mul(pd.Series({**W, "Revenue": 10, "Profit": k}), axis=0).sum().idxmax() for k in (1, 3, 5, 7, 9, 10)})
