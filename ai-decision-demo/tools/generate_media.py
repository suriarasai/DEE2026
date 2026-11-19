#!/usr/bin/env python3
"""Draw the image and PDF inputs for the AI decision demo.

    data_extra/images/ops_dashboard_nov2025.png      screenshot of the fulfilment dashboard
    data_extra/images/competitor_shopnova_1212.png   competitor flyer (prices in SGD)
    data_extra/images/competitor_kedaiprime_1212.png competitor flyer (prices in RM)
    data_extra/docs/swiftlane_scheduleB_and_finance_note.pdf

The numbers agree with the orders data and with each other on purpose, so the
AI can be checked: received counts come from orders.csv, capacity from the PDF.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "data_extra"
IMG, DOCS = OUT / "images", OUT / "docs"
IMG.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)

CAPACITY = 1200
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ------------------------------------------------------------- ops dashboard
orders = pd.read_csv(DATA / "orders.csv")
days = pd.date_range("2025-11-09", "2025-11-20").strftime("%Y-%m-%d")
received = orders.groupby("order_date").size().reindex(days).astype(int)
dispatched, backlog, carry = [], [], 0
prev = int(orders.groupby("order_date").size().get("2025-11-08", 50))
for d in days:
    todo = carry + received[d]
    out = min(CAPACITY, todo)
    dispatched.append(out); carry = todo - out; backlog.append(carry)
on_time = [97, 96, 97, 93, 46, 44, 58, 79, 91, 95, 96, 97]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(12, 6.6), dpi=110, facecolor="#f4f5f7")
fig.text(0.02, 0.955, "Fulfilment Ops  ·  Daily view  ·  Swiftlane DC-1", fontsize=15, weight="bold", color="#1f2933")
fig.text(0.02, 0.92, "9 – 20 Nov 2025   |   source: WMS export, refreshed 21 Nov 06:00", fontsize=9, color="#616e7c")
tiles = [("Peak backlog", "1,800 orders", "end of 11 Nov"),
         ("Lowest on-time delivery", "44%", "14 Nov (target 95%)"),
         ("CS contacts 11–17 Nov", "2,140", "+410% vs previous week"),
         ("Late-delivery vouchers", "1,630 × S$5", "issued 13–20 Nov")]
for k, (t, v, s) in enumerate(tiles):
    ax = fig.add_axes([0.02 + k * 0.245, 0.74, 0.225, 0.15]); ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="white", transform=ax.transAxes))
    ax.text(0.06, 0.72, t, fontsize=9, color="#616e7c", transform=ax.transAxes)
    ax.text(0.06, 0.34, v, fontsize=18, weight="bold", color="#c0392b" if k < 2 else "#1f2933", transform=ax.transAxes)
    ax.text(0.06, 0.1, s, fontsize=8, color="#7b8794", transform=ax.transAxes)
ax = fig.add_axes([0.06, 0.1, 0.86, 0.55], facecolor="white")
x = range(len(days)); labels = [d[8:] + " Nov" for d in days]
ax.bar([i - 0.2 for i in x], received.values, width=0.4, color="#3e7cb1", label="Orders received")
ax.bar([i + 0.2 for i in x], dispatched, width=0.4, color="#9fb3c8", label="Orders dispatched")
ax.plot(x, backlog, color="#e67e22", marker="s", lw=1.5, label="Backlog (end of day)")
ax.axhline(CAPACITY, color="#7b8794", ls="--", lw=1)
ax.text(len(days) - 0.6, CAPACITY + 40, "standard capacity 1,200/day", ha="right", fontsize=8, color="#52606d")
for i, v in enumerate(received.values):
    ax.text(i - 0.2, v + 25, f"{v:,}", ha="center", fontsize=7, color="#334e68")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("orders")
ax.set_ylim(0, 3300); ax.spines[["top"]].set_visible(False)
ax2 = ax.twinx(); ax2.plot(x, on_time, color="#c0392b", marker="o", lw=2, label="On-time delivery %")
for i, v in enumerate(on_time):
    ax2.text(i, v + 3, f"{v}%", ha="center", fontsize=7, color="#c0392b")
ax2.set_ylim(0, 135); ax2.set_yticks([0, 20, 40, 60, 80, 100]); ax2.set_ylabel("on-time delivery %"); ax2.spines[["top"]].set_visible(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=True, ncol=2)
fig.savefig(IMG / "ops_dashboard_nov2025.png", facecolor=fig.get_facecolor()); plt.close(fig)

# ------------------------------------------------------------- competitor flyers
def flyer(path, bg, accent, lines):
    W, H = 900, 900
    im = Image.new("RGB", (W, H), bg); dr = ImageDraw.Draw(im)
    for k in range(0, W + H, 60):                       # diagonal stripes
        dr.line([(k, 0), (k - H, H)], fill=tuple(min(255, c + 12) for c in bg), width=18)
    y = 70
    for text, size, bold, colour in lines:
        if text == "---":
            dr.rectangle([80, y, W - 80, y + 6], fill=accent); y += 40; continue
        f = ImageFont.truetype(FONT_B if bold else FONT, size)
        for part in text.split("\n"):
            w = dr.textlength(part, font=f)
            dr.text(((W - w) / 2, y), part, font=f, fill=colour); y += int(size * 1.25)
        y += 18
    im.save(path)

flyer(IMG / "competitor_shopnova_1212.png", (18, 32, 74), (255, 196, 0), [
    ("ShopNova", 64, True, (255, 255, 255)),
    ("12.12 MEGA SALE", 72, True, (255, 196, 0)),
    ("---", 0, False, None),
    ("ELECTRONICS", 54, True, (255, 255, 255)),
    ("UP TO 60% OFF", 88, True, (255, 196, 0)),
    ("phones · monitors · earbuds · smartwatches", 28, False, (220, 228, 255)),
    ("---", 0, False, None),
    ("FREE NEXT-DAY DELIVERY", 46, True, (255, 255, 255)),
    ("on orders over S$50 (Singapore)", 28, False, (220, 228, 255)),
    ("10 – 12 DEC  ·  3 DAYS ONLY", 44, True, (255, 196, 0)),
    ("While stocks last. T&Cs apply.", 22, False, (160, 170, 200)),
])
flyer(IMG / "competitor_kedaiprime_1212.png", (252, 228, 236), (194, 24, 91), [
    ("Kedai Prime", 64, True, (136, 14, 79)),
    ("12.12 Beauty & Fashion Fest", 46, True, (194, 24, 91)),
    ("---", 0, False, None),
    ("BUY 2, GET 25% OFF", 70, True, (136, 14, 79)),
    ("serums · sunscreen · sneakers · dresses", 28, False, (120, 60, 90)),
    ("Bundles from RM 49", 52, True, (194, 24, 91)),
    ("---", 0, False, None),
    ("PRIME PLATINUM", 44, True, (136, 14, 79)),
    ("EARLY ACCESS", 60, True, (194, 24, 91)),
    ("11 Dec, 8pm – one day before everyone else", 28, False, (120, 60, 90)),
    ("Malaysia & Singapore. Prices in Malaysian Ringgit.", 22, False, (150, 100, 120)),
])

# ------------------------------------------------------------- PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ss = getSampleStyleSheet()
doc = SimpleDocTemplate(str(DOCS / "swiftlane_scheduleB_and_finance_note.pdf"), pagesize=A4,
                        leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50,
                        title="Swiftlane Schedule B and 11.11 finance note")
grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e7eb")),
                   ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP")])
P = lambda t, s="BodyText": Paragraph(t, ss[s])
story = [
    P("Swiftlane Logistics Pte Ltd", "Title"),
    P("Master Services Agreement — <b>Schedule B: Capacity, Rates and Service Levels</b> (excerpt)"),
    P("Client: [the retailer] · Facility: DC-1 (Tuas) · Effective 1 Jan 2025 · Page 14 of 22", "Italic"),
    Spacer(1, 12),
    Table([["Item", "Term"],
           ["B.1 Standard capacity", "1,200 orders dispatched per day from DC-1."],
           ["B.2 Surge capacity", "Up to 2,000 orders per day, only if booked in writing at least\n5 working days before the surge day."],
           ["B.3 Rates", "S$4.20 per order up to standard capacity.\nS$6.00 per order above 1,200 on a booked surge day."],
           ["B.4 Over-capacity orders", "Orders above available capacity are carried to the next\ndispatch day in order received."],
           ["B.5 Service levels", "Dispatch within 1 business day of order.\nDelivery within 3 days (Singapore) or 6 days (regional)."],
           ["B.6 Service credits", "None payable where volume exceeds booked capacity."],
           ["B.7 Peak season", "1 Nov – 31 Dec: surge subject to availability.\nFor 12.12, surge must be confirmed by 5 Dec."]],
          colWidths=[130, 360], style=grid),
    Spacer(1, 10),
    P("Note: 11 Nov 2025 was not booked as a surge day. Standard capacity applied."),
    PageBreak(),
    P("INTERNAL MEMO — Finance", "Heading1"),
    P("<b>Subject:</b> 11.11 post-mortem: cost inputs (draft v2)<br/><b>From:</b> FP&amp;A · <b>Date:</b> 24 Nov 2025"),
    Spacer(1, 8),
    P("The CFO has asked a direct question: <b>did 11.11 make money once discounts are taken into account, "
      "and in which categories?</b> Marketing has proposed repeating the same format (sitewide 30–50% off, one day) for 12.12. "
      "Please use the inputs below together with unit costs from the product cost file."),
    Spacer(1, 8),
    Table([["Cost line", "Value", "Notes"],
           ["Marketing spend for 11.11", "S$85,000", "Paid social 52k, influencers 21k, banners 12k"],
           ["Payment gateway fee", "2.2% of revenue", ""],
           ["Fulfilment", "S$4.20 per order", "Schedule B, standard rate"],
           ["Late-delivery vouchers", "1,630 × S$5", "Issued 13–20 Nov"],
           ["Kite Smartwatch E59", "Write off all 11.11 units", "Safety recall, units not resellable"],
           ["Other returns", "Resold at cost", "Assume no loss on goods"]],
          colWidths=[150, 120, 220], style=grid),
    Spacer(1, 10),
    P("All figures in SGD and exclude GST. Revenue in the order system is net of discount (unit_price is the price paid)."),
    P("Open item: marketing claims 11.11 'brought in new customers'. Please check this against sign-up dates before it goes into the board pack."),
]
doc.build(story)
print("media written to", OUT)
