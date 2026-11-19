# Prompt script: "Should we repeat 11.11 for 12.12?"

Paste one prompt at a time into Copilot Chat in **Agent** mode, with the `ai-decision-demo` folder open.
Each prompt is tagged with the step of the eight-step decision process from the Day 1 deck, the kind of
data it uses, and what a correct answer looks like. Run `answer_key/answer_key.py` beforehand so you have
the reference numbers in front of you.

---

## Step 1: Identify the problem  (PDF + text)

> Read `data_extra/docs/swiftlane_scheduleB_and_finance_note.pdf` and the six transcripts in
> `data_extra/support_calls/`. In five bullets: what decision is being asked of us, by whom, by when,
> and which things are symptoms rather than the underlying problem?

**Expect:** decide the 12.12 format; the CFO asks whether 11.11 made money by category; marketing wants
to repeat it; surge capacity must be booked by 5 Dec. Late deliveries, complaints and refunds are symptoms.
Underlying problems are discount depth versus margin, and volume versus logistics capacity.

**Say to the room:** slide 18, "make sure it's a problem and not just a symptom".

## Steps 2 and 3: Criteria and weights  (human judgement)

> Propose decision criteria for choosing the 12.12 format. Give each a weight out of 10, and say which
> files would give evidence for each one. Put it in a markdown table in the notebook.

**Then edit the weights yourself, live.** This is the step the AI cannot own. Use the answer key's
weights (Profit 9, Customer experience 8, Operational risk 7, Competitive position 6, Revenue 5,
Loyalty fairness 4) or let the room vote.

## Step 4: Develop alternatives  (images)

Drag both competitor flyers from `data_extra/images/` into the chat, then:

> These are two competitor flyers for 12.12. Extract their offers into a table: competitor, categories,
> offer, dates, currency, delivery promise. Convert any RM prices to SGD and state the rate you used.
> Then list four alternatives for our 12.12 format, including not running a sale.

**Expect:** ShopNova offers up to 60% off Electronics, free next-day delivery over S$50, 10–12 Dec.
Kedai Prime offers buy 2 get 25% off Beauty and Fashion, bundles from RM 49, platinum early access on
11 Dec at 8pm. **Check** that it noticed the currency is ringgit and stated a rate.

## Step 5a: Analyse the structured data  (CSV)

> Using `data/orders.csv`, `order_items.csv`, `products.csv` and `data_extra/product_costs.csv`,
> build the 11.11 profit and loss by category: revenue, cost of goods, gross margin, payment gateway fee,
> fulfilment, contribution. Apply the cost rules in the finance memo in the PDF. Print total revenue across
> all orders first, as a check.

**Expect:** total revenue 16,255,674.43 (the same figure the three databases print in Demo 2).
11.11 lost about **S$183k** after marketing and vouchers. Electronics alone lost about **S$189k**, although
it was the biggest revenue category.

**Traps to watch for:**
- Did it include refunded and cancelled orders? It should use delivered orders only. Ask: *"Which order statuses did you include, and why?"*
- Did it write off the Kite Smartwatch E59 units, as the memo says?
- Did it subtract the S$85,000 marketing spend and the 1,630 × S$5 vouchers?

## Step 5b: Analyse the unstructured text  (reviews in four languages)

> Analyse `data_extra/reviews.csv`. First check data quality (duplicates, order ids that do not exist,
> ratings outside 1 to 5) and tell me what you removed. Then compare 11.11 orders with all other days:
> average rating, and the share of reviews that complain about delivery, by category. The reviews are in
> English, Malay, Indonesian and Chinese, so handle all four. Flag any product with a cluster of safety complaints.

**Expect:** 15 duplicates, 5 orphan order ids, 4 bad ratings. Average rating 3.47 on 11.11 versus 3.73 on
other days. Delivery complaints in 38% of 11.11 reviews versus 5% otherwise. **Kite Smartwatch E59:** 29 of
53 reviews on orders from 11 Nov to 31 Dec mention heat or battery, and none on other dates, which points
to a bad batch.

## Step 5c: Read a chart from a screenshot  (image + cross-check)

Drag `data_extra/images/ops_dashboard_nov2025.png` into the chat:

> Read this dashboard screenshot. Put the daily orders received, orders dispatched and on-time delivery %
> into a dataframe. Check the received numbers against orders.csv and the capacity line against Schedule B
> in the PDF. Do the three sources agree?

**Expect:** yes. 3,000 orders received on 11 Nov matches orders.csv, and the 1,200 per day line matches
Schedule B. On-time delivery fell to 44% on 14 Nov.

## Step 5d: Test a claim  (the correlation trap)

> Marketing says 11.11 buyers spent 2.2 times more than other customers in the 90 days afterwards, so the
> sale created loyal customers. They also say it brought in new customers. Test both claims.

**Expect:** flash buyers already spent about 2.1 times more *before* the sale, so this is selection, not
lift. And there are no sign-ups after 30 Sep 2025, so the "new customers" claim has no support in the data.
**Say to the room:** slide 30, management filters. Would anyone have checked if the number agreed with what they hoped?

## Step 5e: Sensitivity analysis  (slide 37)

> Calculate the break-even discount for each category: the discount at which price after the 2.2% gateway
> fee equals unit cost. Then show contribution margin as a % of list price at 10, 20, 30, 40 and 50% off.
> Which categories should never go deeper than 25%?

**Expect:** Electronics breaks even at about 25.6% off and Groceries at 28.3%. Beauty can go to 68.7% and Fashion to 60.6%.

## Step 6: Select the best alternative  (weighted scoring + sensitivity)

> Score the four alternatives from 1 to 10 against our weighted criteria, citing the evidence for each score.
> Then run a sensitivity check: how does the winner change as the Profit weight moves from 1 to 10, for a
> manager who gives Revenue a weight of 10?

**Expect:** "targeted discounts, spread over three days, platinum early access" wins. For a manager who
puts revenue first, "same discounts over three days" wins until the Profit weight reaches about 4.
The decision depends on what the organisation values, and the AI cannot settle that for you.

## Steps 7 and 8: Implement and evaluate

> Draft a one-page recommendation for the CFO: the decision, the three numbers that justify it, the risks,
> the 5 December logistics deadline, and the KPIs we will check on 13 December to judge whether it worked.

## Close: make it show its working

> List everything in your analysis that is an assumption rather than a fact taken from the files.

**Expect:** the volume response to a smaller discount, the RM to SGD rate, review sentiment as a stand-in
for all customers, the fulfilment cost allocation, and the scores themselves.
