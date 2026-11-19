# AI-assisted decision demo: "Should we repeat 11.11 for 12.12?"

A companion to `flash-sale-demo`. Same users, products and orders, plus the other material a real
post-mortem would have on the table: unit costs, customer reviews in four languages, call transcripts,
competitor flyers, an ops dashboard screenshot and a logistics contract. An AI assistant (GitHub Copilot
in VS Code, agent mode) works through the eight-step decision process from the Day 1 deck, and the
audience checks its work.

Everything is synthetic. ShopNova, Kedai Prime and Swiftlane are invented names.

## The story in one paragraph

On 11 Nov 2025 the store ran a one-day sitewide sale at 30–50% off. It took 3,000 orders and S$1.55M,
39 times a normal day. Marketing wants to repeat it for 12.12. The CFO wants to know whether it made
money. It did not: once unit costs, fees and campaign costs are included it lost about S$183k, and
Electronics, the biggest revenue category, lost about S$189k on its own. The warehouse could only
dispatch 1,200 orders a day, so on-time delivery fell to 44%. A bad batch of smartwatches went out,
and customers complained in English, Malay, Indonesian and Chinese. Revenue went up while profit went
down, and the AI has to work that out from all of these sources together.

## Setup (10 minutes, once)

1. Put this folder next to `flash-sale-demo/data` (or anywhere; it carries its own copy of the CSVs in `data/`).
2. Python 3.10+ with `pip install pandas matplotlib jupyter`.
3. VS Code with the GitHub Copilot extension. **Open this folder as the workspace root**, so Copilot
   picks up `.github/copilot-instructions.md`.
4. Open Copilot Chat, switch to **Agent**, and create an empty `decision.ipynb`.
5. Run `python answer_key/answer_key.py` and keep `answer_key/answer_key_output.txt` open on your own screen.

## Files

| Path | Modality | Used for |
|---|---|---|
| `data/*.csv` | structured | orders, lines, products, users (copy of the shared dataset) |
| `data_extra/product_costs.csv` | structured | unit cost per product, from "finance" |
| `data_extra/reviews.csv` | free text, 4 languages | 2,019 reviews with planted quality problems |
| `data_extra/support_calls/*.txt` | transcripts | 5 support calls and 1 ops voice memo |
| `data_extra/images/competitor_*.png` | image | two competitor 12.12 flyers (one priced in RM) |
| `data_extra/images/ops_dashboard_nov2025.png` | image of a chart | backlog and on-time delivery, 9–20 Nov |
| `data_extra/docs/swiftlane_scheduleB_and_finance_note.pdf` | PDF | capacity and rates, finance cost rules |
| `PROMPTS.md` | | the prompt script, one prompt per decision step, with expected answers |
| `answer_key/` | | reference numbers; your safety net if the live run goes wrong |
| `tools/` | | regenerate `data_extra/` (`generate_extra.py`, then `generate_media.py`) |

## Planted things for the audience to catch

- Revenue includes refunded and cancelled orders unless you filter to delivered ones.
- 15 duplicate reviews, 5 reviews for orders that do not exist, 4 ratings outside 1–5.
- Some reviews say the customer returned the item, while the order still says `delivered`.
- Kedai Prime's prices are in ringgit.
- Flash buyers spend more afterwards, but they already spent more before. That is selection, not lift.
- "11.11 brought in new customers": there are no sign-ups after 30 Sep 2025.
