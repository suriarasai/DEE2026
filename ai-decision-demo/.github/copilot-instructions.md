# Instructions for Copilot in this folder

You are helping a commercial team make a decision from data. The audience is senior programmers
who will read your code, so be explicit and checkable.

- Work in `decision.ipynb` with pandas and matplotlib. One question per cell.
- After every load, join or filter, print the row count, so a reader can follow what changed.
- Name the file each number comes from. If a number comes from an image or a PDF, say so.
- Separate facts from assumptions. Put assumptions in a list at the top of each answer.
- Never invent data that is not in the files. If the files cannot answer a question, say that.
- All money is in SGD unless a source says otherwise. If you convert a currency, state the rate.
- Prefer short, readable code to clever code. No hidden helper modules.

Data:
- `data/` holds a copy of the shared flash-sale dataset (users, products, orders, order_items, user_profiles).
- `data_extra/` holds product costs, reviews (four languages), call transcripts, images and a PDF.
