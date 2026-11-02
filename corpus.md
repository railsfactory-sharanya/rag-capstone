# Mealbox corpus

Part 1 deliverable for the support-ticket and FAQ assistant. The entries that will be embedded live in [`corpus/entries.json`](corpus/entries.json) only. This file is a reviewer note. Do not ingest it. Indexing it would put the questions below into the vector store.

Mealbox is a fictional weekly meal-kit service. A box holds 3 recipes, in a 2-serving plan ($66.00 before tax) or a 4-serving plan ($111.00 before tax). Delivery is Tuesday or Thursday, 8:00am–8:00pm, inside the contiguous United States and Washington, D.C.

48 entries: 37 `faq`, 11 `ticket`. Categories: `account` (9), `billing` (10), `subscription` (7), `delivery` (10), `orders` (7), `bugs` (5).

## Intentionally unanswerable questions

These are realistic things a customer might ask. No entry describes the service they ask about. Later parts must decline them instead of inventing a policy.

1. Can Mealbox send a team to cook and serve dinner at my venue for a wedding of 80 guests?
2. Can I book weekly calls with a Mealbox dietitian who will build a marathon-training menu for me?
3. I run a cafe. Can I buy Mealbox ingredients wholesale and resell the cooked plates?

Nearby articles are easy to over-read, and they still do not answer these:

- Plan-size articles (`faq-019`, `faq-022`) describe a 2-serving or 4-serving box shipped to a home or business door. They say nothing about on-site staff, venues, or a guest count of 80.
- Menu filters (`faq-032`) only show or hide vegetarian, vegan, and gluten-free recipes. They do not create a personal training menu.
- `faq-029` allows a normal weekly box to a business street address. It does not open a wholesale or resale account.

## Overlaps written on purpose

Same situation, different wording, different correct article. Plain similarity search should be able to confuse these. That is what later retrieval work is for.

| Pair | Entries | What a careful answer must tell apart |
|------|---------|----------------------------------------|
| Getting back in | `faq-001` password reset, `faq-002` change email, `ticket-001` unfamiliar device | A new password, a new inbox, and a session the customer did not open |
| Stopping boxes | `faq-016` cancel, `faq-017` pause, `faq-018` skip one week | End the plan, turn off 1–8 weeks, or turn off one week |
| Login vs plan | `faq-004` delete account, `faq-016` cancel | Remove the login, or keep the login and stop boxes |
| Where it goes | `faq-024` future address, `faq-025` already shipped, `faq-036` Tuesday vs Thursday | Rewrite a future label, a label already printed, or only the day |
| Money off | `faq-010` promo codes in general, `ticket-003` code `SAVE20`, `faq-014` referral credit, `ticket-011` iOS paste bug | Rules for any code, one expired first-box code, a $20 referral, or a field that ignores paste |
| Credit for a bad box | `ticket-004` late, `ticket-005` missing ingredient, `ticket-006` warm protein | $15 late credit, $10/$16 ingredient credit, or a full recipe credit of $22/$37 |
| iOS checkout | `ticket-007` crash `CHECKOUT-409`, `ticket-011` paste bug `PAY-218` | App closes when two discounts are combined, or the field ignores paste on the version that already fixed the crash |
