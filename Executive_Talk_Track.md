# Function Sequencing: Executive Talk Track

## What We’re Solving

We have a large set of **customers** (enterprises) and a long list of **functions** they use (e.g. Payroll, Inventory, GL). Each customer uses many functions, and each function can be **modernized** or not—today we know for each customer/function whether it’s “Yes,” “Partial,” or “No.”

We can’t modernize everything at once. The question is: **in what order should we modernize functions** so we unlock the most value (customers fully modernized and/or ARR) as quickly as possible?

This exercise answers that with a **prioritized roadmap**: a sequence of functions to modernize, step by step, so we see clearly how many customers get “fully modernized” and how much ARR gets unlocked over time.

---

## How We Think About “Unlocked”

- A **customer is “unlocked”** when every function they use (that we care about—Daily, Weekly, or Monthly usage only) is modernized. Until then, they’re still “in progress.”
- **ARR unlocked** is the total annual recurring revenue from customers who have just become fully modernized. We track both **how many customers** we unlock and **how much ARR** we unlock at each step.

So the roadmap doesn’t just say “do Payroll, then Inventory.” It says: “If we do Payroll first, we unlock X customers and $Y ARR; then if we do Inventory next, we unlock Z more customers and $W more ARR,” and so on.

---

## How We Pick the Order (The “Greedy” Logic)

We use a **greedy** approach—at each step we ask: “Which single function, if we modernize it next, unlocks the most value *right now*?”

- **Value** can be defined in different ways. We support:
  - **100% customers:** Maximize number of customers unlocked.
  - **100% ARR:** Maximize ARR unlocked.
  - **Blended (e.g. 50/50, 75/25, 25/75):** A mix of customers and ARR, so we can tune whether we care more about logo count or revenue.

We run **multiple views** (e.g. 100% customers, 75% customers / 25% ARR, 50/50, 25% customers / 75% ARR, 100% ARR) so leadership can compare strategies and choose the one that fits our goals.

---

## What “Completeness” Means (Daily / Weekly / Monthly Only)

We only count a customer as “complete” for a function when that function is modernized **for the ways they use it that we care about**: **Daily, Weekly, and Monthly** usage. We ignore less frequent usage (e.g. quarterly, yearly) for this roadmap, so the plan is focused on the usage that matters most.

---

## What the Outputs Give You

1. **Step-by-step roadmap**  
   For each step: which function to modernize next, how many customers get unlocked that step, and cumulative customers unlocked. So you can see “by step 10 we’ve unlocked 50 customers,” etc.

2. **Customer-level unlock detail**  
   Which specific customers unlock at each step, plus each customer’s ARR and cumulative ARR unlocked. Good for storytelling and account-level planning.

3. **Tranche summary**  
   A rollup by step: customers unlocked and ARR unlocked per step and cumulative. Easy to chart and present.

4. **Customer remaining by scope**  
   Three views (Daily+Weekly only; Daily+Weekly+Monthly; all in-scope): for each customer, how many functions are left to modernize and what percent of their function use is still remaining. Surfaces who is “almost done” vs. “still a long way to go.”

5. **Functions not yet modernized, by customer**  
   For Daily/Weekly/Monthly only: a simple list per customer of which functions are still not modernized. Useful for delivery and account teams.

All of this is in **one Excel file**, with multiple tabs for the different views and blend weights (e.g. 100% customers vs. 100% ARR vs. 50/50), so executives can flip between scenarios without switching tools.

---

## One-Sentence Summary

**We prioritize which functions to modernize first so that, step by step, we unlock the most customers and ARR as quickly as possible, using only Daily/Weekly/Monthly usage, with clear outputs that show the roadmap, who unlocks when, and who still has work left.**

---

## Optional One-Liner for Slides

*“A data-driven roadmap that sequences function modernization to maximize customers fully modernized and ARR unlocked, with multiple strategies (customer-heavy, ARR-heavy, or blended) in one place.”*
