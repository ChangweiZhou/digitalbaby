# A trust of its own and a step of its own for every cell: can a strictly local reading reach the shared one on both texts? (21 November 2026)

**Authorisation and steer.** Mr. Xu's words: "我们反复强调是local learning" (my translation: "we have stressed again and again that it is local learning") and "这些你自己决定，不需要问我。一切从第一原理和事实出发。" (my translation: "Decide these yourself, you need not ask me. Everything from first principles and from facts."). The choice and design of this step are mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written after the outcome of `PLAN_11.md`; before any run of `own16.py` other than its check (`check16_output.txt`, passed). Development level; no checker gate, seal or review is claimed.

## 1. The facts this follows (`PLAN_11.md`, five seeds)

The counting learner read through trusts shared by kind and level (S1) names 82.4% of never-shown sums, reads English at 1.509 nats a byte, and loses 5.4 points of the sums through the English, all of it in the shared trusts. With a trust of its own for each cell (P) it loses one point, but pays elsewhere: at the step that suits the sums (0.3) it is 4.6 points under S1 and reads English at 2.77 nats; at the step that suits English (0.03; 1.559 nats) it is 11.6 points under on the sums. **One step number does not serve both texts.**

Why (mine). In the sums a cell keyed on the right places is followed by the same byte every time; its trust should rise fast and far. In English a young cell is followed by many bytes; its trust should move little. The shared trust settles this for a cell from the record of all cells of its kind. A cell alone has only its own record, and in P it was given the same step whatever that record said.

**The first principle (mine): a cell's own record does say which case it is in.** If the changes the cell's trust has received keep going one way, a larger step would have been better; if they keep cancelling, a smaller one. A step that the cell adjusts by this sign is local in the strictest sense: the cell keeps a few numbers of its own and no number is shared. The rule exists in the literature of learning with one step for each weight (Sutton's IDBD of 1992 and its normalised form, Autostep, by Mahmood, Sutton and others, 2012; from my memory of them, I have not checked the papers tonight, and I claim no result from them).

## 2. The learner, exactly (`own16.py`)

The cells, kinds, messages and counts are `PLAN_11.md`'s learner 2 (256 kinds on sets of at most four of the last nine places; smoothing 2; trusts born at 0.02; no second answering cell). Each cell holds its trust, its step (born at 0.03), a trace h of its own recent changes, and a yardstick v. When the cell speaks and the byte arrives, with g its driver (its message at the byte that came less the expected message):

* yardstick: v = max(|g h|, v + (step x g² / tau) x (|g h| - v));
* step: multiplied by exp(mu x g h / v), so by at most e^mu up or down;
* the answering cell: if the sum over the speaking cells of step x g² is above one, every speaking cell's step is divided by that sum (**form A**: the cell keeps the divided step, as I remember the published rule; **form B**: the division holds for this byte only);
* trust: plus step x g;
* trace: h = h x max(0, 1 - step x g²) + step x g.

**Numbers.** tau = 10,000 (the published default as I remember it); mu from 0.01, 0.03, 0.1. **The choice** is made on the development seed among the six (two forms, three values of mu): of those that read the English within 0.1 nats a byte of S1 on that seed (last tenth), the one that names the most never-shown sums at the end of the sums; if none reads the English that well, the one that reads it best. Then frozen.

The life and the score are `PLAN_11.md`'s (40,000 lines of sums, 400,000 bytes of English, 5,000 lines of sums). S1, P and F are that plan's runs and are not run again; with the new rule off the kernel is `aba.py`'s bit for bit (checked).

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** A cell that learns its step from its own record reads both texts as the shared trusts do, with no number shared: at the median of five seeds the chosen form ends the sums within 3 points of S1, reads the English within 0.05 nats a byte of S1, and loses 2 points or fewer of its never-shown sums through the English.

**Refuted if** it ends the sums more than 5 points under S1, or reads the English more than 0.1 nats above S1: then, by this rule at least, a cell cannot learn from its own record what the shared trust learns from all cells of its kind, and strict locality keeps its price in this learner.

| | Statement (five reported seeds; median; S1: 82.4 and 1.509) | Chance |
|---|---|---|
| T1 | Sums within 3 points of S1 | 0.40 |
| T2 | English within 0.05 nats a byte of S1 | 0.45 |
| T3 | Both (the first two parts of the hypothesis) | 0.25 |
| T4 | 2 points or fewer lost through the English | 0.80 |
| T5 | Better than every P of `PLAN_11.md` on both texts at once: above 77.8 on the sums and below 1.559 on the English | 0.45 |
| T6 | Form B is the one chosen on the development seed | 0.60 |

## 4. Runs, size, cost, what is not run

Development seed 1: the six. Reported: the seeds 73129, 93683, 116089, 406364, 545353, the chosen one. Size: 1.2 million cells after the sums and 14.5 million after the English, as in `PLAN_11.md`, each with three more numbers. Cost: 2 to 3 CPU-minutes a run; eleven runs. **Not run:** the life over six streams (the next plan if the hypothesis holds); a second answering cell (its parts are shared numbers); a rule with no mu; subtraction after addition.

## 5. Outcome

(to be entered after the runs)
