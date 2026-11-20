# Eleventh plan: two-digit numbers with letters between them (from R3 towards R4) (20 November 2026)

**Authorisation.** As for the sixth to tenth plans (Mr. Xu, 20 November 2026). **Source.** This branch at 6d4388ad, with the text `g2` added to `hold.py` (the earlier texts give the same readings as before). Written before any run on this text. Development seed only; a small case; no checker gate.

**Why.** The seventh plan showed one digit held across 0 to 8 letters; the ninth showed two-digit sums learned with two delay lines and cells that hold. The first family has both at once: two digits, an operator, letters, two digits, `=`, the answer. This plan puts the two together, still without the task byte and the modulus.

**Text `g2`.** Lines `aa+<letters>bb=ccc` with an end mark: 0 to 8 letters (four letter values, drawn afresh for every line) between the operator and the second number. 2,000 of the 10,000 pairs are never shown. Score: whole answer right on 500 never-shown pairs, at gaps of 0, 4 and 8 letters (and on 500 shown pairs); mean of the last three readings.

**System.** `hold.py`, 256 cells, K = 16, eta1 = 1, the present byte and the two before it on lines (`chain=2`); `cross=0` and `cross=1 ret=exact`; 2,400,000 lines; development seed 1.

**Hypothesis.** With one crossing the cells hold both digits of the first number across the letters and answer never-shown sums at every length of gap. *Refuted if* the crossing arm is at or below 15% of never-shown sums at the gap of 8 letters.

| | Prediction (mine, before any run) | Chance |
|---|---|---|
| T1 | crossing arm, gap of 8 letters: at least 50% of never-shown sums | 0.35 |
| T2 | crossing arm: the gap of 8 letters is within 15 points of the gap of none | 0.55 |
| T3 | arm without the crossing: at most 20% at the gap of 8 letters | 0.65 |

**Cost.** About fifteen CPU-minutes a run. Not run: the reported seeds; gaps beyond 8; the task byte and the modulus of the first family.
