# nano/skip, ninth plan: bytes to pass over that are not a set of byte values (a digit that matters in one place and is filler in another) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at 97284af, with `ctx.py` added (the system and the kernel of `skip.py` unchanged; `ctx.py` run with the numpy program and with the kernel on 300 lines gives the same strengths to rounding and the same bytes passed over, for the three clocks used below). **Status.** Development level, development seed 1 first; no checker gate, seal or outside review is claimed. Written before any run on this text except those 300-line cases.

**Why.** `REPORT.md`, section 4, first item: in every text so far the bytes to pass over are a set of byte values. A clock with one entry for each byte value cannot keep a 7 in a number and pass over a 7 elsewhere. This plan takes the smallest text in which that is asked.

**The text `h2`.** Lines `aa+<0 to 4 fillers>bb=ccc`, a filler being one of the four letters followed by a digit drawn by chance: `12+y5w6z2y634=046`. The filler digits say nothing about the answer and are the same byte values as the digits that do. Which digit is filler can be told from the byte before it (a letter), so the smallest context that suffices is one byte. Never-shown pairs as before (2,000 of 10,000 kept back, 500 scored), with 0, 2 and 4 fillers, the fillers fixed by the pair.

**One change: what an entry belongs to.** The trial clock of `PLAN_2.md`, second form, nothing retuned (alpha 2, p0 0.2, period 256, floor 0.01, fading 1/3,000), with entries of two kinds, beside each other: `key=value`, one entry for each byte value (18 entries), as before; `key=pair`, one entry for each byte value after each byte value (324 entries, 4 numbers each): when the byte v that arrived after the byte p is to leave the present place, the entry (p, v) decides whether it enters the chain. In both, and this differs from `skip.py`, an entry's evidence changes only in a period in which its byte came (with 324 entries many do not come in every period). What an entry reads is unchanged: its own state and the one number sent out by the answering cell. No byte is named; the program does not tell the system what a filler is. To be closed for the text to be the plain one: the 40 entries (letter, digit) and the 44 entries of a letter after `+` or after a digit; and none of the other 150 or so entries that occur.

**System.** 256 cells, 55,570 numbers that learn; at the entries 72 (`key=value`) or 1,296 (`key=pair`).

**Runs (development seed 1).** 2,400,000 lines: `key=pair`; `key=value`; `clock=none`. 800,000 lines: `clock=oracle` (letters and the digit after each letter passed over by the program: the check). If `key=pair` reaches 50% with every number of fillers, the five reported seeds follow as an addition written before them.

**Question.** Does the same trial rule find what to pass over when it must be found for each byte in its context, with five times as many entries to close and each of them rarer?

**Hypothesis H9.** With `key=pair` the system has at least 50% of never-shown pairs with 0, 2 and 4 fillers within 2,400,000 lines (development seed; last three readings). **Refuted if** under 50% with any.

*What I expect (Reading, mine, before the runs).* Harder than anything so far, for two reasons I can name. A line is aligned only when every entry its fillers use is closed, and a filler uses two; with entries born closed one period in five, one line in 125 with a single filler is aligned by chance, against one in 5 of the lines with a single letter on `g2`. And closing half of a filler (the letter but not its digit) puts a stray digit in the chain with nothing to mark it. On the other side, `PLAN_5.md` showed entries closing in parallel. I have been too dark twice on this rule.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| X1 | H9 holds | 0.40 |
| X2 | `key=value`: under 10% with 2 and with 4 fillers (it cannot do it by construction) | 0.90 |
| X3 | `key=value` closes the four letters all the same | 0.40 |
| X4 | `key=pair`: at the end at least 35 of the 40 (letter, digit) entries are closed | 0.45 |
| X5 | `key=pair`: at the end no entry of a digit after a digit, after `+` or after `=` is closed | 0.50 |
| X6 | `clock=oracle`: at least 85% | 0.90 |

**Cost.** About 4 CPU-minutes a run of 2,400,000 lines; about 15 CPU-minutes. **Not run.** Contexts longer than one byte; fillers whose kind cannot be told from the byte before; the expectation rule on this text.
