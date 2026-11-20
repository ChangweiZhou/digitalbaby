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

## Outcome of the runs above (entered 20 November 2026; development seed 1; `res/p9_h2_*_s1.json`)

`h2`, never-shown pairs with 0 / 2 / 4 fillers, mean of the last three readings:

| Clock | Never-shown pairs | Closed at the end, of the entries that occur |
|---|---|---|
| trial, `key=pair` (2,400,000 lines) | 55.0 / 1.4 / 0.5 | 0 of the 40 (letter, digit); 0 of the 44 letters after `+` or a digit; 0 of the 152 others (one other entry closed at two readings and reopened) |
| trial, `key=value` (2,400,000) | 62.3 / 1.7 / 0.8 | nothing |
| `clock=none` (2,400,000) | 60.9 / 1.9 / 0.5 | |
| `clock=oracle` (800,000) | 95.7 / 95.7 / 95.7 | the 40 and the 44, by the program |

**How many entries come to exist** (the coordinator's question): of the 324 possible pairs 236 occur on this text (40 filler digits after a letter, 44 letters, 152 others), 4 numbers each.

**H9 is refuted**: 1.4 and 0.5% with fillers, where 50% was asked; the rule closes nothing at all in 2,400,000 lines and the system stays where it is without a clock. **X1, X3, X4 failed. X2, X5 (emptily), X6 held.**

**Where it fails (Reading, mine).** Not in the chain's recovery (the oracle arm is at 95.7%) and not by closing wrong entries (none is closed). The entries never gather evidence. I see two causes and the runs above do not part them. (i) Half a filler closed earns nothing: letters closed alone leave stray digits in the chain, and filler digits closed alone leave the letters, which is the text `g2` again; `key=value`, which can only do the first, indeed closes nothing. Only the two entries of one filler closed in the same period align a line. (ii) Each entry reads a number averaged over a period of 256 bytes, about 18 lines, while its own byte comes in about one of them: the 26 letters of `PLAN_5.md` each came about three times a period, these entries less than once. In `PLAN_5.md` I wrote that entries close in parallel; here they dilute each other's periods as well.

## Addition 1 (20 November 2026, written before its runs): an entry listens only after its own byte; and what the expectation rule would propose here

Received from the coordinating session while the runs above were going: it asks for the number of entries (given above) and for one further arm, the expectation rule as proposer of what to try closing and the mismatch as judge.

**(a) A measurement first: what would the expectation rule propose on `h2`?** The table of `PLAN_7.md` (runs of up to 4 bytes, the next two bytes, counting across the end mark) after 400,000 lines; the share of bytes it would pass over by kind, at theta 0.2, 0.3, 0.5. *My reasoning before the run:* a filler has an inside. After its letter a digit is certain where before it a letter or the number was expected, and after its digit a letter or the number is expected where a digit was; so both bytes of a filler change the expectation and the rule should propose neither. The filler as a whole leaves the expectation where it was, but the rule looks at single bytes. Prediction Y1: at theta 0.2 under 10% of the filler letters and under 10% of the filler digits are passed over (chance 0.75). If so, the combined arm has nothing to combine on this text and is not run; I say so rather than build it.

**(b) One change to the trial rule: when an entry draws and what it listens to.** `listen=16`: an entry draws its state anew each time its byte comes, and enters into its closed or open mean the answering cell's number averaged over only the 16 bytes that follow (in this text the answer is over within 16 bytes of any filler). Everything else as before (alpha 2, p0 0.2, floor 0.01; the tendencies brought up to date every 2,048 bytes; the fading the same in bytes). The number sent out is the same one, to all entries alike; what is new is that an entry attends to it only for a while after it has acted, which is local to the entry. This answers cause (ii) and not cause (i). T = 16 is a number set by hand for these texts and is to be reported as such. The z of two entries in one line share bytes, so the evidence is more confident than it has a right to be; alpha is left alone all the same.

Runs (development seed 1): `g2`, `key=value`, 1,200,000 lines (the check that the listening form still does what the period form did: 95.1%); `h2`, `key=pair`, 2,400,000 lines; `h2`, `key=value`, 2,400,000 lines.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| Y2 | `g2`, `key=value`, listening: at least 85%, exactly the letters closed at the end | 0.65 |
| Y3 | `h2`, `key=pair`, listening: at least 50% with 0, 2 and 4 fillers | 0.35 |
| Y4 | `h2`, `key=pair`, listening: at least 20 of the 40 (letter, digit) entries closed at the end | 0.45 |
| Y5 | `h2`, `key=pair`, listening: some entry of a digit after a digit, `+` or `=` is closed at the end (a wrong closing) | 0.40 |
