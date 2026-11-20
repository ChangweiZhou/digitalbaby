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

## Outcome of addition 1 (entered 20 November 2026; development seed 1; `res/p9_crit_h2_s1.json`, `res/p9_*_listen_s1.json`)

**(a) The expectation rule proposes nothing on `h2`.** Share passed over after 400,000 lines: filler letters 0.00 and filler digits 0.00 at theta 0.2 and at 0.3 (0.13 and 0.18 at 0.5, with the answer's first digit at 0.08); every other kind 0.00. **Y1 held**, for the reason given. One thing the measurement shows that I had not asked: the fillers' total variation is lower than every other kind's (medians 0.75 for the letters and 0.53 for the digits, against 0.88 to 1.00), so the rule would rank the fillers first if it were used to order what to try and not to decide. The combined arm as asked (the rule proposes, the mismatch judges) has nothing proposed here and was not built.

**(b) The listening form.** Never-shown pairs, last three readings:

| Run | Never-shown pairs | Closed at the end |
|---|---|---|
| `g2`, `key=value`, `listen=16`, 1,200,000 lines | 93.7 at every gap (the period form: 95.1) | exactly the letters, from the first reading |
| `h2`, `key=pair`, `listen=16`, 2,400,000 lines | 57.7 / 1.0 / 0.6 | nothing (one other entry at the first reading) |
| `h2`, `key=value`, `listen=16`, 2,400,000 lines | 61.3 / 19.3 / 0.3 | `+`, from the first reading; nothing else |

The mean evidence of the entries in the `h2`, `key=pair` run, in units of its own uncertainty, at four readings across the run: filler digits -2.5, -3.3, -3.4, -3.1; letters -3.6, -4.2, -3.9, -3.4; other entries -4.3 to -4.7.

**Y2 held** (the listening form does on `g2` what the period form did). **Y3, Y4 failed**: on `h2` it closes no filler entry. **Y5 failed** (no wrong closing either).

**Where it fails, now located (Reading, mine).** The listening form removes the dilution (cause (ii) above) and the entries still do not close; and their evidence is not near nothing, it is steadily against closing, at three to four times its uncertainty. So it is cause (i), and stronger than I put it: closing one half of a filler does not merely earn nothing, it is punished. A filler digit closed while its letter is in the chain, or a letter closed while its digit is in the chain, makes the line worse to name than leaving both. Each entry tries alone, each finds that closing does harm, and the pair that would pay together is never held long enough to be seen. The trial rule finds a byte whose passing over pays by itself. It does not find two bytes that pay only together.

(`key=value` closing `+` and reaching 19.3% with two fillers, that is one filler pair, is a thing I did not foresee and have not examined.)

## Addition 2 (20 November 2026, written before its runs): an entry knows whether the byte before it was passed over

*One change.* With `key=pair` and the listening form, each entry has two tendencies with their own evidence: one used when the byte before its own entered the chain, one used when that byte was passed over (`bit=1`). What it reads beyond before is one thing, and it is local: whether the chain stepped at the last byte. 472 tendencies where there were 236.

*Why this and not something else (Reading, mine).* The failure is that two closings pay only together. If the digit of a filler can lean one way when its letter went into the chain and the other way when its letter was passed over, then "closed after a letter that was passed over" is a trial that completes a filler whenever chance has closed the letter, and it should be rewarded by itself; once that tendency is up, closing a letter is closing a whole filler and should pay as closing a letter paid on `g2`. The second number's first digit, which follows a filler digit that was passed over, must then learn to stay open after a byte passed over; whether it does is part of the test.

*Run (development seed 1).* `h2`, `key=pair`, `listen=16`, `bit=1`, 2,400,000 lines. If it reaches 50% with 0, 2 and 4 fillers, the five reported seeds with `clock=none` and `clock=oracle` beside it follow at once, as registered here.

| | Prediction (mine, before the run) | Chance |
|---|---|---|
| Z1 | at least 50% of never-shown pairs with 0, 2 and 4 fillers (development seed) | 0.45 |
| Z2 | at the end at least 35 of the 40 entries of a filler digit after a letter that was passed over are closed | 0.50 |
| Z3 | at the end no entry of a digit after a digit is closed in either state | 0.50 |

## Outcome of addition 2 on the development seed, and the reported seeds as run (20 November 2026; the second written before its runs)

Development seed 1, `h2`, `key=pair`, `listen=16`, `bit=1`, 2,400,000 lines (`res/p9_h2_pair_listen_bit_s1.json`): **95.2%** of never-shown pairs with 0, 2 and 4 fillers alike (readings every 200,000 lines: 4, 17, 35 / 48 / 16, 87, 90, 94, ...). Closed at the end: all 40 entries of a filler digit after a letter that was passed over (22 of them at the first reading, all 40 from the third); all 40 entries of a letter after a filler digit that was passed over, and the 4 of a letter after `+`; no entry of a filler digit after a letter that entered the chain; no other entry in either state. **Z1, Z2, Z3 held.**

*The reported seeds, as registered in addition 2, with one arm more:* `h2`, 2,400,000 lines, seeds 73129, 93683, 116089, 406364, 545353: (1) `key=pair`, `listen=16`, `bit=1`; (2) the same without the bit (what it changed from); (3) `clock=none`; (4) `clock=oracle`, 800,000 lines. Hypothesis: arm (1) has at least 50% with 0, 2 and 4 fillers at the median; refuted if under 50%. Prediction Z4 (mine): arm (1) at least 85% on at least four of five seeds (chance 0.6); Z5: arm (2) under 5% with 4 fillers on every seed (chance 0.85).
