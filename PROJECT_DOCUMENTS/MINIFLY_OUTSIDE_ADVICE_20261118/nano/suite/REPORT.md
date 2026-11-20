# Line `nano/suite`: one system, one setting, six byte streams of different kinds (20 November 2026)

Branch `claude/nano-suite-8mmhfm`, folder `nano/suite`; source `claude/friendly-darwin-8mmhfm` at 32b631d91. Development level. No checker gate, seal or review is claimed; the checks named are executed checks by the author of the programs. Readings are marked as mine. Mr. Xu's words that set the line are quoted in `PLAN_1.md`; the tasks were set by the coordinating session of the outside-advice work. Seven plans, each committed before its runs, each with its outcome and every prediction scored (`PLAN_1.md` to `PLAN_7.md`).

## 1. What was asked and what was built

The programme judged its ideas on two-digit sums and on one or a few English texts, with numbers set for each. This line built a test-bed of six real byte streams of five kinds (`make_streams.py`, `manifests/`): tiny Shakespeare; Moby Dick; Python source (the machine's standard library); C headers; a data file with numbers in rows (the Unicode collation table); markup with short texts in some fifty languages (the shared MIME database). 1.1 to 1.9 million bytes each; the first nine tenths read once, the last tenth the test; at most 128 values a stream. Two measures on the tenth: **nothing changing**, and **everything going on** (each byte answered before it is read, cells and trusts moving). References on the same split: xz, bzip2, gzip at their strongest (they go on learning); Kneser-Ney tables of the last 3 to 6 bytes (they change nothing).

The system is the programme's "runs of 0 to 6 bytes" (cells that count; an answering cell whose branches compete on the sum of trust times message; nothing but local rules), in a new kernel where a cell stores only the counts it has (`suite.py`). It is checked against a plain-Python reference in fourteen cases, against `nano/table/gate.py` itself (tiny Shakespeare's last tenth: 1.5226, 459,129 cells), and by a test that no answer moves when its own byte and all later ones are replaced (`check_output.txt`). **One setting on every stream** (e = 0.02; trusts of the runs born at 0.25, all others at nothing). Every number was fixed by an earlier line, by the coordinating session, or on the two prose streams; nothing was chosen on the other four.

## 2. The table (`report_final.txt` has it with sizes and cost; nats a byte on the last tenth)

**Everything going on**

| system | shakespeare | book | py | c | table | xml |
|---|---|---|---|---|---|---|
| base: runs of 0 to 6 bytes, first cell | 1.4136 | 1.4476 | 1.1503 | 1.0301 | 0.8141 | 0.4570 |
| + second cell (parts by kind and byte) | 1.3876 | 1.4184 | 1.1228 | 0.9470 | 0.6944 | 0.4398 |
| + one recalling sender | 1.3783 | 1.4084 | 1.0351 | 0.8448 | 0.5700 | 0.3707 |
| + the rule for the step | 1.3705 | 1.4006 | 1.0210 | 0.7952 | 0.5253 | 0.3547 |
| + last-byte senders | 1.3427 | 1.3953 | 0.9657 | 0.7172 | 0.3483 | 0.2953 |
| Z: with three recalling senders | 1.3389 | 1.3910 | 0.9496 | 0.7068 | 0.3415 | 0.2844 |
| **ZA: Z with trusts by age** | **1.3344** | **1.3894** | **0.9358** | 0.6908 | 0.3431 | 0.2846 |
| ZP: Z with since and pool cells | 1.3497 | 1.3972 | 0.9677 | 0.5970 | 0.2494 | 0.2790 |
| **ZPA: ZP with trusts by age** | 1.3459 | 1.3959 | 0.9582 | **0.5921** | **0.2479** | **0.2777** |
| base with classes (`nano/merge`'s program) | 1.4003 | 1.4332 | 1.1324 | 0.9397 | 0.6373 | 0.4484 |
| base with made kinds F (`nano/merge`'s program) | 1.3993 | 1.4328 | 1.1250 | 0.8942 | 0.6210 | 0.4494 |
| xz -9e | 1.7496 | 1.7876 | 1.1265 | 0.7244 | 0.3814 | 0.4333 |
| bzip2 -9 | 1.6621 | 1.7035 | 1.2079 | 0.9426 | 0.5976 | 0.3845 |
| gzip -9 | 2.1463 | 2.2479 | 1.3972 | 1.0363 | 0.8122 | 0.6184 |

**Nothing changing**

| system | shakespeare | book | py | c | table | xml |
|---|---|---|---|---|---|---|
| base, first cell | 1.5226 | 1.4676 | 1.4568 | 2.1222 | 1.7567 | 0.8077 |
| + second cell | 1.4978 | 1.4388 | 1.4349 | 2.1137 | 1.8376 | 0.7915 |
| Z (second cell) | 1.5470 | 1.4580 | 1.4661 | 2.2396 | 2.0140 | 0.8431 |
| ZA, second cell | 1.5454 | 1.4591 | 1.4333 | 2.1829 | 1.7527 | 0.8179 |
| ZA, first cell | 1.5123 | 1.4586 | 1.4211 | 2.0939 | 1.6558 | 0.7723 |
| ZPA, second cell | 1.5471 | 1.4716 | 1.5171 | 2.5265 | 2.3660 | 0.9326 |
| ZPA, first cell | 1.4868 | 1.4322 | 1.3852 | 2.1363 | 1.7228 | 0.7410 |
| base with classes | 1.5116 | 1.4536 | 1.4300 | 2.1532 | 1.7906 | 0.8019 |
| base with made kinds F | 1.5099 | 1.4530 | 1.4307 | 2.1454 | 1.7406 | 0.8056 |
| best Kneser-Ney table (k) | 1.5165 (6) | 1.4570 (6) | 1.3974 (6) | 2.0578 (6) | 1.8000 (4) | 0.7893 (6) |

**Size and cost.** Base, Z, ZA: 0.30 to 0.56 million cells, 0.45 to 1.0 million numbers stored (counts above zero and parts), 2 to 7 CPU-seconds a stream. ZP, ZPA: 1.8 to 4.0 million cells, 7.8 to 16 million numbers, 111 to 191 CPU-seconds. Classes: 1.8 to 3.3 million cells, 2.9 to 6.2 million numbers, 99 to 179 CPU-seconds; made kinds: 1.3 to 3.3 million cells, 124 to 782 CPU-seconds. Two-digit sums (section 4): 1.21 million cells, 50 CPU-seconds a run.

## 3. Which ideas are general (the rule of `PLAN_2.md`: lower by at least 0.001 on at least three of the five kinds of stream, no stream harmed by more than 0.005)

| idea | where it was found | going on | nothing changing | cost |
|---|---|---|---|---|
| second cell, parts by kind and byte | English, sums | kept (5 of 5) | not (table +0.081) | none |
| since-cells | English | kept (5 of 5) | not (table +0.195) | 1.3 to 3 x cells |
| pool | English | kept (5 of 5) | not (c +0.050) | 3 to 10 x cells |
| classes found by search | sums, English | kept (5 of 5) | not (c, table +0.03) | 4 to 7 x cells, 30 x CPU |
| made kinds born with the pooled past (F) | sums, English | kept (5 of 5) | not (c +0.023) | as classes, up to 5 x their CPU |
| birth only where surprised | English, sums | (a saving) 18 to 32% fewer cells for 0.000 to 0.012 | 0.002 to 0.052 | fewer cells |
| a sender that recalls (step 3, the coordinator's) | here | kept (5 of 5): 0.009, 0.010, 0.088, 0.102, 0.124, 0.069 | not (c +0.116) | no cells |
| the rule for the trust's step (step 4) | here | kept (5 of 5) | not (table +0.172) | none |
| a cell says what it counted last (step 5, mine) | here | kept (5 of 5): 0.028, 0.005, 0.055, 0.078, 0.177, 0.059 | not (every stream +0.02 to +0.08) | one value a cell |
| trusts of the senders of recency by age (step 7, mine) | here | kept (3 of 5, largest harm +0.002) | **kept** (4 of 5, largest harm +0.001) | one number a cell |

**English or sums only:** by the rule, none of the ideas tried. What is per-text is not an idea but two numbers: the smoothing e with the trust at birth (section 4).

## 4. The per-text numbers (`PLAN_4.md`)

One rule for the step (each trust's change = eta x its driver / the root of the sum of its own squared drivers; eta = 0.03 chosen on the two prose streams, flat from 0.01 to 0.1) is lower than the prose step 0.00005 on all six streams going on. On two-digit sums as one stream among others (40,000 lines, five reported seeds, never-shown sums named whole, first cell / second cell at the median): the programme's per-text setting 79.4 / 88.0; **only the step replaced by the rule 81.3 / 91.1**; the one setting of the streams (e = 0.02, born 0.25, the rule) 75.0 / 77.7; the prose setting as it was 62.4 / 68.7. So the rule replaces both settings of the step; the smoothing does not transfer and remains the number set for a text.

## 5. The best general system and its distance from the compressors

There is no single best. **ZA** (runs of 0 to 6 bytes; second cell; three recalling senders; last-byte senders; the rule; trusts by age; no more cells than the base) is the best on prose and Python source and is below xz on all six streams, going on: by 0.42, 0.40, 0.19, 0.03, 0.04, 0.15 nats a byte. **ZPA** (the same with the since and pool cells, four to twelve times the cells, about twenty-five times the CPU) is far better on C headers and number rows (0.59 and 0.25 against xz's 0.72 and 0.38) and 0.007 to 0.022 worse than ZA on prose and Python. The better of the two stands at 76%, 78%, 83%, 82%, 65% and 64% of xz's cost. The base stood at 81%, 81%, 102%, 142%, 213% and 105%.

What this comparison can say: xz is the strongest general compressor on this machine, and nothing was downloaded; it is a dictionary coder. Compressors that mix the opinions of many contexts, the family this system resembles, are not installed and are known to be clearly better than xz on text. "Below xz" means past the dictionary coders, not at the state of the art. The system also holds far more memory per byte read than a compressor and is slower.

## 6. My readings

1. **The programme's one text hid a weakness and its remedy.** On prose the base was already 0.34 nats below xz; on source code, rows of numbers and markup xz was below it everywhere, by up to 0.43. What those streams asked for was hearing the present: going on at all (0.31 to 1.09 nats on the four), the recalling sender, the last byte of each cell, a faster trust. Each is general going on and costs no cells.
2. **The ideas found on English did transfer.** I predicted the since-cells, classes and made kinds would not be kept (chances 0.30, 0.30, 0.15) and was wrong on all three. The fear that the prose line was fitting English is not supported on these streams.
3. **The two measures part on streams whose end is unlike their beginning** (C headers joined from files; a table sorted by key). There, with nothing changing, whatever learns more from the nine tenths is more wrong about the tenth, and every idea but one is "not kept". On the four streams whose end is like their beginning, every idea of the day's list but birth is lower on both measures. A learner that is meant to go on should be judged going on; the figure with nothing changing remains the fair one against a fixed table or a network that is not updated.
4. **A sender of recency should be trusted by the age of what it says.** With that index the first answering cell gains from recency on both measures on every stream (1.5123 against the base's 1.5226 on tiny Shakespeare with nothing changing). The second cell's fast-learned parts remain a sender of the present.
5. **Of the two per-text numbers carried longest, the step can go and the smoothing cannot yet.** And the rule's one number may not be free of the number of senders (ZP is worse than Z on prose; sums with 256 kinds): untested.

## 7. Faults found and repaired, and what limits the claims

* The recalling sender first saw its own byte (the table of places written one byte early; 0.0002 nats on the development streams). The reference had the same fault, so the check had passed. Repaired before any claim; the check now has a test for it.
* Under the rule, a driver whose square is zero in the machine's numbers moved a trust to infinity on a reported seed of sums. Repaired; the arm was rerun on all five seeds.
* The Kneser-Ney reference's published discounts fall outside their range on the collation table; they are now bounded.
* Limits. One run of each system on each stream (the system is deterministic; there are no seeds but on sums). The four streams other than prose were each scored seventeen times by this line (once for each system of the table and of the steps); nothing was chosen on them, but the order of the work was mine and later steps were shaped by earlier results (the last-byte senders and the index by age were thought of after seeing where the system lost). Tiny Shakespeare's tenth is worn by the whole programme. The compressors are not the strongest of their trade. Five kinds of stream are not all kinds: no binary data, no audio or images, nothing longer than two million bytes. Classes and made kinds could not be put into the combined system (another program).

## 8. Not run

A rule for the smoothing; the rule for the step with its number scaled by the senders that speak; parts of the second cell indexed by age; classes and made kinds in this kernel, with the second cell and the senders of recency; counts that fade; longer streams, and a life over several streams in turn (keeping); the systems with word cells (their ends are given by hand).
