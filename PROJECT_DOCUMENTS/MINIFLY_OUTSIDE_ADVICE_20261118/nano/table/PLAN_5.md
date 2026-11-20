# What the system keeps and what it gains when the text changes (roadmap second revision, tracks K and L) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at 162f09a01; `trust.py`, `words.py`, `table.py`, `score.py` as they stand. Written before any run. Development level; no checker gate.

## 1. The evidence this follows

1. `PLAN_4.md`, reading 4: the largest difference measured on this line is between a system that stops changing and the same system going on (0.11 nats on the last tenth).
2. `loop_small_case`: the delay-chain system keeps 1.8% of addition after 200,000 lines of subtraction; the lines `nano/keep` and `nano/replay` work on that with sums. Nothing has yet been measured about keeping on prose.
3. A cell that counts is active only in its own context and its counts only grow. If keeping is anywhere easy, it is here; if it fails here, that is worth knowing first.

## 2. The question

**After the system has read a second, different text, what has it kept of the first, and did the first help it with the second?**

*The life.* Text A: the stream of the test (the first nine tenths of tiny Shakespeare, 1,003,854 bytes). Text B: three other texts already in the repository (`minifly_v148_pretrain_probe_dev_20261025/data/`: Bryant's stories, Burgess's Buster Bear, Carroll's Alice; 478,497 bytes together), read one after another, the first nine tenths of each; the last tenth of each is never read. A byte of B that is not among the 65 byte values of A is replaced by a space, and carriage returns are removed (the system has 65 lines; this is said, not hidden). One reading of each text, in the order A, B.

*Scored, each time with nothing changing during the scoring:* the last tenth of A, and the last tenths of the three texts of B, (i) after A, (ii) after A and B. A second system reads B only and is scored on the same parts.

*Systems, with the settings their plans chose, none chosen again here.* T2: counting cells read with a learned trust, the table's cells (K = 6) with the cells that step once a word (`PLAN_3.md`, arm 2; constant step 0.00005, e = 0.02). T0: the same without the word cells. M0: the summing form of `PLAN_1.md` (strengths changed by the mismatch, one reading, eta = 0.5), table's cells only, as a system of cells that does not count.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* T2 keeps A: its loss on the last tenth of A rises by less than 0.02 nats after B has been read. *Refuted if* it rises by 0.02 or more.

| | Statement | Chance |
|---|---|---|
| V1 | T2: the loss on A rises by less than 0.02 after B | 0.55 |
| V2 | M0: the loss on A rises by at least 0.01 more than T0's does | 0.55 |
| V3 | T2: having read A first, the loss on B's held-out parts is at least 0.02 lower than for the system that read B only | 0.80 |
| V4 | T2 after A alone is at least 0.3 nats worse on B's held-out parts than after A and B | 0.80 |

## 4. Size, cost, what is not run

Cells as in the earlier plans, more of them (cells are made for B's contexts as they occur); reported. A life takes under a minute of CPU. Not run here: references taught by back-propagation through the same life (to be asked of the reference line once this stands); the relay system through the same life (the loop line); a return to A after B with learning on; more than two texts.

## 5. Outcome (entered 20 November after the runs; `res/life_*.json`)

Loss in nats a byte on parts never read, nothing changing during the scoring. "B" is the three held-out parts together, weighted by their length (47,699 bytes). Of B's bytes 0.2 to 0.8% were not among A's 65 values and were replaced by a space.

| System | A after A | A after A and B | rise | B after A | B after A and B | B, system that read B only |
|---|---|---|---|---|---|---|
| T2, counting cells with trust, with the word cells | 1.4973 | 1.5067 | **0.009** | 1.8128 | 1.4152 | 1.4300 |
| T0, the same without the word cells | 1.5226 | 1.5306 | 0.008 | 1.8133 | 1.4454 | 1.4525 |
| M0, the summing form (no counting) | 1.5616 | 1.6190 | 0.057 | 1.8189 | 1.4530 | 1.5106 |

B by text, T2, after A and B against B only: Bryant (read first) 1.4579 against 1.5170; Burgess 1.2971 against 1.2553; Carroll 1.4109 against 1.3836. Cells that have counted, T2: 1,826,213 after A, 2,419,727 after A and B. A life took 5 to 15 CPU-seconds.

**The hypothesis holds**: T2's loss on the last tenth of A rises by 0.009 nats after B has been read, and under 0.02 was asked. V1 held. V2 held (the summing form's rise is 0.049 more than T0's). V3 failed: having read A first is worth 0.015 on B, and 0.02 was predicted. V4 held (0.40).

Readings (mine).
1. **On prose this system keeps what it has read.** After 430,000 bytes of three other books its loss on the first text is 0.6% higher. The same cells with strengths changed by the mismatch and summed lose seven times as much (0.057). No reference taught by back-propagation and no system of relay cells has yet been taken through this life; until one has, "seven times" is a statement about two forms of the same cells and nothing more.
2. **What it keeps costs it something on the new text, and the cost shows only after a while.** Against a system that read only B, having read A is worth 0.059 nats on the first new book and costs 0.042 and 0.027 on the second and third. Reading (mine): the cells of short runs are shared between the texts; a million bytes of A stay in their counts and outweigh what B says. A system that only ever adds to its counts cannot tell that the text has changed.
3. This is the old question of the roadmap's track K in a new place: on sums the system forgot; here it keeps and is slow to take the new. Mr. Xu, 20 November: cells "having different memory ability". The next plan gives each cell of the table a companion whose counts fade, and lets the answering cell's trust settle how the two are used (`PLAN_6.md`).
