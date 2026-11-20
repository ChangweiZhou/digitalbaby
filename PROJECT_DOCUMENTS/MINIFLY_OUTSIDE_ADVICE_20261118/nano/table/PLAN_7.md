# Do the relay's cells add anything to the cells that count? (roadmap second revision, step L-TR, in the trust form) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at 2617c8fa4; the relay system is the program of the line `nano/loop` (`loop.py`, `kernp.c`, `kernt.c` at a87f720 of `claude/nano-loop-8mmhfm`, copied into this branch unchanged). Written before any run. Development level; one text; no checker gate.

## 1. The evidence this follows

The line `nano/loop` (its first plan, five seeds): the relay system on the test text gives 1.768 nats with 1,024 cells after eight readings, against 1.52 for the table's cells. Cells that count, read with a learned trust, with the cells that step once a word: 1.4973 (`PLAN_3.md`). The roadmap's step L-TR asks whether the two kinds together are better than the cells that count alone, by at least 0.02 nats. That step was given to the loop line in the summing form and is waiting behind its runs on scale; `PLAN_2.md` showed that the summing form is the wrong place to add a kind of cell, and the trust form exists now. This plan asks the question in the trust form.

## 2. The question

**Read as one more sender with its own trust, does the relay system lower the loss of the cells that count?**

*The system.* T2 of `PLAN_5.md` (the table's cells, K = 6, and the cells that step once a word; e = 0.02; the trust's step 0.00005), and beside it the relay system of `nano/loop` with the settings its first plan chose (1,024 cells, ten delay lines, branches that compete, eta1 = 1, eta2 = 1/4, fading 1e-5). The relay system reads the same stream at the same time, **once**: one life for both. Its message at a byte is the log of the probabilities it gives; the answering cell of the whole holds a trust for it by the level of its confidence (its largest probability, in ten equal bins), changed by the mismatch as every other trust. The relay system learns from its own mismatch as in `nano/loop`; nothing of the counting cells reaches it.

One simplification in the program, stated: the relay system's answers are computed in blocks of 512 bytes from its strengths at the start of the block, and it then learns the block. An answer therefore never uses a byte it is asked about, and is at most 512 bytes out of date.

*Arms.* (0) T2 alone (must give 1.4973). (1) T2 with the relay system whose return strengths are born at nothing (the setting the loop line's development seed chose; in a system of one relay step this is back-propagation). (2) T2 with the relay system whose return strengths are born at the earlier size and grow with the forward ones (the relay proper). (3) Each relay system alone after its one reading, for reference. Relay seed 1 (the development seed); the reported seeds only if an arm gains at least 0.01.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis (the roadmap's, which I do not expect to hold).* Arm 1 or arm 2 is at least 0.02 nats below arm 0 on the last tenth. *Refuted if* both gain less than 0.02.

| | Statement | Chance |
|---|---|---|
| R1 | arm 1 gains at least 0.02 nats over arm 0 | 0.25 |
| R2 | arm 1 gains at least 0.005 | 0.60 |
| R3 | arm 2 gains less than arm 1 | 0.60 |
| R4 | the relay system alone, after one reading, is above 1.85 nats in both forms | 0.60 |

## 4. Size, cost, what is not run

The relay system: 1,024 cells, 733,249 numbers that learn (the loop line's count). The counting cells as `PLAN_3.md`, arm 2. Ten more trusts. One reading of the relay system with its answers kept takes a few CPU-minutes; the rest seconds. Not run here: a relay system that has read the stream several times before the life (its answers on the stream would then be for bytes it has seen, and the trust learned from them would not hold on unread text); more cells; the summing form (the loop line's plan).

## 5. Addition 1 (20 November; written after the two arms on the development seed were read and before the runs it describes)

On the development seed arm 1 gains 0.0115 and arm 2 gains 0.0078 (section 6 will give them). By the plan's rule arm 1 is run on the five reported seeds (73129, 93683, 116089, 406364, 545353; the seed is that of the relay system's birth; the cells that count have none). Beyond the rule, and said here before the runs: arm 2 is run on the same five seeds as well, because it is the arm with the relay proper (return strengths born at a size, growing with the forward ones) and the comparison of the two forms on one seed carries nothing. Prediction (mine): the median gain of arm 1 over five seeds is between 0.008 and 0.015 (chance 0.7), and arm 1 is better than arm 2 on at least four seeds of five (chance 0.6).

## 6. Outcome (entered 20 November after all runs; `res/relay_arm*.json`)

Loss on the last tenth, nats a byte, nothing changing. The seed is that of the relay system's birth. The compiled loop with a given sender equals numpy to 1.3e-15 on a prefix; arm 0 gives the figure of `PLAN_3.md`.

| Arm | Development seed | Five reported seeds | Median | Gain over arm 0 | The relay system alone, after its one reading |
|---|---|---|---|---|---|
| 0, T2 alone | 1.4973 | (no seed) | | | |
| 1, with the relay system born at nothing | 1.4858 | 1.4858, 1.4859, 1.4860, 1.4857, 1.4861 | 1.4859 | **0.0114** | 1.915 to 1.920 |
| 2, with the relay proper (return strengths born at a size) | 1.4895 | 1.4897, 1.4895, 1.4897, 1.4893, 1.4894 | 1.4895 | 0.0078 | 1.968 to 1.972 |

The trust the answering cell ends with for the relay system, by its level of confidence (development seed, arm 1): 0.21 to 0.27 where its largest probability is under one half, falling to 0.16 where it is above 0.8. A run took about 90 CPU-seconds on one thread, nearly all of it the relay system's reading.

**The hypothesis is refuted**: read as one more sender with its own trust, the relay system lowers the loss by 0.011 (born at nothing) and 0.008 (the relay proper), and 0.02 was asked. R1 failed (as I expected). R2 held. R3 held, on every one of the five seeds. R4 held. Addition 1: both predictions held (median gain 0.0114; arm 1 better on five seeds of five).

Readings (mine).
1. **The relay's cells add something to the cells that count on prose, and it is small**: about one hundredth of a nat, from a system that alone stands at 1.92 after its one reading. It is the same size as each of the other kinds added today (`PLAN_6.md`, `PLAN_8.md`, `PLAN_9.md`).
2. The form born at nothing adds more than the relay proper, on every seed, as the loop line found for the relay system alone.
3. The answering cell trusts the relay system less where the relay system is surer. Reading (mine, untested): where it is sure the cells that count are sure too and need no second voice; it is heard where it is unsure, which is where the counting cells have least to go on.
4. *A look inside, not a registered measure* (`where.py`, `res/where_T2.json`): 35% of T2's loss on the last tenth is at the first letter of a word and 10% at punctuation, that is, in which word comes next. That is where a learner that generalises across words would have to add; this relay system reads ten bytes and does not.
5. What this does not show: that a larger or longer-taught relay system would add no more (not run: more cells, more readings before the life); anything about sums, where the relay's cells carry what cells that count lack.
