# The kinds of cells found today, together (roadmap second revision, track L) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at 59f2f4342; the programs of `PLAN_3.md` to `PLAN_9.md`. Written before any run, and before the last arms of `PLAN_9.md` and the seeds of `PLAN_7.md` were read. Development level; one text; no checker gate.

## 1. The evidence this follows

Beside the table's cells and the cells that step once a word (T2, 1.4973 nats on the last tenth), four kinds of sender have each been added alone today, each read with its own trust: companions whose counts fade over 10,000 bytes (1.4892, `PLAN_6.md`); the relay system born at nothing, one reading (1.4858 on the development seed, `PLAN_7.md`); cells that time how long ago each byte value last came, born with no trust (1.4873, `PLAN_8.md`); a pool of cells keyed on two or three of the last eight bytes, born with no trust (1.4886, `PLAN_9.md`). Singly they gain 0.008 to 0.012. `PLAN_4.md` found that two gains did not add in full. Mr. Xu asked on 20 November whether we are ready for an experiment at the level of nanoGPT; its full setting is described at about 1.47, and the reference line's own run of it is not yet in.

## 2. The question

**Offered together, do these kinds add up, and where does the system then stand?**

*The system.* T2 as before (K = 6; e = 0.02; the trust's step 0.00005; one reading), with all four additions at once, each as its own plan had it: the companions (half-life 10,000 bytes, trust born at 0.25); the relay system of `nano/loop` (1,024 cells, born at nothing, development seed 1, reading the stream once alongside; trust born at 0.25 by ten levels of its confidence); the 65 kinds that time a byte value and the 82 kinds of the pool (trust born at nothing). Nothing is chosen here; the last tenth is scored once for each arm, with nothing changing. A second figure, marked as a different measure, lets everything go on changing through the last tenth (the relay system's answers there stay those of its strengths at the cut).

*Arms.* (All) everything; and four arms that each leave one addition out, to see what each is worth in the presence of the others.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The additions add up to at least 60% of the sum of their single gains (0.0081 + 0.0115 + 0.0100 + 0.0087 = 0.0383, so at least 0.023): arm All is at most 1.4743. *Refuted if* it is above 1.4743.

| | Statement | Chance |
|---|---|---|
| T1 | arm All is at most 1.4743 | 0.50 |
| T2 | arm All is at most 1.4700 | 0.30 |
| T3 | leaving out any one addition makes the system worse (four of four) | 0.60 |
| T4 | the addition whose leaving out costs most is the relay system | 0.35 |
| T5 | going on through the last tenth, arm All is below 1.35 | 0.55 |

## 4. Size, cost, what is not run

About 4.8 million cells that count (reported exactly), the relay system's 733,249 numbers, and 6,890 trusts. A run takes a few CPU-minutes and about four gigabytes. Not run here: runs of eight bytes (worth 0.003 beside the word cells); the reported seeds for the relay system (its five seeds in `PLAN_7.md` differ by 0.0002); any choice of a setting.

What this plan cannot show, said beforehand: whether the system is below nanoGPT's full network on this measure, since that network's figure by the reference line's own run is not yet in; and that anything here generalises beyond what cells that count can do.
