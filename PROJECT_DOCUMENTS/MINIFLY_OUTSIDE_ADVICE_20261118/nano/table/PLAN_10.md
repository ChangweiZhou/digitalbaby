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

## 5. Outcome (entered 20 November after the runs; `res/all*.json`)

Loss on the last tenth, nats a byte. The unified loop gives back the loops of `two.py` and of `relay.py` exactly on a prefix (largest difference 0). Each arm was scored once.

| Arm | Nothing changing | Cost of leaving it out | Going on through the last tenth (a different measure) | Cost of leaving it out |
|---|---|---|---|---|
| All | **1.4707** (56.4% of bytes named right) | | **1.3470** (59.1%) | |
| without the cells that time a byte value | 1.4791 | 0.0084 | 1.3588 | 0.0118 |
| without the pool keyed on a few bytes | 1.4750 | 0.0043 | 1.3513 | 0.0043 |
| without the relay system | 1.4740 | 0.0033 | 1.3462 | -0.0008 |
| without the fading companions | 1.4715 | 0.0008 | 1.3673 | 0.0203 |

Size of arm All: 4,489,825 cells that have counted, 164 kinds, the relay system's 733,249 numbers; 168 CPU-seconds, of which about 90 are the relay system's reading.

**The hypothesis holds**: all together the additions gain 0.0266 nats over T2 (1.4973), which is 69% of the sum of their single gains (0.0383); 60% was asked. T1 held. T2 failed by 0.0007 (1.4707 against 1.4700). T3 held (each addition costs something when left out). T4 failed: what costs most to leave out is the cells that time a byte value, not the relay system. T5 held.

Readings (mine).
1. **The kinds found today add up to about two thirds of what each gives alone.** The system stands at 1.4707 with nothing changing, at the level at which nanoGPT's full setting is described, and at 1.347 if it goes on changing. Beside it, measured by the reference line on the same bytes: the small GPT after 10,000 steps 1.536, a GRU 1.574, the smoothed table 1.517.
2. **What each kind is worth beside the others is not what it is worth alone.** With nothing changing the companions are worth almost nothing beside the rest (0.0008) and the cells that time a byte value the most (0.0084); for a system that goes on, the companions are worth the most (0.020).
3. The relay system is worth 0.003 beside the others with nothing changing. In the going-on measure it is worth nothing, but that comparison is not fair to it: there its answers stay those of its strengths at the cut while every other sender goes on.
4. **What tempers this figure.** (a) The ends of words were given; found by the system they cost 0.002 (`nano/ends`). (b) Each arm was scored once, but the last tenth of this one text has now been scored for some forty arms in the course of the day, and what was carried from one plan to the next was chosen with those scores in sight. The gain of all the kinds over T2 therefore needs a text on which nothing was chosen; addition 1 asks for that. (c) The system holds 4.5 million cells; the small GPT about 0.8 million numbers and nanoGPT's full network about 10.7 million. (d) nanoGPT's figure on this measure by the reference line's own run is not yet in.

## 6. Addition 1 (20 November; written after section 5 and before the runs it describes): the same systems on texts where nothing was chosen

The three other books (`PLAN_5.md`: Bryant, Burgess, Carroll) have been used today only to measure keeping; no kind, setting or threshold was chosen on them. A fresh system reads the first nine tenths of each in turn (the life "B only" of `PLAN_5.md`) and each book's last tenth is scored right after its stream, nothing changing. Arms: T2 (already run: 1.5598, 1.2769, 1.3836; `res/two_T2_H0_moments.json`) and all the additions but the relay system, whose program reads only the test text. On the test text that arm gains 0.0233 over T2 (1.4740 against 1.4973).

*Prediction (mine):* on each of the three books the gain over T2 is at least half of that, 0.0117 (chance 0.5 for all three; 0.8 for at least two). *If the gain is under 0.005 on two books or more*, the gains of sections 5 were fitted to the last tenth of one text and are withdrawn as general statements.
