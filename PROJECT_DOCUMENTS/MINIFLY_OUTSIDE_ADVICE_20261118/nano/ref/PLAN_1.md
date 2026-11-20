# Line L0, first plan: the references of the test at the level of nanoGPT (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as recorded in `ROADMAP_SECOND_REVISION_20261120.md`, section 1: "can you revise the road map and keep doing some experiments? Spin off some agents to speed up the experiment process." and "Note you can scale the network or do whatever you like." The roadmap's section 6 gives line L0 and the folder `nano/ref` to a separate agent session; this is that session. **Source.** Branch `claude/friendly-darwin-8mmhfm` at fe397ca6; this line's branch is `claude/nano-ref-8mmhfm`. Development level; one seed; no checker gate, seal or review is claimed. Nothing outside `nano/ref/` is changed. The adopted core is not touched; none of these programs is the loop: they are comparators and tables.

**Written before any run that shows a loss.** Before this plan was committed I ran only: `score.py`'s self-check (equal probabilities; the stream's byte frequencies); the table program on a made-up text of 60,000 bytes drawn from a random chain, to check its counts against a direct count; the GPT and one-layer programs with a rate of zero (nothing learned), to see that they run; and timing tests of a few steps of several GPT sizes, in which no loss is printed. No table was counted on the test text and no network was taught on it.

## What is measured

The measure is `MEASURE.md` (`score.py`): on the 111,540 bytes of the last tenth, each predicted from the bytes before it, the stream loss in nats and bits a byte and the share of bytes named right. For the GPTs also nanoGPT's own estimate (200 batches of random blocks of the last tenth).

1. **Tables of the last k bytes**, k = 0 to 8, counted on the stream (`tables.py`): unsmoothed for the share named right; interpolated Kneser-Ney (three discounts an order) and interpolated Witten-Bell for the loss. Also the same tables counted without the stream's last 50,000 bytes and scored on them (the development part used in item 3).
2. **GPTs by back-propagation** (`gpt.py`, written here in the manner of nanoGPT; AdamW 0.9/0.99, weight decay 0.1, gradient clipped at 1, rate 1e-3 with 100 steps of warm-up and a cosine fall to 1e-4 over the run; seed 1337; random blocks of the whole stream):
   * `small2k`: 4 layers, 4 heads, width 128, 64 bytes back, no dropout, batch 12, 2,000 steps (804,096 numbers that learn; 1.5 million bytes seen).
   * `small10k`: the same for 10,000 steps, as a run of its own with the rate falling over 10,000 steps (7.7 million bytes seen).
   * `large`: 6 layers, 6 heads, width 384, 256 bytes back, dropout 0.2 (nanoGPT's full network: 10,745,088 numbers that learn), **batch 16 in place of 64 and 4,000 steps in place of 5,000**: 16.4 million bytes seen against nanoGPT's 82 million. The timing test gave 4.0 s a step at batch 64 (5,000 steps: five and a half hours) and 0.90 s a step at batch 16, so 4,000 steps are about an hour of the four cores. This is not nanoGPT's full setting and is not to be called so.
   The exact stream loss (every byte from a full window) is taken for the small GPTs on every byte, and for the large one on every eighth byte; the cheaper reading with windows moved on by half a block is taken on every byte for all three.
3. **The loop's network by back-propagation** (`onelayer.py`): the last D = 10 bytes one-hot (650 lines), H logistic cells, a softmax answering layer; H = 256, 1,024, 4,096; batches of 256, Adam, rate 1e-3 falling by a cosine to 1e-4, 12 passes; seed 1337. The stream's last 50,000 bytes are kept out of learning as a development part; the strengths of the pass with the lowest development loss are the ones scored (the last pass is scored too). A ceiling for one layer on ten bytes by back-propagation with minibatches and Adam; not a twin of the loop's rule, which reads one byte at a time.
4. **If time allows**: D = 32 (2,080 lines) with H = 1,024 and 4,096; a GRU of 256 cells (and of 512 if an hour is left), taught on 32 lanes of the stream read in order, 64 bytes at a time with the state carried, Adam 2e-3, 20 passes, the same development part; scored with its state carried along the text.

## Hypothesis

The roadmap says (section 3) "A well-made table of five to seven bytes is, on a text of this size, already near the small setting's level". As a hypothesis of mine: **the best Kneser-Ney table of at most 8 bytes has a lower stream loss than the small GPT after 2,000 steps.** *Refuted if* the best table's stream loss is above `small2k`'s exact stream loss.

## Predictions (mine, before any run)

| | Prediction | Chance |
|---|---|---|
| T1 | the lowest Kneser-Ney loss over k = 0..8 lies between 1.50 and 1.65 nats | 0.60 |
| T2 | it is reached at k = 5, 6 or 7 | 0.75 |
| T3 | the Kneser-Ney loss at k = 8 is within 0.03 nats of the lowest | 0.70 |
| T4 | Witten-Bell is above Kneser-Ney at every k from 3 to 8 | 0.80 |
| T5 | the best unsmoothed table names between 50 and 56% right | 0.65 |
| G1 | `small2k`: nanoGPT's estimate on the last tenth between 1.83 and 1.95 nats (the coordinator recalls 1.88) | 0.65 |
| G2 | `small10k`: nanoGPT's estimate between 1.60 and 1.75 | 0.55 |
| G3 | for both small GPTs the exact stream loss is below nanoGPT's estimate, by 0.02 to 0.10 nats | 0.70 |
| G4 | `large`: nanoGPT's estimate at the last step between 1.50 and 1.62 (nanoGPT's full run is recalled as 1.47) | 0.50 |
| G4b | `large`: below 1.50 | 0.20 |
| G5 | `large` has a lower stream loss than the best table | 0.55 |
| G6 | `small10k` has a higher stream loss than the best table | 0.75 |
| H | the hypothesis above holds (best table below `small2k`) | 0.90 |
| M1 | one layer, D = 10, H = 256: stream loss between 1.82 and 1.98 | 0.60 |
| M2 | H = 1,024: between 1.72 and 1.86 | 0.55 |
| M3 | H = 4,096: between 1.64 and 1.80 | 0.50 |
| M4 | the loss falls from 256 to 1,024 to 4,096 | 0.85 |
| M5 | the best one-layer network on ten bytes has a higher stream loss than the best table | 0.80 |
| M6 | the best one-layer network on ten bytes names between 47 and 53% right | 0.60 |
| R1 | (if run) the GRU of 256: stream loss between 1.50 and 1.65 | 0.50 |

## Sizes and cost

Tables: a few minutes of one core. Small GPTs: 0.03 s a step on four cores (one and five minutes). The one-layer networks: seconds to a minute a pass. The large GPT: about 60 minutes of the four cores for learning, some minutes more for its estimates and scoring. The small runs and the one-layer runs are run side by side on the four cores first; the large GPT then has the four cores to itself. CPU times are process times (all threads), learning apart from scoring.

## What is not run

nanoGPT's full setting as published (batch 64, 5,000 steps): five and a half hours here. More than one seed. Any search over rates, sizes or schedules (each network is run once as written above). GPTs with other widths or windows. The loop itself, or any local rule (line L1). Tables longer than 8 bytes; tables with fitted discounts. Text generated by any model. No checkpoint or wheel is committed.

## Addition 1 (entered 20 November, after the tables, `small2k`, the one-layer runs with 256 and 1,024 cells and the GRU of 256; before the runs it adds)

**What was seen.** The networks that keep the stream's last 50,000 bytes out of learning score much worse on the last tenth than on that development part (one layer, 1,024 cells: 1.761 nats on the development part, 1.926 on the last tenth; GRU of 256: 1.607 and 1.763), while the tables do not (Kneser-Ney, k = 6: 1.548 and 1.517). A look at the text gives a reason: the play that fills the first two thirds of the last tenth begins about 25,000 bytes before the cut (its speakers' names first appear at bytes 978,546 to 989,789), that is, inside the development part. A network that does not learn from that part has never met these names; the tables and the GPTs, which read the whole stream, have. The development part as I set it in item 3 therefore handicaps the one-layer network and the GRU against the tables and the GPTs. This is a fault of my plan, found after its first runs; the runs stay in the record as they are.

**What is added.** An arm `whole=1` of `onelayer.py`: the whole stream is learned from, with the same seed, rate, schedule and number of passes as before (12 for one layer, 20 for the GRU); there is no development part and nothing is chosen; the last pass is scored. Run for one layer on ten bytes with 256, 1,024 and 4,096 cells and for the GRU of 256 (and for D = 32 if that is run). These, and not the runs with a development part, are the figures for the report's table; both are reported. A check with the tables: the same tables counted without the stream's last 50,000 bytes and scored on the last tenth. Also added to `gpt.py`, as a look and not a score: the mean loss at each place of a block (`bypos`), because `small2k`'s stream loss from a full window (1.906) came out above the reading from half a window and more (1.869); `small2k` is run once more with it (`small2k_again`), and `small10k` was stopped and started again with it before it had finished.

| | Prediction (mine, before these runs) | Chance |
|---|---|---|
| A1 | one layer, 1,024 cells, whole stream: stream loss at or below 1.85 (0.07 below the run with a development part) | 0.70 |
| A2 | GRU of 256, whole stream: stream loss at or below 1.66 | 0.60 |
| A3 | Kneser-Ney, k = 6, counted without the last 50,000 bytes: at least 0.05 nats above 1.517 on the last tenth | 0.70 |
| A4 | M1 to M6 above are scored on the whole-stream runs; I keep their chances as written, though M1 to M3 now look too low a range for 256 and 1,024 cells (the first runs gave 1.959 and 1.926) | - |

## Addition 2 (entered 20 November while the large GPT was still learning; before the runs it adds)

**Asked for by the coordinating session** (two messages, 15:20 and 15:26 UTC): (i) a reference by back-propagation that goes on learning through the last tenth, each block named before it is used for a step; (ii) the small GPT through the life of two texts of `nano/table/PLAN_5.md`, with the note that if only one can be done it should be (ii). I do (ii) now, on one thread beside the large run; (i) is **not run** unless time is left after the report, and is then a further addition.

**The life** (`life_gpt.py`; `load_all()` copied from `nano/table/life.py` on `claude/friendly-darwin-8mmhfm` at fcc174a, so the bytes and parts are the same). A: the stream of the test. B: the first nine tenths of Bryant's stories, Burgess's Buster Bear and Carroll's Alice (430,155 bytes; held-out tenths 47,131 bytes together), a byte not among A's 65 values made a space, carriage returns removed. The small GPT (4 layers, 4 heads, width 128, 64 bytes back, batch 12, seed 1337) is taught on A as `small10k` was (10,000 steps, 7.68 million bytes seen, 7.65 readings of A). It is then taught on B for 4,285 steps of the same size (12 blocks of 64 bytes a step: 3.29 million bytes seen, the same 7.65 readings of B), on random blocks of B's three stream parts joined; **no byte of A is shown again**; the optimiser's state is carried over. Two arms for the rate on B, since the recipe does not say what a second text's rate should be: `fresh_schedule` (the recipe again: 100 steps of warm-up to 1e-3, cosine fall to 1e-4) and `constant_1e-4` (the rate at which A ended). A third network sees B only (the recipe from a new start, 4,285 steps). Scored with nothing changing, each byte from the 64 bytes before it in its own text: the last tenth of A and the three held-out tenths of B, after A, after A and B (both arms), and for B only. One seed; a first figure only. Unlike the counting systems of `nano/table`, which read each text once in order, this learner reads random blocks, each text about 7.65 times; that difference is part of the comparison and is not removed.

| | Prediction (mine, before these runs) | Chance |
|---|---|---|
| L1 | `fresh_schedule`: the loss on A's last tenth rises by at least 0.15 nats after B | 0.75 |
| L2 | `constant_1e-4`: it rises by between 0.03 and 0.15 nats | 0.50 |
| L3 | B's held-out parts after A and B (`fresh_schedule`) are at least 0.05 nats below the network that saw B only | 0.80 |
| L4 | B's held-out parts after A and B (`fresh_schedule`) are below 1.4152 nats (the counting cells with a learned trust, `nano/table`) | 0.50 |
| L5 | B's held-out parts after A alone are at least 0.3 nats above those after A and B | 0.85 |

**Correction to addition 2 (entered at once, before any figure of it was seen).** I miscounted B's stream above. Its three stream parts are 219,510, 74,692 and 129,955 bytes: 424,157 together, not 430,155. The program takes the number of steps on B from the lengths themselves: round(10,000 x 424,157 / 1,003,854) = 4,225 steps (3.24 million bytes seen), not 4,285. The text above stands as written; these are the figures that hold.

## Outcome (entered 20 November, 16:20 UTC; the account is in `REPORT.md`; the large GPT was still learning and its predictions are scored in a later entry)

Stream loss on the last tenth, nats (named right, %). Tables, Kneser-Ney, k = 0..8: 3.347, 2.484, 2.038, 1.719, 1.561, 1.523, **1.517** (54.8), 1.519, 1.522; unsmoothed, best 53.9% at k = 6; Witten-Bell best 1.669 at k = 4. Small GPT: 2,000 steps 1.906 (43.7), nanoGPT's estimate 1.895; 10,000 steps 1.536 (54.2), estimate 1.577. One layer on ten bytes, whole stream, 256 / 1,024 / 4,096 cells: 1.818 (47.2) / 1.731 (49.9) / 1.695 (51.6); with the development part of the plan as first written: 1.959 / 1.926 / 1.942. One layer on 32 bytes, 1,024 cells: 1.879. GRU of 256: 1.574 (54.9) whole stream; 1.763 with the development part. Tables counted without the stream's last 50,000 bytes, k = 6: 1.677. The life of two texts: A 1.537, after B 2.302 (recipe's rate) and 2.030 (constant 1e-4); B's held-out parts 2.005 after A, 1.403 and 1.414 after A and B, 1.584 for B only.

| | Outcome | |
|---|---|---|
| T1 | lowest Kneser-Ney loss 1.5165 | held |
| T2 | reached at k = 6 | held |
| T3 | k = 8: 1.5220, 0.006 above the lowest | held |
| T4 | Witten-Bell above Kneser-Ney at every k from 3 to 8 (by 0.05 to 0.45) | held |
| T5 | best unsmoothed table 53.9% | held |
| G1 | `small2k`: nanoGPT's estimate 1.895 | held |
| G2 | `small10k`: nanoGPT's estimate 1.577, below my range of 1.60 to 1.75 | **failed** |
| G3 | `small10k`: stream loss 0.040 below the estimate; `small2k`: 0.011 above it | **failed** |
| G4, G4b, G5 | the large GPT has not ended | not yet scored |
| G6 | `small10k` 1.536 against the best table 1.517 | held (by 0.020) |
| H | the best table (1.517) is below `small2k` (1.906): the hypothesis is not refuted | held |
| M1 | 256 cells, whole stream: 1.8175, 0.003 below my range of 1.82 to 1.98 | **failed** |
| M2 | 1,024 cells: 1.731 | held |
| M3 | 4,096 cells: 1.695 | held |
| M4 | the loss falls from 256 to 1,024 to 4,096 | held |
| M5 | best one layer on ten bytes 1.695 against the best table 1.517 | held |
| M6 | best one layer on ten bytes names 51.6% | held |
| R1 | GRU of 256, whole stream: 1.574 | held |
| A1 | one layer, 1,024 cells, whole stream: 1.731 | held |
| A2 | GRU of 256, whole stream: 1.574 | held |
| A3 | Kneser-Ney, k = 6, without the last 50,000 bytes: 1.677, 0.160 above 1.517 | held |
| L1 | the recipe's rate on B: A rises by 0.765 | held |
| L2 | constant 1e-4: A rises by 0.493, far outside my 0.03 to 0.15 | **failed** |
| L3 | B after A and B 1.403 against B only 1.584: 0.181 below | held |
| L4 | 1.403 is below 1.4152 | held (by 0.012) |
| L5 | B after A alone 2.005, 0.602 above | held |

M1 to M6 and R1 are scored on the whole-stream runs, as addition 1 said; on the runs with the development part as first planned, M1 would hold (1.959) and M2, M3, M4 and R1 would fail (1.926, 1.942, not falling, 1.763). Of the 25 predictions scored here 21 held and 4 failed. Where I was wrong: I gave the small GPT too little for 8,000 further steps; I assumed a full window is always worth more than half of one, which is false for a network after 2,000 steps; and I expected a small constant rate to protect the first text, which it did not. The plan's own fault (the development part) is described in addition 1 and in the report's section 5.

**Departures from the plan.** `small10k` was stopped and started again once, to add the look at each place of a block. One layer on 32 bytes was run with 1,024 cells and the whole stream only; 4,096 cells and the GRU of 512 were not run. The large GPT took far longer than the hour I gave it (1.4 s a step alone, 4 to 5 s when other runs shared the cores), and the report was written before it ended.
