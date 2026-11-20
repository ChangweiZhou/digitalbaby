# Line L0: the references of the test at the level of nanoGPT (20 November 2026)
\n**Interim state: the large GPT is still learning; the entries written LARGE_... and TABLE_HERE are open and are filled in when it ends.**

By the agent session that owns `nano/ref` (roadmap, second revision, section 6). Development level; one seed for each network; repeated by no one; no checker gate, seal or review is claimed. Nothing outside `nano/ref/` was changed. None of these programs is the loop or the adopted core: they are tables and comparators taught by back-propagation. The plan, `PLAN_1.md`, was committed before the runs (ad361c5); an addition to it was committed before the runs it adds (9378f61); its outcome section scores every prediction.

## 1. In brief

* **The measure** is defined once in `MEASURE.md` and computed by `score.py` (numpy): on the 111,540 bytes of the last tenth, each predicted from the bytes before it, the stream loss in nats and bits a byte and the share of bytes named right. The first bytes of the last tenth take the end of the stream as their context; every byte is scored.
* **The best table**: interpolated Kneser-Ney on the last 6 bytes, **1.517 nats (2.188 bits), 54.8% right**. Tables of 5 to 8 bytes are within 0.006 nats of it.
* **nanoGPT's small setting, taught here**: nanoGPT's own estimate after 2,000 steps is **1.895 nats**, against the 1.88 that the coordinator recalls from nanoGPT's description: they agree. Its stream loss is 1.906 nats, 43.7% right. Run to 10,000 steps the same network reaches 1.536 nats, 54.2% right (nanoGPT's estimate 1.577).
* **A larger GPT** (nanoGPT's full network, but at a quarter of its batch and for 4,000 steps): LARGE_BRIEF
* **The loop's network taught by back-propagation** (ten bytes one-hot, one layer of logistic cells, softmax): 1.818, 1.731 and 1.695 nats with 256, 1,024 and 4,096 cells (47.2, 49.9 and 51.6% right).
* **A GRU of 256 cells**: 1.574 nats, 54.9% right.
* **A fault of my plan, found and repaired**: I first kept the stream's last 50,000 bytes out of learning for the one-layer network and the GRU. The play that fills most of the last tenth begins inside those bytes, and the figures were 0.14 to 0.25 nats worse for it. The figures above are from the added runs that learn from the whole stream (section 5).

## 2. The table

Stream loss and share named right on the last tenth (111,540 bytes), by `score.py`. Bytes seen: for a table, the stream once; for a network, steps times batch times block, or passes times the bytes learned from. CPU time is process time over all threads, for learning only, on four cores without a GPU; several runs shared the cores, which inflates the figures of the runs marked †.

TABLE_HERE

Notes to the table. (a) For the two small GPTs the stream loss is exact: every byte from the full 64 bytes before it. For the large GPT it is exact on every eighth byte (13,943 bytes); the reading with windows moved on by half a block, on every byte, is beside it. (b) "nanoGPT's estimate" is the mean loss over 200 batches of random blocks of the last tenth, every place of each block counted, the network in evaluation mode; "stream part" is the same on the stream. (c) The one-layer and GRU rows learn from the whole stream and score the last pass; nothing is chosen.

## 3. The tables of the last k bytes (`tables.py`, `res/tables.json`)

Counted on the stream; formulas in `MEASURE.md`. Kneser-Ney's discounts come from the counts of counts of each order; nothing is fitted to the last tenth or to any held-out part.

| k | different runs of k bytes seen | unsmoothed: named right, % | Kneser-Ney: nats | bits | right, % | Witten-Bell: nats | bits | right, % |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 14.90 | 3.3473 | 4.8292 | 14.90 | 3.3473 | 4.8292 | 14.90 |
| 1 | 65 | 26.98 | 2.4836 | 3.5830 | 26.98 | 2.4858 | 3.5862 | 26.98 |
| 2 | 1,380 | 38.09 | 2.0381 | 2.9404 | 38.08 | 2.0492 | 2.9564 | 38.14 |
| 3 | 11,228 | 47.15 | 1.7191 | 2.4801 | 47.22 | 1.7738 | 2.5590 | 47.17 |
| 4 | 48,539 | 52.52 | 1.5612 | 2.2523 | 52.77 | 1.6688 | 2.4076 | 52.70 |
| 5 | 133,293 | 53.65 | 1.5226 | 2.1967 | 54.20 | 1.6981 | 2.4498 | 54.14 |
| 6 | 264,623 | **53.89** | **1.5165** | **2.1879** | **54.80** | 1.7792 | 2.5668 | 54.57 |
| 7 | 413,620 | 53.51 | 1.5192 | 2.1917 | 54.78 | 1.8758 | 2.7062 | 54.41 |
| 8 | 559,503 | 53.07 | 1.5220 | 2.1958 | 54.70 | 1.9701 | 2.8423 | 54.05 |

What I read from it (my readings). The unsmoothed table is best at 6 bytes (53.9%) and loses a little beyond; the Kneser-Ney table names slightly more right than the unsmoothed one from k = 3 on, because where the long context was seen only once or twice the shorter ones are heard as well. Kneser-Ney is flat from 5 to 8 bytes; Witten-Bell is at its best at 4 bytes (1.669) and worsens quickly, so the smoothing matters by 0.15 nats and more at these lengths. The tables counted without the stream's last 50,000 bytes are 0.16 nats worse on the last tenth (k = 6: 1.677 nats, 51.3% right; `res/tables_without_last50000.json`): see section 5.

## 4. The GPTs (`gpt.py`)

Written here in the manner of nanoGPT, without fetching code: pre-norm blocks, GELU, no bias terms, the byte table tied to the answering layer, learned positions; AdamW (0.9, 0.99), weight decay 0.1 on the matrices, gradient clipped at 1, rate 1e-3 with 100 steps of warm-up and a cosine fall to 1e-4 over the run; random blocks of the whole stream; seed 1337.

**The small setting (4 layers, 4 heads, width 128, 64 bytes back, no dropout, batch 12; 804,096 numbers that learn, 795,904 without the positions, which is nanoGPT's way of counting).**

| | 2,000 steps (`small2k`) | 10,000 steps (`small10k`) |
|---|---|---|
| bytes seen | 1,536,000 | 7,680,000 |
| nanoGPT's estimate on the last tenth (on the stream) | 1.895 (1.781) | 1.577 (1.359) |
| stream loss, exact: nats, bits, right % | 1.906, 2.750, 43.7 | 1.536, 2.217, 54.2 |
| stream loss, half-window reading | 1.869, 44.4% | 1.531, 54.4% |
| CPU time, learning (two threads) | 170 s | 829 s † |

* **Against nanoGPT's published figure.** The coordinator recalls about 1.88 nats for this setting at 2,000 steps. I measure 1.895 by nanoGPT's own kind of estimate. They agree to 0.015 nats, which is within what a different seed and a different sample of 200 batches would give; I have not measured that spread. `small10k` is a run of its own whose rate falls over 10,000 steps, not `small2k` continued; its curve is in `res/gpt_small10k.json` (at step 2,000 it stands at 1.897).
* **The stream loss against nanoGPT's estimate.** At 10,000 steps the stream loss is 0.040 below the estimate, as expected: the estimate counts the first places of each block, where the network has one, two, three bytes to go on (loss 2.51, 2.11, 1.84 at the first three places; flat near 1.53 from about the sixteenth place on). At 2,000 steps the stream loss from a full window is 0.011 *above* the estimate and 0.038 above the half-window reading. The look at each place of the block (`loss_by_position` in `res/gpt_small2k_again.json`, a repeat run that reproduced `small2k` to the last digit) shows why: the loss is flat near 1.87 from the eighth place on, so this network gains nothing from bytes further back than about eight, and at the last place of the block it is 1.898. My reading, not tested: the last place is the one no later place attends to, so its position is the least taught after so few steps. At 10,000 steps the last place is no worse than its neighbours (1.520). The exact figure is the measure; the half-window reading is within 0.006 of it at 10,000 steps.
* **Against the tables.** At 2,000 steps the small setting (1.906) is between the tables of 2 and 3 bytes (2.038, 1.719) and 0.39 nats above the best table. At 10,000 steps (1.536) it is 0.020 nats above the best table (1.517) and names 54.2% right against 54.8%.

**The large GPT (`large`).** LARGE_SECTION

## 5. The loop's network by back-propagation, and a GRU (`onelayer.py`)

**The network.** The last D bytes one-hot (65 D lines; 650 for D = 10), one layer of H logistic cells, a softmax answering layer. Batches of 256 drawn without replacement, Adam, rate 1e-3 falling by a cosine to 1e-4, 12 passes, seed 1337. This is a ceiling for the shape under back-propagation with minibatches and Adam at these settings, with no search over them; it is not a twin of the loop's rule, which reads one byte at a time and changes strengths locally.

| | numbers that learn | whole stream, last pass: nats, bits, right % | last 50,000 bytes kept out (first runs): last tenth; development part |
|---|---|---|---|
| D = 10, H = 256 | 183,361 | 1.818, 2.622, 47.2 | 1.959, 44.2%; 1.836 |
| D = 10, H = 1,024 | 733,249 | 1.731, 2.497, 49.9 | 1.926, 46.4%; 1.761 |
| D = 10, H = 4,096 | 2,932,801 | 1.695, 2.446, 51.6 | 1.942, 47.5%; 1.737 |
| D = 32, H = 1,024 | 2,197,569 | 1.879, 2.710, 47.1 | not run |
| GRU of 256 (20 passes) | 264,769 | 1.574, 2.270, 54.9 | 1.763, 51.2%; 1.607 |

**The fault and its repair.** The plan kept the stream's last 50,000 bytes out of learning as a development part, to choose the pass to score. In all three one-layer runs the development loss was still falling at the twelfth pass, so nothing was in fact chosen; for the GRU the twelfth of twenty passes was chosen. On the last tenth these runs were far worse than on the development part (third column), while the tables are not (Kneser-Ney, k = 6: 1.548 on the development part, 1.517 on the last tenth). The reason is in the text: the play that fills about two thirds of the last tenth begins some 25,000 bytes before the cut (its speakers' names first appear between bytes 978,546 and 989,789 of 1,003,854). A learner that skips the end of the stream has never met those names. Three checks agree: the tables counted without those 50,000 bytes lose 0.16 nats on the last tenth (1.677 at k = 6); the one-layer networks gain 0.14 to 0.25 nats when they learn from the whole stream; the GRU gains 0.19. **For the other lines: on this text, a learner must read the stream to its end before it is scored, and no part of the end of the stream should be held out; a figure from a learner that stopped earlier is not comparable with the references here.** The whole-stream runs score their last pass, so they have no protection against reading too long; with 4,096 cells the first runs' development loss was still falling at the twelfth pass, and for the GRU it rose by 0.007 from the twelfth pass to the twentieth, so the GRU's 1.574 may be slightly above what an earlier stop would give.

**What I read from it (my readings).** One layer on ten bytes, taught by back-propagation, ends between the tables of 3 and 4 bytes with 4,096 cells (1.695 against 1.719 and 1.561) and at the table of 3 bytes with 1,024 cells (1.731); with 256 cells it is between the tables of 2 and 3 (1.818). Sixteen times the cells buy 0.12 nats. With 32 bytes on lines and 1,024 cells the same recipe is worse than with ten (1.879 against 1.731); I did not find out whether that is reading too long with three times the strengths or something else, and no development run was made for it. The GRU of 256 cells, with a third of the numbers of the one-layer network of 1,024, is 0.16 nats below it and 0.06 above the best table.

## 6. Where the references stand, in one line each

In order of stream loss (nats; right %): LARGE_ORDER best table, Kneser-Ney on 6 bytes 1.517 (54.8); small GPT at 10,000 steps 1.536 (54.2); table of 4 bytes 1.561 (52.8); GRU of 256 1.574 (54.9); one layer on ten bytes with 4,096 cells 1.695 (51.6); table of 3 bytes 1.719 (47.2); one layer with 1,024 cells 1.731 (49.9); one layer with 256 cells 1.818 (47.2); small GPT at 2,000 steps 1.906 (43.7); table of 2 bytes 2.038 (38.1); byte frequencies 3.347 (14.9); equal probabilities 4.174.

The roadmap's sentence "A well-made table of five to seven bytes is, on a text of this size, already near the small setting's level" is, on these figures, an understatement at 2,000 steps (the table is 0.39 nats better) and about right at 10,000 steps (0.02 better). LARGE_READING

## 7. Predictions

Scored in `PLAN_1.md`, "Outcome". In short: LARGE_PRED

## 8. Commands, seeds, cost, and what was not run

```
python3 score.py                                   # self-check of the measure
python3 tables.py; python3 tables.py without       # res/tables.json; res/tables_without_last50000.json
python3 gpt.py small2k threads=2                   # res/gpt_small2k.json
python3 gpt.py small10k steps=10000 every=500 threads=2 bypos=7
python3 gpt.py small2k_again threads=2 bypos=7     # the repeat with the look at each place
python3 gpt.py large layers=6 heads=6 width=384 block=256 dropout=0.2 batch=16 steps=4000 every=500 evalbatches=50 finalbatches=200 exact=8 bypos=61 threads=4
python3 onelayer.py d10_h256 H=256 threads=1       # and H=1024, H=4096: the first runs, with a development part
python3 onelayer.py d10_h256_whole H=256 threads=1 whole=1          # and H=1024, H=4096; d32_h1024_whole D=32 H=1024
python3 onelayer.py gru256 gru=1 H=256 threads=1;  python3 onelayer.py gru256_whole gru=1 H=256 threads=1 whole=1
python3 make_table.py                              # the table of section 2 from the result files
```

Seeds: 1337 for every network (the start of the strengths and the order of batches); 4242 and 4243 for the blocks of nanoGPT's estimate on the stream and on the last tenth. One seed each; the roadmap's five reported seeds were not run, so nothing here is more than development level. PyTorch 2.15.0 (CPU wheel), numpy 2.5.3, Python 3.13, four cores, no GPU. `small2k` was run twice and gave the same figures to the last digit.

Cost: tables 20 s; the small GPTs 170 s and 829 s of CPU for learning, and 261 s and 328 s for the exact stream loss; the one-layer networks 115 s to 1,350 s; the GRU 628 s; LARGE_COST

**Not run.** nanoGPT's full setting as published (batch 64, 5,000 steps: about five and a half hours here at 4.0 s a step). A second seed for anything. Any search over rates, sizes, schedules or numbers of passes. One layer on 32 bytes with 4,096 cells; a development run for one layer on 32 bytes; a GRU of 512 cells. Tables longer than 8 bytes or with fitted discounts. The exact stream loss of the large GPT on every byte (taken on every eighth). The loop itself or any local rule: that is line L1. No checkpoint and no wheel is committed.
