# Line L0: the references of the test at the level of nanoGPT (20 November 2026)

By the agent session that owns `nano/ref` (roadmap, second revision, section 6). Development level; one seed for each network; repeated by no one; no checker gate, seal or review is claimed. Nothing outside `nano/ref/` was changed. None of these programs is the loop or the adopted core: they are tables and comparators taught by back-propagation. The plan, `PLAN_1.md`, was committed before the runs (ad361c5); an addition to it was committed before the runs it adds (9378f61); its outcome section scores every prediction.

## 1. In brief

* **The measure** is defined once in `MEASURE.md` and computed by `score.py` (numpy): on the 111,540 bytes of the last tenth, each predicted from the bytes before it, the stream loss in nats and bits a byte and the share of bytes named right. The first bytes of the last tenth take the end of the stream as their context; every byte is scored.
* **The best table**: interpolated Kneser-Ney on the last 6 bytes, **1.517 nats (2.188 bits), 54.8% right**. Tables of 5 to 8 bytes are within 0.006 nats of it.
* **nanoGPT's small setting, taught here**: nanoGPT's own estimate after 2,000 steps is **1.895 nats**, against the 1.88 that the coordinator recalls from nanoGPT's description: they agree. Its stream loss is 1.906 nats, 43.7% right. Run to 10,000 steps the same network reaches 1.536 nats, 54.2% right (nanoGPT's estimate 1.577).
* **A larger GPT** (nanoGPT's full network, but at a quarter of its batch and for 4,000 steps): **still learning when this report was written** (step 1,500 of 4,000: nanoGPT's estimate, on 50 batches, 1.659 on the last tenth and 1.466 on the stream; 2.074 at step 500 and 1.793 at step 1,000). Its row is added when it ends; see section 4.
* **The loop's network taught by back-propagation** (ten bytes one-hot, one layer of logistic cells, softmax): 1.818, 1.731 and 1.695 nats with 256, 1,024 and 4,096 cells (47.2, 49.9 and 51.6% right).
* **A GRU of 256 cells**: 1.574 nats, 54.9% right.
* **A fault of my plan, found and repaired**: I first kept the stream's last 50,000 bytes out of learning for the one-layer network and the GRU. The play that fills most of the last tenth begins inside those bytes, and the figures were 0.14 to 0.25 nats worse for it. The figures above are from the added runs that learn from the whole stream (section 5).

## 2. The table

Stream loss and share named right on the last tenth (111,540 bytes), by `score.py`. Bytes seen: for a table, the stream once; for a network, steps times batch times block, or passes times the bytes learned from. CPU time is process time over all threads, for learning only, on four cores without a GPU; several runs shared the cores, which inflates the figures of the runs marked †.

| Model | Numbers that learn | Bytes seen | Stream loss, nats | bits | Named right, % | CPU time (learning) | Beside it |
|---|---|---|---|---|---|---|---|
| equal probabilities | 0 | 0 | 4.174 | 6.022 | 1.5 | 0 |  |
| table of the last 0 bytes, Kneser-Ney | 65 counts | 1,003,854 | 3.347 | 4.829 | 14.9 | 20 s for all nine | unsmoothed: 14.9% right |
| table of the last 1 bytes, Kneser-Ney | 1,380 counts | 1,003,854 | 2.484 | 3.583 | 27.0 |  | unsmoothed: 27.0% right |
| table of the last 2 bytes, Kneser-Ney | 11,228 counts | 1,003,854 | 2.038 | 2.940 | 38.1 |  | unsmoothed: 38.1% right |
| table of the last 3 bytes, Kneser-Ney | 48,539 counts | 1,003,854 | 1.719 | 2.480 | 47.2 |  | unsmoothed: 47.1% right |
| table of the last 4 bytes, Kneser-Ney | 133,293 counts | 1,003,854 | 1.561 | 2.252 | 52.8 |  | unsmoothed: 52.5% right |
| table of the last 5 bytes, Kneser-Ney | 264,623 counts | 1,003,854 | 1.523 | 2.197 | 54.2 |  | unsmoothed: 53.6% right |
| table of the last 6 bytes, Kneser-Ney | 413,620 counts | 1,003,854 | 1.517 | 2.188 | 54.8 |  | unsmoothed: 53.9% right |
| table of the last 7 bytes, Kneser-Ney | 559,503 counts | 1,003,854 | 1.519 | 2.192 | 54.8 |  | unsmoothed: 53.5% right |
| table of the last 8 bytes, Kneser-Ney | 684,640 counts | 1,003,854 | 1.522 | 2.196 | 54.7 |  | unsmoothed: 53.1% right |
| GPT 4 layers, 4 heads, 128 wide, 64 back, 2,000 steps | 804,096 | 1,536,000 | 1.906 | 2.750 | 43.7 | 170 s | nanoGPT's estimate 1.895 (stream part 1.781); half-window reading 1.869, 44.4% |
| the same, 10,000 steps | 804,096 | 7,680,000 | 1.536 | 2.217 | 54.2 | 829 s † | nanoGPT's estimate 1.577 (stream part 1.359); half-window reading 1.531, 54.4% |
| one layer, 10 bytes, 256 cells | 183,361 | 12,045,312 | 1.818 | 2.622 | 47.2 | 115 s | with the last 50,000 bytes of the stream kept out: 1.959, 44.2% (development part 1.836) |
| one layer, 10 bytes, 1,024 cells | 733,249 | 12,045,312 | 1.731 | 2.497 | 49.9 | 281 s | with the last 50,000 bytes of the stream kept out: 1.926, 46.4% (development part 1.761) |
| one layer, 10 bytes, 4,096 cells | 2,932,801 | 12,045,312 | 1.695 | 2.446 | 51.6 | 1,350 s † | with the last 50,000 bytes of the stream kept out: 1.942, 47.5% (development part 1.737) |
| one layer, 32 bytes, 1,024 cells | 2,197,569 | 12,045,312 | 1.879 | 2.710 | 47.1 | 738 s † |  |
| GRU of 256 cells | 264,769 | 20,070,400 | 1.574 | 2.270 | 54.9 | 628 s † | with the last 50,000 bytes kept out: 1.763, 51.2% (development part 1.607, pass 12) |
| GPT 6 layers, 6 heads, 384 wide, 256 back, batch 16, 4,000 steps | 10,745,088 | (16,384,000 when done) | pending | | | | still learning; see section 4 |

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

**The large GPT (`large`).** nanoGPT's full network (6 layers, 6 heads, width 384, 256 bytes back, dropout 0.2; 10,745,088 numbers that learn), at batch 16 in place of 64 and for 4,000 steps in place of 5,000: 16.4 million bytes seen against nanoGPT's 82 million. It is not nanoGPT's full setting and must not be called so. **The machine is slow for it.** My timing test (without dropout, cores free) gave 0.90 s a step; the run itself takes about 1.4 s a step with the four cores to itself and 4 to 5 s a step whenever another run shares them, which my own one-layer runs and then the life of section 6 did. It was started at 14:45 UTC and stood at step 1,500 at 16:14.

| step | nanoGPT's estimate on the last tenth (50 batches of 16 blocks of 256) | on the stream |
|---|---|---|
| 500 | 2.074 | 1.973 |
| 1,000 | 1.793 | 1.621 |
| 1,500 | 1.659 | 1.466 |

The stream loss, the final estimate on 200 batches and the loss at each place of the block are entered here when the run ends.

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

## 6. The small GPT through a life of two texts (`life_gpt.py`, `res/life_gpt.json`; plan addition 2)

Asked for by the coordinating session, to stand beside `nano/table/PLAN_5.md`. A is the stream of the test; B is the first nine tenths of Bryant's stories, Burgess's Buster Bear and Carroll's Alice (424,157 bytes; the three held-out tenths are 47,131 bytes together), prepared by a copy of `nano/table/life.py`'s `load_all()`. The small GPT is taught on A exactly as `small10k` (10,000 steps), then on B for 4,225 steps of the same size (12 blocks of 64 bytes: 3.24 million bytes seen, the same 7.65 readings as of A), on random blocks of B's stream only: no byte of A is shown again; the optimiser's state is carried over. Two arms for the rate on B: the recipe again (warm-up to 1e-3, cosine fall to 1e-4), and a constant 1e-4 (where A ended). A third network sees B only (the recipe, 4,225 steps). Scored with nothing changing, each byte from the 64 bytes before it in its own text. One seed (1337): a first figure only.

| Stream loss, nats (right, %) | last tenth of A | B's three held-out tenths together | Bryant | Burgess | Carroll |
|---|---|---|---|---|---|
| after A | 1.537 (54.2) | 2.005 (45.1) | 1.914 | 2.039 | 2.139 |
| after A and B, the recipe's rate again | 2.302 (41.5) | 1.403 (58.6) | 1.498 | 1.236 | 1.339 |
| after A and B, constant 1e-4 | 2.030 (44.2) | 1.414 (57.5) | 1.476 | 1.296 | 1.378 |
| B only | 2.383 (38.4) | 1.584 (52.8) | 1.668 | 1.441 | 1.525 |

* **What it kept of A.** The loss on A's last tenth rises by 0.77 nats with the recipe's rate and by 0.49 at the constant small rate; with the recipe's rate it ends 0.08 below a network that never saw A. Beside it, as the coordinating session reports them from `nano/table` (I have not rerun them): the counting cells with a learned trust rise by 0.009 and the summing form by 0.057.
* **What A gave to B.** Having read A first, the network is 0.18 nats better on B's held-out parts than the network that saw B only (1.403 against 1.584), and at 1.403 it is 0.012 below the 1.4152 reported for the counting cells after A and B.
* **Limits.** One seed. The two kinds of learner do not read alike: the counting cells read each text once, in order; this network reads random blocks, each text about 7.65 times. No rate between the two arms was tried, nor any means of keeping (showing A again, slowing chosen strengths). My reading: on these figures back-propagation at the small setting gains more from the second text and loses most of the first, and the loss is not an effect of the rate being raised again, since it is 0.49 nats at the rate A ended on.

## 7. Where the references stand, in one line each

In order of stream loss (nats; right %): best table, Kneser-Ney on 6 bytes 1.517 (54.8); small GPT at 10,000 steps 1.536 (54.2); table of 4 bytes 1.561 (52.8); GRU of 256 1.574 (54.9); one layer on ten bytes with 4,096 cells 1.695 (51.6); table of 3 bytes 1.719 (47.2); one layer with 1,024 cells 1.731 (49.9); one layer with 256 cells 1.818 (47.2); small GPT at 2,000 steps 1.906 (43.7); table of 2 bytes 2.038 (38.1); byte frequencies 3.347 (14.9); equal probabilities 4.174.

The roadmap's sentence "A well-made table of five to seven bytes is, on a text of this size, already near the small setting's level" is, on these figures, an understatement at 2,000 steps (the table is 0.39 nats better) and about right at 10,000 steps (0.02 better). The large GPT's place in this order is not yet known.

## 8. Predictions

Scored in `PLAN_1.md`, "Outcome". In short: of the 17 predictions of the first plan that can be scored now, 14 held and 3 failed (G2: the small GPT at 10,000 steps did better than I gave it; G3: at 2,000 steps the stream loss is above nanoGPT's estimate, not below; M1: one layer with 256 cells came out 0.003 below my range); G4, G4b and G5 wait for the large GPT. Of the additions' eight, seven held and one failed (L2: at a constant small rate the small GPT still loses 0.49 nats on the first text).

## 9. Commands, seeds, cost, and what was not run

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
python3 life_gpt.py threads=1                      # res/life_gpt.json (section 6)
python3 make_table.py                              # the table of section 2 from the result files
```

Seeds: 1337 for every network (the start of the strengths and the order of batches); 4242 and 4243 for the blocks of nanoGPT's estimate on the stream and on the last tenth. One seed each; the roadmap's five reported seeds were not run, so nothing here is more than development level. PyTorch 2.15.0 (CPU wheel), numpy 2.5.3, Python 3.13, four cores, no GPU. `small2k` was run twice and gave the same figures to the last digit.

Cost: tables 20 s; the small GPTs 170 s and 829 s of CPU for learning, and 261 s and 328 s for the exact stream loss; the one-layer networks 115 s to 1,350 s; the GRU 628 s; the life of two texts 2,432 s; the large GPT 13,405 s of CPU for its first 1,500 steps, much of it lost to sharing the cores.

**Not run.** nanoGPT's full setting as published (batch 64, 5,000 steps: about five and a half hours here at 4.0 s a step). A second seed for anything. Any search over rates, sizes, schedules or numbers of passes. One layer on 32 bytes with 4,096 cells; a development run for one layer on 32 bytes; a GRU of 512 cells. Tables longer than 8 bytes or with fitted discounts. The exact stream loss of the large GPT on every byte (taken on every eighth). A reference that goes on learning through the last tenth, each block named before it is used for a step (asked for by the coordinating session at 15:20 UTC; second in its order of the two requests): not run. The loop itself or any local rule: that is line L1. No checkpoint and no wheel is committed.
