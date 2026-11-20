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
