# nano/ref2, plan 1: the reference network again, with seeds and with its answers kept byte by byte (20 November 2026)

By the agent session that owns `nano/ref2` (branch `claude/nano-ref2-8mmhfm`). Development level. No gate, seal or review is claimed. Nothing outside `nano/ref2/` is changed. This line does not touch the system of cells; it makes the reference it is compared with trustworthy.

**Source commit:** `0b1c9d9ac85de8d28414c7da7a4d78e9c4cf28f3` (branch `claude/friendly-darwin-8mmhfm` as checked out here). **The word that set the work** (Mr. Xu, in Chinese, exactly): "继续大胆尝试，小心求证。" The coordinating session's translation: "Go on trying boldly, and verify with care." The task itself was given to this line by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc).

**Why.** `ROADMAP_SECOND_REVISION_20261120.md` section 14 and `nano/table/PLAN_16.md` section 7: the system of cells' 1.4421 nats a byte on the last tenth was set against "1.454" of the reference network; 1.454 is that network's figure on every eighth byte only (13,943 bytes, each with 256 bytes before it); its figure on all 111,540 bytes (1.4562) is from windows of 129 to 256 bytes; one seed; its answers byte by byte were not kept, so no difference with an error could be given.

## 1. The network and the schedule (unchanged from `nano/ref`'s run `large`)

`gpt.py` here is a copy of `nano/ref/gpt.py`. 6 layers, 6 heads, width 384, 256 bytes back, dropout 0.2, no bias terms, tied byte table, learned positions: 10,745,088 numbers that learn. AdamW (0.9, 0.99), weight decay 0.1 on the matrices, gradient clipped at 1, rate 1e-3 after 100 steps of warm-up, cosine fall to 1e-4 at step 4,000; batch 16 random blocks of the first nine tenths; 4,000 steps (16,384,000 bytes seen). Text and split: `nano/data`, `score.py` (a copy of `nano/ref/score.py`, identical to `nano/table/score.py`). Not nanoGPT's published setting (batch 64, 5,000 steps).

**What the copy adds, and nothing else:** (a) `ckpt=`, `ckevery=`, `stop=`: the whole state (strengths, optimiser, both random generators, curve, CPU time) saved every 250 steps; a run started again goes on from it. Tested on a small setting before this plan: a run broken twice gives the same curve and the same per-byte file as an unbroken one, bit for bit. (b) `perbyte=8`: `stream_stride()`, see section 3.

## 2. What was measured before this plan (timing only; no loss of the large network was looked at)

Machine: 4 cores (Xeon 2.1 GHz), 15 GB, no GPU; PyTorch 2.15.0 CPU wheel, numpy 2.5.3, Python 3.13 (the versions `nano/ref` names). Seconds a step of the large network, 10 to 12 steps each:

| runs side by side | s a step, each |
|---|---|
| one run, 4 threads / 2 threads / 1 thread | 1.64 / 2.14 / 3.86 |
| two runs, 2 threads each | 2.30 |
| three runs, 1 thread each | 3.4 |
| four runs, 1 thread each | 3.3 |
| three runs, 2 threads each | 3.3 |
| one run of 4 threads beside one, two, three runs of 1 thread | 3.5, 4.4, 5.4 (the 1-thread runs: 3.4, 3.9, 4.9) |

So four runs on one thread each carry four seeds in the wall time of one (about 3.7 h for 4,000 steps); a run of four threads is twice as costly a step in CPU.

**The number of threads changes the arithmetic.** Six steps of the large network from seed 1337 give the same first loss with 1, 2 and 4 threads (4.226068019866943) and differ from the second or third step on in the seventh digit. `nano/ref`'s run had 4 threads. So only a run of 4 threads can repeat it to the digit; a run of 1 thread from the same seed starts from the same strengths and sees the same batches, but is, after rounding has grown, another draw.

## 3. Step 1: the runs

**0. The exact check (first, alone on the machine).** `python3 gpt.py check1337 <the large setting> every=500 evalbatches=50 threads=4 seed=1337 ckpt=... ` to step 500 only (about 14 minutes). Compared with `nano/ref/res/gpt_large.json`: at step 0 the estimates 4.223835563659668 (stream) and 4.229513664245605 (last tenth); at step 500, 1.973288643360138 and 2.0743672847747803. If they agree to the digits, the copy teaches as the original did, and what could differ later is only what the thread count does. If they do not agree, I find out why before the seeds. A whole run of 4 threads alone would take 1.8 h and the three or four seeds another 3.7 h: more than the budget, so the exact check stops at step 500 (its checkpoint is kept, in case the budget allows more of it at the end).

**1. Four seeds side by side, one thread each:** 1337 (the same start and batches as `nano/ref`; one thread), 1338, 1339, 1340. The large setting; `every=1000 evalbatches=50 finalbatches=200 exact=64 perbyte=8 ckevery=250`. The estimates on the way (which do not touch the teaching: their blocks come from their own generators and dropout is off) are taken every 1,000 steps and not every 500, to save about ten minutes.

**The scores, for each seed** (on the strengths at step 4,000; nothing is chosen on the last tenth):
* `stream_stride(8)`: windows of 256 bytes moved on by 8; the outputs at the last 8 places are used. **Every one of the 111,540 bytes is scored with 249 to 256 bytes before it**, and the bytes 0, 8, 16, ... of the tenth (13,943 bytes: the set of `nano/ref`'s 1.454) have exactly 256, by the same forward pass on the same window as `nano/ref`'s `stream_exact(every=8)`; the last three bytes get windows of their own. The first bytes of the tenth take their context from the end of the teaching text. **This is not the full 256 bytes for every byte**: that costs eight times as much (about 2.4 CPU-hours a seed by `nano/ref`'s figures). `nano/ref`'s look at the places of a block found no gain beyond 128 bytes back (1.457, 1.457, 1.456 for places 65-128, 129-192, 193-256), so I expect the difference to be under 0.001 nats; it is not measured unless budget remains (one seed).
* The per-byte minus-log-probabilities kept as `res/nll_<label>.f32` (float32, 111,540 numbers, 446 kB).
* `stream_half` as in `nano/ref` (the 1.4562 reading), nanoGPT's estimate on 200 batches (the 1.471 reading), and `nano/ref`'s own `stream_exact` on every 64th byte (1,743 bytes), as a second path that must agree with the same bytes of the kept file.
* Reported: loss on all bytes, loss on the every-eighth set, share named right, CPU time; mean and spread over seeds.

**Cost.** Check: about 0.3 CPU-hours. Seeds: 4 x 4,000 x 3.3 s = about 14.7 CPU-hours (3.7 h of wall time), estimates and scoring about 0.5 CPU-hours a seed. Checkpoints 130 MB a run, deleted when scored; no strengths are committed.

## 4. The questions and my predictions (chances are mine)

| | Prediction | Chance |
|---|---|---|
| R1 | the exact check agrees with `nano/ref`'s curve at step 500 in all digits of both estimates | 0.6 |
| R1b | if not in all digits, it agrees within 0.01 nats | 0.95 |
| R2 | seed 1337 on one thread: every-eighth loss within 0.006 of 1.4541 | 0.7 |
| R3 | all four seeds: loss on all bytes between 1.440 and 1.470 | 0.8 |
| R4 | largest minus smallest of the four seeds on all bytes: at most 0.010 | 0.6 |
| R5 | for each seed the loss on all bytes is below its loss on the every-eighth set (the cells have 1.4421 against 1.4463 there: that eighth is the harder) | 0.65 |
| R6 | the mean over seeds on all bytes is above 1.4421 (the cells' figure as reported; not rerun by me yet) | 0.7 |
| R7 | `stream_exact(every=64)` agrees with the same bytes of the kept file to 1e-5 nats in the mean | 0.95 |

Falsifier of the working hypothesis "the earlier 1.454 is a fair single draw of this setting": 1.4541 lying outside the range of the four seeds' every-eighth figures by more than their range.

## 5. Not run in this step

The full 256 bytes before every byte (see above). A whole run of 4 threads. More than four seeds. Any change of the setting; any choice on the last tenth. Step 2 (the comparison byte by byte with the systems of cells) and step 3 (a longer schedule) have plans of their own, committed before their runs.

## 6. Outcome

(appended below as the runs end)

**Entry 1 (21:54 UTC, 20 November; a first writing of this entry gave the hour wrongly as 22:53): the exact check.** `check1337` (seed 1337, 4 threads, the large setting, alone on the machine) to step 500, broken once at step 250 and gone on from its checkpoint: 1,920 s of CPU, 682 s of wall time (1.36 s a step). Its estimates (50 batches, `res/check1337_step500.json`):

| step | this copy, stream / last tenth | `nano/ref/res/gpt_large.json` |
|---|---|---|
| 0 | 4.223835563659668 / 4.229513664245605 | 4.223835563659668 / 4.229513664245605 |
| 500 | 1.973288643360138 / 2.0743672847747803 | 1.973288643360138 / 2.0743672847747803 |

Equal in every digit. **R1 held.** So on this machine the copy with its checkpoints teaches exactly as `nano/ref`'s program did through 500 steps, and a run of 4 threads continued to step 4,000 would be expected to repeat 1.4541 to the digit (about 80 more minutes of the whole machine; not run now, the checkpoint is kept). A slip of mine in the command: the run left before the estimate at step 500, so that estimate was taken afterwards from the checkpoint by the program's own `estimate()` with the same blocks.

The four seeds were started at 21:55 UTC (`jobs1.sh`).

**Entry 2 (23:48 UTC).** The four runs left at step 2,000 after 105 minutes, as written, and were started again from their checkpoints at 23:47 UTC. At step 1,000 nanoGPT's estimate on the last tenth (50 batches) was 1.787, 1.801, 1.778 and 1.772 for seeds 1337 to 1340 (`nano/ref`'s run: 1.793). The pace was 3.1 CPU-s a step; the cells' runs of plans 2 and 3 took a core for about 18 minutes in all during this stretch.
