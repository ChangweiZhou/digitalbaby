# Draft for hostile audit: E29, slow units and transmission delays in the GRU (13 November 2026, 14:51 EST by the clock)

> **Status.** Draft. **No worker starts from this file** until a hostile audit has read it and the coordinator has answered. Implementation and runs are an Opus 5.5 worker's. The auditors are fresh sessions of the same model family: second readers, not independent ones. Parts A and B are measurements and a fixture under backprop. They make no claim of a learning-core improvement and no claim about any learning rule; no E-level is claimed for them.

## Where this comes from

Mr. Xu, 13 November: a neuron's decision at one moment is shaped by several earlier moments, and not by a fixed number of them. Two things do this in a nervous system, and both may matter:

- **Integration.** A slow neuron's state is already a blend of what it received over recent steps. Nothing is stored outside the neuron.
- **Delay.** Transmission takes time, so a neuron hears others as they were some steps ago, and different neurons hear different moments.

What is on file:

- The two-bank network had units at five fixed retention rates, 0.5 to 0.999 (`grep -n "RATES" MOD97_LOCAL_RULE_20261106/arch17.py`). Its units did not form a circuit: the first bank reads the current byte, the second reads the first. It did not compose under backprop (review 9.101 A). It is not a fair test of integration in a recurrent circuit.
- The GRU has a learned gate per unit, `h = (1 - z) * n + z * h` (`sed -n '62,66p' MOD97_LOCAL_RULE_20261106/arch26.py`). A unit whose z stays near 1 is a slow integrator. It is built by `nn.GRU` with torch's default initial values (`grep -n "nn.GRU" arch20.py`; no other initialisation in `arch20.py`, `arch26.py` or `sz_learner.py`), so at birth every gate sits near one half and every unit is fast.
- No experiment here has given a recurrent network slow units at birth, or transmission delays of any kind (`git grep -il "conduction delay\|transmission delay\|delay line"` finds nothing).

## The questions

- **A.** Do the trained GRUs on file have slow units, and does the sweep make fewer than backprop?
- **B.** Does a GRU born with a spread of unit speeds, with a spread of transmission delays, or with both, learn the copy fixture at distances the plain GRU does not, or sooner?

Part C, the same variants in E26's births and lives under the sweep and under backprop, is **not in this brief**. It is written after A and B are read, and needs its own statement of how a sign backward path crosses a delay.

## Part A. Read-only measurement on saved weights

Files: the untrained G1 for each seed; E26's six births `exp45/e26/e26_dev_{S,B}_<seed>.pt`; E26's six lives `e26_life_{S,B}_<seed>.pt` (branch `claude/adoring-cori-b07aog`, commit ce409d64; each checked against `weights_sha256` in its receipt). Seeds 73129, 93683, 116089.

For each file, run `arch26.forward` with exact arithmetic over the 300 items of the sealed development row add89 at k = 12 and record z for every unit at every real position. Per unit:

- **keep** = mean z over positions, and **timescale** = 1 / (1 - keep);
- **hold share** = the share of positions with z above 0.9;
- the same two over distractor-letter positions only.

Report per file: the quartiles and the 90th and 99th percentiles of timescale, and the share of units with timescale at least 4, 8 and 16.

*Check before reading:* on the untrained file with the update-gate biases set by hand so that z would be 0.9 with zero weights, the measured keep is within 0.05 of 0.9 for at least 90% of units. Otherwise the measurement does not see what it is meant to.

**Predictions for A** (copy unchanged into `E29_PREDICTIONS.md` before any file is read):

- **P29.1** Untrained: no unit has timescale at least 4, all three seeds. About 90%.
- **P29.2** Backprop-born: at least 10% of units have timescale at least 8, in at least two of three seeds. About 55%.
- **P29.3** Sweep-born files have a smaller share of units with timescale at least 8 than backprop-born files, in at least two of three seeds. About 50%.

**Readings.** P29.2 holding says training makes slow units from fast ones at this task. P29.3 holding is a difference between the two births' networks and no more; it does not show that slowness is why the sweep composes less. A mean of z is a crude summary of a gate that depends on its input; the hold share is reported beside it for that reason.

## Part B. The copy fixture with slow units and delays at birth

**Built on the span job.** Fixture, dose and classes are those of `OPUS_TASK_SIZING_SPAN_20261113.md` revision 2 and its `sz_span.py` (branch `claude/target-test-sizing-stage-1-drmnlq`). Start from that branch's tip **after its roster is complete**. Backprop, every byte a target, credit cut every 256 targets, state carried, AdamW at 0.001 with decay 0.3.

**The learner.** A new class in a new file `sz_variants.py` with the GRU step written out as in `arch26.forward`, same thirteen tensors, same initial values per seed. Two additions, both fixed at birth, neither learned, neither in the optimizer:

- **Speed.** A buffer `speed` of one number per unit, added to the update gate before the sigmoid: `z = sigmoid(i_z + h_z + speed)`. A unit with speed s and nothing else has keep sigmoid(s). It is a buffer and not an initial bias because every bias is under decay 0.3 (`sed -n '57,60p' sz_learner.py`: only `book` is exempt), which would shrink an initial bias by half within about 2,300 updates.
  - *spread:* timescales drawn log-uniformly from 2 to 256 (256 is the credit length), `speed = log(timescale - 1)`, from a generator seeded `e29-speed|<seed>`.
  - *uniform:* every unit at the geometric mean of that range, timescale 22.6.
- **Delay.** A buffer `delay` of one whole number per unit. The recurrent map reads unit j as it was `delay[j]` steps ago: `gh = W_hh · [h_{t - delay[j]}[j]]_j + b_hh`. A unit's own keep term `z * h` uses its own last state, since integration is local. Delay 1 for every unit is the plain GRU.
  - *spread:* half the units at delay 1; the rest log-uniform on 2 to 64, rounded, from a generator seeded `e29-delay|<seed>`.
  - The states in transit are held in a ring of the last 64 states, carried across chunks and detached at each cut as the state is. This is 64 × 256 numbers, fixed, with no search and no copy of the weights. In a nervous system the signal is in the axon; here it has to be held somewhere while it travels.

**Arms. Seeds 1104, 1105, 1106. Fixed distances 12, 24, 60; 235 × (D + 1) chunks each.**

| Arm | Speed | Delay | Runs |
|---|---|---|---|
| plain | none | 1 | the span job's block A records, **if** admission item 2 is bit-equal; otherwise rerun here (9) |
| slow | spread | 1 | 9 |
| slow-uniform | uniform | 1 | 3, at D = 24 only |
| delay | none | spread | 9 |
| both | spread | spread | 9 |

30 new runs, about 227,000 chunks. Forecast the CPU from admission item 5 and stop if it passes fourteen CPU-hours.

**Admission.**
1. Print Python, torch, architecture.
2. With speed zero and every delay 1, the new learner equals `sz_learner.StreamLearner` on seed 1104, D = 12: the same last-quarter counts as the span job's own record on this host after 3,055 chunks, and bit-equal weights after the first 10 chunks.
3. Speed: on an untrained net with the recurrent and input update-gate weights zeroed, each unit's z equals sigmoid(speed) to 1e-6. The spread's quartiles are printed.
4. Delay: a hand-built net in which unit 0 copies its input and unit 1 reads only unit 0 through a delay of 5 shows unit 1's input equal to unit 0's state five steps earlier, across a chunk boundary as well as inside one. The ring holds 64 states and no more.
5. A 300-chunk dry run of each new arm at D = 12 completes; CPU per chunk reported.

**Records:** as the span job's (windows of 250 chunks, the first chunk before any update, repeats to 50%, class on the last tenth), plus, at the end of each run, part A's timescale summary of that run's network on its own stream.

**Forecasts for B** (copy unchanged before the first roster run). "Passes" is the span brief's class on the last tenth, in at least two of three seeds.

- **F29.1** The slow arm passes at 24. About 60%.
- **F29.2** The slow arm passes at 60. About 35%.
- **F29.3** The delay arm passes at 24. About 45%.
- **F29.4** At D = 24 the slow arm needs fewer repeats to 50% than slow-uniform, or passes where it does not, in at least two of three seeds. About 50%.
- **F29.5** At D = 12, where plain passes, neither the slow nor the delay arm needs more than twice plain's repeats to 50%, in at least two of three seeds. About 60%.

**Readings, fixed now.**

- *A variant helps at a distance* only if it passes there where plain does not, in at least two of three seeds, at the same dose. Where both pass, repeats to 50% are compared and read only if they differ by a factor of two or more in all three seeds.
- *Spread against uniform (F29.4)* is what speaks to "not a fixed number": if uniform does as well as the spread at 24, the gain is from slowness and not from a range of speeds, at this one distance. One distance cannot show the range matters; a uniform timescale of 22.6 is close to 24 by construction, and that is said.
- *Both against each alone:* reported; an interaction is claimed only if "both" passes at 60 and neither alone does, in at least two of three seeds.
- *Cost (F29.5):* a variant that slows the easy case by more than a factor of two is reported as a cost beside any gain.
- *If plain already passes at 60 in the span job,* the fixture cannot show a gain in class, and only repeats to 50% are read.
- *Nothing here is about composition, about lives, or about the sweep.* A gain on a copy over random filler says the variant carries a byte further under backprop, and that is all.

## Source check, read-only, alongside

Open and cite, marking each claim confirmed, corrected or not found: work on initialising recurrent gates to a range of timescales (the coordinator recalls Tallec and Ollivier, 2018, from memory and has not opened it); work on heterogeneous time constants in recurrent or spiking networks; work on fixed or learned transmission delays in recurrent networks. For each: what was changed, what it was tested on and against, seeds, and whether credit was exact. One page. It decides nothing.

## What this cannot say

One fixture, one learner, backprop. The ranges 2 to 256 and 1 to 64 are choices tied to the credit length and to the fixture's largest distance; no other range is run. Per-unit delays are one of several ways a delay can enter (per connection is another). Three seeds.

## Running it

One host, at most three processes on four cores, one thread each. New files only: `MOD97_LOCAL_RULE_20261106/e29_timescales.py` for part A, `TARGET_TEST_STEP0/sizing/sz_variants.py` for part B; records in `exp45/e29/` and `runs/e29/`. Part A may run while the span roster finishes; part B waits for it. Own branch; nothing to the default branch.

## Questions for the audit

1. Is a fixed buffer the right form for speed, or does it make the comparison unfair to plain (plain's units can learn to be slow only through decayed weights)?
2. Does a per-unit delay with a local keep term do what "neurons hear different moments" means, or does it only add skip connections in time?
3. Is the copy fixture, with its exact distance, a fair place for integration, which blurs time? Would slow units be expected to help a copy at all?
