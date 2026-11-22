# Plan: learning on several timescales (22 November 2026)

Written before the runs it governs. Development level; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 22 November 2026, this session: "做一做，试试看。……无论如何我们可以从两个开始做起。" (do it, try; in any case we can
start from two). His question, why two systems and not ten given the brain's many rhythms, is taken up in the design: K stages, each
learning more slowly than the one before and taught by the one before's replay; K = 2 is the start.

**Source.** `NEWNET/` on `claude/magical-albattani-sei232` at the commit of this file (`net8.py`, `t11_gate.py`, `t13_life8.py`).

## Why

Measured in this folder: sparse codes that do not overlap learn fast and keep (NEWNET order cells) but do not generalise; overlapping
codes generalise but are rewritten by a new task (net8: A on letters 1.000 -> 0.000 through text). The two needs conflict in one set of
codes. The principle taken (complementary learning, a cascade of timescales): a fast stage with separated codes learns new things at
once; slower stages with overlapping codes learn slowly, online and from the fast stage's replay, and keep what is common.

## Step 1 (this file's first runs): is net8's output end a fast learner where it should be a slow one?

**Hypothesis H1.** net8's online share on text (0.34) partly follows local repetition. With a slower output end, the share on 2,000
bytes of the same text never read, learning off, rises above 0.23, while the online share falls.

**Refuted if** the held-out share does not rise above 0.25 at any slower rate (0.1, 0.02, 0.005, against 0.5): then "it follows local
repetition" is wrong, and the slow stage needs something else than a slower step.

**Runs.** net8 (N=1024, expansion 8,192, out_var 1000, no tissue learning), 16,000 bytes of tinyshakespeare, seed 1; rates 0.5, 0.1,
0.02, 0.005; the held-out test of `t13_life8.py life=texts` after A; the online last quarter. About 1 minute a run, 4 cores.

## The fixed battery (every later design is scored on all of it; a step counts only if nothing in it falls)

1. Memory across a gap: context stream G = 8, 16, 32 (`t11_gate.py`), at the byte after the gap.
2. Text: online last quarter and held-out with learning off.
3. A -> B -> A with A on letters (`t13_life8.py life=ctx`): A after B.
4. Two texts (`life=texts`): A and B after each phase, with the B-only control.
5. Never-shown combinations: open; the task is to be agreed with the owner.

## Not run in this step

The fast stage, replay between stages, K > 2, yes/no, other seeds.

## Outcome of step 1 (entered 22 November 2026; seed 1; `t13_life8.py life=texts TB=0 TR=0`, `t11_gate.py stream=text`)

| eta_o | online, last quarter | held-out, learning off | online - held-out |
|---|---|---|---|
| 0.5 | 0.339 | 0.230 | 0.109 |
| 0.1 | 0.333 | 0.302 | 0.031 |
| 0.02 | 0.298 | 0.291 | 0.007 |
| 0.005 | 0.239 | 0.260 | -0.021 |

**H1 held**: the held-out share rises above 0.25 (0.302 at 0.1), and the gap between the online and the held-out share shrinks with
the rate (0.11 to 0.007): at 0.5 the output end followed local repetition. My expectation that the online share falls at once was
wrong at 0.1 (0.333 against 0.339): 0.5 was simply too fast; 0.1 is better on both.

## Step 2 (before its runs): do a slower output end and replay keep A without a fast stage?

**Hypothesis H2.** In `life=ctx` (A on letters, then 16,000 bytes of text), a slower output end (0.1, 0.02) with replay from a store of
1,000 episodes at 1/20 keeps A after B at 0.75 or more. **Refuted if** A after B stays below 0.5 in every arm: then slowness and replay
on overlapping codes are not enough, and the fast stage with separated codes is needed first. Arms: eta_o 0.1 and 0.02, rho 0 and
1/20; seed 1.

## Outcome of step 2 (entered 22 November 2026; seed 1; `t13_life8.py life=ctx`)

| eta_o | A after B, no replay | A after B, replay 1/20 | B test after B (no replay / replay) |
|---|---|---|---|
| 0.1 | 0.050 | 0.500 | 0.266 / 0.272 |
| 0.02 | 0.000 | 0.560 | 0.280 / 0.283 |

**H2 refuted** (best 0.56 < 0.75; replay brings A only to chance). Slowness and replay on overlapping codes do not keep A. (The slower
output end raises the text test after B from 0.175 to 0.27-0.28.)

## Step 3 (before its runs): a fast stage with separated codes beside the slow one

**Design.** Slow stage: net8 as in step 1 (expansion 8,192, eta_o 0.1). Fast stage: a minicolumn tissue whose cells connect to the
cells that fired at the byte before (net5's rules in its C form `net7`, `ctx=0 hold=0 n_grow=8`, one region: a cell is predicted when its active strong slots are at least 3/4 of the active
sources, matches at 3/5; a predicted cell fires, else a matching one, else the least used; the firing cell strengthens its active slots
and makes slots to the cells it has none for): codes separate by history and repeat where a sequence repeats. Each output cell has
two compartments, one for each stage; each compartment learns by the delta rule on its own prediction (target - its own output), and
the answer is the largest sum. Replay (when on) teaches the slow stage only. `net9.py`.

**Hypothesis H3.** In `life=ctx`, A after B is 0.9 or more, while text (online and held-out) is within 0.02 of the slow stage alone and
memory at G=16 stays at 1.000. **Refuted if** A after B is below 0.75, or text falls by more than 0.02. Seed 1 first; seeds 2-3 if held.

## Outcome of step 3 (entered 22 November 2026; seed 1; `t13_life8.py net=9`)

| Arm | A after B (`life=ctx`) | text held-out after 16,000 bytes (`life=texts`) |
|---|---|---|
| slow stage only (`fast=0`) | 0.050 | 0.302 (step 1) |
| slow + fast | 0.000 | 0.185 |
| slow + fast, replay 1/20 to the slow stage | 0.000 | |

**H3 refuted**: the fast stage does not keep A, and it costs text 0.12. Not yet diagnosed; my reading, to be checked before anything
else: (1) with `ctx=0` the fast tissue fires its default cells when a prediction is broad, so A's start and first bytes are coded as
text codes the same letters, and the chain that would separate A begins from a code text shares; (2) the fast compartment learns at 0.5
on codes that do repeat in text (the default codes, one per byte value), so it adds a noisy one-byte vote to every answer.

## Diagnosis of step 3 (22 November 2026; `t15_fastcode.py`, seed 1)

1. My first reading was wrong in its first half: after A, the fast stage's codes for A are separated (default cells at A's start
   only; A's gap code, 8 cells, never active at the same letter in 4,000 bytes of text; their output strengths unchanged after B).
2. After B, A no longer reaches those cells: at A's gap the fast compartment votes 0.018 for the right byte and 0.429 for a wrong one,
   and from the first middle byte on, A's fast cells are not those of before B (share the same: 1.00 at the start, 0.00 after).
3. Why (probe at the first middle byte): A's cell there still has all 8 of its active slots strong, so it is predicted; but the region
   signals surprise, because after text the start letter's code predicts many columns (text's successors of that letter), and the
   surprise rule of net5 makes every column fire its default cell. Without the weakening of failed predictions (`f_eta_minus=0`) the
   same happens: it is not that weakening.
4. `f_nodef=1` (surprise never forces the default cell; added to net7 C and checked identical where off): text with the fast stage
   recovers (held-out 0.284, against 0.185 with the reset and 0.302 for the slow stage alone), but A itself is learned less well
   (0.87 / 0.89 on seeds 1 / 2) and is lost through B (0.000 / 0.280): without resets the fast codes split by the whole past (the start's
   code depends on the sequence before), so A does not take one path each time.

So the fast stage meets a dilemma of its own: reset at uncertain bytes, and its chains start from codes text shares and are rerouted;
no reset, and its codes split by the whole past and do not recur.

## Step 4 (22 November 2026): the fast stage bound to a slowly drifting context (seed 1)

Fast cells' context also requires the slow context code (`f_slow`, net7 `slow`), resets as before (`f_nodef=0`). The A test was also
run after a warm-up of A sequences read with learning off (`warm`), so that a slow context could return before A is scored.

| Arm | A after A | A after B | text held-out |
|---|---|---|---|
| slow stage only (warm 100) | 0.990 | 0.400 | 0.302 |
| fast, context tau 300 / 1,000 (no warm) | 0.800 / 0.920 | 0.000 / 0.000 | 0.202 / 0.204 |
| fast, context tau 300 (warm 50 / 100) | 0.760 / 0.720 | 0.000 / 0.000 | |
| fast, context tau 1,000 (warm 100) | 0.880 | 0.000 | |
| fast reads only cells with context (`f_skip0=1`), no slow context (warm 100) | 1.000 | 0.400 | 0.302 |
| the same with slow context tau 300 | 0.740 | 0.000 | 0.202 |

Readings: (1) the fast compartment's default cells (context-free, one code per byte, taught fast in text) cast a strong wrong vote at
A's gap; reading only cells with context removes the harm (text back to 0.302, A learned 1.000) but the fast stage then adds nothing
(A after B 0.400, as the slow stage alone). (2) Binding to the slow context hurts in every form tried: A is learned less well and text
falls to 0.20 (my reading, not checked: contexts split by the slow code are each learned once at a fast rate, and loose matching lets
a cell fire, with its vote, in contexts close to the one it was taught in). No form of the fast stage tried keeps A.

## Step 5 (before its runs): the fast stage as a hierarchy of temporal receptive windows

After the owner's ruling of 22 November (WHAT_A_NEURON_HAS section 21: a short window at the level of a cell is allowed; no cell records
a long text). Fast stage: 8 regions (net7 `ctx=2 hold=0`, C=128, M=16, S=64): each cell reads the present byte's columns and region d-1's
cells one byte late, so region d codes the last d + 2 bytes; the span comes from the hierarchy, as temporal receptive windows lengthen
along the cortex. Its compartment learns at 0.2 (normalised over its active cells); the slow stage as in step 1 (eta_o 0.1).

**H5.** `life=ctx` (A on letters): A after B 0.9 or more; text held-out 0.31 or more; context stream G=16: 0.99 or more. **Refuted if**
A after B is below 0.75 or text held-out below 0.30. Also run: `life=texts` with its B-only control (no prediction made). Seed 1 first.

## Outcome of step 5 (entered 22 November 2026; seeds 1-3)

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| `life=ctx`: A after A / after B / after the return | 1.000 / **1.000** / 1.000 | 1.000 / **1.000** / 1.000 | 1.000 / **1.000** / 1.000 |
| `life=ctx`: text test after B / after the return | 0.343 / 0.340 | 0.333 / 0.340 | 0.351 / 0.342 |
| text held-out after 16,000 bytes of it | **0.350** | 0.337 | 0.340 |
| context stream G=16, 40,000 bytes | **0.995** | 0.998 | 0.998 |
| `life=texts`: A test after A / after B / after the return | 0.350 / 0.334 / 0.335 | 0.337 / 0.335 / 0.347 | 0.340 / 0.334 / 0.342 |
| `life=texts`: B test after B / after the return | 0.392 / 0.387 | 0.408 / 0.408 | 0.401 / 0.398 |
| control, B only: A test / B test after B | 0.310 / 0.388 | 0.316 / 0.398 | |

**H5 held on three seeds** (A after B 1.000, text held-out 0.337-0.350, G=16 0.995-0.998). Two texts: what A adds over the B-only
control is 0.040 after A and 0.024 after B on seed 1, 0.021 and 0.019 on seed 2 (before, with one stage and a constant step: 0.073 and
0.024; 0.033 and 0.007); B is learned as well as by a learner that read only B. Run time 1-2.5 minutes.

## Step 6 (before its runs): replay to the slow stage

The store of step 2 (1,000 episodes of 40 bytes, every episode read with the same chance to be kept); replay teaches the slow stage
only (`replaying`), the fast stage's activity being put back afterwards. **H6.** In `life=texts`, at rho 1/20 the share of A's own part
kept after B (A test after B minus the B-only control, over the same after A) rises above step 5's (seed 1: 0.024 / 0.040 = 0.60), with
the B test within 0.01 of step 5. **Refuted if** the kept share does not rise or B falls by more than 0.01. Also run: rho 1/100; `life=ctx`.

## Outcome of step 6, first part (2,000-byte tests; seeds 1-2)

| | kept share of A's own part, seed 1 / seed 2 | B after B, seed 1 / seed 2 |
|---|---|---|
| no replay (step 5) | 0.60 / 0.90 | 0.392 / 0.408 |
| replay 1/20 | 0.68 / 0.82 | 0.393 / 0.411 |
| replay 1/100 (seed 1) | 0.64 | 0.393 |

Not decidable at this size: a 2,000-byte test has a standard error of about 0.011 at 0.34, and A's own part is 0.02-0.04. The step is
repeated with 10,000-byte tests (standard error about 0.005).

## Outcome of step 6 (entered 22 November 2026; 10,000-byte tests, standard error about 0.005; seeds 1-2)

| | seed 1 | seed 2 |
|---|---|---|
| B-only control: A test / B test | 0.328 / 0.403 | 0.321 / 0.399 |
| no replay: A after A -> after B (kept share of A's own part) | 0.356 -> 0.346 (0.64) | 0.349 -> 0.342 (0.75) |
| replay 1/20 to the slow stage | 0.357 -> 0.345 (0.59) | 0.348 -> 0.343 (0.81) |
| B after B: no replay / replay | 0.409 / 0.410 | 0.414 / 0.412 |

**H6 refuted**: replay to the slow stage changes the kept share by -0.05 and +0.06, within the noise. Two facts measured at this size:
about two thirds to three quarters of what A adds over the control is kept through 16,000 bytes of B; and B is learned better after
A than alone (0.409 against 0.403, 0.414 against 0.399): reading Shakespeare first transfers to Emma.
Not yet known: where the lost third is (fast or slow compartment); why replay at this dose does nothing (the slow stage's rate times
1/20 may simply be too small to matter).

**Where the lost part is** (`putback=1`, 10,000-byte tests): A after A / after B / after B with the fast compartment of after A put back /
with the slow one put back: seed 1 0.356 / 0.346 / 0.358 / 0.357; seed 2 0.349 / 0.342 / 0.349 / 0.354. Either compartment put back
restores A fully: the tissue (fast cells, slow states) keeps everything, and both compartments' readings of the contexts the two texts
share have moved towards B. The loss is the reading of shared contexts following the latest text, in both compartments.

## Step 7 (22 November 2026): yes/no feedback alone (seed 1 unless stated)

`fbm=yesno`: the answer given (the largest sum of the two compartments; while learning, with chance 0.1 a random byte) is told right
or wrong; only that byte's output cell learns, in both compartments (target 1 if right, 0 if wrong). Replayed episodes are the system's
own, so their bytes are known. A first run broke ties by the lowest byte (all outputs start at 0, so the same byte was answered again
and again); ties are now broken at random (noise in the competition).

| Test | yes/no, ties at random | teach (step 5) |
|---|---|---|
| `life=ctx`, A 20,000 bytes: A after A / after B / after the return | **1.000 / 0.990 / 1.000** | 1.000 / 1.000 / 1.000 (A 4,000) |
| context G=8, 20,000 bytes: all bytes, last quarter / after the gap | 0.834, rising / 0.672 | 0.95 / 1.000 |
| context G=16, 40,000: all / after the gap | 0.856 / 0.453 | 0.97 / 0.995 |
| text held-out after 32,000 bytes | 0.225 | 0.350 (16,000) |

Yes/no carries one bit an answer where teaching tells every output cell; learning is slower, and not finished at these lengths.
Longer lives (G=8 60,000; G=16 120,000) and seeds 2-3 for `life=ctx` are running.

**Step 7, longer lives and seeds.** `life=ctx` with yes/no (A 20,000 bytes): A after B 0.990 / 1.000 / 1.000 on seeds 1 / 2 / 3 (after
the return 1.000 on all). Context G=8, 60,000 bytes: all bytes 0.854, after the gap 0.898 (with 10% of answers tried at random the
ceiling is about 0.86 and 0.90: learned). G=16, 120,000 bytes: all bytes 0.853 (at the ceiling), after the gap 0.466 (online).
`t16_split.py` (learning off after the life, choosing between the two end bytes): G=12, 60,000 bytes, yes/no: right from the slow
compartment alone 1.00, the fast alone 0.52, the sum 1.00 (teach the same); the fast compartment gives the two end bytes the same value
(difference 0.000), so it does not drown the slow one. G=16 and G=24 frozen splits are running.

**Frozen splits under yes/no** (choosing between the two end bytes only): G=12, 60,000 bytes, seed 2: slow alone 1.00, sum 1.00;
G=16, 120,000 bytes: slow alone 0.96 / 1.00 (seeds 1 / 2), sum the same; G=24, 120,000 bytes: 1.00. So under yes/no the slow stage
does learn the branch across 16 and 24 bytes; the online 0.466 at G=16 counts answers over all byte values, so at the gap another
byte is often answered. Which, is being measured.

**The online 0.466 at G=16 explained in part** (`t16_split.py`, yes/no, 120,000 bytes, seed 1): with 32 byte values online 0.906,
frozen 1.00, slow alone 1.00; with 64: 0.905 / 0.96 / 0.96; with 128 (the default of net9, used by `t11_gate.py` above): 0.466 / 0.44 /
0.48, and the fast compartment now gives the two end bytes very different values (0.600) although it cannot see the start. So the
failure is with 128 byte values, not with yes/no as such; and every context-stream result with teaching so far used 32 or 64 values,
while text uses 128. Whether the slow stage carries a gap of 16 with 128 values when taught is being measured.

**128 byte values** (`t16_split.py`): teach, G=16, 60,000 bytes: online 0.998, frozen 1.00, slow alone 1.00; teach, G=8, 40,000: 1.000 /
1.00; teach, G=16 without the fast stage: online 1.000; yes/no, G=16, 120,000, seed 2: online 0.906 (the ceiling), frozen 0.99, slow
alone 0.99. So 128 values are not the cause; yes/no fails at G=16 with 128 values on seed 1 only. There the fast compartment gives the
two end bytes very different values without seeing the start (0.600), which steers the answer; under yes/no only the answer given is
taught, and exploration at random over 128 values tries the other end byte at the gap about once in a thousand answers (my reading,
not checked). Exploration in proportion to the outputs, not uniform over all bytes, would be the general remedy to test if this recurs.

**Exploration by noise in the competition** (`noise=0.1 eps=0`: each answer while learning is the largest of the outputs plus normal
noise of s.d. 0.1, instead of a uniformly random byte one time in ten), yes/no:

| | eps 0.1 (uniform) | noise 0.1 |
|---|---|---|
| G=16, 128 values, 120,000 bytes, seed 1: online at the gap / frozen | 0.466 / 0.44 | **0.998 / 1.00** |
| the same, seed 2 | 0.906 / 0.99 | 0.989 / 0.98 |
| text held-out after 32,000 bytes (seed 1) | 0.225 | **0.294** (teach: 0.350 after 16,000) |
| `life=ctx`, A 20,000: A after B / text test after B (seed 1) | 0.990 / 0.154 | 1.000 / 0.275 |

The reading of the failure is borne out: uniform exploration over 128 values hardly tries the other end byte at the gap; noise in the
competition tries the answers close to the best. Text and `life=ctx` on seeds 2-3 not yet run.

**Seeds 2-3, yes/no with noise 0.1:** `life=ctx`, A 20,000: A after B 1.000 / 1.000 (seed 1: 1.000); text test after B 0.237 / 0.246.
Text held-out after 32,000 bytes: 0.305 / 0.280 (seed 1: 0.294). Confirmed on three seeds: with yes/no alone the system learns A, keeps
it through text, and reads never-seen text at 0.28-0.31 (teach: 0.34-0.35 after half the bytes).

## Step 8 (before its runs): how long must the fast stage's hierarchy be? (teach, seed 1)

f_L regions code the last f_L + 1 bytes. **Predictions.** (P8a) A on letters with G=8 (its start is 9 bytes before the answer) is lost
through text with f_L = 2 or 4 unless the slow stage carries it. (P8b) A with G=4 is kept with f_L = 4 (A after B 0.9 or more).
(P8c) Text held-out with f_L = 4 is within 0.01 of f_L = 8 (0.350). **Refuted if** text falls by more than 0.02 at f_L = 4 (longer
hierarchies matter for text) or A with G=4 at f_L = 4 is below 0.75 (keeping does not come from the hierarchy reaching A's start).

## Outcome of step 8 (entered 22 November 2026; teach, seed 1)

| f_L (bytes coded) | A with G=4: after A / after B | A with G=8: after A / after B | text held-out |
|---|---|---|---|
| 2 (3) | 1.000 / **0.000** | 0.560 / 0.470 | 0.361 |
| 4 (5) | 1.000 / **1.000** | 0.560 / 0.470 | 0.361 |
| 8 (9) (step 5) | | 1.000 / 1.000 | 0.350 |

P8a held (with G=8, hierarchies of 3 and 5 bytes do not reach A's start: A is learned only by the slow stage, and in 4,000 bytes not
even learned, 0.56). P8b held (G=4 with 5 bytes: kept 1.000; with 3 bytes learned by the slow stage and lost through text, 0.000).
P8c held, and more: text is 0.361 with 3 or 5 bytes against 0.350 with 9. So a context is kept exactly when the fast hierarchy reaches
the byte that tells it apart; text needs only the last 3-5 bytes; the slow stage holds longer gaps but not through later text.
