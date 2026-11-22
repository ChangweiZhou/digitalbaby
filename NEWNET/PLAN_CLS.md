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
