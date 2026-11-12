# Draft for audit: a sizing sweep for the target test (11 November 2026, 20:26 EST by the clock)

> **Status.** Draft for Codex's audit. **No worker starts from this file.** Coordinator's design, adopted from the audit session's note of 11 November (its section 4.3; review 9.101 D.3). **This is a sizing exercise. It registers no prediction for judgment, gives no verdict, and concludes nothing about any learning rule.** Its only product is a table that says at what sizes a learner can acquire anything at all from this stream.

## Why

Step 0 is done (reviews 9.101 C, 9.105, 9.106): at the proposed sizes every held answer of the target stream is determined by the bytes before it. That is identifiability. It does not show that a learner can acquire those answers, and the brief of step 0 asked this itself as its first question.

The proposed sizes are small in a way this project has not tried: 216 taught pairs per family, each shown four times before the first panel, with letters in place of digits, so that the learner has to find 36 hidden numbers and the addition that joins them from outcomes alone. The audit session expects that no learner starting from nothing acquires the held pairs at these sizes under any update. If that is right, the design would stop at its reference step after several more rounds of audit, at sizes that have to change. A sweep that costs minutes per cell says so now.

## What is varied

The generator is `TARGET_TEST_STEP0/target_stream.py`, unedited. Its sizes are set through its `SIZES` table (and its modulus constant) in a copy made at run time; every cell records the full table it used.

| Factor | Values |
|---|---|
| Modulus | 97 (proposed), 31, 11 |
| Presentations of each taught pair before the first panel | 4 (proposed), 32, 256 |
| Disclosure lines in the stream | none (the ordinary stream); the aligned lines (`X:vv c:uu` after each episode) |

Eighteen cells. **World seeds 101, 102 and 103**, which are never scored in any later step. Seeds 1, 2 and 3 are not touched.

## The learner, one for every cell

A GRU of E20-B's shape (byte embedding, one layer to 512, a GRU of 256 units, the two-layer readout; the library layer), trained by backprop with AdamW at the settings of E20-B's births, **starting from random weights, with every next byte as the target and no answer mask.** No rule, no sign path, no replay store.

Two ways of feeding the stream, because they answer different questions:

- **A. Episode by episode.** Each episode is one item and the state starts from zero. The network can use only what is in the episode and what its weights have stored. In the ordinary stream that is the context letter and the cue; an answer on a taught pair then shows the weights have learned that pair, and an answer on a held pair shows they have learned the values and the addition. This arm says whether the hidden structure is learnable from outcomes at all at a given size. Disclosure lines cannot help here by construction, so arm A runs on the ordinary stream only: nine cells.
- **B. As one continuing stream.** The bytes are read once, in order, with the state carried from byte to byte and credit cut every 256 bytes. Nothing is reset. This is how the target test itself runs. It can use a line read earlier only if the state still holds it. Both streams: eighteen cells.

One pass over the stream in arm B, since a continuing learner sees its stream once. In arm A the same episodes in the same order, once.

## What is read, at the first panel and at the end of the stream

- **Taught pairs**, on fresh renderings with the context ten or more bytes back: whole outcome, two digits and the end byte.
- **First-encounter pairs**, at their first occurrence, before the reveal.
- **Sealed twins**: both members right; also single members.
- Beside each, from the step 0 checker: the ten-byte-suffix table, the exact-key table and the recency table on the same probes, and the prefix-only solver, which is the ceiling.
- The mean next-byte loss on outcome bytes and on all other bytes, per tenth of the stream.
- CPU-seconds per cell.

In arm B the probes are scored the way step 0 fixes: the learner's state is saved before a probe and restored after it, and both members of a twin are answered from the same state.

## What the table is for

It is read by eye and decides three things about the design, none about a learner:

1. **Whether the proposed sizes can stay.** If no cell at modulus 97 and four presentations shows the taught pairs above the tables, the proposed sizes are too small for any learner that starts from nothing.
2. **What a learner must bring.** If only the long, small-modulus cells learn anything, a learner that starts from nothing needs a stream of that length, and the test becomes a test of slow rule learning. The alternative is a learner born on a declared curriculum that shares no pair and no value with the test world. That choice is the coordinator's and Mr. Xu's to make after this table, and it is the main reason for the sweep.
3. **Whether the stated values should be the base and their absence the harder arm.** If in arm B the cells with lines acquire held pairs and the cells without do not, the order of the design's arms is turned round, and the line schedule is then made sparser so that the disclosure ledger can differ between arms.

## What this is not

- Not a reference for step 1. The learner here is not qualified against anything, and its scores are not baselines.
- Not evidence about backprop against the rule. One learner, one update.
- Not on the scored worlds.
- Not a search for sizes that make a result look good. The grid is fixed above; cells are not added after the table is seen. If the table shows the whole grid is in the wrong place, that is reported and a second grid is briefed and audited.

## Admission

- The stream builder at the proposed sizes, for world seed 1, still gives the hashes on file (`TARGET_STEP0_CHECK.json`), which shows the copy-and-set route changes nothing by itself.
- At every changed size the step 0 checks that do not depend on a learner (identification, oracle, exposure, twins, determinism) pass, or the cell is reported as not built and why. A smaller modulus makes equal outcomes more likely; twins are redrawn as step 0 requires.
- A fixture learner that answers from the oracle scores every probe through the sweep's own scoring code, and one that answers from the ten-byte suffix scores no twin.
- State save and restore around a probe leaves the continuing learner's state unchanged, shown on a fixture.

## Running it

- New files only, in `TARGET_TEST_STEP0/sizing/`. The arithmetic harness is not imported; the GRU is built directly.
- One host, at most three processes on four cores, one thread each.
- Cost: a cell at the proposed size is a stream of about 46 KB and should take seconds to minutes; a cell at 256 presentations is about 64 times longer. Time the first cell of each size and forecast the rest before running them. If the whole grid would pass twelve hours of CPU, stop and report the forecast.
- Work on your own branch. One commit when the table is complete.

## Report

One table per arm (cells by modulus and presentations, three world seeds each, the four readings with the tables' and the solver's figures beside them), the loss curves, the sizes used in every cell, the admission record, CPU per cell, and whatever was unclear. No conclusion about any learner or rule.

## Questions for the audit

1. Arm A resets the state at every episode, which the target test never does. Is it still the right probe of "learnable from outcomes at all", or does it answer a question the design does not ask?
2. Arm B reads the stream once. Should a second pass be allowed in some cells, marked as such, to separate "too little data" from "too few updates"?
3. Is credit cut every 256 bytes long enough for a line to be used at the next episode of its context, which is some hundreds of bytes later? If not, arm B with lines would fail for a reason that has nothing to do with the sizes.
4. Should one cell with a born learner (pretrained on other worlds, then run on a sweep world) be in this sweep, or is that a design of its own?
