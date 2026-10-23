# Review of MiniFly v4 (design and Codex's implementation) — 2026-10-20

Subject: `minifly_v4_design_20261020/DESIGN.md` (Claude, commit 190a26c) and Codex's implementation
`minifly_v4_dev_20261020/` (codex branch, commit a6ce358b, verdict NO_ADVANCE). Read-only review; nothing re-run.

## Summary

v4 never tested its central hypothesis — that consolidating only what recurs lets rules win over item memorisation.
The runs stopped at prerequisites that could not pass at the doses used, and several of those prerequisites failed
because of omissions or choices in my design. The result is uninformative about the hypothesis in either direction.

## What was run

* Raw-bit input; segment state with three closed slots of right-aligned bytes; 65,536 / 262,144 memory cells with
  fan-in 12 (random signed weights on raw bit lines, topographic or random windows); nine native output channels
  (8 bits + end-of-answer); recurrence-gated consolidation; boundary unit learned by a reward-modulated rule; drift-
  gated warmup starting at 1/16.
* **G0 coverage:** external ridge readout fitted on **128** problems (≈ 900 target events), tested on 64.
  Training bytes 100%; unseen answer bytes 5.4–9.1%; unseen exact records 0%. Neutral deterministic boundary: zero
  segment closures; exploratory boundary: ≈ 6 random closures per problem (`cycles/cycle2/G0_*.json`, `REVIEW.md`).
* **G1 learning sanity:** **64 fresh problems** (+8 replays) per history, two worlds, seven histories. Every arm 0%.
  Generated output was `00 00 00 …` until the 16-byte cap, never an end marker (`results/DIAGNOSIS.json`).
  Boundary probabilities 0.49985–0.50012; boundary weight norm 2×10⁻⁴; warmup multiplier still 1/16–1/8 at the end.
* A teaching-sign error in the new port was found and fixed during the run; affected histories were discarded.

## Findings, most serious first

1. **Dose (test design).** One native teaching event moves an output value by ≈ 0.01–0.02
   (`ops/PORT_BASELINE_DIAGNOSIS.json`), while every channel starts at ≈ −0.98. A bit flips only after dozens of
   consistent writes to the same cells. With 64 one-off problems and teaching scaled to 1/16, even the answer format
   (end-of-answer timing, digit bytes) could not be learned. Column patterns — the units the hypothesis needs — occurred
   about once or twice each.
2. **Output reference point missing (my omission).** The design said the emitted byte is "the sign pattern of the bit
   channels" without a reference. Native values carry a large common offset (−0.98 at birth on all nine channels), so
   every bit read 0 and end-of-answer never fired. v0's 16-component code hid this by centring across compartments;
   independent bit channels cannot be centred that way.
3. **Boundary learning was circular (my design choice).** Boundaries were learned from the answer error, but answers
   cannot improve before meaningful segments exist, and segments cannot be learned before answers improve. The
   segment-relative position codes — the core of the design — never existed in any run.
4. **G0 was mis-sized (my omission).** No fitting sample size was specified. A linear readout over 65k–262k features
   fitted on 128 problems memorises them and cannot generalise whatever the features are; it also ran without
   segments. Its failure says nothing about coverage.
5. **Raw bits feed sparse random combinations poorly (design flaw).** A digit is a 4-bit pattern and an aligned digit
   pair an 8-bit pattern; answer bits are high-order functions of those bits. Each cell summed 12 random signed bit
   lines from random slots, many of them useless for digits (constant high bits `0011`, the flag channel). A cell
   selective to one aligned digit pair arises only by accident. A stage that forms byte-value units from the raw bits
   (unsupervised) is needed before cross-position combinations.
6. **A sanity check that contradicted the hypothesis.** G1 scored recall of problems seen once; the hypothesis predicts
   such traces fade. The right early checks are recurring regularities: answer format, then digits by column.
7. **Warmup starved learning.** The drift-gated ramp never advanced (64 records gave too few stable checkpoints), so
   teaching stayed at 1/16. Slowness belongs to what becomes permanent and to code formation, not to the fast store's
   ability to respond — fast traces decay and cannot "bake" the brain.
8. **Minor.** The consolidation sigmoid allocated 0.9% to the slow store at zero confirmation (not strictly
   "nothing permanent before confirmation").

## Process findings (v3 and v4)

* Cycles test designs at doses where they cannot succeed, then record NO_ADVANCE; v3 and v4 both failed on unreached
  prerequisites.
* Exact whole answers are the only score, which hides partial learning (format, single columns) and gives no
  direction after a failure.
* Prerequisites were checked in the wrong order (coverage before boundaries; learning before the output could fire).
* Qualification overhead dominated; the learning runs themselves stayed tiny.

## Recommendation

A v5 revision that fixes findings 1–8 without adding new kinds of mechanism, run as an ordered ladder with doses
computed in advance and partial-ability curves reported at every rung. See `DESIGN.md` in this folder.
