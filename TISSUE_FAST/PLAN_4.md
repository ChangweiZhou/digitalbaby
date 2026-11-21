# Plan 4: a target of activity that the cells can reach (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 21 November 2026: "好的，多试一试" to the revised order, whose step 4 is this plan as the outside reading corrected it: not a set point taken from the cell's own long mean (which drifts with the tissue), but a fixed, low target the cells can reach. **Program:** `TISSUE_EVENT/` at the commit of this plan; the target is E63's a* (`--astar`), which sets the threshold's control (formula 9: the threshold rises when the output is above a* and falls when below, never below 0). With the turnover rule of plan 2, making no longer depends on a*, so a* now acts only through the threshold.

**Why.** At a* = 1/8 no cell reaches the target, so every threshold falls to 0 and stays there: the control does nothing. Plan 1 lowered a* to 1/128 with E63's making and S(1,0) rose (0.687 to 0.950 first form; 0.304 to 0.889 coincidence form), but there a* also drove making, so the two effects were mixed.

**H4.** In the system of plan 3's N75T (scaling at 3/4 and 1/4, turnover with trial and withdrawal by use), lowering a* to a value the cells reach makes thresholds rise above 0 in most cells that are ever active, and raises S(2,0) in A2 above N75T's at a* = 1/8 (0.577 first form, 0.758 coincidence form) in at least one of the two values, in both forms. **Refuted if** neither value raises S(2,0) in a form, or if the tissue falls silent (S(1,0) below 0.1103).

| No. | Prediction | Chance |
|---|---|---|
| E1 | H4 holds in the first form | 0.35 |
| E2 | H4 holds in the coincidence form | 0.3 |
| E3 | S(4,0) above 0.05 in some arm | 0.2 |
| E4 | a* = 1/128 has fewer cells sending a step than a* = 1/8 in both forms | 0.7 |

**Arms.** Seed 73129; 16,384 bytes; N75T of each form with a* = 1/32 and 1/128: 4 tissues. Reference: plan 3's N75T (a* = 1/8). **Readings added:** the share of cells with a threshold above 0 at the end, and its mean (from the final state). **Cost** about 2 to 4 ms a byte: under 10 minutes. **Not run:** a biphasic guard on structural change (no sign yet that the tissue falls silent; it is the next change if it does); the cell's own gain; other seeds.

## Outcome

(to be entered after the runs)
