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

## Outcome (entered 21 November 2026; four runs, exit 0; `p3b_p4_tables.py`)


| Form | a* | Connections at the end | Last 4,096 bytes | Cells sending a step | Cells with mean threshold above 0 in A2 | Mean threshold in A2 | S(1,0), S(2,0), S(4,0) | ms a byte |
|---|---|---|---|---|---|---|---|---|
| first | 1/8 (plan 3) | 17.5 | +1.2% | 31.6 | 0.116 | 0.0001 | 0.992, 0.577, 0.022 | 1.68 |
| first | 1/32 | 17.8 | +1.9% | 33.8 | 0.213 | 0.0018 | 0.987, 0.522, 0.022 | 1.49 |
| first | 1/128 | 18.2 | +2.4% | 19.5 | 0.276 | 0.0087 | 0.855, 0.344, 0.022 | 1.12 |
| coincidence | 1/8 (plan 3) | 18.8 | +4.1% | 52.3 | 0.115 | 0.0001 | 0.982, 0.758, 0.057 | 3.19 |
| coincidence | 1/32 | 17.9 | +2.7% | 45.4 | 0.225 | 0.0020 | 0.972, 0.608, 0.040 | 2.89 |
| coincidence | 1/128 | 17.0 | +2.5% | 21.0 | 0.291 | 0.0086 | 0.911, 0.346, 0.022 | 2.19 |

The thresholds are read as their mean over window A2 (`theta_mean` of `A2.npz`), not from the final state as the plan says: the runner does not save the final state. Plan 3's N75T at a* = 1/8 also has some cells with a mean threshold above 0 (12%): the threshold rises in a cell above 1/8 for a while before falling back.

| | Outcome |
|---|---|
| H4 | **Refuted in both forms.** Lowering a* makes more cells hold a threshold above 0 (21 to 29% against 12%) and fewer cells send (first form 19.5 against 31.6 a step at 1/128), but S(2,0) falls: 0.577 to 0.522 and 0.344 in the first form, 0.758 to 0.608 and 0.346 in the coincidence form |
| E1 | Failed |
| E2 | Failed |
| E3 | Failed: S(4,0) at most 0.040 |
| E4 | Held |

**Reading.** In this system a reachable target makes the tissue sparser and the cells two connections out less informative. The rise of S(1,0) at a* = 1/128 in plan 1 came, at least in part, through making (which a* drove there), not through the threshold. a* = 1/8 stays the setting of the system.
