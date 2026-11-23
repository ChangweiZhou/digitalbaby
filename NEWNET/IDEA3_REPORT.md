# IDEA3 report: a relay gate on content-free inputs (23 November 2026)

Branch claude/newnet-idea3-sei232 (from claude/magical-albattani-sei232 @ 6dc127e8). Development level, **seed 1 only**, one machine;
no gate, seal or outside review is claimed. Plan and criterion written before the runs: `IDEA3_PLAN.md`.
Code: `net12_cf.py` (subclass of net12.Tissue; option gate_in = all | match | match+pos; the relay gate reads only those inputs,
the direct readout reads lag + match cells as in net12; position cell p = "the last newline was p bytes back", p = 0..7).
Test: `toys/vig_toy_cf.py` (a copy of toys/vig_toy_lesion.py importing net12_cf; 4 regions; copy 2000 -> rev 2000 -> succ 3000
-> copy 1000 lines; cued "=", "~", "+"; probes 60 items per rule, learning off). Logs: `idea3_runs/toy_*.log`, `idea3_runs/run_toy.sh`.

## Results (probe after each phase; first200 = right share of answer bytes in the phase's first 200 lines, learning on)

| variant | copy first200 / probe c,r,s | rev first200 / probe | succ first200 / probe | final copy first200 / probe c, r, **s** |
|---|---|---|---|---|
| net12 (no match; = vig_toy_lesion baseline) | 0.73 / 1.00 0.36 0.00 | 0.34 / 1.00 1.00 0.01 | 0.01 / 0.53 0.97 1.00 | 0.79 / 1.00 1.00 **0.58** |
| match=4, gate_in=all | 0.70 / 1.00 0.35 0.00 | 0.35 / 1.00 1.00 0.01 | 0.01 / 0.72 0.97 1.00 | 0.85 / 1.00 1.00 **0.81** |
| match=4, gate_in=match | 0.72 / 0.97 0.36 0.02 | 0.65 / 0.15 0.98 0.01 | 0.06 / 0.00 0.34 1.00 | 0.58 / 0.78 0.63 **0.52** |
| match=4, gate_in=match+pos | 1.00 / 1.00 0.36 0.00 | 0.79 / 0.46 1.00 0.02 | 0.05 / 0.01 0.01 1.00 | 0.79 / 1.00 0.33 **0.04** |

Lesions after the whole life (probe copy, rev, succ with one compartment's strengths set to 0):

| variant | without relay | without lagro | without ctx | without hold |
|---|---|---|---|---|
| net12 (no match) | 0.04 0.36 0.97 | 1.00 1.00 0.01 | 1.00 0.96 0.52 | 1.00 1.00 0.24 |
| match=4, all | 0.02 0.34 0.92 | 1.00 1.00 0.03 | 1.00 0.98 0.80 | 1.00 1.00 0.64 |
| match=4, match | 0.02 0.34 0.77 | 0.81 0.67 0.04 | 0.76 0.34 0.69 | 0.78 0.64 0.82 |
| match=4, match+pos | 0.01 0.04 0.98 | 1.00 0.34 0.00 | 1.00 0.34 0.03 | 1.00 0.36 0.04 |

The net12_cf run without match cells reproduces the earlier toy result (succ 0.58 after the final copy; without relay 0.97), which
checks that the subclass behaves as net12 when gate_in=all.

## Against the criterion (succ >= 0.85, copy and rev >= 0.95 after the final copy; copy first200 >= 0.6)
- gate_in=all, match=4: copy 1.00, rev 1.00, first200 0.70, **succ 0.81 < 0.85: fails** (narrowly; it is the best variant).
- gate_in=match: succ 0.52, copy 0.78, rev 0.63: **fails** on every count but first200 (0.72).
- gate_in=match+pos: succ 0.04, rev 0.33: **fails**, and worse than net12.

No variant passed, so the full-size runs (t23 cue=1, t22) were **NOT_RUN**. `t22_rules_cf.py` and `t23_life_rules_cf.py` are
ready (they import net12_cf) but have not been run.

## What it shows (seed 1, 4-region toy)
1. The hypothesis in this form is refuted. A gate that reads only content-free inputs is not protected against copy. It is
   rule-blind instead, so copy training reopens it on succ lines as much as, or more than, before. With match+pos it reopens
   fully and also wipes rev: lag j's gate is set by line position, which is the same for every rule. With match only it cannot
   relay copy's first answer byte (no match cell is active there), so copy stays partly in the direct readout and every rule
   interferes (copy 0.78, rev 0.63, succ 0.52 at the end).
2. In all four variants the direct readout keeps succ, with succ between 0.77 and 0.98 when the relay is lesioned. The loss is the
   relay overriding the direct readout. It is not the mapping being erased.
3. What keeps succ apart in the gate is rule information that only the symbol lag cells carry (the delimiter at lag 0, as a
   symbol). Removing those inputs removes what the gate needs to stay shut on succ lines. Adding match cells while keeping all
   inputs (gate_in=all) lifts succ from 0.58 to 0.81, the only gain seen here. It is still below the criterion and was not taken
   to full size.

## Verdict
None of gate_in = match or match+pos (nor match cells added to the full gate) meets the criterion. The learned succ mapping is
still lost through the relay when copy returns. Full-size runs not done.
