# IDEA3 plan: a relay gate that reads only content-free inputs (23 November 2026, written before any run)

Source: branch claude/magical-albattani-sei232 @ 6dc127e8 (NEWNET/net12.py unchanged); work and commits on claude/newnet-idea3-sei232.
Development level, seed 1, one machine; no gate, seal or outside review is claimed.

## Hypothesis
In copy -> rev -> succ -> copy (cued "=", "~", "+"), net12 lagro=1 relay=1 loses succ when copy returns (4-region toy: probe
1.0 -> 0.58); with the relay lesioned the direct readout still gives succ 0.93, so the loss is the relay's gate being reopened by
copy training through the symbol lag cells that copy and succ share. If the gate reads only content-free inputs, copy training
cannot reopen it for succ lines.

## Mechanism (net12_cf.py, a subclass of net12.Tissue; net12.py not edited)
gate_in = "all" (net12 as is) | "match" | "match+pos"; the relay gate G reads only these inputs; the direct readout B reads what it
read before (lag cells + match cells).
- match: net12's match cells m[k, j] (byte k back == byte k+j back), match=4.
- pos: W cells, cell p active when the last "\n" was p bytes back (p = 0..W-1; none when it is further back). A general structural
  cell (line position), not chosen for any symbol.
Learning stays local (the gate's own delta rule, as in net12); nothing tuned to the test.

## Expectation written in advance (not a criterion)
At the first answer byte of a line no match cell is active (nothing has repeated yet) and the position is the same for every rule,
so neither "match" nor "match+pos" can tell copy from succ there; the relay can help only on answer bytes 2-3 (copy: lag0 == lag4),
and the first byte must come from the direct readout. Copy's first-200 learning may therefore drop.

## Tests
1. toys/vig_toy_cf.py (copy of toys/vig_toy_lesion.py importing net12_cf; 4 regions; copy 2000 -> rev 2000 -> succ 3000 -> copy
   1000), match=4, gate_in = all / match / match+pos: per phase first-200 and probes, and the lesion lines.
2. Only if a variant passes: t23 cue=1 seed 1 (t23_life_rules_cf.py, a copy importing net12_cf) against the baseline in
   LIFE_RULES_REPORT.md (succ 0.98 -> 0.07, copy/rev 1.00 at end, bits 828,527), and t22 (t22_rules_cf.py; baseline P0 b/c
   0.99/0.83).

## Success criterion (toy, decided now)
A variant passes when, at the probe after the final copy phase: succ >= 0.85 AND copy >= 0.95 AND rev >= 0.95, AND first-200 copy
(phase 1) >= 0.6. Full size (if run): succ at end clearly above 0.07 with copy/rev at end >= 0.95 and t22 layers P0 b/c not much
below 0.99/0.83 (taken as: b >= 0.95, c >= 0.75).
Anything else is reported as a failure of the hypothesis in this form.
