# E2 and E3 in one life: rules and a text (t23_life_rules.py, 22-23 November 2026)

Development level, one machine, **seed 1 only**; no gate, seal or outside review is claimed. Results: `life_rules_runs/res.jsonl`.
Life: copy (6,000 lines) -> rev (6,000) -> 40,000 bytes of tinyshakespeare -> succ (6,000; a learned symbol mapping) -> copy (3,000)
-> rev (3,000); 40-symbol alphabet; cue=1: each rule its own delimiter, cue=0: all "=". Probes every 1,000 lines with learning off.

## The whole life in one number (bits of every byte read, prequential)

| | cue=1 | cue=0 |
|---|---|---|
| storage (back-off n-gram) | 1,984,668 | 1,925,623 |
| K8 (net10) | 1,149,490 | 1,133,344 |
| net12 K8 + lag cells, direct readout + relay | **828,527** | **845,244** |
| the same + context groups (ctxg=16) | 883,734 | 909,695 |

## E3: first-met answer bytes, new combinations (layer b) / never-answered symbols (layer c, 40 bytes)

| | P0 copy b / c | P1 rev b | P3 succ b | P4 copy b | P5 rev b |
|---|---|---|---|---|---|
| storage | 0.06 / 0.08 | 0.00 | 0.02 | 0.00-0.02 | 0.00 |
| K8 | 0.14 / 0.00 | 0.01-0.02 | 0.11-0.13 | 0.10 | 0.01-0.03 |
| net12 relay, cue=1 | 0.98 / 0.73 | 0.80 | 0.51 | 0.99 | 1.00 |
| net12 relay, cue=0 | 0.98 / 0.73 | 0.74 | 0.35 | 0.80 | 0.96 |
| net12 relay + ctxg, cue=1 | 0.86 / 0.43 | 0.75 | 0.41 | 0.97 | 0.97 |

## E2: probes (learning off), after the phase that taught it and at the end of the life

| | text after text -> end | succ after succ -> end | copy at end | rev at end |
|---|---|---|---|---|
| storage, cue=1 | 0.37 -> 0.37 | 0.13 -> 0.10 | 0.15 | 0.71 |
| K8, cue=1 | 0.36 -> **0.09** | 0.11 -> 0.00 | 0.00 | 0.35 |
| net12 relay, cue=1 | 0.35 -> **0.24** | 0.98 -> **0.07** | 1.00 | 1.00 |
| net12 relay + ctxg, cue=1 | 0.36 -> 0.22 | 0.75 -> 0.34 | 0.93 | 0.93 |
| net12 relay, cue=0 | 0.36 -> 0.28 | 0.89 -> 0.02 | 0.64* | 1.00 |

*cue=0: the first answer byte cannot tell copy from rev (the right guess follows the recent rule), so probes of the rule not in force
understate what is kept; the later answer bytes are fixed once the first is seen.

Savings (first 250-line window when a rule returns, cue=1): net12 relay copy 0.95 (first time 0.48), rev 1.00 (first time 0.35);
K8 0.06 and 0.38.

## What it shows (seed 1)
1. One number: net12 with the relay needs 42% of storage's bits and 72% of K8's for the same life.
2. Rules that repeat earlier bytes are learned as rules (new combinations and new symbols) and kept through text and other rules when
   cued (copy and rev probes 1.0 to the end); when a rule returns it is right at once.
3. Not solved: the learned mapping (succ) is forgotten when copy returns (0.98 -> 0.07: it shares the direct readout with copy); the
   text is partly lost (net12 keeps 69% of its text probe, K8 25%, storage 100%). Context groups by a fast byte trace (ctxg) keep succ
   better (0.34) but learn less and generalise worse; not a solution.
4. Small tests (toys/, 4 regions, seed 1; text kept after 20,000 bytes of copy lines): nothing 0.85; context-gated cells (xdg=8)
   0.88; the lag readout's strengths per context group (lag_xdg=1) 0.93; both 0.945; replay of 500 stored 40-byte episodes at 25% 0.94;
   all three 0.93. The full-size lives with xdg are run by a helper (claude/newnet-xdg-sei232).
5. Battery (helper claude/newnet-battery12-sei232, in progress): net12's new compartments change Moby Dick / code by at most 0.002 at
   seed 1 and keep every keeping/memory test at 1.0: they cost nothing on natural text, and gain nothing there either.
