# Idea 12: targeted replay and slow consolidated gate strengths (plan, written before the runs, 23 November 2026)

Helper branch `claude/newnet-idea12-sei232`, from `claude/magical-albattani-sei232` @ 6dc127e8. Development level, one machine,
seed 1 only; no gate, seal or outside review is claimed. Learning stays local (net12's delta rules, no backprop); the owner allows a
small episode store for rehearsal (500 lines of 8 bytes, or 40-byte text episodes; no long window of past bytes).

## Problem (from LIFE_RULES_REPORT.md and the toys)
Life copy -> rev -> succ -> copy (cued). net12 lagro=1 relay=1: succ probe after the final copy 0.58; random replay of a 500-line
store at 1 line per 4 lines read: copy after succ 0.86 (was 0.53), succ after copy 0.68. Succ is lost in the relay gate, reopened by
copy training through shared symbol inputs. Full size (t23, cue=1, seed 1): text probe 0.35 -> 0.24, succ 0.98 -> 0.07, bits 828,527.

## Mechanisms (new code only: `net12_rep.py`, `toys/idea12_toy.py`, `t23_life_idea12.py`; no existing .py edited)
1. Targeted replay: each stored line keeps its last error (mean over its answer bytes of 1 - p(answer)), set when read, updated when
   replayed; lines are drawn with probability proportional to error^2.
2. Slow gate strengths (net12_rep slow=1|2): relay gate (optionally the direct readout, slowB=1) = fast + slow.
   slow=1 (owner's form): during replay slow += sk * fast, fast *= 1 - sk; slow=2: replay's own changes go to the slow part.
   Note before running: without a decay of the fast part the sum evolves exactly as in net12, so the split alone cannot change
   anything; the variants run with the fast part decaying (ftau bytes). slow=1 with ftau=0 is run once as a check that it matches the
   plain replay run byte for byte.
3. References: random replay at rho 0.25 and 0.5, same store (500 lines). The toy script stores a line after it is read (the
   earlier toy stored it before, so it could be replayed just before being read); the references are rerun in the new script.

## Toy runs (4 regions, toys/idea12_toy.py, seed 1, ~2 min each, 1 CPU)
random 0.25; random 0.5; targeted 0.25; slow-gate + random 0.25; slow-gate + targeted 0.25 (slow=1 and slow=2, ftau 500, a few
values of sk/ftau at most); the check slow=1 ftau=0. Report first200 and probes per phase.

## Success criteria (stated before the runs)
- Success: at replay rho <= 0.25, succ probe after the final copy >= 0.85, with the copy and rev probes after the final copy >= 0.95.
- Partial: succ after final copy improves on random replay 0.25 (rerun here) by >= 0.10 with copy/rev >= 0.95.
- Failure: anything else, or a gain bought by learning less (first200 of a phase lower by > 0.10 than random replay 0.25).

## Full size
The best toy variant (by the criteria; if none succeeds, the best by succ after final copy with copy/rev >= 0.95) is run in a t23
life (`t23_life_idea12.py`: t23_life_rules.py plus the same store/replay, net12_rep; struct K8, lagro=1 relay=1, cue=1, seed 1),
with random replay at the same rho as its reference if CPU allows. Episodes: each rule line (8 bytes), and 40-byte text episodes in
the text phase (one candidate per 40 bytes; every 8 text bytes count as one line for the replay clock). Replayed bytes learn but are
not scored (bits count only bytes read in the life). Cost: ~30 min per baseline life plus replay (~+25-50%).
Compare with LIFE_RULES_REPORT.md: text probe 0.35 -> 0.24, succ 0.98 -> 0.07, bits 828,527.
Full-size success: succ at end >= 0.5 with copy/rev at end >= 0.95 and bits not worse than 828,527 by more than 2%.
NOT_RUN unless run: other seeds, cue=0.
