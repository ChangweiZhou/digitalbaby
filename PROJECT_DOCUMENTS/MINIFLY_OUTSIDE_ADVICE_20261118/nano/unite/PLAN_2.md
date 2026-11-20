# nano/unite, second plan: keeping, on the text with letters (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, in Chinese, in the coordinator's translation as given to this line: "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." and then "Good, keep trying." The step is the coordinating session's. **Source.** `claude/nano-unite-8mmhfm` at the commit that carries this file; `unite.py` as checked in `check_output.txt` (ten cases PASSED). **Status.** Development level; no checker gate, seal or outside review is claimed. Written while the reported-seed runs of `PLAN_1.md` are still going; what is known is the development seed's outcome there (the clock found within 20,000 lines; 89 to 94% at every gap at 40,000 to 60,000 lines; a fall with further reading that the oracle arm shares in part).

**The condition for this step** was that step 1 holds at 60% or more. On the development seed the united system is at 94% at 40,000 to 60,000 lines and at 58% at 200,000. I take the condition as met for a life of the length at which the system holds (the length `nano/sparse` and `nano/ahead2` used on plain sums), and say here that at 200,000 lines it is not met on that seed. The reported seeds' figures will be entered in `PLAN_1.md`; if their median at 40,000 lines is under 60% this plan's runs are void and will be marked so.

## 1. Question

Does the united system keep never-shown addition through subtraction, nothing of addition shown again, on the text with letters?

## 2. System, text, life

The united system of `PLAN_1.md` unchanged (`clock=trial`, `skip.py`'s numbers, look-ahead of three, no cap), with 18 byte values (`-` now has a line and an entry; 13,345,794 plain cells and 2 x 9,030,870 cells keyed on a guess can be addressed; about 4.8 gigabytes). Life: 40,000 lines of addition `aa+<0 to 8 letters>bb=ccc`, then 20,000 lines of subtraction `aa-<0 to 8 letters>bb=ccc` (the larger number first, as `cells2.py`), nothing of addition shown during subtraction. The clock goes on trying throughout; the `-` entry is born with the others and first meets its byte in the second phase. Split as `cells2.py`: for each operator 500 pairs scored and never shown, 8,000 shown. Readings every 5,000 lines: never-shown additions at gaps 0, 4, 8, own answer bytes; subtraction at the end of each phase and through the second, on all 500 held-out pairs and on the clean ones (those whose written line no shown pair writes; `ERRATA_20261120.md`).

Arms, five reported seeds: (a) the united system; (b) `clock=oracle` (letters passed over by the program), the comparison for what the clock costs in keeping. If `PLAN_1.md`'s V3 holds (the cap holds the level on the development seed), also (c) the united system with `ncap=16`. Beside them, not rerun: `nano/ahead2` on plain sums (with look-ahead: 91.2 before, 86.8 after, 94.7% kept, clean subtraction 89.4).

## 3. Hypothesis, and what refutes it

**H2 (the coordinator's).** Never-shown addition after the 20,000 lines of subtraction is at least 85% of its level before them: median over the five reported seeds of each seed's ratio, last reading of each phase, at the gap of 8 letters (the gaps of 0 and 4 beside it). **Refuted if** that median is under 85%.

## 4. Predictions (mine, before the runs)

| | Prediction | Chance |
|---|---|---|
| K1 | the united system before subtraction: at least 85% at every gap (median) | 0.65 |
| K2 | H2 holds for the united system | 0.50 |
| K3 | the oracle arm keeps at least 85% of its level | 0.70 |
| K4 | the united system's ratio is within 8 points of the oracle arm's | 0.50 |
| K5 | the letters stay closed through subtraction on every seed (w x y z closed at the last reading) | 0.75 |
| K6 | no digit and no `=` is closed at the last reading on any seed | 0.65 |
| K7 | the `-` entry is open at the last reading on at least four seeds of five | 0.60 |
| K8 | clean subtraction at the end, the gap of 8, united system: at least 70% (median) | 0.45 |

What I do not know: whether the trials of the clock during subtraction (a new operator arriving, the mismatch high at first) disturb the entries that were settled; whether the `-` entry closes (a line without its operator in the chain is still a line whose numbers are aligned, so closing it may cost nothing, and it would then keep subtraction's cells apart from addition's, which could help keeping; I do not know which way it goes).

## 5. Cost, and what is not run

A run is 60,000 lines, about 3 to 4 CPU-minutes; two at a time for memory; 10 to 15 runs. **Not run:** a third task from `nano/life3`; replay; the life of 400,000 lines; subtraction first.
