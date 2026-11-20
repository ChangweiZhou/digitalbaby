# Fourth plan: three pushes along the revised roadmap (20 November 2026)

Authorisation: Mr. Xu, 20 November: "Keep working, see how much you can push along the road map. Feel free to revise anything you like." System: that of PLAN_2.md and PLAN_3.md (two-digit sums, 256 cells in between, relay with learned return connections, eta1 1/4, eta2 1/16, lambda 1e-5), runner `cells2.py`, which must first reproduce `keep.py`'s plain run on seed 1 exactly. Written before any run. Development on seed 1; what holds there is run on the five reported seeds. Score: whole answer right on 500 never-shown pairs, mean of the last three readings at 50,000-line intervals, 400,000 lines.

## A. As a cell has it (roadmap R2)

- `fan=K`: a cell in between reads K of the 140 lines, drawn at birth. `rewire=T`: every T bytes each cell withdraws the weakest of its connections that are at least four periods old and makes one, at strength 0, to a line it does not read (Mr. Xu, WHAT_A_NEURON_HAS sections 8 and 19: a cell keeps adjusting which cells it is connected to, and cuts a connection that brings it nothing). The cell compares only its own strengths.
- `dale=1`: a strength from a line is never negative; each cell in between is excitatory or inhibitory (a quarter inhibitory, as at the tissue's birth) and its strengths onto the answering cell keep that sign.
- Predictions: A1 (0.65) with 32 lines a cell and rewiring, at least 60% (fixed 32 lines gave 25% in the earlier system); A2 (0.5) with 16 lines and rewiring at least 40%; A3 (0.6) `dale=1` at least 60%; A4 (0.45) both together at 32 lines at least 50%.

## B. The relay over two steps (the roadmap's step 3)

Lines, then cells A, then cells B, then the answering cell. The answering cell's mismatch reaches B over learned return connections; B passes on what it received, times its own sensitivity, to A over a second set of learned return connections. Sizes 128 + 128 and 256 + 256.
- Predictions: B1 (0.6) 128 + 128 reaches at least 60%; B2 (0.3) some two-step system has the tens digit right more often than the one-step system on the same seed.

## C. Keeping, in small parts (roadmap R5)

- *Where the loss is written.* After addition then subtraction (`plain`), addition is scored with the answering cell's strengths put back to their state after addition and the cells in between left as subtraction made them, and the other way round.
- *How little rehearsal.* One line of addition in 32 and in 128 during subtraction (one in 8 kept 71%).
- *How fast it returns.* After subtraction, addition is taught again; lines needed to reach 70% against the lines needed at first.
- Predictions: C1 (0.6) the loss is mostly in the answering cell (putting it back restores at least half); C2 (0.5) one line in 32 keeps at least 50%; C3 (0.8) addition returns in at most a tenth of the lines it first needed.

## Entered on 20 November after the development runs (seed 1), while the reported seeds were running and before any of them was read

Development seed, never-shown addition, mean of the last three readings (`res/c2_d_*.json`). In this runner the one-step system that reads every line gives 81.5 on this seed (by place at the end: 91, 91, 100).

- **A.** 32 lines a cell, fixed: 21.7. 32 lines with rewiring every 4,096 bytes: 43.4 (52 at the last reading, still rising). 16 lines with rewiring: 10.9. 64 lines with rewiring: 76.5. One sign for each cell: 66.6. One sign, 32 lines and rewiring: 41.8. On this seed A1, A2 and A4 fail and A3 holds.
- **B.** 128 + 128: 68.9 (by place 82, 82, 100). 256 + 256: 84.3 (94, 89, 100). B1 holds on this seed; B2 fails on it (tens digit 82 and 89 against 91).
- **C.** After subtraction, addition is at 1.4%; 49.8% with the answering cell put back; 3.6% with the cells in between put back. One addition line in 32 keeps 12%, one in 128 keeps 2%. Taught again, addition is at 53% after 2,000 lines and 71% after 6,000; at first it reached 71% after 50,000.

**Reported seeds.** Seven arms, queued before these lines were written: 32 lines fixed; 32 lines with rewiring; 64 lines with rewiring; one sign; 128 + 128; addition, subtraction, addition again (with the two put-back scores); one addition line in 32. Arms that failed on the development seed are run all the same, so that their size is known on five seeds. Not run on the reported seeds: 16 lines; one sign with 32 lines; 256 + 256.

**Added after the plan** (no registered prediction; each is named as an addition wherever it is reported):

- `r_plain`: the one-step system on the five reported seeds in this runner, which B2's comparison by place needs.
- `lam2=0`: no fading of the answering cell's strengths in the second phase, to see whether the fading by itself is what loses addition.
- `ans2=1`, with `freeze=1` or `stiff1=k`: two answering cells, the program choosing by the line's operator which one answers and learns; the cells in between free, unchanged after the first phase, or stiffening.
- `reserve=128`, alone and with one addition line in 32 or in 128: 128 further cells in between, silent and unchanged in the first phase; in the second phase they alone change, and everything the first phase wrote is left as it is. A control in which the program decides where the second task is written. Run on the five reported seeds alone and with one line in 32.

*An error of mine, found before any of its runs was read.* In the first form of `reserve` the answering cell's strengths from the first phase's cells went on fading through the second phase, at a rate that leaves nothing of them after 200,000 lines whatever is learned. The runs were stopped and discarded and the option corrected (those strengths, and the answering cell's constants, are now left untouched in the second phase).
