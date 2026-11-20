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
