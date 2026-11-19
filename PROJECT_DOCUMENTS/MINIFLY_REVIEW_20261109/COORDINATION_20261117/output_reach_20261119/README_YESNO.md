# The output end learning from yes/no alone, on recorded cell outputs (19 November 2026)

**Question** (the reviewer's message of 06:19 UTC): can the output end learn to emit the next byte from the cells near it when all it is told, after each emission, is yes or no? The rule may use only what one driver cell has: the outputs arriving at its connections, its own output, and the yes/no signal.

The output end is simulated alone, online, in one pass through the stream in order. The tissue runs as in `reach.py` and is not changed by the learning.

## Programs
- `yesno.py`: rules R1, R2, R3 and the supervised delta rule D.
- `yesno2.py`: rules A and B, added after R1 to R3 stayed below 1%.
- `reach_nbhd.py`: an offline ridge reader in which each byte's output uses only its own driver's neighbourhood.
- `make_table_yesno.py` builds `TABLE_YESNO.md`.
- Batches: `batch_yesno_64k.txt`, `batch_yesno2_64k.txt`, `batch_yesno2_256k.txt`, `batch_nbhd.txt`; logs in `out/`, results in `yesno/` and `reach/nbhd_*.json`.

## Set-up (choices of this program, not of the definition)

**The units.**
- Each of the 256 output lines' drivers is a linear unit.
- Its inputs are the set-(a) cells in its own neighbourhood, the driver included and feedback cells excluded: 32 to 106 cells, median 78, depending on the seed.
- It also has a constant input 1. Strengths start at 0.

**The cell inputs.**
- Each cell's output at the reading step is standardised by a running mean and deviation that the cell itself could hold: m += r(x − m), s² += r((x − m)² − s²), with r = 1/min(n, 1024) and n the bytes so far.
- There is no floor; a deviation of 0 gives an input of 0.

**Emission.**
- The most active unit emits its byte, with ties going to the lowest value.
- Every position has an emission. These are linear units, so the definition's level of 1/2 is not used.
- Yes if the emitted byte equals the next byte; no otherwise.

**Step size.**
- eta is divided by the unit's expected squared input length: number of cell inputs + 1 for the cells; 2, 3 or 4 for one, two or three one-hots.
- With strengths starting at 0, eta only scales the strengths in R1, R3 without random emitters, A and B. Their results do not depend on eta.

**Rules from yes/no.**
- R1: the emitter moves by +eta·u after yes and by −eta·u after no.
- R2: as R1, but with probability eps (1/64 or 1/16) the emitter is a driver drawn at random among the bytes seen so far.
- R3: as R2, and after a no every other driver moves by +(eta/46)·u_j.
- A: +eta·u after yes, −kappa·eta·u after no, with kappa 1/46 or 1/8; eps 0 or 1/16.
- B: eta·(r − rbar_e)·u, where r = 1 after yes and 0 after no, and rbar_e is the emitter's own running share of yes (rate 1/64, starting at 0); eps 0 or 1/16.

**The supervised ceiling, D.** Every driver moves by eta·(target − activity)·u/|u|², normalised by the squared length of its own input at that step; the target is 1 for the next byte and 0 otherwise.
- The first version stepped by eta/P instead and overflowed on the cell inputs, whose running standardisation gives large values in the first bytes. It was stopped and replaced before any result was kept.

**Inputs compared.** U and M cells (as in reach.py); oh1, the one-hot present byte; oh2, two one-hots; oh3, three one-hots.

**Scoring.**
- Share right in windows of 4,096 life positions.
- References in each window: the scorer's online ref_1, ref_2 and ref_3, the share of the window's most frequent byte, and the share of the end mark. These last two are the same at 6.3%: the end mark is the most frequent byte.
- "f" and "r1" in the table are the end of the first window whose share reached the end mark's share and ref_1 of that window.

## Results
See `TABLE_YESNO.md` and the section added below after the 262,144-byte runs.
