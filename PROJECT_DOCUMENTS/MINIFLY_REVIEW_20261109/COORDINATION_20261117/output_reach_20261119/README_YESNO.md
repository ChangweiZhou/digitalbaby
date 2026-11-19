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

Share right in the last window of 4,096 bytes, %, range over the three seeds, for the best setting of each rule family on each seed. Every window, setting and seed is in `TABLE_YESNO.md` and `yesno/*.json`.

| Input | R1, R2, R3 (65,536) | A (65,536) | B, eps 1/16 (65,536) | D, supervised (65,536) | A (262,144) | B, eps 1/16 (262,144) |
|---|---|---|---|---|---|---|
| One-hot present byte | 0.5–0.7 | 7.1–7.3 | 8.9–9.2 | 8.4–10.1 | 8.5–9.2 | 8.3–8.9 |
| Two one-hots | 0.4–1.0 | 8.0–8.7 | 6.7–10.9 | 11.6–12.6 | not run | not run |
| Three one-hots | not run | 12.1–13.7 | 13.8–16.2 | not run | 11.8–13.5 | 17.7–20.4 |
| U cells near own driver | 0.3–0.9 | 5.2–6.5 | 1.6–4.0 | 4.5–6.6 | 5.5–6.5 | 2.1–4.1 |
| M cells near own driver | 0.3–0.9 | 4.3–5.8 | 2.8–5.6 | 4.9–5.2 | 4.4–5.9 | 3.1–4.4 |
| End mark's share (the most frequent byte) | 6.3 | | | | 6.2 | |
| ref_1 / ref_2 / ref_3 | 10.0–10.8 / 11.9–12.5 / 25.6–26.0 | | | | 10.5–10.9 / 12.9–13.5 / 27.0–28.2 | |

**Offline check of the inputs** (`reach_nbhd.py`, 65,536 bytes, fitted on the first three quarters, scored on the last quarter).

| Tissue | Own neighbourhood (8.3–12.3 overall) | All of set (a) (12.3–15.3 overall) |
|---|---|---|
| U | 8.8–10.3 | 12.8–14.7 |
| M | 8.3–12.3 | 12.3–15.3 |

The reference on the same positions: ref_1 10.6, ref_2 12.1–12.3, end mark 6.2–6.3. The all-of-set-(a) figures agree with `reach.py`'s 65,536-byte runs.

**What the runs measured.**
1. **R1, R2 and R3 as specified** (+eta after yes, −eta after no) stay at 0.3–1.0% for every input, eta, eps and seed through 65,536 bytes. That is below the end mark's share. They were not extended, because nothing rose.
   - With equal steps, an emitter's expected change in a context is eta(2p − 1), where p is the share of that context after which its byte comes. It gains only where p > 1/2.
2. **With a smaller step after no (A, kappa 1/8) or an own baseline (B)**, yes/no alone takes the one-hot present byte to 7–9%. That is above the end mark's share (passed after 4,096 to 69,632 bytes) and below ref_1 in the last window.
   - Some windows of B reach ref_1, after 32,768 bytes on two seeds.
   - On three one-hots, B reaches 17.7–20.4% by 262,144 bytes and is still rising. That is above ref_2 (passed after 12,288 to 28,672 bytes) and below ref_3.
3. **On the cells near each driver, no rule from yes/no reaches the present-byte level** (ref_1) at 65,536 or 262,144 bytes. Only one last window reaches the end mark's share: A, kappa 1/46, U, seed 116089, 6.5 against 6.2. The supervised delta rule reaches only 4.5–6.6.
   - Yet a ridge reader fitted offline on the same per-driver inputs reaches 8.3–12.3%, around ref_1.
   - So the information is in each driver's neighbourhood. The one-pass rules on these inputs do not extract it in this many bytes; that includes the supervised one, with running standardisation.
4. **Settings that never learn.**
   - B with eps 0 never changes: the baseline starts at 0, so a no gives a step of 0.
   - A with kappa 1/46 gives the same windows on oh1, oh2 and oh3 of a seed. That is consistent with the emitter settling on one byte early; I have not checked it.

**Not measured.**
- Rules with exploration other than a fixed eps.
- Standardisation with other rates.
- Inputs beyond the driver's own neighbourhood for the online rules.
- Feedback reaching the drivers through feedback cells, as the definition has it. Here the yes/no reaches the emitting unit directly.

**Process notes.**
- The first `yesno.py` batch was stopped once to change the step of D (see above); no result of the stopped runs is kept.
- One wait command was moved to the background past two minutes and was stopped; it only waited.
