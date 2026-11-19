# The tissue as run on 19 November 2026 (tissues N0, N64, N256)

Written by the reviewer from the program (`reviewer_tissue_20261119/tissue.c`, `tis.py`), so that the rule can be read, criticised and
written again by someone who has not seen the program. It is not a new revision of the definition: revision 5 stands until Mr. Xu
decides. Where this text and the program differ, the program is what was run and this text is at fault.

What is kept from revision 5: the birth (2,048 cells with random coordinates in the unit square; neighbourhood radius 1/8; 256 input
lines on x = 0 and 256 output lines on x = 1; 64 feedback cells, silent in step 1; a quarter of the cells inhibitory; 8 starting
connections of strength 1/8 from random members of the neighbourhood), 32 steps a byte, the input line of the present byte held at 1
for the 32 steps, the output function, and the threshold's formula. What is new is marked NEW.

## 1. Quantities

Of a cell i: potential v_i (starts 0); threshold θ_i (starts 0); gain g_i (starts 1) NEW; time constant τ_i NEW; fast trace F_i of its
output (starts 0) NEW; long mean M_i of its output (starts 1/64) NEW; peak of its own output P_i and peak of what arrives Q_i (start 0)
NEW; the step of its last making (starts 0). Of an input line: its value, F and M as for a cell.
Of a connection from sender j to receiver i: strength w_ij; the step it was made; two sums, lead_ij and lag_ij (start 0) NEW.
σ_j is +1 for an excitatory cell and for an input line, −1 for an inhibitory cell.

## 2. One step k (φ = k mod 32)

z_j denotes the output of cell j at step k − 1, or the value of line j at step k. At φ = 0 the line of the present byte is set to 1
and every other line to 0.

For every cell i that is not a feedback cell, in the order of the cells' numbers:

1. Arrival. A_i = g_i Σ_j σ_j w_ij z_j over its connections.
2. Potential. v_i ← v_i + (A_i − v_i) / τ_i. (τ_i = 2 gives revision 5's formula.)
3. Output. x_i = min(1, max(0, v_i − θ_i)).
4. NEW. Order in time. With F_i and F_j the fast traces before this step: if x_i ≠ 0 or F_i ≠ 0, then for every connection of i from
   an excitatory cell or from an input line:
   lead_ij ← (1 − 1/16384) lead_ij + F_j x_i;  lag_ij ← (1 − 1/16384) lag_ij + F_i z_j.
   No strength is changed by this. When x_i = 0 and F_i = 0 nothing happens to the two sums, not even the decay.
5. NEW. Conservation. The strengths of i's connections from excitatory senders (lines included) are multiplied by one factor so that
   they sum to 3/4; those from inhibitory senders by one factor so that they sum to 1/4; a strength above 1 is set to 1. This is
   done at every step, after the arrival of that step (so the arrival of step 0 uses 1/8 each); connections from feedback cells
   count with their sign; a sign with no senders, or whose strengths sum to 0, is left alone.
6. Threshold. θ_i ← max(0, θ_i + (x_i − 1/8) / 128).
7. NEW. Gain. Let m_i be the largest z_j among i's excitatory senders (lines included).
   Q_i ← max(m_i, (1 − 1/16384) Q_i);  P_i ← max(x_i, (1 − 1/16384) P_i);
   g_i ← g_i (1 + (1/4096) (Q_i − P_i) / (Q_i + P_i + 1e-30)), then held between 1/64 and 8. (m_i = 0 for a cell with no
   excitatory sender; the new Q_i and P_i are used; the arrival of step 1 used the gain as it was before this.)

After all cells:

8. NEW. Traces, for every cell and input line j: F_j ← F_j + (z_j − F_j) / 4;  M_j ← M_j + (z_j − M_j) / 16384, with z_j as
   defined above (for a cell its output at step k − 1, not the output just computed). So the F used in step 4 of step k holds
   outputs up to step k − 2.
9. For every cell i that is not a feedback cell, in the order of the cells' numbers:
   (a) Withdrawal. Its connections from cells (never from lines) are looked at from the last of its list to the first. One that is at
       least 128 steps old is withdrawn if its strength is below 1/128; otherwise, NEW, if its sender is excitatory, it is at least
       4,096 steps old, lag_ij > 0 and lag_ij > lead_ij, it is withdrawn. (The last connection of the list takes the place of a
       withdrawn one.)
   (b) Making. NEW in its condition: if θ_i = 0, M_i < 1/4096, at least 1,024 steps have passed since i last made a connection (or
       since step 0), and some member of i's neighbourhood is not yet a sender to i, then one such member is chosen with equal chance
       and a connection from it is made with strength 1/8 and both sums 0. (The member may be an input line. θ_i and M_i are the
       values after steps 6 and 8 of this step. Age is counted from the step of making; the starting connections were made at step 0.)
10. x_i becomes the output of step k. The reading of a byte is the outputs at its step φ = 31.

No strength changes in any other way: only by conservation (5), withdrawal (9a) and making (9b). (In the program this is the
option `prune_only` of the rule named `timing`; the name is historical.)

The reading: the window is the last 4,096 bytes of the life; outputs below 0.01 are set to 0 before the readings tool. The distance of
a cell is counted on the connections as they are at the end of the window: 1 if it has a connection from an input line whose byte
occurs in the window; n if it is not yet numbered and has a connection from a cell at n − 1; feedback cells have none.

## 3. The three tissues

- N0: τ_i = 2 for every cell.
- N64: τ_i drawn for each cell with equal chance from {2, 4, 8, 16, 32, 64}.
- N256: from {2, 4, 8, 16, 32, 64, 128, 256}.

## 4. Numbers, and where each comes from

| Number | Value | Basis |
|---|---|---|
| totals of strength | 3/4 excitatory, 1/4 inhibitory | the starting state of revision 5 (6 excitatory and 2 inhibitory connections of 1/8); 1/2 and 1/2, 5/8 and 3/8, 1 and 1/4 were also run |
| ratio at which a connection is withdrawn | lag > 1 × lead | measured: at 1.25 and above loops remain and two copies of the tissue diverge (helper, three seeds) |
| age before that test | 4,096 steps (128 bytes) | arbitrary; not varied |
| rate of the two sums, of the peaks, of the long mean | 1/16384 a step (512 bytes) | arbitrary; one value for all three; not varied |
| time constant of the fast trace | 4 steps | arbitrary; 16 was run by the helper in another base with little change |
| bound of the gain | 8 | 4 and 16 were run (helper, three seeds, without making): 4 reads the far cells best but leaves some below 1/2; 16 is worse |
| rate of the gain | 1/4096 a step | taken over from the first control tried; not varied |
| level of the long mean below which a cell seeks a source | 1/4096 | 1/1024, 1/4096, 1/16384 run by E52's worker: all hold |
| steps between two makings of one cell | 1,024 | 128 and 1,024 run by E52's worker: both hold |
| strength of a new connection | 1/8 | revision 5; 1/64 also holds |
| time constants | powers of two up to 64 or 256 steps | a first trial; 16, 128 and a quarter of the cells at 16 were also run |
| target in the threshold's formula | 1/8 | revision 5; unchanged |

None of these was chosen by fitting the scores of step 1 over a grid; each was a first value or one of two or three tried, as stated.

## 5. For an independent program

The birth, the stream and the readings are those of E63's `tissue_step1.py` (functions `birth(seed)`, `stream(seed, n)`) and of the
helper's `readings.py`; copies are in `reviewer_tissue_20261119/deps/`. The draw of τ: numpy `default_rng(seed + 20261119).choice(values, size=2048)`.
The draw of a new sender: a generator x (64 bits, starting at 88172645463325252 + seed), advanced by x ^= x >> 12; x ^= x << 25;
x ^= x >> 27, giving x × 2685821657736338717 (mod 2^64); the member chosen is number (that value mod n) among the n members of i's
neighbourhood that are not senders, taken in the neighbourhood's order. Arithmetic is in double precision with numbers below the
smallest normal number treated as zero. A program written from this text should give, on seed 73129 and 16,384 bytes, the same
connections and the same outputs as `drive.py prm_ref_pk8 rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1
gmax=8 mk_long=1 a_make=1/4096 make_period=1024` up to rounding: 3.234 connections a cell over 2,048 cells, 1,219 made, 10,467
withdrawn with the program as it stands. (An earlier form of this sentence gave 3.24, 1,252 and 10,482: those were the figures of
the program before the potential was written as v + (A − v)/τ in place of v/2 + A/2, the same formula rounded otherwise; E62's
worker found the mismatch.)

Comparison with a program written from this text alone (E62's worker, 09:11 and 09:19 UTC). The two programs follow the same rule.
Over 2,048 bytes the outputs of every cell at the reading step of every byte are equal to the last bit of a 32-bit float, and both
programs hold 7,365 connections. They first differ at byte 2,376 (one cell, 7.5e-9), and by byte 3,072 the two lives are different
lives. The cause, tested: the reviewer's program was built with `-march=native`, which lets the compiler join a multiplication and
an addition into one operation rounded once; built with `-ffp-contract=off` and nothing else changed it equals the independent
program at every byte of 16,384 for every cell (6,638 connections, 3.241 a cell, 1,240 made, 10,474 withdrawn). So:

- "Double precision" above means: every multiplication and every addition rounded separately. From 09:31 UTC on 19 November the
  reviewer's drivers build with `-ffp-contract=off`, and a program written from this text should reproduce them exactly. Every
  figure of the reviewer's program before that time (and the figures 3.234, 1,219, 10,467 above) came from the other rounding and
  can be reproduced only from tissue.c with that compiler on that machine.
- A difference in the last bit of one sum, early in the life (while the cells still make and withdraw many connections), grows
  into a different life. The readings of the two lives agree within a few hundredths: S(d,0) at d = 1, 2, 4, 8, 10 is 0.979 /
  0.774 / 0.632 / 0.376 / 0.291 with separate rounding and 0.979 / 0.814 / 0.639 / 0.385 / 0.305 with joined rounding. A
  difference of 0.04 between two settings on one seed is therefore within what rounding alone gives, and the divergence test of
  section 6 (two copies at the END of a life) does not speak for the early life.

## 6. What has been measured on it (see STATUS_OF_STEP_1_20261119.md for all rows and who measured each)

- N256: all nine entries of roadmap step 1 above their levels on 15 of 15 seeds (E52's worker, 76b00600); N64 on 14 of 15.
- N0: the present byte above its level at 8 connections on 15 of 15 seeds, at 10 on 12, among the cells near the output lines on 11.
- Two copies of one tissue differing by 1e-9: no difference after 512 bytes on 43 of 45 tissues; 9e-10 on one; 0.29 on one (N0, seed
  20261211). So the rule does not exclude dependence on the past; it is rare in these runs.
- What it does not give: the present byte and the bytes before it together among the cells near the output lines.
