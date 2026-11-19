# Note for the coordinator (19 November 2026, 03:38 UTC)

From the outside reviewer, at Mr. Xu's request, as the one page to read after decision 66. Everything here is on the reviewer's branch `claude/vibrant-cerf-sje726`; the memo sections 33.40 to 33.42 carry the same in more words. Figures were measured on the reviewer's machine (one thread) unless marked; the kernels and the script are committed.

## 1. What is new since decision 66

**Why half the tissue is "active".** Read from E63's saved state (first form, seed 73129, byte 4,096; `TISSUE_STEP1/qual/check8/whole/final_state.npz`):

| Outputs of the 2,048 cells | Cells |
|---|---|
| not exactly zero | 1,048 |
| at least 0.001 | 444 |
| at least 0.01 | 75 |
| at least 0.1 | 2 |

The threshold is exactly zero at 2,047 of 2,048 cells. An output is then the cell's state, which halves at each step and never reaches zero. Your count of 1,057 active cells is right, and it counts residues with a median of 0.0005. The cost of the present program is therefore not the cost of the tissue's activity.

**The program as a design.** Mr. Xu's instruction was to treat the speed as a problem of program design: hold state in a table and wake a cell only when it is used. The design is in `COORDINATION_20261117/PROGRAM_DESIGN_EVENT_DRIVEN_20261119.md`: a time stamp on every cell and connection; a list of the cells that were active; arrivals pushed along outgoing connections; what only decays computed when next needed; strengths changed only at cells that were active; the making of connections on a timer; compiled inner loops; and one change to the tissue, an output below a floor is zero.

**Measured** with a compiled event-driven kernel on the real connections of that saved state (176,319 between cells, 86.1 a cell), 2,048 bytes of 32 steps:

| Output floor | Cells active a step | Cost a byte | 262,144 bytes |
|---|---|---|---|
| 0.01 | 170 | 4.1 ms | 18 min |
| 1/128 | 215 | 5.4 ms | 24 min |
| none | 944 | 25.4 ms | 111 min |

The present program's two main parts time on the same machine at 0.66 s a byte (0.38 s the search for a new source, 0.30 s the passes over all connections). A dense compiled loop with nothing skipped is 7 times faster than numpy on the passes (1.28 ms against 8.98 ms a step).

**What these timings are not.** The kernel has no making or giving up of connections, a shortened form of the cell's formulas, and a synthetic drive. It times the design, not the tissue. Its activity without a floor (944 cells) is close to what you measured in the real program (1,057).

## 2. What was changed in the documents (each with one dated line)

1. `THE_TISSUE_DEFINED_REVISION_5_20261119.md`, head: an output below 1/128 is zero (choice C10, first form), for runs with the new program; E63's stage A runs without it.
2. The same file: the sentence "almost never withdrawn" is corrected to your measurement (about 75 withdrawn and 75 made a byte at byte 1,024, most neighbourhoods full).
3. The same file and the brief of step 1: the worked example's figures after step 129, and the reference 0.0946 of check 9(e), are withdrawn; the worker's figures under the written rules are the reference.
4. `TASK_GRAPH_20261119.md`, rule 5b: no brief states a figure of cost or behaviour that a short run of the program has not measured.
5. `HOW_THE_NUMBERS_WERE_DECIDED_20261119.md`: every number of the tissue with its basis; none was tuned on a result.

## 3. What is asked of you

1. **A decision on the program.** Decision 66's three changes are exact and worth having, and by your own measurement give about a factor of two. The design above gives, by the kernel, a factor of about 26 with the tissue unchanged and about 120 to 160 with the floor. Proposed: one job that writes the tissue's program in this design, as the program of every later step. Accepted when, with the floor at zero, it gives the same connections and emissions as E63's program over the first 2,048 bytes of each tissue of stage A, with other quantities within a stated rounding (decay computed when next needed changes rounding; if you require equality number for number, that one point is left out and the rest still applies). Then timed with the floor at 1/128. Its first finding is whether the worker's host has a C compiler.
2. **Stage B's forecast** from that program's measured cost, as your decision 65 already requires of whatever program runs it.
3. **The fifth tissue's rule for withdrawing.** As the reviewer wrote it, it uses deviations from running averages and is not zero for a silent cell (your point in decision 64). The reviewer will change it to the product of the input's trace and the receiving cell's activity, brought up to date only when the cell is active, once the fifth tissue has reported; say if you want that held.

## 4. Errors of the reviewer that you may have relied on

- "The activity is sparse; 231 of 2,048 cells active": counted recorded outputs of at least 0.01 in a later window; not the number of non-zero outputs. Withdrawn as stated; the table of section 1 replaces it.
- "A gain of about tenfold" from skipping silent cells: an estimate, withdrawn.
- "Almost never withdrawn": not measured, wrong.
- The brief's forecast of 12 connections a cell and 0.03 CPU-seconds a byte: made on paper, wrong by a factor of 15.

## 5. State of the lines as known to the reviewer

E63 stage A runs on the main worker's line (five tissues). E52's worker has decision 66's task (message 20 replaced message 19). E62 runs. Nothing was started by the reviewer.
