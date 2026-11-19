# The program of the tissue as a design problem: work only where something happens (19 November 2026, 03:34 UTC)

By the outside reviewer, at Mr. Xu's instruction to treat the speed of the program as a problem of program design and solve it. Every figure below was measured on this machine (one thread, gcc -O2, Python only to load the data); the kernels and the timing script are in `event_driven_kernel_20261119/`. No figure is an estimate unless it says so.

## 1. What was wrong

The program of E63 computes, at each of 32 steps for every byte, every connection of every cell. Local learning was chosen because locality makes activity sparse and sparse activity makes the work small; the program made no use of this, and the reviewer's brief did not ask it to.

Two facts were measured on E63's own saved state (first form, seed 73129, byte 4,096; `TISSUE_STEP1/qual/check8/whole/final_state.npz` on the worker's branch):

| Outputs of the 2,048 cells at one step | Cells |
|---|---|
| not exactly zero | 1,048 |
| at least 0.001 | 444 |
| at least 0.01 | 75 |
| at least 0.1 | 2 |

The threshold of 2,047 of the 2,048 cells is exactly zero, so a cell's output is its state, and a state that decays by half at each step never reaches zero. **Half the tissue counts as active only because of such residues**, with a median of 0.0005. The coordinator's count of 1,057 active cells and the reviewer's of 231 (read from records that store values below 0.01 as zero) are both right; they count different things.

## 2. The design

1. **Tables with a time stamp.** Every cell and every connection holds the step at which it was last brought up to date. With cells numbered 0 to N − 1 these tables are arrays addressed by the number; where cells are not densely numbered, or the tissue grows, the same design uses a hash table from the cell's identifier to its record.
2. **A cell is woken only when something reaches it.** The program keeps the list of cells whose output at the previous step was not zero. For each of them it follows the cell's outgoing connections, adds to the arrival of each receiving cell, and notes that cell as reached. Only reached cells, and cells that were active, are computed at this step.
3. **What only decays is computed when it is next needed.** A state, a prediction, a trace or a running average of a cell that receives nothing changes by a known factor at each step (the threshold by a known amount). When the cell or the connection is next visited, the factor for the steps missed is taken from a table. Nothing is done at the steps in between.
4. **A strength changes only where its rule is not zero.** The change is a product that contains the receiving cell's activity at the previous step; the program visits the incoming connections of the cells that were active and no others. The test for giving a connection up is made where a strength has just changed.
5. **Making a connection is a timer, not a search at every step.** A cell is looked at when its period has passed; a cell whose neighbourhood is full is taken out until one of its connections is given up.
6. **An output below a floor is zero.** This is the one change to the tissue itself, and it is what makes points 2 to 4 effective: a cell either sends something or sends nothing. The floor is a number fixed by us; 1/128 is the grid value nearest to the 0.01 below which the measurement already stores zero.
7. **The inner loops are compiled.** They are short loops over connections with indirect addressing, which numpy does poorly.

Points 1 to 5 and 7 change no figure of the tissue beyond rounding in point 3. Point 6 changes the tissue and is entered in the definition as such.

## 3. What was measured

**Dense, compiled, nothing skipped** (`dense_compiled_kernel.c`; 186,368 connections; outputs equal to the numpy form's): 1.28 ms a step against 8.98 ms in numpy, a factor of 7.

**Event-driven, compiled** (`ev.c`, `ev_time.py`), on the real connections between cells of E63's saved state (176,319, 86.1 a cell), 2,048 bytes of 32 steps, five cells of the input end driven for each byte:

| Output floor | Cells active a step | Connections visited a step, as a share of two passes over all | Cost a byte | 262,144 bytes |
|---|---|---|---|---|
| 0.01 | 170 | 8.5% | 4.1 ms | 18 min |
| 1/128 | 215 | 10.7% | 5.4 ms | 24 min |
| 0.001 | 350 | 17.6% | 9.0 ms | 39 min |
| none | 944 | 46.0% | 25.4 ms | 111 min |

For comparison, the two parts of the present program timed here at 0.66 s a byte together (0.38 s for the search of point 5, 0.30 s for the passes over all connections), and E63's worker measured 0.474 s a byte on its host. With a floor of 0.01 the event-driven kernel is about 160 times faster here than those two parts; with no floor, about 26 times.

**What the kernel does not contain,** so that these are timings of the design and not of the tissue: the making and giving up of connections (point 5 is designed, not timed), the coincidence form, the fifth tissue's rule, the recording, and the true drive from the input lines; its cell dynamics are a shortened form of the definition's. The level of activity it reached without a floor (944 cells) is close to what the coordinator measured in the real program (1,057), which is the reason to trust its timings as to order.

## 4. What follows

1. One tissue of stage B (262,144 bytes) would cost tens of minutes and not 34 CPU-hours. Stage B, and the steps after it, become cheap enough to run several forms side by side.
2. The output floor is entered in the definition (choice C10) as a change of the first form, dated today, with 1/128 as its first value. It has to be run: it changes which cells are active and therefore what is learned.
3. The program is to be written once in this design, as the program of every later step, and accepted against E63's present program on the first 2,048 bytes of each tissue with the floor set to zero (connections and emissions identical; other quantities within a stated rounding).
4. The fifth tissue's rule for withdrawing, as the reviewer wrote it, uses deviations from running averages and is not zero for a silent cell. In this design it is changed to a form that is: the product of the input's trace and the receiving cell's activity, brought up to date only when the cell is active. That is entered in the definition when the fifth tissue has reported.

## 5. Second kernel (03:43 UTC): 262,144 bytes in about one minute

Mr. Xu asked for another factor of ten, and asked two questions: is there a number that makes little sense, and is there an improvement of the algorithm. Both have answers, and the second kernel (`ev2.c`, `ev2_time.py`) measures them on the same real connections.

**Numbers that make little sense.**

1. *The input is held on, and everything recomputed, for 32 steps a byte.* Nothing new arrives after the first step of a byte; the tissue settles within a few steps and the remaining steps repeat it.
2. *A target of activity of 1/8.* No cell reaches it; every threshold but one is zero, so the control does nothing except drive the making of connections until the neighbourhood is full.
3. *About 90 connections a cell,* the consequence of 2 and of a neighbourhood of 94 members. Every active cell costs 90 visits out and 90 in.
4. *Strengths changed at every one of the 32 steps,* from a mismatch that changes slowly.
5. *No floor on an output* (section 1).

**Changes of the algorithm.**

1. *Only changes are sent.* Each cell keeps its arrival from step to step. A sending cell passes on the difference between its output now and the output it last sent, and only when that difference exceeds a tolerance. A cell is worked on only while its arrival or its state is still moving. Once the tissue has settled within a byte, the remaining steps cost nothing; the number of steps a byte stops mattering.
2. *One trace for a sending cell,* not one for each of its connections: 2,048 numbers in place of 176,000, and nothing to update along connections.
3. *Strengths are changed once a byte,* from a sum that each receiving cell keeps of its mismatch times its activity; only cells with such a sum visit their incoming connections, and they skip inputs whose trace is zero.

**Data structures used:** connections stored once, sorted by receiving cell, with a second index by sending cell; 16-bit cell numbers and 32-bit strengths (6 bytes a connection); a work list of cells still moving, with a membership flag for each cell; a list of cells that have something to learn this byte; the arrival, state, prediction, threshold, last output sent, learning sum and trace of each cell in arrays addressed by the cell's number.

**Measured** (4,096 bytes, output floor 1/128, tolerance 1/1024, five cells of the input end driven a byte, one thread):

| Setting | Connections a cell | Cell updates a byte | Connection visits a byte | Cost a byte | 262,144 bytes |
|---|---|---|---|---|---|
| 32 steps a byte, all connections | 86.1 | 6,846 | 31,197 | 0.271 ms | 1.2 min |
| 8 steps a byte | 86.1 | 6,193 | 63,981 | 0.444 ms | 1.9 min |
| 32 steps, the 32 strongest connections of each cell | 30.8 | 10,653 | 36,126 | 0.424 ms | 1.9 min |
| 32 steps, the 16 strongest | 15.5 | 3,133 | 5,920 | 0.090 ms | 0.4 min |
| 32 steps, tolerance 1/128 | 86.1 | 4,406 | 27,221 | 0.211 ms | 0.9 min |
| 32 steps, 20 cells driven a byte | 86.1 | 17,493 | 78,954 | 0.706 ms | 3.1 min |

The first kernel visited about 30,000 connections at every step, 960,000 a byte; this one visits about 31,000 a byte. Against the present program's two main parts on this machine (0.66 s a byte) the first row is about 2,400 times faster.

**What these changes do to the tissue.** Change 2 alters only the trace of a connection in the steps after it is made. Changes 1 and 3 are approximations: an output that moves by less than the tolerance is not passed on, and strengths follow the sum over a byte and not each step. They are to be accepted by running the full program in both ways on one tissue and comparing what is learned, not by equality of numbers. As before, the kernel's cell formulas are shortened and it does not make or give up connections, so these are timings of the design; the level of activity in it (about 54 cells sending at the end of a byte) is lower than in the first kernel and has to be checked in the full program.

## 6. More than one core, and a graphics processor (04:07 UTC)

Mr. Xu asked whether several cores or a graphics processor can make it faster still.

**Measured** (this machine, 4 cores, no graphics processor; `par.py`): independent tissues run side by side, one process each, scale with the number of cores: one tissue of 8,192 bytes takes 3.7 s of wall time including the start of Python, and four at once take 3.7 s, four times the bytes a second. An experiment of the roadmap is several independent tissues (seeds and arms), so this gain is available at once: the nine tissues of a full run of step 1 fit in three rounds of about 1.2 minutes on four cores. With `-O3 -march=native` in place of `-O2` the kernel costs 0.241 ms a byte against 0.284, 15 in 100 less.

**Not measured, reasoned.** Inside one tissue of this size there is little for further cores to do: a step is about 8 microseconds of work (0.27 ms a byte over 32 steps), which is of the order of what it costs to hand work to another thread. For the same reason a graphics processor does not help one small tissue: starting one of its operations costs tens of microseconds, more than the whole step. Both would pay in two cases that the roadmap reaches later: a tissue of hundreds of thousands of cells, where one step is large; and very many tissues at once (hundreds of seeds or settings of a number), computed as one batch. No machine of this project has a graphics processor that the reviewer knows of, so no figure is given.
