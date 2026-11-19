# Status of roadmap step 1 (the tissue reads the stream and stays informative)

Written by the reviewer for Mr. Xu, 19 November 2026, 06:40 UTC, while he rests. It is changed as results arrive; the last change
is dated at the foot. Nothing in it has been verified by the coordinator, who on Mr. Xu's word is not brought in until he returns.
Every figure names who measured it and on how many seeds. Scores S(d, lag) are the nearest-centroid scores of the helper's readings
tool for the cells d connections from the input lines; chance is 0.022; the level for the present byte is 0.1103; "the tool as it
is" sets outputs below 0.01 to zero.

## 1. In one paragraph

Step 1 is not reached by any tissue of the definition as written (revision 5): this is now closed on 48 settings and on nine
tissues of full length. The cause was found and it is not what the briefs assumed. In the tissue whose strengths do not change, the
byte is not lost on its way through the tissue; its amplitude falls about 4.5-fold a connection, below anything a reading or a cell
could use. Two changes, each a quantity of one cell or of one connection, remove most of the fault: (i) a cell holds the total
strength of its incoming connections constant and (ii) a connection grows when its sender is active before the receiver and falls
when after. With them the present byte is read, by the tool as it is, at 1, 2 and 4 connections on three seeds, and two copies of the
tissue never diverge. A spread of time constants among the cells then carries the bytes one and two back at 1 to 6 connections, on
three seeds. Three things remain open: the variety of bytes carried by the cells far from the input lines, the making of new
connections, and the learning of the output end from yes or no.

## 2. What was measured

| # | Finding | Measured by | Seeds | Repeated by another line |
|---|---|---|---|---|
| 1 | The rules of the definition (first form, coincidence; floor, target of activity 1/8 to 1/128, withdrawal by use): S(4,0) at most 0.076 in 48 tissues; no score at lag 1 or 2 above its level. At full length (262,144 bytes) the same, and S(1,0) falls over the life. | E52's worker (d5e4a175, claude/round2-runner-sdga8c); helper (840511e5, claude/prep-step1-full) | 1 (grid); 3 (full length) | — |
| 2 | Unchanging tissue, no control: the present byte explains 1.000 of a cell's output variance to 6 connections and 0.96 at 8; the mean output falls 0.0046, 0.00088, 0.00022, 0.00005 at 1 to 4 connections. With no floor and each cell divided by its own deviation, S(d,0) is 0.74 to 0.96 out to 8 connections, and the previous byte is carried at 9 to 11. | reviewer; E57's worker (f4de9cb79); helper (4bda7aa1) | 3 | yes, twice |
| 3 | A gain that drives every cell to a fixed activity makes cells active on their own; the threshold rises above zero and cuts off the part of the potential that varies with the byte; two copies differing by 1e-9 differ by 1.0 after 512 bytes. | reviewer; E57's worker | 1 | yes |
| 4 | A gain by ratio (a cell passes on the mean activity of its sources) gives S(d,0) above the level at d = 1, 2, 4 (0.98 to 1.00 / 0.81 to 0.85 / 0.18 to 0.24) but depends on the tissue's past unless the gain is bounded (at 4: divergence 0). | reviewer; E52's worker (7002c939) | 3 | yes |
| 5 | All-or-none output under the definition's rule of the threshold: activity holds at every distance and does not depend on the past, but does not carry the byte (S(1,0) at most 0.077), with or without a floor of the threshold. With a threshold held fixed the cells one connection away carry the byte (0.198) and beyond them the tissue either fills or stays silent. Closed. | main worker (c312c8f2, claude/e63-allornone) | 3 | — |
| 6 | Adaptation of the cell and depression of sending, time constants 32 to 2,048 steps: neither carries the previous byte. Closed. | E55's worker (f8a24ee1, claude/e55-slow-variables-017dSEpW) | 1 (3 for the best) | — |
| 7 | Total incoming strength conserved (excitatory 3/4, inhibitory 1/4), strengths changing by order in time, connections from the input lines changing too, no connection made, no gain: by the tool as it is S(d,0) = 0.93 to 0.98 / 0.69 to 0.78 / 0.38 to 0.48 / 0.13 to 0.21 at d = 1 / 2 / 4 / 8; 4.9 to 5.2 connections a cell; no cell differs between two copies after 16 to 512 bytes in 21 runs. At rate 8: above 0.1103 at every distance from 1 to 11. | reviewer; main worker (claude/e63-norm-timing) | 3 | yes, exactly |
| 8 | In tissue 7 the largest output falls about 1.4-fold a connection (0.70 at d = 1, 0.01 to 0.03 at d = 11). With the excitatory total at 7/8 it is 0.12 at d = 11; with the excitatory total at 1 and the inhibitory at 1/4 the mean output does not fall with distance (0.0216 at d = 1, 0.0137 at d = 11). At rate 8 with these totals, or with no inhibition, every cell carries the same signal and the byte is lost. | reviewer (totals 1 and 1/4: one seed so far); main worker (7/8) | 1 to 3 | in progress |
| 9 | What is lost in tissue 7 with distance is the variety of bytes: each far cell is well explained by the byte (0.90 or more of its variance), but the cells together tell few bytes apart (S at d = 8 to 11 with no floor: 0.17 to 0.28 at rate 1, 0.12 to 0.18 at rate 8, 0.23 to 0.53 at rate 1/8). The slower the rate, the more variety and the less amplitude. | main worker; reviewer | 3 | — |
| 10 | Memory by delay: on the bases that do not depend on their past, only a delay near one byte (32 steps) on a quarter of the connections carries the previous byte (S(d,1) = 0.145 / 0.651 / 0.148 at d = 1 / 2 / 4); delays of 8 or 16 steps do not. | E57's worker (cb554a546 and later, claude/e63-tissue-event-017dSE) | 1 | — |
| 11 | Memory by a spread of time constants of the potential (each cell its own, from 2 to 64 steps): in the unchanging tissue with bounded gain the byte one back is carried at d = 1 to 4 and the byte two back at d = 2 to 4, at the cost of the present byte beyond d = 4; divergence 0. In tissue 7 the tool as it is reads the present byte at d = 1, 2, 4 (0.93 to 0.96 / 0.63 to 0.67 / 0.31 to 0.43), the byte one back at d = 1 to 6 (0.18 to 0.41) and the byte two back at d = 2 to 6 (0.14 to 0.23), each above what the nearer bytes give plus 0.02. | helper (8c02831b and later) | 3 | — |
| 12 | Making connections by the definition's rule (a cell below the target activity makes one every 128 steps) raises the number to 20 to 43 a cell in tissue 7 and loses the byte (S(1,0) 0.23 to 0.56), also when making is slowed or new connections start weak. | reviewer | 1 | E52's worker is on it |
| 13 | A linear reader (a measuring device, told the right byte) on the 495 to 519 cells near the output lines predicts the next byte at 12.2 to 14.7% in the unchanging tissue with no control and 11.7 to 15.3% with the bounded gain: above the present byte alone (10.1 to 11.3%), about what two bytes give, far from three (27 to 29%). | E55's worker (5ddfbaad, claude/e55-output-reach-017dSEpW) | 3 | — |
| 14 | The output end told only yes or no, with a step up after yes and down after no at the emitting cell, stays at 0.2 to 1.0% right over 65,536 bytes. | E55's worker (in progress) | 1 | — |

## 3. Against step 1 as the roadmap states it

Step 1 asks for the present byte and the bytes one and two back at 1, 2 and 4 connections. Row 11 (tissue 7 with a spread of time
constants) gives, on three seeds and by the tool as it is: the present byte at 1, 2, 4; the byte one back at 1, 2, 4; the byte two
back at 2 and 4 but not at 1. It is the first tissue to come this near, and it does not depend on its own past. It is not yet a
result: it comes from one line, from 16,384 bytes, with no connection made, and its far cells (8 to 11 connections, where the output
lines are) read the present byte at 0.13 to 0.21 at most.

## 4. What I propose to change in the definition (for Mr. Xu to decide)

Each is a quantity of one cell or of the two ends of one connection. None is tuned beyond the values named; the values are first
trials and are marked so in the workers' tables.

1. A cell conserves the total strength of its incoming connections (excitatory and inhibitory apart). It replaces the starting
   strength 1/8 as the thing that bounds a cell's input, and it bounds the number of connections a cell can hold at useful strength.
2. Strength changes by the order in time of the two ends' activity, for the connections from the input lines as well (ruling 10:
   the input end learns). It replaces the first form and the coincidence rule, both of which lose the byte.
3. Each cell has its own time constant of the potential, drawn at birth.
4. The making of connections needs a new condition; the present one is wrong for cells that are rightly active only rarely.
5. The readings of step 1: the floor at 0.01 and the fixed distances 1, 2, 4 hid what the tissue carried. I propose to read every
   distance, with and without the floor, and to ask the bytes back of the tissue as a whole and of the cells near the output lines.

## 5. Lines at work at the time of writing

| Line | Task |
|---|---|
| main worker | tissue 7: totals 1 and 1/4 on three seeds; lives to 262,144 bytes; connections made |
| helper | tissue 7 with the spread of time constants: other spreads, the base with totals 1 and 1/4, 65,536 bytes |
| E57's worker | delays as a property of each connection, on the bases that do not depend on their past |
| E52's worker | a rule of making and withdrawing connections under which tissue 7 keeps the byte |
| E55's worker | the output end learning from yes or no on recorded outputs |
| E62's worker | E62 (the earlier programme), about two and a half hours left |
| reviewer | keeping the variety of bytes at a distance (target of activity near one byte's frequency; a change divided by the sender's own mean activity) |

## 6. Corrections of my own statements of tonight

- I told Mr. Xu that in tissue 7 "the present byte explains 0.92 or more of the variance at every distance" as if the byte were
  carried whole to the far cells. The figure is right for single cells; the cells together carry few bytes (row 9). The main
  worker's replication showed it.
- The readings of every earlier report of step 1 ("silent or constant beyond two connections", "S(4,0) at chance") were products
  of the floor of the reading and of the fall of amplitude, not of a loss of what the cells carry (row 2).

## 7. Where the material is

- Reviewer's program and results: `COORDINATION_20261117/reviewer_tissue_20261119/` on `claude/vibrant-cerf-sje726` (README.md, tissue.c, drive.py, diag.py, out/, diag/).
- Log of every task sent and report taken: `COORDINATION_20261117/LOG.md`.
- Workers' branches and commits: in the table of section 2.

Last change: 06:40 UTC, 19 November 2026.
