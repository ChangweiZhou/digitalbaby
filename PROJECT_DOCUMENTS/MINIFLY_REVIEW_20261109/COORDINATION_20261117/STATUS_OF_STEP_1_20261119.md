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
| 12 | Making connections by the definition's rule (a cell below the target activity makes one every 128 steps) raises the number to 20 to 43 a cell in tissue 7 and loses the byte (S(1,0) 0.23 to 0.56), also when making is slowed or new connections start weak; about 3 million are made and withdrawn in 16,384 bytes and the tissue then depends on its past. The cause is the mean the condition uses (four bytes long). With the cell's own mean over about 500 bytes in its place and the level 1/4096: 3.1 to 5.5 connections a cell, S(d,0) at d = 1, 2, 4 at or above the tissue with no making, and above 0.1103 at d = 1, 2, 4, 8, 10 in 6 of 8 settings. | reviewer; main worker (2157849d); E52's worker (942bfabd, claude/round2-runner-sdga8c) | 1 (three seeds and a life are running) | the fault yes; the remedy not yet |
| 13 | A linear reader (a measuring device, told the right byte) on the 495 to 519 cells near the output lines predicts the next byte at 12.2 to 14.7% in the unchanging tissue with no control and 11.7 to 15.3% with the bounded gain: above the present byte alone (10.1 to 11.3%), about what two bytes give, far from three (27 to 29%). | E55's worker (5ddfbaad, claude/e55-output-reach-017dSEpW) | 3 | — |
| 14 | The output end told only yes or no. With equal steps up after yes and down after no it stays at 0.3 to 1.0% right. With a smaller step after no, or the cell's own share of yes as baseline, it learns from clean inputs (three bytes given: 17.7 to 20.4% at 262,144 bytes and rising, above the two-byte level 12.9 to 13.5%) but not from the cells near the drivers (4 to 6.5%), although a fitted reader finds the present byte in each driver's own neighbourhood (8.3 to 12.3%); a one-pass rule told the right byte does not extract it either. | E55's worker (7a129fd7) | 3 | — |
| 15 | Tissue 7 with the excitatory total at 1 and the inhibitory at 1/4, rate 1: S(d,0) = 0.92 to 0.95 / 0.60 to 0.66 / 0.32 to 0.39 at d = 1 / 2 / 4; largest output 0.52 to 0.57 at d = 8 and 0.35 to 0.50 at d = 11; cells with a largest output above 1/2: 47 to 60% at d = 4, 2 to 10% at d = 8, none at d = 11; no divergence. Tissue 7 (totals 3/4 and 1/4, rate 8) over a life of 262,144 bytes: settles, S moves by at most 0.04, no cell differs between two copies. | reviewer; main worker (2157849d and later, claude/e63-norm-timing) | 3 | yes |
| 16 | Why the far cells tell few bytes apart (variety.py): in tissue 7 every cell comes to answer to one byte, and the bytes preferred at a distance fall from 40 of 46 at d = 1 to 18 at d = 4 and 6 at d = 8. With conserved totals of at most 1 a cell can keep its amplitude only by passing on a single source, so the lines of single bytes die out with distance. Learned inhibition, a lower target of activity, a change divided by the sender's own mean activity, and a cap on one connection's share with a gain do not help. | reviewer | 1 | — |
| 17 | Inhibitory total equal to the excitatory total (1/2 and 1/2), unchanging tissue: the present byte is told apart to the far end (score with no floor and each cell divided by its deviation: 0.84 / 0.81 / 0.78 at d = 9 / 10 / 11, against 0.74 / 0.49 / 0.28 with 3/4 and 1/4); amplitude falls tenfold a connection. A gain on a tissue with loops destroys this at any bound from 4 to 32. | reviewer | 1 | E55's worker and the helper are on it |
| 18 | Withdrawal by order in time with strengths left shared (a connection is withdrawn when its sender is active after the receiver more than before; no strength is changed by the rule), then a gain that matches peaks, bound 8: the largest mean output is 0.39 to 0.92 at every distance to d = 13; score with no floor 0.98 / 0.82 / 0.72 / 0.63 / 0.51 / 0.42 / 0.36 / 0.30 / 0.28 / 0.20 / 0.13 at d = 1 to 11 with totals 3/4 and 1/4 (0.98 / 0.84 / 0.67 / 0.58 / 0.35 / 0.31 / 0.28 and 0.10 to 0.15 beyond with 1/2 and 1/2); divergence 0 at every checkpoint. Its fault: the cells that answer grow few with distance (70 of 141 at d = 8, 7 of 172 at d = 10), having lost their sources. | reviewer | 1 | the helper is on it |
| 19 | Delays in tissue 7 (a delay of its own on every connection between cells): the bytes one, two and three back are above their levels somewhere in the tissue (best 0.4 to 0.6 / 0.3 to 0.4 / 0.25 to 0.35) with the present byte read by the tool at d = 1, 2 and, for delays 0 to 16 or 32 on a quarter, at d = 4; among the cells near the output lines the bytes one and two back reach their levels in some settings and seeds only; divergence 0 in every row. | E57's worker (claude/e63-tissue-event-017dSE, TISSUE_DELAY) | 2 so far | — |
| 20 | Tissue 7 with the bounded gain (4) added: the mean output stays 0.014 to 0.032 out to d = 12 and the tool reads the present byte at the far cells on all seeds (0.13 to 0.23 at d = 8, 0.09 to 0.17 at d = 10 to 11), with no divergence. With the spread of time constants as well: present byte 0.93 to 0.97 / 0.68 to 0.75 / 0.29 to 0.43 at d = 1 / 2 / 4 and 0.10 to 0.15 at d = 8; the byte one back by the tool at d = 2 to 8 and two back at d = 4; one seed weak at d = 10 to 11 (0.040). | helper (4897e426) | 3 | — |

| 21 | Row 18 with making on the cell's long mean (level 1/4096, at most one making in 1,024 steps), totals 3/4 and 1/4, gain bound 8, withdrawal ratio 1. By the tool as it is: S(d,0) = 0.98 to 1.00 / 0.79 to 0.86 / 0.56 to 0.61 at d = 1 / 2 / 4 on three seeds, and 0.25 to 0.38 / 0.18 to 0.32 at d = 8 / 10 on the two seeds read so far (without making 0.23 to 0.27 / 0.07 to 0.19). Every cell at d = 8 and d = 10 answers (without making 41 to 70 of 141 to 177, and 7 to 21 of 172 to 193). Seed 73129: the largest output is 1.000 and 95 to 100% of the cells exceed 1/2 at every distance to d = 12; 17 to 23 different bytes are preferred at d = 8 to 11; a tenth of the cells at d = 9 to 12 are explained to 0.52 to 0.64 by the previous byte. 3.2 to 3.6 connections a cell; about 1,300 made and 10,400 withdrawn in 16,384 bytes; two copies do not diverge (0 at every checkpoint). Over 65,536 bytes (seed 73129): 0.975 / 0.812 / 0.642 at d = 1, 2, 4. Withdrawal ratios 2 and 4 leave loops and diverge (three seeds). | reviewer; E52's worker (claude/round2-runner-sdga8c, PREP/tissue6); helper (ratios) | 3 (d = 1, 2, 4); 2 (far cells) | yes: E52's worker reproduces the reviewer's figures on two seeds; third running |
| 22 | The merged program (the four parts as options, each checked step for step against its author's program: largest difference 0.0). Tissue 15 with making on the long mean over 65,536 bytes, three seeds: S(d,0) = 0.91 to 0.96 / 0.58 to 0.68 / 0.29 to 0.42 in every window, 3.1 to 3.5 connections a cell, 250 to 800 made and about 9,800 withdrawn, no cell differing between two copies; no byte before the present one. | main worker (b8e62c94, claude/e63-merged-tissue) | 3 | — |
| 23 | Making on the long mean in tissue 7, final rule (level 1/16384): S(d,0) above 0.1103 at d = 1, 2, 4, 8, 10 on three seeds; 3.8 to 4.3 connections a cell; over 65,536 bytes 3.7 to 3.8 with 250 made after the first quarter. | E52's worker (80401826) | 3 | — |
| 24 | Delays in tissue 7, all 30 runs, three seeds, divergence 0: with a delay of 0 to 32 steps on every connection the tool reads the present byte at d = 1, 2, 4 and the bytes one to three back are above their levels somewhere in the tissue on every seed; among the cells near the output lines the byte three back is above its level in no setting. | E57's worker (DELAY_TABLES.md, being committed) | 3 | — |

## 2a. What the table comes to (07:55 UTC)

The parts of a tissue that would meet step 1 have now each been measured, by different lines, and are being put into one program by
the main worker: (i) conserved total incoming strength with strength change by order in time (rows 7, 15: carries the present byte,
settles, never diverges); (ii) making of connections on the cell's long mean (row 12: 3 to 5 connections a cell, the byte kept);
(iii) memory by a spread of time constants (rows 11, 20) or by delays of the connections (rows 10, 19): two to three bytes back;
(iv) a bounded gain, safe once the tissue is directed (rows 18, 20). Since 07:30 row 21 gives, on the seeds read so far, the present byte by the tool as it is at 1, 2, 4, 8 and 10 connections,
with every far cell answering at an output above 1/2, the cells making and withdrawing their own three to four connections, and no
dependence on the past; it has no memory yet (time constants and delays are being put into it by three lines). What was missing
until then, on three seeds, was the present byte
told apart by the cells near the output lines at an output above 1/2: tissue 7 keeps the amplitude and loses the variety of bytes
(row 16), the balanced tissue keeps the variety and loses the amplitude (row 17), and row 18 is the first to keep both to about
nine connections, on one seed. And the output end does not yet learn from yes or no on what the tissue gives it (row 14).

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

## 5. Lines at work at the time of writing (07:25 UTC)

| Line | Task |
|---|---|
| main worker | one program with the four parts (base with totals 1 and 1/4; making on the long mean; time constants; delays); each checked against its author's figures; lives of 65,536 bytes on three seeds |
| helper | row 18 on three seeds: ratios of withdrawal, bounds of the gain, totals, then with its time constants |
| E57's worker | delays in tissue 7: the third seed, then 65,536 bytes for the best |
| E52's worker | the chosen rule of making on two more seeds and over a life of 65,536 bytes |
| E55's worker | the reader and the yes/no rules on the balanced tissue and on the tissue with totals 1 and 1/4; a driver that first removes what its inputs share |
| E62's worker | E62 (the earlier programme), about two hours left |
| reviewer | milder withdrawal of backward connections; whether making restores the far cells' sources |

## 6. Corrections of my own statements of tonight

- I told Mr. Xu that in tissue 7 "the present byte explains 0.92 or more of the variance at every distance" as if the byte were
  carried whole to the far cells. The figure is right for single cells; the cells together carry few bytes (row 9). The main
  worker's replication showed it.
- I told Mr. Xu at 06:20 that the variety was lost because frequent bytes win at every connection. The measurement (row 16)
  shows that with totals 1 and 1/4 the preferred bytes are not the frequent ones before ten connections; the loss is that every
  cell passes on one source.
- The readings of every earlier report of step 1 ("silent or constant beyond two connections", "S(4,0) at chance") were products
  of the floor of the reading and of the fall of amplitude, not of a loss of what the cells carry (row 2).

## 7. Where the material is

- Reviewer's program and results: `COORDINATION_20261117/reviewer_tissue_20261119/` on `claude/vibrant-cerf-sje726` (README.md, tissue.c, drive.py, diag.py, out/, diag/).
- Log of every task sent and report taken: `COORDINATION_20261117/LOG.md`.
- Workers' branches and commits: in the table of section 2.

Last change: 07:55 UTC, 19 November 2026.
