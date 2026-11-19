# How the numbers of the tissue were decided (19 November 2026, 01:24 UTC)

By the outside reviewer, at Mr. Xu's request, so that later decisions are not built on a misreading of where a number came from. A *number fixed by us* (in other fields called a hyperparameter) is a quantity of the tissue that neither the stream nor the cells' own changes determine and that the experimenters therefore chose. The tissue is the one defined in `THE_TISSUE_DEFINED_REVISION_5_20261119.md`; its section 16 lists these numbers, and this note says how each was arrived at.

## 1. The plain statement

**No number of the tissue has been tuned on a result, because no tissue had been run when they were chosen.** Every value below was chosen on paper between 18 November, 22:25 UTC, and 19 November, 01:20 UTC, by the agents who wrote and read the definition and by the reviewer. None is Mr. Xu's, except where marked. They are starting values of a search, not findings. A reader who meets one of them later should not take it as established.

## 2. What Mr. Xu laid down (these are not ours)

| What | His words (section of `WHAT_A_NEURON_HAS_20261118.md`) |
|---|---|
| The fewer numbers fixed by us the better; a cell should if possible decide its own number of connections | 12 |
| A number that must be fixed is tried on the grid 2, 4, 8, ..., 128 | 12 |
| Each cell has a coordinate drawn at random | 15 |
| The feedback for predicting the next byte is yes or no; richer forms are allowed later | 13, 14 |
| Dedicated cells receive the feedback; how it is passed on is learned | 16 |
| A cell makes connections and withdraws those that bring it noise | 19 |

## 3. The rules by which a value was chosen

They were applied in this order.

1. **A living cell must be able to do it.** A form that no living cell has was struck out before any number was considered.
2. **The simplest form first.** For each choice the form with the fewest parts was taken as the first form.
3. **A value on the grid.** A count is a power of two from 2 to 128 (numbers of cells go beyond 128 by further powers of two); a fraction or a rate is the reciprocal of such a number.
4. **Consistency on paper.** Where arithmetic showed that a first value made a step of the roadmap impossible before any run, the value was changed, and the change is recorded below. This is the only way in which a value has been revised so far.
5. **Where nothing decided between grid values, a middle value was taken** and the number is marked below as *arbitrary*.

## 4. The numbers, one by one

*Basis* is one of: **ruling** (Mr. Xu's word); **biology** (a textbook figure); **arithmetic** (a calculation on paper or with the standard library, with what it secured); **arbitrary** (a grid value taken with no reason stronger than rule 5). Where the reviewer has not found the writer's reason on record, it says so.

| Number | Value | Basis | How it was arrived at | Changed since first written? |
|---|---|---|---|---|
| Cells | 2,048 | arithmetic | Large enough that, with the radius below, cells exist at every number of connections from 1 to 10 from both ends (138 to 247 a distance in three draws) | Yes: revision 2 had 128 encoder, 128 inner and 256 output-end cells, with which nearly every cell lay one connection from the output end |
| Dimensions of the space | 2 | arithmetic | One dimension gives either about 45 connections between the ends or neighbourhoods of about 400 cells | New in revision 3, with Mr. Xu's random coordinates |
| Radius of a neighbourhood | 1/8 | arithmetic | With 2,048 cells in a unit square it gives about 90 cells in a neighbourhood and about 8 to 10 connections between the ends | New in revision 3 |
| Share of inhibitory cells | 1/4 | biology | About one cell in five of mammalian cortex is inhibitory; 1/4 is the nearest grid value | No |
| Starting connections of a cell | 8 | arbitrary | A small grid value; the number is the cell's own after birth | Yes: 16 and fixed in revision 2, against Mr. Xu's ruling |
| Starting strength, and strength of a line connection | 1/8 | arbitrary | Chosen so that eight inputs at full activity sum to 1; the reviewer has found no other reason on record | No |
| Time constant of a cell's state | 2 steps | arithmetic | With 4 steps the output end received under 3 in 100 of the effect of the byte just read | Yes: 4 until revision 3 |
| Time constant of traces and of the prediction | 32 steps | arithmetic | One byte; long enough for a relay over four connections (9 steps) | Yes: 8, then 16 |
| Rate of change of a strength | 1/8 | not found | Raised from 1/32 in revision 4; the reviewer has not found the writer's reason on record | Yes |
| Rate of the threshold that controls a cell's activity | 1/128 | arbitrary | Slow beside one byte | No |
| Target of a cell's activity | 1/8 | arbitrary | A grid value; a reader has since shown on paper that no cell can reach it at birth, so that it drives the making of connections without end under the rules as first written | No; 1/32 is named as the next value, and the same reader holds that it does not help |
| Period for making and withdrawing | 128 steps | arbitrary | Four bytes | No |
| Level below which a connection is withdrawn by strength | 1/128 | arbitrary | The smallest grid fraction | The rule around it has changed twice (see section 5) |
| Level and period for withdrawing a connection that brings noise | 1/128 of the cell's largest; 128 steps | arbitrary | Entered by the reviewer at 01:18 UTC on Mr. Xu's ruling of section 19; the form is the reviewer's, the two values are the same grid values as above for want of a reason to differ | New |
| Steps of exchange for each byte | 32 | arithmetic | Enough for the byte just read to reach the output end over 8 to 11 connections (0.975 to 0.999 of its effect) | Yes: 4, then 16 |
| Level of activity at which an output line gives a byte | 1/2 | arbitrary | Half the largest activity | No |
| Feedback cells | 64 | arbitrary | Half receive yes and half no | A reader proposed 128 for step 2 |
| Input, output and emission lines | 256 each | follows | One for each value of a byte; nothing about digits | No |

## 5. What has already gone wrong with these numbers

1. **The rule for withdrawing a connection** was first written with a test on the sending cell's activity and the level 1/128. On paper this removes about 98 in 100 of all connections at the fourth byte. It was replaced by a test on strength alone; the reviewer wrote here that under it almost nothing is ever withdrawn, which was not measured and is wrong (measured by the coordinator at byte 1,024 of the first form: about 75 withdrawn and 75 made a byte, with most neighbourhoods full; corrected 03:27 UTC). It is now a test of whether what arrives is of use to the receiving cell (Mr. Xu, section 19). Each of the three used the same value, 1/128; the value was never the fault, the rule was.
2. **The first size** (512 cells with neighbourhoods drawn from all of them) left no cell more than one connection from the output end, so that distance had no meaning.
3. **The first timing** (4 steps a byte) meant that the tissue's output could not depend on the byte just read.

In all three cases the fault was found by computing the consequence over more than one byte, not by reading.

## 6. How a number is changed from now on

1. A number is changed when a run or a calculation shows a reason, and for no other cause.
2. One number is changed at a time, to the next value on the grid, unless a calculation gives a value directly.
3. The change is entered in the definition with one dated line saying what changed and why (Mr. Xu, section 18: these documents are changed in real time as the experiments require).
4. A number that a cell can set from its own quantities is handed to the cell as soon as a rule for that exists, and is then struck from this list.
5. A value marked *arbitrary* or *not found* above carries no evidence. When a result depends on it, the report says so.

## 7. What this note does not cover

The numbers of the measurement (lengths of windows, the levels of the rule of success, the classifier) are in the brief of each step, with their reasons there. The numbers of the comparator's network belong to the record of 6 to 18 November and are not numbers of the tissue.
