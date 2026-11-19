# The loop at the smallest size, second plan: a text whose unseen items are new combinations of seen parts

**Date:** 19 November 2026. Same authorisation, source and system as `PLAN.md`. Written after the first plan's reported result and after one probe of the comparator on this text on the development seed; before any run of the relay on this text and before any reported seed.

## 1. What the first plan showed, and why a second text

First plan, five reported seeds, 60,000 lines of single-digit sums (`report1.txt`): with learning cells in between, the whole answer is right on 97.5% of sums shown under the local relay with learned return connections and on 96.2% under the comparator (back-propagation), against 12.5% with no cells in between and 2.5% with cells that do not learn. Sums never shown: 0% at the median in every arm, the comparator included. So the relay teaches the cells in between as well as back-propagation does, and neither answers an unseen single-digit sum.

My reading of the second fact: with one line for each byte value, an unseen pair of single digits is not a new combination of anything taught. Only an order of the digits could carry an answer over to it, and 80 facts do not force a learner to form one; a table fits them. That holds for back-propagation as for the cells. The record's own roadmap asks for something else at its step 5: "a never-taught instance is a new combination of parts taught in other instances". The smallest text of that kind is numbers of two digits: a pair of numbers never shown is made of digits that were shown in every place.

*Probe (development seed 1, comparator only, 400,000 lines, 256 cells):* whole answer right on 62% of sums shown and 64% of sums never shown, the two equal at every reading; by place on sums never shown: hundreds 82%, tens 80%, units 98%; the readings swing by 20 to 30 points between one reading and the next.

## 2. The question

On two-digit numbers, **do cells in between that learn from the relayed mismatch answer sums never shown as the comparator does?**

*Hypothesis.* Yes: within 15 points of the comparator on sums never shown, and never-shown sums as well as shown ones.

*Refuted if* the comparator has at least 50% of sums never shown right at the median and the relay 15% or less.

## 3. The text, the system, the score

Lines `aa+bb=ccc` with an end mark, a and b from 0 to 99, each written with two digits and the sum with three (the first family's own way of writing a number). Of the 10,000 pairs, 2,000 drawn by the seed are never shown. One stream of lines drawn with replacement from the 8,000 pairs shown; one byte at a time; the system names the next byte at each of the nine positions before the end mark.

The system is that of `PLAN.md`, section 3, with nine delay cells in place of four (a line has ten bytes; 10 x 13 lines) and more cells in between. Nothing about digits, places, carrying or addition is laid down.

Score, learning off: 500 pairs shown and 500 never shown, fixed by the seed; the whole answer is the three digits after `=`; given also by place.

## 4. Arms, seeds, numbers

The six arms of `PLAN.md`. The five reported seeds of `PLAN.md`, each once in every arm.

Development on seed 1 (and seed 2 for the setting chosen), arm `relay-learned`: 256 cells in between; eta1 in 1/16, 1/4, 1; eta2 in 1/16, 1/4; lambda in 0, 1e-5; 400,000 lines. The setting with the highest mean of the last three readings on sums never shown is taken for every arm and entered in section 7 before the reported seeds are run. If the readings still swing by more than 15 points between readings, one remedy may be added and is named in section 7 before the reported seeds: a smaller step, or a step that each cell divides by the recent size of the signal it receives.

## 5. Success, failure, predictions (mine; made after the comparator's probe and before any run of the relay on this text)

*Holds:* at the median over the five seeds, `relay-learned` is within 15 points of the comparator on sums never shown, and at least 50%. *Refuted:* as in section 2. Between: partial.

Each reading below is the mean of a run's last three readings (at 50,000-line intervals).

| | Statement | Chance |
|---|---|---|
| Q1 | `relay-learned` is within 15 points of the comparator on sums never shown at the median | 0.75 |
| Q2 | In every arm that has at least 30% of sums shown right, sums never shown are within 10 points of sums shown at the median | 0.85 |
| Q3 | `none` has at most 10% of sums never shown right, and `fixed` at most 30% | 0.80 |
| Q4 | `relay-learned` has at least 60% of sums never shown right at the median | 0.50 |
| Q5 | `relay-learned` under yes or no has at least 30% of sums never shown right at the median | 0.30 |
| Q6 | `relay-fixed` has at least 40% of sums never shown right at the median | 0.60 |

## 6. Cost, and what is not run

A run of 400,000 lines is 3.6 million bytes, about two minutes of one core. The development grid is 12 runs; the report 30 runs, about 20 minutes on four cores. Not run: numbers written without leading zeros (the reviewer's form); more than two digits; the first family; any tissue; anything about keeping.

## 7. The setting chosen on the development seed

Entered on 19 November after the development runs and before any reported seed was run on this text.

Development seed 1, arm `relay-learned`, 256 cells, 400,000 lines, the grid of section 4 (12 runs): the best setting is **eta1 = 1/4, eta2 = 1/16, lambda = 1e-5**, with the whole answer right on 88.1% of sums shown and 84.8% of sums never shown (mean of the last three readings; the eight readings on sums never shown: 71, 80, 81, 82, 83, 86, 82, 86; by place: hundreds 99, tens 87, units 100). On development seed 2 the same setting gives 82.5% and 84.2%. The readings no longer swing, so no remedy is added. This setting is used for every arm.

**One arm is added before the reported seeds, in fairness to the comparator.** At the relay's setting the comparator gives 72.5% (seed 1) and 56.1% (seed 2) on sums never shown, which may only mean that the setting was chosen for the relay. The same grid was therefore run for the comparator on seed 1 (12 runs); its own best setting is eta1 = 1, eta2 = 1/16, lambda = 0 (79.3% shown, 77.9% never shown). The report gives the comparator at both settings, and Q1 and the rule of section 5 are read against the better of the two.
