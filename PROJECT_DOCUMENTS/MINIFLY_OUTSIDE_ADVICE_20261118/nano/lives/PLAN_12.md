# A strength changes only while its own cell speaks: four local changes to the relay network, and whether the rule it learned is then kept through English (21 November 2026)

**Authorisation and steer.** As `PLAN_11.md`, where Mr. Xu's words are quoted in full; the one that governs this step: "这些你自己决定，不需要问我。一切从第一原理和事实出发。" (my translation: "Decide these yourself, you need not ask me. Everything from first principles and from facts."). The choice and design of this step are mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written after the development runs of `PLAN_11.md` and with three of its five reported seeds in; before any run of `relay12.py` other than its check (`check12_output.txt`: the sixteen forms at two settings against a reference in numpy, the form with no switch against `nano/loop/kernp.c`, and silent cells keeping their strengths bit for bit; all passed). Development level; no checker gate, seal or review is claimed.

## 1. The facts this follows (`PLAN_11.md`, sections 4b and 5)

The relay network learns the rule of two-digit addition (96.6% of never-shown sums at the median of the three reported seeds in so far) and answers none of them after 400,000 bytes of English. Where the loss sits was measured by putting parts back:

| After the English, with ... of the end of the sums put back | Never-shown sums (three reported seeds: median, smallest to largest) |
|---|---|
| nothing | 0.0 (0.0 to 0.0) |
| the cells in between (their strengths from the lines and their thresholds) | 0.0 (0.0 to 0.0) |
| the answering strengths | 86.6 (86.3 to 89.0) |
| the answering strengths and the cells' thresholds | 96.1 (95.7 to 97.5), against 96.6 before the English |

So the rule is still there after the English, in the cells' strengths from the lines. English cannot move those: a strength from a line moves only when its line is active, and the lines of digits are not active in English. What English destroys is the reading of the rule (all of it) and the cells' thresholds (ten points). The return says the same: after the English the network is back at 92.7% in 1,000 lines and 97.4% in 5,000, where a network at birth reaches 9.8% in 5,000 lines.

Why the reading is destroyed, from the program itself (`nano/loop/kernp.c`): (1) every answering strength fades by a hundred-thousandth at every byte, whichever cells were active; over 400,000 bytes that alone leaves a fiftieth; (2) every cell's threshold moves at every byte; (3) every branch of the answering cell is drawn to nought, from above and from below, at every byte: through the English, the ten digit branches are taught 400,000 times to say exactly nothing, through whichever cells are active; (4) a cell is born with its threshold at nought, so it is half-active for a state it has never met: before the English is read, a state of the English has 84 of the 256 cells above one half (a state of the sums 111), and the teaching of (3) reaches the digit branches through all of them. The digit branches' strengths changed by 0.998 of their size.

The counting learners of `PLAN_11.md` show the other side. What they lost through the English (5 and 8 points) they lost in the trusts that all cells of a kind share, and all of it came back when those trusts were put back; with a trust for each cell they lost 0.7 points; with no learned trust, nothing. Their counts lost nothing: a count changes only when its own context is present.

**The principle I take from this (mine):** in every learner of this programme, what a text with other bytes destroyed was held in numbers that are changed while the cell they belong to is not in use; what survived was held in numbers that change only while their own sending cell (or line, or context) is active. The first condition for a change at a synapse in a nervous system is activity at that synapse (my reading of textbook knowledge; I cite no paper for it). The relay network departs from it in the four places above.

## 2. The four changes, each local, each switchable alone (`relay12.py`)

| Letter | As the network is | Changed |
|---|---|---|
| g | Every answering strength and return strength fades by lam = 1e-5 at every byte | It fades by lam x the activity of its sending cell at that byte: a strength from a silent cell does not fade |
| t | A cell's threshold moves at every byte by what the cell receives over its return strengths | The thresholds stay where they were born |
| q | Thresholds are born at nought: a cell is half-active for any state it has not met | Thresholds are born at -theta: a cell is nearly silent until its lines have grown strengths to it (mean activity at birth 0.15, 0.066, 0.026 for theta = 2, 3, 4; 0.5 as it is) |
| r | The mismatch of a branch whose byte did not arrive is nought less its output, whatever the sign of the output | An output below nought counts as silence (nought): a branch that is silent and whose byte did not arrive is not taught. The branch whose byte arrived is taught as before (one less its output, the output taken as nought if below) |

Nothing else is changed: 256 logistic cells, ten places, one step of relay, the steps 1 and 1/16, the least-squares answering cell with its normalised step, return strengths born uniform in -1..1. Every strength is still its own; no number is shared between connections. The kernel is `nano/loop/kernp.c` with three switches added by stated replacements of its text; with no switch it does `kernp.c`'s arithmetic (checked: 2e-14), and each of the sixteen forms is checked against a reference written out in numpy.

**Sixteen forms**: every combination of the four letters, from R (the network as it is, which is `PLAN_11.md`'s R and must repeat its figures) to Rgtqr (all four). **theta** is chosen on the development seed among 2, 3 and 4 by the first phase alone: the largest theta with which Rgtqr ends the sums within 3 points of R. If none does, a quiet birth costs this learner its rule; that is recorded, the letter q is dropped and the eight forms without it are reported, Rgtr standing for "all".

**The life and the score** are `PLAN_11.md`'s: 400,000 lines of sums, 400,000 bytes of English, 5,000 lines of sums again; never-shown sums, whole answer, in the network's own bytes with nothing changing. Added marks: the return after 10, 20, 50, 100 and 200 lines; the same score when only the ten digit branches may answer (it separates a reading that is lost from one that is outvoted by branches of the other text); and, at every mark, the bytes named right on 20,000 bytes of the English that are never read (so that what the English keeps when the sums return is measured as well).

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With the four changes (Rgtqr) the relay network keeps the rule it learned through the English: at the end of the English, with nothing changing, it answers at least three quarters of the never-shown sums it answered at the end of the sums (median of the five reported seeds).

**It counts only if** the form learned the rule (it ends the sums within 3 points of R at the median) **and** it learned the English (bytes named right on the held-out English at the end of the English within 3 points of R's): keeping bought by not learning is not keeping.

**Refuted if** Rgtqr, having learned both, keeps under a quarter. Between a quarter and three quarters the hypothesis is not supported and the principle of section 1 is at best a part of the answer.

| | Statement (five reported seeds; median; "kept" = never-shown sums at the end of the English over those at the end of the sums) | Chance |
|---|---|---|
| Y1 | Rgtqr learns the rule: it ends the sums within 3 points of R | 0.55 |
| Y2 | The hypothesis holds: Rgtqr keeps three quarters or more | 0.30 |
| Y3 | Rgtqr keeps a quarter or more | 0.55 |
| Y4 | With only the digit branches answering, Rgtqr keeps three quarters or more | 0.45 |
| Y5 | Rgtqr is back within 5 points of its level after 100 lines of the return (R needs more than 1,000) | 0.45 |
| Y6 | No single change keeps a quarter: Rg, Rt, Rq and Rr are each under a quarter | 0.60 |
| Y7 | Rgtqr learns the English as well as R: held-out English named right at the end of the English within 3 points of R's | 0.50 |
| Y8a | R loses the English when the sums return: its held-out English falls by more than 10 points over the 5,000 lines | 0.60 |
| Y8b | Rgtqr keeps the English through the return: it falls by less than 3 points | 0.40 |
| Y9 | The form that keeps most of the sixteen has both q and r | 0.55 |

## 4. Runs, size, cost, what is not run

Development seed 1: R (it must repeat `PLAN_11.md`'s 83.5, 96.0 and 0.0), and Rgtqr at theta = 2, 3, 4 for the choice of theta by the first phase alone. Reported: the seeds 73129, 93683, 116089, 406364, 545353, all sixteen forms, nothing chosen among them. Size: 256 cells, about 230,000 strengths (75 byte values). Cost: about 1.5 CPU-minutes a run; 84 runs, about two CPU-hours. **Not run:** subtraction after addition (the same bytes asking for other answers, which no silence can keep apart); a longer English; branches that compete; cells in between that compete; this learner on the six streams; more than one value of theta in the reported runs.

## 4a. The development runs, the choice of theta, and two additions (development seed 1; before any reported run)

Never-shown sums, whole answer, per cent; in brackets the same with only the digit branches answering (`out/relay12_dev*.txt`):

| Form | sums: 5,000 lines | 40,000 | end of the sums | end of the English | the same with the answering strengths of the end of the sums put back | return: 10 lines | 50 | 100 | 500 | 5,000 | held-out English named right: end of the English, end of the return |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R (as it is) | 5.4 | 83.5 | 96.0 | 0.0 (0.0) | 80.5 | 0.0 (2.6) | 0.0 (6.5) | 0.0 (12.9) | 46.0 (67.7) | 94.2 (94.2) | 41.9, 18.6 |
| Rgtqr, theta 2 | 17.9 | 91.7 | 98.0 | 0.0 (52.7) | 95.4 | 1.9 (77.3) | 69.8 (92.1) | 82.7 (93.4) | 95.0 (95.9) | 96.8 (96.8) | 45.6, 41.2 |
| Rgtqr, theta 3 | 21.6 | 91.3 | 97.8 | 0.0 (62.9) | 92.9 | 1.3 (84.6) | 72.0 (92.7) | 84.0 (92.9) | 94.7 (95.2) | 96.7 (96.7) | 46.4, 41.5 |
| Rgtqr, theta 4 | 34.7 | 92.6 | 98.9 | 0.7 (88.1) | 91.0 | 23.6 (91.0) | 77.4 (91.7) | 88.0 (93.1) | 97.5 (97.6) | 98.4 (98.4) | 46.3, 42.9 |
| Rgtqrb, theta 4 (addition 1) | 32.3 | 90.5 | 98.3 | 0.0 (87.4) | 90.1 | 0.0 (90.1) | 13.9 (90.9) | 57.6 (92.6) | 95.5 (95.9) | 96.3 (96.3) | 47.4, 44.0 |

**R repeats `PLAN_11.md`** (5.4, 83.5, 96.0, 0.0). **theta.** All three values end the sums within 3 points of R (they end above it), so by the rule of section 2 **theta = 4**, the largest of the set. It is at the edge of the set; a larger value was not tried.

**What the development seed already says of the hypothesis (mine; one seed; not a result).** On its own measure it fails: straight after the English the network with the four changes names 0.7% of the never-shown sums. But the reading is not lost: with only the digit branches answering it names 88.1% (98.9% before the English), and it is back at 88.0% after 100 lines of the return, where the network as it is names none after 100 lines and 46.0% after 500. It also learns both texts better than the network as it is.

**Addition 2 (measurements only; no learner is changed).** (i) *Who answers in the states of the sums*: the output of the right digit's branch, the largest output among the branches that are not digits, and how often such a branch is the loudest. On the development seed, form Rgtqr at theta 4: before the English the right digit's branch gives 0.95 and the loudest other branch 0.00; after the English 0.07 and 0.42, and another branch is the loudest in 78% of the states. So two things happened: the digit branches sank together (their order mostly kept), and branches of the English answer in states of the sums, where they were never taught. (ii) *Shown sums beside never-shown ones* at every mark (500 of the shown pairs, after the same read lines): the helper's reading of the second literature report asked for cases beside the rule (`OUTSIDE_LITERATURE_ROUND2_AGAINST_THE_PROGRAMME_20261121.md`, section 3).

**Addition 1 (a fifth letter, b: the branches' own constants held at nought).** Written after the run at theta 2 showed the whole answers outvoted. I had overlooked, in section 1, that a branch's own constant is a fifth number that belongs to no sending cell and moves whenever the branch is taught. Tried once on the development seed at theta 4 (the last row of the table): it is worse. After 100 lines of the return it names 57.6% against 88.0%; the digit branches' strengths changed by 0.47 of their size against 0.23; the loudest other branch stands at 2.05 against 0.42, with its constant at 0.05 where constants move. So the constants are not what outvotes: the branches of the English are loud in the states of the sums through cells that are active in both texts, and a constant takes up corrections that otherwise fall on the strengths. The letter is carried to the reported seeds in one form only (Rgtqrb), to see whether the direction holds; I state no chance for it. `relay12.py`'s check covers all thirty-two forms (`check12_output.txt`).

**Reported runs, as planned:** the sixteen forms at theta 4 on the five seeds, and Rgtqrb.

## 5. Outcome

(to be entered after the runs)
