# Line `nano/merge`, third plan: a cell is withdrawn when its class foretells it at least as well as it foretells itself (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written while step 2 was running, with one reported seed of sums seen (73129: 79.0 without classes, 91.8 with, 93.1 with replacement) and no book seen; before any run of `merge3.py` other than its check. Development level; no gate, seal or review is claimed.

## 1. The evidence this follows (step 1, development material)

Replacement of every R-cell whose X-cell is in a class lifts sums (90.8 to 93.8) and harms prose (1.5488 to 1.5612, where the base has 1.5566). My reading: on sums the cell of a whole pair says exactly what its class cell says, so nothing is lost by silencing it and the trust then learns on shown lines; on prose a run of six bytes that has been met many times says something its class, pooled over other runs, does not.

## 2. The question and the one change

**Can the system itself tell which cells a class may stand for?** One change from replacement: at each finding, for each R-cell (x, z) whose X-cell is in a class of a kept pair, two numbers are compared, both in the measure J of the search: what the class's other members foretell of this cell's counts, and what this cell's own counts foretell of each other (each count from the rest). **The cell is silent only if the class foretells it at least as well as it foretells itself**; otherwise it goes on speaking beside its class cell. A cell judged silent by one kept pair is silent. A cell born after the finding, with its X-cell in a class, is silent until the next finding (it has no record of its own). Silent cells go on counting, as before. No new number. `merge3.py`, a subclass of `merge.py`'s system; its check is in `check3_output.txt`.

A cell met once foretells nothing of itself, so it is silent wherever any class foretells it better than 1/S: this is the statement "a cell met once holds no regularity", applied by the system and not by a rule about counts.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* On development material this form is above the frozen setting on sums (90.8) and not more than 0.002 nats above it on prose (1.5488). *Refuted if* either fails. If it holds, it is run on the five reported seeds and on the life as a second setting, named as chosen after step 2 had begun; the frozen setting of step 2 stays the one the hypothesis of `PLAN_1.md` is scored on.

| | Statement | Chance |
|---|---|---|
| Q1 | sums, development seed: at least 92.5 | 0.50 |
| Q2 | prose, last twentieth of the stream: at most 1.5508 | 0.40 |
| Q3 | prose: below the frozen setting's 1.5488 | 0.20 |
| Q4 | on prose more than half of the judged cells are silent | 0.60 |
| Q5 | both parts of the hypothesis | 0.25 |

## 4. Size and cost

As step 2; the judgment adds a pass over the member cells at each finding (seconds). The cells that speak are counted beside all cells. Not run: a trust that learns from a cell's own record; anything else of `PLAN_2.md`'s list.

## 5. Outcome (entered 20 November after the two development runs; `res/dev3_sums.json`, `res/dev3_prose.json`, `out_dev3_*.txt`, check in `check3_output.txt`)

| | sums, seed 1, last three readings | prose, last twentieth | cells (speaking) | CPU s |
|---|---|---|---|---|
| no classes | 76.8 | 1.5566 | | |
| frozen setting (classes beside the cells) | 90.8 | 1.5488 | | |
| replacement of every member's cell | 93.8 | 1.5612 | | |
| **replacement cell by cell (this plan)** | **89.7** | **1.5551** | sums 1,236,269 (1,171,706); prose 2,292,678 (1,900,794) | 55; 107 |

At the last finding 64,197 of 74,273 judged cells were silent on sums and 300,579 of 353,761 on prose.

**The hypothesis is refuted in both parts**: below the frozen setting on sums (89.7 against 90.8) and 0.0063 nats above it on prose. Q1 failed (89.7; 92.5 predicted), Q2 failed (1.5551; at most 1.5508 predicted), Q3 failed, Q4 held (85% of the judged cells silent on prose), Q5 failed. It is not run on the reported seeds or on the books. One seed on sums; the difference from 90.8 is within what seeds differ by, so of sums I say only that it did not gain.

Reading (mine). On prose the cells that the class foretells better than they foretell themselves are mostly cells met once or twice, and silencing them still costs 0.006 nats: a cell met once, read at the small trust the answering cell has learned for such cells, adds something that its class cell, read at another trust, does not replace. "Met once holds no regularity" is true of the cell and not of what the answering cell makes of many such cells together. Withdrawal by this test is therefore not the route to a smaller store on prose.
