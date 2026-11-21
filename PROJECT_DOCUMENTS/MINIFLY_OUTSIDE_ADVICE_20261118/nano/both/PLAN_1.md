# One system with both kinds of cell on two-digit sums: the relay system as one more sender to the two answering cells (step 1) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). The line was set by the coordinating session; the hypothesis is the coordinator's.

**Source.** Branch `claude/nano-both-8mmhfm`, made from `claude/friendly-darwin-8mmhfm` at c3f0418cd. Copied unchanged into this folder from that commit: `loop_small_case/cells2.py`, `fastk.py`, `fastk3.py`, `kern3.c` (the relay system and its fast kernel) and `nano/merge/merge.py` (the cells that count, with classes). The kernel of `nano/table/gate.py` (the second answering cell, gate 0) is copied into `both.py` and given one more sender. Development level; no gate, seal or review is claimed. Written before the registered runs; one trial of the program on the development seed is declared in section 6.

## 1. The evidence this follows (never-shown two-digit additions, whole answer, own bytes, after a read line; as the lines report)

The relay system: 82.9% after 40,000 lines, 95.5% after 400,000; it gains with reading. Cells that count with classes and a second answering cell: 97.4% after 40,000 lines; it loses with reading (76.5 and 91.9 on two seeds at 400,000). The earlier joinings (`nano/sparse`) were scored with an empty past, one answering cell, no classes.

## 2. The question and the system

**Is one system with both kinds of cell at least as good as the better of the two at every length of reading?**

*The system.* `cells2.py`'s plain relay system (256 logistic cells in between, ten delay lines, one answering cell; eta1 = 1/4, eta2 = 1/16, fading 1e-5; 43,278 numbers that learn) reads the stream of lines beside `merge.py`'s cells that count (256 kinds of at most four of nine delays, 32 class kinds found by search, that line's frozen setting; about 1.2 million cells and 6.3 million counts above zero after 400,000 lines) with `gate.py`'s two answering cells (11,520 trusts; 1,451,520 parts in the second cell). The relay system is one more sender:

- its message at a byte is the log of its fourteen branches, each cut off below at nothing, with 0.01 added, made to sum to one (its branches are fitted to one and nothing, so they stand for shares);
- its level is its largest share in ten equal bins;
- the first cell holds one trust for each level; the second cell holds for each of the nine delays a part indexed by (level, byte at that delay), as for every other sender; all born at nothing; changed by the mismatch with the rule and the step of every other sender (0.1 over the root of one plus the uses);
- for a block of 50 lines its messages come from its strengths at the start of the block; then it learns the block (as `nano/table/relay.py` does on prose). Nothing of the counting cells reaches it;
- it speaks at the bytes it is taught: `cells2.py` teaches it the nine bytes of a line after the first, so at a line's first byte it is silent. This is `cells2.py`'s own property, taken over, and it is knowledge of the line's length that the counting system does not have; stated as a limit.

*Arms, same stream, same held-out pairs.* The relay system alone; the counting system alone (first cell, second cell); joined (first cell, second cell); and, from the joined run, the joined cells with the relay system silent at the score (what the counting cells' reading is worth inside the joined system). Readings at 5,000, 10,000, 20,000, 40,000 lines and every 50,000 to 400,000. Reported seeds 73129, 93683, 116089, 406364, 545353. The score: the 500 never-shown additions, whole answer in the system's own bytes, each line read after a line drawn by chance from the shown lines (mean of five draws; the same five draws as `past.py` and `gate.py`), the relay system's message formed after the same read line; the empty past is kept beside it.

*Checks.* `both.py check`: the compiled answering loop with the given sender against numpy; with no given sender the first cell against `nano/merge`'s kernel; the relay system read in blocks against `cells2.py`'s numpy loop and its messages against a plain computation. Parts against their own lines: the relay system alone against `loop_small_case/res/past_seed*_{40000,400000}.json`; the counting system alone against `nano/table/res/gate_sums_gate0_modeclasses_seed*.json` (40,000 lines, five seeds; 400,000 lines, two seeds).

## 3. Hypothesis, what refutes it, predictions (mine, before the registered runs)

*Hypothesis (the coordinator's).* At the median of five seeds the joined system (second cell) is at least 96 at 40,000 lines and at least 95 at 400,000 lines. *Refuted if* it is more than 3 points under the counting system alone at 40,000 lines or more than 3 under the relay system alone at 400,000.

My expectation is that it is refuted at 400,000 lines: a trust changes only where the answer is not yet certain, and on the shown lines of a long reading the cells that hold the line exactly are certain, so the relay system, which becomes good late, is given little trust when it has earned it.

| | Statement | Chance |
|---|---|---|
| B1 | joined, second cell, median at 40,000 lines at least 96 | 0.80 |
| B2 | joined, second cell, median at 400,000 lines at least 95 | 0.20 |
| B3 | joined at 400,000 lines not more than 3 under the relay system alone (median) | 0.25 |
| B4 | joined above the counting system alone at 400,000 lines by at least 3 points (median, second cell) | 0.50 |
| B5 | the relay system's trust in the first cell at its top level, at 400,000 lines, is under 0.2 (median) | 0.50 |
| B6 | with the relay system silent at the score, the joined second cell at 400,000 lines is within 3 points of the counting system alone (the fall is still there) | 0.70 |

## 4. Cost, what is not run

Ten runs of 400,000 lines (five seeds, alone and joined), about ten CPU-minutes each on one thread, three at a time; about 0.4 GB each. Not run in this step: the second task (step 3), composed lines (step 4), any change to the trust's rule (step 2 follows the outcome), other floors or blocks than those stated (one floor, one block; a choice between them would be made on seed 1 only and is not made here).

## 5. Reporting

The last reading and the mean of the last three beside it; five seeds and the median; size and CPU beside the figures; every prediction scored.

## 6. Declared before the runs

To test the program, the joined arm was run once on the development seed 1 for 40,000 lines before this plan was committed (105 CPU-seconds): the relay system alone 3.2, 12.4, 71.8, 75.3 at 5,000, 10,000, 20,000 and 40,000 lines; joined second cell 92.9, 95.9, 97.0, 96.6; the same with the relay system silent at the score 92.8, 95.7, 96.9, 96.6. Prediction B1 was written knowing this.

## 7. Outcome (entered 20 November after all ten runs; `res/s1_*.json`, the whole table with every seed and reading in `report_s1.txt`, made by `report.py`)

Never-shown two-digit additions, whole answer, own bytes, after a read line, %. Five reported seeds, median (range). One thread; a run of 400,000 lines takes 351 to 384 CPU-seconds for the counting system alone and 412 to 445 joined. Sizes at 400,000 lines: 1.24 million cells that have counted, 6.3 million counts above zero; 11,530 trusts; 1,452,780 parts in the second cell, 778,587 used; the relay system 43,278 numbers.

| Lines | Relay alone | Counting alone, first cell | Counting alone, second cell | **Joined, second cell** | Joined, first cell | Joined second cell, relay silent at the score |
|---|---|---|---|---|---|---|
| 5,000 | 3.9 | 90.4 | 94.0 | 94.2 | 90.5 | 94.2 |
| 10,000 | 19.6 | 91.4 | 95.0 | 95.3 | 91.6 | 95.1 |
| 20,000 | 66.0 | 92.6 | 96.0 | 96.6 | 93.0 | 96.6 |
| 40,000 | 82.9 (81.2 to 86.5) | 92.9 | **97.4** (96.4 to 98.2) | **97.1** (96.4 to 98.1) | 93.4 | 96.9 |
| 100,000 | 91.9 | 85.4 | 87.0 | 86.5 | 86.1 | 86.0 |
| 200,000 | 94.4 | 82.9 | 90.4 | 89.9 | 84.6 | 88.4 |
| 300,000 | 95.0 | 80.0 | 77.8 | 81.8 | 82.5 | 78.7 |
| 400,000 | **95.5** (95.0 to 97.0) | 77.2 | 75.5 (73.1 to 91.6) | **79.0** (75.1 to 92.2) | 80.5 | 75.0 |
| mean of the last three | 95.4 | 78.8 | 76.5 | 80.4 | 81.7 | 76.8 |

(Every figure is the median of the five seeds, as `report_s1.txt` gives it.)

*Checks.* `both.py check` passed (compiled against numpy to 1e-15; with no given sender the first cell equals `nano/merge`'s kernel exactly; the relay system read in blocks equals `cells2.py`'s numpy loop to 1e-14). Each part alone gives its own line's figures to the digit: the relay system against `loop_small_case/res/past_seed*_{40000,400000}.json` on all five seeds at both lengths (82.92, 82.32, 81.16, 86.48, 83.16 and 95.00, 95.00, 97.00, 95.48, 96.08), the counting system against `nano/table/res/gate_sums_gate0_modeclasses_seed*.json` on all five seeds at 40,000 lines and on the two seeds that line ran at 400,000 (both cells).

**The hypothesis is refuted at 400,000 lines and holds at 40,000.** At 40,000 lines the joined system has 97.1, level with the counting system alone (97.4; not more than 3 under). At 400,000 lines it has 79.0, 16.5 points under the relay system alone (95.5); at least 95 was asked and not more than 3 under allowed. The joined system is the counting system with a small addition: it falls with reading as the counting system does, and is 1 to 4 points above it from 300,000 lines on.

**Why, from the trusts** (the table's last parts in `report_s1.txt`). The relay system's trust in the first cell at 400,000 lines, by level of sureness (its largest share in ten bins; the tenth bin is never reached, since the floor of 0.01 keeps the largest share under 0.89): 0.26, 1.08, 0.69, 0.19, 0.11 and 0.03 at levels 3 to 8. It is used most at level 7 and 8 (1.06 million and 471 thousand bytes), which is where it is sure and right, and there its trust is 0.11 and 0.03: next to nothing beside 288 other senders. The trust grows where the relay system is half sure (levels 4 and 5) and those are few bytes. Silencing the relay system at the score takes 4 points from the joined answer at 400,000 lines and 0.2 at 40,000. So the relay system is given trust where the others leave a mismatch, and on lines read many times they leave none where it is sure.

Predictions. B1 held (97.1). B2 failed (79.0; I gave it 0.20). B3 failed (16.5 under; I gave it 0.25). B4 held, narrowly (79.0 against 75.5: 3.5 above). B5 held (0.03 at the top level reached). B6 held (75.0 with the relay system silent against 75.5 for the counting system alone).

Readings (mine).
1. **Joining the relay system as one more sender, with trusts taught by the mismatch, does not give the better of the two.** It gives the counting system with its fall. The coordinator's hypothesis fails in the form stated; my expectation in section 3 held, and for the reason given there as far as the trusts show.
2. The counting system's own fall is still whole in the joined answer (B6).
3. What this does not show: that the two kinds of cell cannot be joined. `PLAN_2.md` locates the fall (a look on the development seed: with the cells that counted fewer than 512 times silent at the score, the same system answers 97.7% at 400,000 lines, above the relay system) and tries two changes to how the trusts are taught.
