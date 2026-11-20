# Line `nano/search`, fourth plan: the score after a read line beside the score with an empty past (20 November 2026)

**Authorisation and steer.** Mr. Xu's steer as quoted at the head of `PLAN_1.md` (unchanged; no new words of his). The request for this addition is the coordinating session's (its message of 20 November, pointing to `loop_small_case/ERRATA_20261120.md`, section 3, on its branch at 8ae14f646, which I have not read; I work from the message). **Source.** `claude/nano-search-8mmhfm` at 0cf8e54; `search.py` gains the option `past=1` (a second score; the first is computed as before, and the stream's and the search's random draws are untouched, so a rerun reproduces the old figure to the digit: checked on development seed 1, 5,000 lines, 60.6 and 60.6). Development level; no gate, seal or review claimed. Written before any run with the new score on a reported seed.

**The fault.** Every sums figure of this line scores a held-out line with line ends in the places that reach before the line ("empty past"). The system never meets that past after its first line. My kinds reach up to 16 places back and the search did draw kinds that reach into the line before (d11 to d16 among the top earners), so the empty past may misstate this line's figures more than those of hand-given kinds.

**The addition.** Beside the empty-past score: the score **after a read line**. Before each held-out line stand the bytes of lines drawn by chance from the lines that may be read (the task being read at that time); the mean of five draws; own answer bytes, whole answer right, as before. Nothing else changes; no setting is chosen again; no finished prediction is re-scored.

**Runs** (the same commands as `PLAN_2.md` and `PLAN_3.md` with `past=1`, new tags `p_*`; a system is rebuilt by rerunning, since no state was saved): two-digit sums, five reported seeds: the life add 40,000 then sub 20,000 at 128 alive; the control at 40,000; 256 alive at 40,000; 400,000 lines at 128 alive. Three-digit sums at 40,000: searched and control. About 60 CPU-minutes. **Not run:** `nrep` 8 and 32; the control at 400,000 lines; anything on prose (not touched by the fault).

**Predictions (mine).** A1: the score after a read line is above the empty-past score for the searched system at 40,000 lines, median of five (chance 0.75). A2: by 5 points or more (0.45). A3: at 400,000 lines the score after a read line is still below the 40,000-line one (so the fall with reading is not an effect of the empty past) (0.6). A4: searched at 128 alive, after a read line, at least 77 (0.2).

## Outcome (entered 20 November; `res/p_*`, raw lines `out_p.txt`, table `report_4.txt`)

| Task (five reported seeds; last-three mean, each seed; median) | Empty past (as before) | After a read line |
|---|---|---|
| Two-digit, 40,000 lines, searched, 128 alive | 68.3, 47.4, 59.5, 43.9, 67.9; **59.5** | 84.5, 82.4, 72.9, 81.1, 75.8; **81.1** |
| Two-digit, 40,000 lines, searched, 256 alive | 78.7, 73.8, 85.4, 87.9, 57.3; **78.7** | 88.4, 87.0, 84.4, 92.5, 85.9; **87.0** |
| Two-digit, 40,000 lines, control (128 random kinds never changed) | 0.5, 45.7, 0.5, 18.9, 1.4; **1.4** | 59.6, 49.6, 1.1, 45.9, 78.5; **49.6** |
| Two-digit, 400,000 lines, searched | 73.5, 28.3, 40.5, 56.7, 50.7; **50.7** | 73.6, 74.2, 52.5, 57.2, 50.2; **57.2** (median by lines: 67.4 at 10,000; 82.3 at 40,000; 79.1 at 100,000; 71.2 at 200,000; 59.0 at 300,000; 60.2 at 400,000) |
| Three-digit, 40,000 lines, searched | 40.3, 65.6, 40.7, 72.3, 71.7; **65.6** | 88.7, 85.7, 90.8, 86.1, 77.8; **86.1** |
| Three-digit, 40,000 lines, control | 0.3, 59.6, 29.5, 0.9, 0.0; **0.9** | 0.4, 53.2, 74.2, 1.6, 0.4; **1.6** |
| Life: addition before subtraction | 64.6 (median, last reading) | 88.6, 83.1, 75.9, 81.8, 75.0; **81.8** |
| Life: addition after 20,000 lines of subtraction (the line before is then a subtraction line) | 43.4; 60.4% of its level | 23.4, 56.7, 43.4, 8.7, 14.2; **23.4**; 26.4% of its level |
| Life: subtraction, clean score | 78.2 | 89.9, 78.1, 81.3, 77.7, 72.1; **78.1** |

Every rerun reproduces its earlier empty-past figure to the digit (30 runs). Sizes and CPU are those of the earlier runs. The other lines' figures after a read line, as the coordinating session reports them: counting cells of at most four delays 80.1 (77.6 at the empty past); the relay system 82.9 at 40,000 lines and 95.5 at 400,000; "with classes" 92.9. Three-digit sums at 400,000 lines, the control at 400,000, `nrep` 8 and 32: NOT RUN with this score.

Predictions: A1 (after a read line above the empty past, searched, 40,000 lines): held (81.1 against 59.5; on every seed). A2 (by 5 points or more): held (21.6). A3 (400,000 lines still below 40,000 after a read line): held (57.2 against 81.1; the fall with reading is not an effect of the empty past). A4 (128 alive at least 77): held (81.1; chance given 0.2, too low).
