# nano/multi: connections of more than two ends (roadmap line C3)

One agent session, branch `claude/nano-multi-8mmhfm`, started from commit 260c5d7 of `claude/friendly-darwin-8mmhfm`. Development level; repeated by no one; no checker gate, seal or review is claimed. Nothing outside this folder is changed.

| Step | Plan | Result (the plan holds the figures, limits and scored predictions) |
|---|---|---|
| 1 | `PLAN_1.md`: a connection of three ends (two lines and a cell) on two-digit sums; five seeds | at equal numbers of strengths the pair arms are below the edge arm on every seed: 48.8 and 48.3 against 80.8 at 64 strengths, 19.5 and 30.7 against 42.7 at 32 (`report1.txt`) |
| 2 | `PLAN_2.md`: the same arms on the text of `nano/data`, 1,024 cells, 8 readings; seeds 1, 2, 3 | the same order: 1.896 nats (edge), 1.938 (half and half), 2.286 (pairs) at 64 strengths; 1.964, 2.006, 2.401 at 32 (`report2.txt`) |
| 3 | `PLAN_3.md`: connections whose two sending ends are cells (the product of their outputs), two steps of relay, and a rule by which each of the three cells can withdraw; five seeds | pairs of cells are level with edges within the scatter (79.5 against 83.6 at 16 strengths, 72.3 against 64.8 at 8); the withdrawal rule as stated collapses at 8 strengths (16.5 against 76.7) and costs 15 points at an eighth of its rate (`report3.txt`) |

In steps 1 and 2 the sending ends are input lines, so only the receiving cell decides about a connection; step 3 is the only one in which a sending cell decides anything. Stopped after step 3, as the brief says, with about two hours of the five used.

**Copied, and from where.** From `loop_small_case` at commit 260c5d7, written again here and changed only here: the class `Sys` and `run` of `cells2.py` (one step of relay, `fan=` and `rewire=`) into `multi.py`; `kern.c` and `kern3.c` into `kernm.c`; `fastk.py` into `fastm.py`; the manner of `check_fast.py` into `check_multi.py` and `check_text.py`. From `nano/loop/loop.py` at commit e0c864c of `claude/nano-loop-8mmhfm`: the loading of the text, the stream and the scoring, into `textm.py`. `check_multi.py` imports `loop_small_case/cells2.py` to compare with it and writes nothing there.

**Files.** `multi.py` (numpy reference and runner for sums), `kernm.c` and `fastm.py` (compiled inner loop; `python3 fastm.py` builds it), `check_multi.py` and `check_multi_output.txt`, `textm.py` (runner for the text), `check_text.py` and `check_text_output.txt`, `jobs_*.txt` (the runs as queued), `res/` (one file a run), `log/` is not kept, `report1.py` and `report1.txt`; `prod.py`, `kernq.c`, `fastq.py`, `check_prod.py` and `check_prod_output.txt` (step 3); `report2.py`, `report3.py` and their tables.
