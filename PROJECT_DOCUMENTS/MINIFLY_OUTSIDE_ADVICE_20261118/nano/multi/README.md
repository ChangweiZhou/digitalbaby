# nano/multi: connections of more than two ends (roadmap line C3)

One agent session, branch `claude/nano-multi-8mmhfm`, started from commit 260c5d7 of `claude/friendly-darwin-8mmhfm`. Development level; repeated by no one; no checker gate, seal or review is claimed. Nothing outside this folder is changed.

| Step | Plan | State |
|---|---|---|
| 1 | `PLAN_1.md`: a connection of three ends (two lines and a cell) on two-digit sums | run on five seeds; outcome in the plan |
| 2 | `PLAN_2.md`: the same arms on the text of `nano/data` | see the plan |
| 3 | `PLAN_3.md`: connections whose two sending ends are cells | see the plan |

**Copied, and from where.** From `loop_small_case` at commit 260c5d7, written again here and changed only here: the class `Sys` and `run` of `cells2.py` (one step of relay, `fan=` and `rewire=`) into `multi.py`; `kern.c` and `kern3.c` into `kernm.c`; `fastk.py` into `fastm.py`; the manner of `check_fast.py` into `check_multi.py` and `check_text.py`. From `nano/loop/loop.py` at commit e0c864c of `claude/nano-loop-8mmhfm`: the loading of the text, the stream and the scoring, into `textm.py`. `check_multi.py` imports `loop_small_case/cells2.py` to compare with it and writes nothing there.

**Files.** `multi.py` (numpy reference and runner for sums), `kernm.c` and `fastm.py` (compiled inner loop; `python3 fastm.py` builds it), `check_multi.py` and `check_multi_output.txt`, `textm.py` (runner for the text), `check_text.py` and `check_text_output.txt`, `jobs_*.txt` (the runs as queued), `res/` (one file a run), `log/` is not kept, `report1.py` and `report1.txt`.
