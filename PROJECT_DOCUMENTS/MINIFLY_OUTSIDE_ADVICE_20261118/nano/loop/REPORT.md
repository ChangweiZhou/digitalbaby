# Line L1, then L2: the loop on the test text (20 November 2026)

By the agent session that owns `nano/loop` (roadmap, second revision, section 6). Development level; repeated by no one; no checker gate, seal or review is claimed. Nothing outside `nano/loop/` was changed. Branch `claude/nano-loop-8mmhfm`, started from commit fe397ca. Four plans (`PLAN_1.md` to `PLAN_4.md`), each committed before its runs; what was entered in a plan during its runs is dated there and nothing earlier in a plan was rewritten. Every figure below is in `tab_output.txt` (made by `tab.py` from `res/*.json`).

## 1. What was built, and what was copied

Copied from `loop_small_case` at commit fe397ca and changed only here: `kern3.c` into `kernp.c` (competing branches as an option, room for 65 byte values, the strengths held by line, one sweep less over the return strengths, the loss of the stream counted); the plain system of `cells2.Sys`, the wrapper of `fastk3.py` and the scoring and tables of `prose.py` into `loop.py`; the manner of `check_fast.py` into `check.py`.

- `loop.py`: `Sys.step` is the numpy reference; `python3 loop.py run <label> ...` makes a run; `python3 loop.py tables` the unsmoothed tables.
- `kernp.c`: the compiled inner loop for both answering cells and any birth of the return strengths. `kernt.c`: the same for return strengths born at nothing, which are then the same numbers as the answering cell's strengths at every byte, so one copy is kept (1.3 times faster at 1,024 cells).
- `check.py`, `check_output.txt`: 15 cases, both arms, D = 4, 10, 32, four births, strong fading, a run made in two calls: the largest difference over the largest strength is 8e-16 to 6e-13; stream loss and bytes named right agree. A wrong term would show as a difference of order one.
- `go.sh`, `go2.sh`, `q.sh`: the queues; `jobs_*.txt`: every run that was made, as it was started; `summ.py`, `tab.py`: tables.

**The system** is the one of `loop_small_case/REPORT.md`, section 2: the last D bytes on lines (65 a delay); H logistic cells that read every line; one answering cell with 65 branches; the mismatch formed when the byte arrives, relayed to each cell over return strengths that the cell holds and that receive the same change as the forward strengths; fading 1e-5 a byte; one byte at a time, no batch, nothing passed backward over the text, nothing reset between readings. The step of the answering cell is eta2 / (h.h + 1); a cell's step from what it receives is eta1.

**The measure** is the one of `nano/ref/MEASURE.md` (I wrote mine before reading it; they agree): every one of the 111,540 bytes of the last tenth, each from the D bytes before it in the text, learning off; loss in nats and bits a byte; share named right. The unsmoothed tables' shares named right are the same in the two lines to the digit (26.98, 38.09, 47.15, 52.52, 53.65, 53.89).

## 2. L1: the answering cell (`PLAN_1.md`)

**Arm LS**: mismatch = indicator of the byte that arrived less the branch outputs; its loss is **crude** (outputs cut off below at 1e-3 and divided by their sum; the cell was never taught to give probabilities). **Arm P**: the branches compete, p = exp(a) / sum of exp(a), mismatch = the indicator less p. Everything else identical.

**Settings, chosen on development seed 1 by loss** (130 runs: a grid of eta1, eta2 and the birth of the return strengths at 256 cells and 2 readings, the best three of each arm at 8 readings, eta2 again at 1,024 cells): LS eta1 = 4, eta2 = 1/64; P eta1 = 1, eta2 = 1/4; **for both arms the best birth of the return strengths is nothing.** The claim runs are on the five reported seeds (73129, 93683, 116089, 406364, 545353); 8 readings, D = 10.

| mean of five seeds | 256 cells: nats; bits; named right | 1,024 cells: nats; bits; named right |
|---|---|---|
| arm LS, born at nothing; loss crude | 2.0593; 2.9710; 47.56% | 1.9846; 2.8632; 48.47% |
| **arm P, born at nothing** | **1.7917; 2.5848; 48.07%** | **1.7682; 2.5510; 48.63%** |
| arm LS, born at the earlier size (not tuned for it) | 2.2111; 3.1899; 42.84% | 2.2133; 3.1931; 41.22% (3 seeds) |
| arm P, born at the earlier size (not tuned for it) | 1.8275; 2.6365; 47.45% | 1.8416; 2.6569; 47.10% (3 seeds) |

The range over seeds is at most 0.011 nats and 0.5 points in every row.

Beside them:

| | nats | bits | named right |
|---|---|---|---|
| unsmoothed table of the last 2 bytes (loss crude, here) | 2.070 | 2.987 | 38.1% |
| unsmoothed table of the last 3 bytes | 1.796 | 2.591 | 47.2% |
| unsmoothed table of the last 4 bytes | 1.755 | 2.531 | 52.5% |
| unsmoothed tables of the last 5; 6 bytes | 1.927; 2.155 | 2.781; 3.109 | 53.7%; 53.9% |
| always the most frequent byte; the byte frequencies | ; 3.347 | ; 4.829 | 14.9% |

- **Arm P has the lower loss at both sizes on every seed** (by 0.268 and 0.216 nats). The hypothesis of the plan is not refuted.
- **The gain is in the loss, hardly in the naming** (0.5 and 0.2 points). Part of the gain is the crudeness of LS's conversion and not a worse knowledge of the text. My reading.
- **Tuned for naming, least squares names more.** REF_X
- **The best birth of the return strengths is nothing, in both arms.** With that birth the rule is back-propagation through this one-layer network (`loop_small_case/REPORT.md`, section 14). On this text the chosen arm therefore contains nothing that back-propagation, one byte at a time, does not. Born at the earlier size, arm P loses 0.036 nats with 256 cells and 0.073 with 1,024.
- Predictions: five of eight held (one emptily); failed: arm P below the best unsmoothed table (1.768 against 1.755); 1,024 cells at least 0.05 nats below 256 (0.024); LS's eta1.

L2_SECTIONS
