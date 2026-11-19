# Long-axon tissue L8-256, does it stand? The helper's runs (19 November 2026)

**Source and provenance.** This is a copy of the reviewer's tissue folder from branch `claude/vibrant-cerf-sje726` at 10876732, made for the reviewer's task of 09:52 UTC.
- The copy includes `tis.py`, `tissue.c` (build `-ffp-contract=off`), `far.py`, `loops.py` and `deps/`.
- `far_rev/` holds the reviewer's own far.py results on L8-256 for comparison.
- No file of the reviewer's is changed.

**The tissue, L8-256.** `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2 tau=u2-256`.

**The new program, `stand.py`.** It measures everything the task asks for in one run of a tissue (see its header).
- **The draw** of long-axon cells and time constants:
  - by default, the reviewer's draw;
  - with `draw=own`, the helper's own draw: Random(seed + 710000) and Random(seed + 700000).
- **Levels:**
  - lag 0 must be above 0.1103 under both levels;
  - lag L ≥ 1 must be above c0(L) + 0.02 (level A) or above C<L(L) + 0.02 (level B).
- **Seeds:**
  - the reviewer's three: 73129, 93683 and 116089;
  - twelve more, drawn by Random(20261120).sample(range(1000, 1000000), 12): 465973, 614332, 545353, 406364, 308924, 754734, 224268, 746958, 146821, 438778, 112290 and 969620.
- **Run list:** `stand_list.txt`; the runs are started by `runs.sh`.

**Check.** With the reviewer's draw on seed 93683, `stand.py` gives the same tool values as the reviewer's `far_rev/L8_256_s93683.json`, at every entry compared:
- the nine entries;
- the cells near the output lines, at lags 0 to 4.

## What the runs measured

Every figure below is in `STAND_ROWS_16k.txt` (one line a tissue), `STAND_TABLE.txt` (all tissues and windows) and `stand/*.json`.
Throughout:
- **Near** means the cells near the output lines (far.py's definition, 569–595 cells), read by the tool as it is, at lags 0–4.
- **The nine** means the nine entries: lag 0, 1, 2 at d = 1, 2, 4.

### 16,384 bytes (window 12288–16383), 15 seeds × 2 draws = 30 tissues

**Nine entries**
- Above level A in all 30.
- Above level B in 27. The three that fail:

  | Draw | Seed | Entry | Value | Level |
  |---|---|---|---|---|
  | reviewer's | 465973 | S(4,2) | 0.100 | 0.110 |
  | own | 116089 | S(4,2) | 0.111 | 0.115 |
  | own | 112290 | S(1,2) | 0.095 | 0.107 |

**Near the output lines, by lag**
- **Lag 0** is below 0.1103 in 7 of 30:

  | Draw | Seed | Lag 0 |
  |---|---|---|
  | reviewer's | 465973 | 0.100 |
  | reviewer's | 112290 | 0.071 |
  | reviewer's | 146821 | 0.102 |
  | own | 406364 | 0.108 |
  | own | 969620 | 0.075 |
  | own | 438778 | 0.081 |
  | own | 224268 | 0.094 |

- **Lag 1** is below its level in 1: own draw, 438778, 0.092 against 0.098.
- **Lag 2** is above level A in all 30. It is below level B in 5:
  - reviewer's draw: 465973, 545353;
  - own draw: 112290, 224268, 116089.
- **Lag 3** is below c0(3) + 0.02 in all 30.

**Every entry above both levels** (the nine and near lags 0–2):
- reviewer's draw, 11 of 15 seeds: 73129, 93683, 116089, 224268, 308924, 406364, 438778, 614332, 746958, 754734, 969620;
- own draw, 9 of 15: 73129, 93683, 146821, 308924, 465973, 545353, 614332, 746958, 754734.

**Other measures**
- Divergence after 512 bytes: 0 in all 30.
- Loops among excitatory cells, in 7 of 30 (largest loop in cells):
  - reviewer's draw: 93683 (2), 112290 (3), 746958 (5);
  - own draw: 406364 (5), 146821 (10), 224268 (3), 116089 (4).
- Near cells: largest output 1.00 in all 30; share above 1/2 0.66–0.97.
- Connections a cell: 2.37–2.82, largest 10–69. Share of connections from long-axon cells: 0.22–0.37.

### 65,536 bytes, seeds 73129, 93683, 116089, four windows (bytes 0, 20480, 40960, 61440 + 4,096)

**Reviewer's draw**
- **From the second window on:** all nine entries and near lags 0–2 are above both levels on all three seeds.
- **First window, seed 93683:**
  - S(4,0) = 0.099 is below both levels;
  - S(4,2) = 0.104 is below level B;
  - near lags 0–2 are below.
- **First window, seed 116089:** S(4,2) and near lag 2 are below level B.

**Own draw**
- **Seed 73129:** everything above both levels from the second window on. In the first window, near lags 0–2 (0.089, 0.092, 0.095) are below; near lag 2 is below level B only.
- **Seed 93683:** everything above both levels from the second window on. In the first window:
  - S(4,2) = 0.079 is below both levels;
  - near lags 0–2 are below.
- **Seed 116089:**
  - First window: S(4,2) is below level B, and near lags 0–2 are below.
  - Second window: S(4,2) = 0.101 is below level B (0.104), and near lag 2 = 0.094 is below level B.
  - Third window: near lag 2 = 0.104 is below level B (0.105).
  - Fourth window (bytes 61440–65535): all above both levels, with near lag 2 = 0.104 against 0.103.

**Both draws**
- Divergence 0 in all six runs.
- Made in the last quarter: 22–289; withdrawn: 23–252.
- Loops among excitatory cells at the end: only own draw 116089 (one loop of 4 cells).

## Plainly
- **Nine entries.** On fifteen seeds and both draws, L8-256 puts the nine entries above level A in every tissue at 16,384 bytes. Under level B, three tissues fail one lag-2 entry.
- **Cells near the output lines.** The present byte, one back and two back are above both levels in 22 of 30 tissues:
  - lag 0 fails in 7 tissues;
  - lag 1 fails in 1 (a tissue where lag 0 also fails);
  - lag 2 fails under level B in 5;
  - lag 3 is never above its level.
- **Which seeds fail.** The failing seeds differ between the reviewer's draw and the helper's draw.
- **The long life.** Over 65,536 bytes, most of the failures are in the first window. From the second window on, all entries are above both levels on the three seeds, with one exception: in the helper's draw of seed 116089, near lag 2 and S(4,2) are near level B in windows 2 and 3.
