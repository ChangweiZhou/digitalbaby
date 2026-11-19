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
