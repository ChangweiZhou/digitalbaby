# Plan 5: carrying the byte deeper: few strong connections, a bounded gain, long axons (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 21 November 2026: "信息无法传递到很远的细胞肯定不是我们想要的。我们至少需要传递到十层以上。…2）继续做实验。" **Program:** `TISSUE_EVENT/` at the commit of this plan. With the new switches off it gives the same numbers as for plans 3 and 4 (checked with `==`, 2,048 bytes).

**Where things stand.** Distances in this tissue run from 1 to 9-11 connections (input lines on the left edge, output end on the right, radius 1/8). S(d,0) in A2 for every d (`depth_read.py`, E63's readings with the list of distances extended), plan 3's N75T: first form 0.992, 0.577, 0.167, then chance from d = 4; coincidence form 0.982, 0.758, 0.391, 0.057, then chance. The signal halves or worse at each connection and is gone by four.

**Three changes, one at a time, each on plan 3's N75T of its form (C_E 3/4, C_I 1/4, turnover with trial and withdrawal by use):**

| Arm | Change | Why |
|---|---|---|
| F | making period 8,192 steps and trial threshold 1 (fewer, stronger connections: with the total held, fewer inputs share it) | the deepest tissue the reviewer measured on 19 November had about 5 connections a cell |
| G | a cell's own gain (`--gain`), moved once a byte so that its mean output goes toward 7/8 of its sources' mean value, at rate 1/1024 a byte, within [1/64, 2] | the reviewer's gain by ratio carried the byte to four connections in a tissue without learning; at a cap of 8 and rates 1/128 to 1/1024 it flooded this tissue in short trials (1,500 to 1,800 of 2,048 cells active), so the cap is 2 |
| L | one excitatory cell in 8 (198 cells) can be a source for cells within 1/2 instead of 1/8, given at birth; such connections are made and kept by the same turnover and trial | a share of long axons is standard in cortex and shortens paths |
| FL | F and L together | |

**H5.** In each form, at least one arm makes the farthest distance at which S(d,0) in A2 is at least 0.1103 larger than N75T's (3 in both forms), while the number of active cells at the end of every 1,024-byte stretch stays below 1,024 and S(1,0) at least 0.1103. **Refuted if** no arm does so in a form.

| No. | Prediction | Chance |
|---|---|---|
| G1 | F raises the farthest distance above level in some form | 0.35 |
| G2 | G raises it in some form | 0.25 |
| G3 | L raises it in some form | 0.3 |
| G4 | Some arm has S(d,0) at least 0.1103 at d = 6 or beyond | 0.1 |
| G5 | G floods (above 1,024 active cells at the end of some stretch) in some form | 0.5 |

**Arms.** Seed 73129; 16,384 bytes; counts every 1,024 bytes; E63's readings and `depth_read.py`. 4 arms × 2 forms = 8 tissues. **Cost** up to about 13 ms a byte if a tissue floods: at most about 15 minutes a tissue; limit 2 CPU-hours. **Not run:** a rule of strength by order in time (STDP); inhibitory plasticity to balance a gain; other seeds; a larger tissue.

## Outcome (entered 21 November 2026; eight runs, exit 0; `p5_tables.py`)

### first

| Arm | S(d,0) in A2 for d = 1 ... 11 | Farthest d at level | Connections at the end | Last 4,096 bytes | Active cells at stretch ends, min / max | ms a byte core | H5 met |
|---|---|---|---|---|---|---|---|
| N75T (plan 3) | 0.992 0.577 0.167 0.022 0.022 0.022 0.022 0.022 0.022 0.022 0.022 | 3 | 17.5 | +1.2% | not recorded | 1.68 | – |
| F | 1.000 0.964 0.688 0.381 0.046 0.049 0.023 0.022 0.022 0.022 0.022 | 4 | 10.2 | +6.7% | 60 / 195 | 1.06 | yes |
| G | 0.903 0.374 0.143 0.048 0.022 0.023 0.022 0.022 0.022 0.022 0.022 | 3 | 23.7 | +1.8% | 18 / 446 | 1.99 | no |
| L | 0.999 0.455 0.075 0.022 – – – – – – – | 2 | 21.2 | +3.2% | 37 / 677 | 3.42 | no |
| FL | 1.000 0.927 0.512 0.369 0.255 – – – – – – | 5 | 10.3 | +12.8% | 14 / 1435 | 2.69 | no |

### coincidence

| Arm | S(d,0) in A2 for d = 1 ... 11 | Farthest d at level | Connections at the end | Last 4,096 bytes | Active cells at stretch ends, min / max | ms a byte core | H5 met |
|---|---|---|---|---|---|---|---|
| N75T (plan 3) | 0.982 0.758 0.391 0.057 0.022 0.022 0.022 0.022 0.022 0.022 0.022 | 3 | 18.8 | +4.1% | not recorded | 3.19 | – |
| F | 1.000 0.987 0.795 0.558 0.268 0.162 0.070 0.022 0.022 0.022 0.022 | 6 | 9.9 | +9.6% | 67 / 498 | 1.78 | yes |
| G | 0.919 0.434 0.227 0.124 0.073 0.044 0.028 0.026 0.022 0.022 0.022 | 4 | 23.5 | -6.3% | 24 / 1869 | 5.02 | no |
| L | 0.993 0.525 0.145 0.026 0.022 0.022 – – – – – | 3 | 20.8 | +5.2% | 55 / 986 | 4.71 | no |
| FL | 0.999 0.919 0.635 0.247 0.139 0.076 – – – – – | 5 | 9.3 | +12.9% | 9 / 1524 | 2.78 | no |


"–": no cell at that distance in window A2. Active cells at the ends of stretches were not recorded for plan 3's runs.

| | Outcome |
|---|---|
| H5 | **Held in both forms, through F.** Fewer, stronger connections (about 10 a cell) carry the byte one connection farther in the first form (S(4,0) 0.381) and three farther in the coincidence form (S(5,0) 0.268, S(6,0) 0.162), with at most 195 and 498 cells active at the end of a stretch |
| G1 | Held |
| G2 | Held as written (coincidence form, d = 4, 0.124), but that tissue floods (up to 1,869 active cells), so it does not meet H5 |
| G3 | Failed: long axons shorten the paths (no cell farther than 4 or 6 connections) but carry less at each distance |
| G4 | Held: coincidence form F, S(6,0) = 0.162 |
| G5 | Held: G floods in the coincidence form |

**Readings (development level, one seed).**

1. **The fan-in is the strongest lever found so far.** With the total incoming strength held, about 10 inputs a cell in place of 18 to 30 make each input about twice as strong, and the signal falls much more slowly with distance: in the coincidence form 1.000, 0.987, 0.795, 0.558, 0.268, 0.162 at d = 1 to 6, against 0.982, 0.758, 0.391, 0.057 then chance. F is also cheaper (1.06 and 1.78 ms a byte).
2. **F has not settled**: +6.7% and +9.6% over the last 4,096 bytes (the making period of 8,192 steps adds slowly). Where it stops is not known.
3. **A gain and long axons both bring floods** (up to 1,435 to 1,869 of 2,048 cells active at the end of some stretch) without inhibition that follows them. Long axons alone do not flood the first form but halve S(2,0).
4. Ten connections are not reached: the farthest at level is 6, and the tissue's farthest cells are 9 to 11 connections from the lines.
