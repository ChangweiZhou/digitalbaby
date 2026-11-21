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

## Outcome

(to be entered after the runs)
