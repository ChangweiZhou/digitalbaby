# Where net13 loses the text in the t23 life: a put-back diagnosis (23 November 2026)

This is development-level work: diagnosis only, with no new mechanism. It ran on one machine, at seeds 1 and 2, with cue=1. **No gate, seal or outside review is claimed.**

- **Source:** `claude/magical-albattani-sei232` @ d0f0a9da. Branch `claude/newnet-textloss-sei232`.
- **New files:** `textloss.py`, `textloss_runs/` (`run.sh`, `s{1,2}_rcall{0,1}.log`, `diag.jsonl`). No existing file was changed.
- **Method:** `textloss.py` executes `t23_life_rules.py` unchanged, with one line added (a hook after each probe), so the life is t23's.
- **Reproduction check:** seed 1's probe lines and results are identical to `net13_runs/full_s1_rcall{0,1}.log`:
  - rc_all=0: 777,024 bits;
  - rc_all=1: 762,455 bits.
- **Settings:** OMP_NUM_THREADS=1. Four lives ran in parallel, about 31 min each, plus about 3 min of diagnosis each.

## Method
- **Snapshots.** The whole net is copied at line 17000 (after the text) and at line 29000 (the end).
- **Put-back.** Each test starts from a fresh copy of the end net:
  1. one part is put back as it was at 17000;
  2. the t23 text probe is run (1,000 never-read bytes, learning off).
  - Because each test starts from a fresh copy, nothing put back stays.
- **"Except X" tests.** Everything plastic is put back except one group.
- **The base probe differs from the logged probe.** The probe here starts right after the t23 probe's own text probe. The holders' clocks still carry that text, so the base number is higher than t23's logged number:
  - end, this probe: 0.257–0.275;
  - end, logged: 0.238–0.255.

  All comparisons below are made within this probe.
- **Parts that were put back:**

  | part | what it contains |
  |---|---|
  | context cells' connections | rsrc, rw, gsrc |
  | reliability | rel |
  | recycling state | `use`: which cell of a column is taken when nothing predicts |
  | Ac | context compartments' readout |
  | holders' A | all together and each of the 16 holders |
  | holders' internal | vy, m: the only other plastic state in net8c's C configuration; Wc, Wg, U, b and gline are fixed |
  | B | lag readout |
  | G | relay gate |
  | rule cells | prototypes Wr/ruse, Gr, Gc |
  | all readouts | Ac, holders' A, B, G, Gr, Gc |
  | everything | everything plastic |

## Text probe with each part put back
Columns are s1 rc1 / s1 rc0 / s2 rc1 / s2 rc0.

| net state | s1 rc_all=1 | s1 rc_all=0 | s2 rc_all=1 | s2 rc_all=0 |
|---|---|---|---|---|
| copy of the net at 17000 | 0.355 | 0.349 | 0.346 | 0.347 |
| **end (nothing put back)** | **0.257** | **0.267** | **0.274** | **0.275** |
| ctx: all (rsrc rw gsrc use rel) | 0.277 | 0.275 | 0.286 | 0.282 |
| ctx: connections (rsrc rw gsrc) | 0.274 | 0.269 | 0.286 | 0.290 |
| ctx: recycling state (use) | 0.263 | 0.262 | 0.274 | 0.277 |
| ctx: reliability (rel) | 0.257 | 0.267 | 0.274 | 0.275 |
| Ac (context compartments' readout) | 0.296 | 0.302 | 0.305 | 0.307 |
| **holders' A (all 16)** | **0.326** | **0.328** | **0.315** | **0.312** |
| one holder's A (range over the 16 holders) | 0.259–0.267 | 0.267–0.275 | 0.278–0.282 | 0.278–0.285 |
| holders' internal (vy, m) | 0.228 | 0.240 | 0.245 | 0.245 |
| holders' vy only | 0.265 | 0.270 | 0.279 | 0.279 |
| holders' m (adaptation) only | 0.241 | 0.248 | 0.260 | 0.258 |
| B (lag readout) | 0.275 | 0.279 | 0.287 | 0.286 |
| G (relay gate) | 0.258 | 0.267 | 0.275 | 0.276 |
| rule cells: all (Wr ruse Gr Gc) | 0.271 | 0.269 | 0.269 | 0.276 |
| rule cells: Gc (gains; only acts with rc_all=1) | 0.264 | 0.267 | 0.273 | 0.275 |
| rule cells: Gr | 0.267 | 0.269 | 0.277 | 0.275 |
| rule cells: prototypes (Wr ruse) | 0.251 | 0.253 | 0.274 | 0.269 |
| **all readouts (Ac, holders' A, B, G, Gr, Gc)** | **0.338** | **0.341** | **0.334** | **0.335** |
| everything plastic | 0.342 | 0.341 | 0.340 | 0.336 |

### Everything plastic put back except one group

| left at its end state | s1 rc1 | s1 rc0 | s2 rc1 | s2 rc0 |
|---|---|---|---|---|
| (nothing: everything put back) | 0.342 | 0.341 | 0.340 | 0.336 |
| **holders' A** | **0.294** | **0.298** | **0.294** | **0.301** |
| Ac | 0.338 | 0.335 | 0.328 | 0.321 |
| ctx (all five) | 0.339 | 0.336 | 0.329 | 0.331 |
| B | 0.335 | 0.337 | 0.329 | 0.332 |
| G | 0.342 | 0.340 | 0.339 | 0.337 |
| rule cells | 0.342 | 0.348 | 0.342 | 0.339 |
| holders' internal (vy, m) | 0.351 | 0.347 | 0.348 | 0.342 |

**Notes**
- **Holders' internal state.** Putting back the holders' internal state (m especially) lowers the probe. It also makes the "except" row higher when it is left out. The likely reason is that m (a slow adaptation mean) put back without the holders' activity h is inconsistent with that activity. I read this as an artefact of the method, not as a place where text is stored.
- **What everything-plastic does not restore.** The remaining 0.006–0.013 below the 17000 copy is the dynamic state.

## Overlap of active cells, text vs rule lines
- **Method.** Learning was off. The rule activity is 60 fixed copy, rev and succ items each; the text activity is the probe text.
- **Cell counts.** Context cells are 16 regions × 1,024 cells; holder expansion cells are 16 × 2,048.
- **rc_all=0 and rc_all=1.** The context cells and the holders' expansion do not depend on the readouts, so their activity is the same with rc_all=0 and 1 (checked: identical numbers).

| | seed 1, 17000 | seed 1, end | seed 2, 17000 | seed 2, end |
|---|---|---|---|---|
| context cells active on text (n) | 6,833 | 5,832 | 6,824 | 5,808 |
| … of these also active on rule lines | 64.1% | 68.6% | 63.1% | 69.2% |
| text activations falling on cells also active on rule lines | 83.3% | 87.1% | 83.1% | 87.3% |
| holder expansion cells active on text (n) | 8,452 | 8,102 | 8,334 | 7,874 |
| … of these also active on rule lines | 73.2% | 78.3% | 71.6% | 78.3% |
| text activations falling on cells also active on rule lines | 82.3% | 84.6% | 82.0% | 84.4% |

**Change between 17000 and the end (seed 1 / seed 2)**

*Context cells.* The 17000 text-active cells are compared with all other context cells.

| measure | text-active at 17000 | all other cells |
|---|---|---|
| got new connections | 80.9% / 80.2% | 76.8% / 77.8% |
| learned as the winner | 94.4% / 93.9% | 95.5% / 96.3% |
| relative change of rw | 0.137 / 0.137 | 0.137 / 0.134 |

- **Same text cells at both times?** The Jaccard overlap of the text-active cell sets at 17000 and at the end is 0.64 / 0.63 (context cells) and 0.59 / 0.59 (holder expansion).
- **Recruited by rule lines.** Of the context cells that were text-active at 17000, 60.7% / 61.1% are active on rule lines at the end.

*Readouts.* Relative change of the readout strengths, rc_all=1 / rc_all=0.

| readout | text-active cells | other cells |
|---|---|---|
| holders' A, seed 1 | 0.86 / 0.89 | 0.86 / 0.89 |
| Ac, seed 1 | 0.62 / 0.62 | 0.82 / 0.84 |

Seed 2 is the same within 0.01.

**rc_all=1 gains** (mean over the probe; 33 compartments: 16 context, 16 holders, lag readout)
- **At 17000, on text:**
  - context compartments 0.974–1.10;
  - holders 0.98–1.03;
  - lag readout 0.96.
- **At 17000, on rule lines:**
  - context compartments 0.99;
  - holders about 0.996;
  - lag readout 1.11.
- **At the end:** text 0.98–1.04; rule lines 0.97–0.98 for the context compartments; lag readout 1.19 on rule lines and 1.04–1.06 on text.
- **What this means:** the gains differ between text and rules by only a few percent. This fits rc_all's small effect on the text.

## Where the loss lies (plain statement)
1. **Mostly in the readout strengths, and most of all in the holders' readout A.**
   - Putting back all readouts restores 83–90% of the loss, measured as the gap between the 17000 copy and the end net.
   - Putting back the holders' A alone restores 51–74% of the loss: 0.257 → 0.326 at s1 rc1, and 0.274 → 0.315 at s2 rc1.
   - The holders' A is the only group whose absence, with everything else put back, still costs a lot: 0.035–0.048.
   - Within the holders the loss is spread: no single holder restores more than about 0.01. The clock bands do not divide it.
   - The context compartments' readout Ac restores 40–44% alone, but is largely redundant once the holders' A is back (0.004–0.015).
   - The lag readout B restores 15–18% alone.
2. **Recycling of context cells is a minor part.**
   - After the text, about 80% of the cells the text used got new connections, and about 94% learned as winners. This is no more than other cells.
   - Putting back the context cells' connections restores only 0.002–0.017 (2–21% of the loss; all five context parts together 10–20%).
   - Putting back the `use` (recycling) state restores about 0.
   - With everything else restored, the end-state context cells cost only 0.003–0.011.
3. **Not in the gates.**
   - The relay gate G, the rule cells' prototypes, Gr and the rc_all gains Gc each restore about 0.
4. **Why the readouts are overwritten.**
   - The text and the rule lines share most active cells: 83–87% of the context cells' text activations and 82–85% of the holders' fall on cells that rule lines also drive.
   - The holders' readout of text-active expansion cells changed as much as everyone else's (relative change 0.86–0.89).
   - The succ/copy/rev training thus rewrites, through the shared mismatch, the readout strengths of cells the text also uses.
   - The rule cells' gains do not separate text from rules enough to prevent this. They differ by only a few percent between text and rule lines.

In short, the text is lost in the readouts, mostly the holders' readout A, and that loss is spread over all 16 holders. It is not lost in the recycling of context cells or in the gates. The same picture holds for rc_all=0 and 1 and for seeds 1 and 2.

## Seed 2 lives (first runs of net13 at seed 2; not in NET13_REPORT)

| | text 17000 → end (logged) | succ end | bits |
|---|---|---|---|
| rc_all=0 | 0.340 → 0.250 | 0.789 | 776,645 |
| rc_all=1 | 0.344 → 0.255 | 0.989 | 762,537 |

The succ-kept effect of rc_all=1 seen at seed 1 repeats at seed 2. That makes 2 seeds; it is not confirmed on 3.

## Limits / NOT_RUN
- **Coverage.** cue=1 only; seeds 1–2 only; one probe (1,000 bytes). Differences of about 0.005 are within the probe's noise; the probe was not repeated.
- **No intermediate snapshot.** There is none at 20000–23000, where the loss happens. The split between the succ phase and the later copy/rev phases is not measured.
- **Holders' m.** It could not be tested cleanly: its put-back is inconsistent with the dynamic state.
- **Overlap numbers** use fixed probe items with learning off. They are not the activity during learning.
