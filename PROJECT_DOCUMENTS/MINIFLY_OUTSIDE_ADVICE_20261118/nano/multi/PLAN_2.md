# nano/multi, second plan: the same arms on the text of nano/data (roadmap C3, step 2)

20 November 2026. Branch `claude/nano-multi-8mmhfm`. Written and committed before any run of this step that is used for a choice or a claim. Run before it: `check_text.py` (3,000 bytes a case; `check_text_output.txt`) and four runs of one reading each to measure the cost (eta1 = 1, eta2 = 1, return strengths at the earlier size: every line read, 256 cells 2.0975 nats and 1,024 cells 1.9392 nats, which are the two figures `nano/loop/PLAN_1.md` gives for the same runs in its own program; 64 lines with rewiring 2.1200; 32 lines + 32 pairs 2.1191). These four are used for no choice. Development level; no checker gate, seal or review is claimed.

Source and authorisation: as in `PLAN_1.md`.

## The question

On real text, at an equal number of strengths a cell, do cells that hold pair-connections (two lines and the cell) reach a lower loss than cells that hold single lines only? On the text a pair of lines at neighbouring delays is a pair of bytes.

## The test

`nano/data/tinyshakespeare.txt` and the split of its README: bytes 0 to 1,003,853 are the stream, read 8 times, one byte at a time, each next byte named before it arrives, nothing reset between readings; the last 111,540 bytes are never read by the learner. After the last reading, with learning off, every byte of the last tenth is scored from the 10 bytes before it: **loss in nats and bits a byte, and the share of next bytes named right**. This is `nano/loop/loop.py`'s stream and scoring (commit e0c864c), written again in `textm.py`.

## The exact learner

`multi.Sys` as in `PLAN_1.md`, with 65 byte values at 10 delays (650 lines) and **1,024 logistic cells**; rewiring every 4,096 bytes by the rule of `PLAN_1.md` (the weakest of all a cell's connections at least four periods old is withdrawn and another of the same kind is tried at strength 0). Pairs at birth are drawn uniformly over pairs of lines at two different delays.

**The answering cell is the one whose branches compete** (arm P of `nano/loop`, branch `claude/nano-loop-8mmhfm`, commit e0c864c: p = exp(a) / sum of exp(a); mismatch = 1 for the byte that arrived and 0 for the others, less p; the step eta2 / (h.h + 1) as before). That branch holds the rule, its numpy reference and its kernel, and no chosen settings or results yet; I use the rule in `kernm.c` (flag `comp`), and `check_text.py` compares `multi.py` with every line read against `nano/loop`'s numpy reference (largest difference 1e-14). The settings are therefore chosen here, on the edge arm.

**Arms** (1,024 cells; a cell also has its threshold; the answering cell has 66,625 numbers and the return strengths are 66,560 in every arm):

| Arm | single lines a cell | pairs a cell | numbers that learn in all |
|---|---|---|---|
| E64, the edge arm | 64 | 0 | 199,745 |
| M64 | 32 | 32 | 199,745 |
| P64 | 0 | 64 | 199,745 |
| E32 | 32 | 0 | 166,977 |
| M32 | 16 | 16 | 166,977 |
| P32 | 0 | 32 | 166,977 |

Beside them, as a reference and not an arm: every line read (650 strengths a cell, 798,785 numbers), seed 1 only.

## Settings chosen on the development seed (seed 1), by the loss on the last tenth after 2 readings

- Stage A, on the edge arm E64: eta1 in {1/4, 1, 4} and eta2 in {1/4, 1, 4}, return strengths born at the earlier size (9 runs). Stage B: at the best of A, return strengths born at 0.1 of that size (1 run). The best is the setting of every arm. If the best lies on the edge of the grid in eta1 or eta2, one further value beyond it is run and this is said.
- Stage C, on M64 at that setting: pairs tried at a turnover drawn uniformly or among the pairs of lines active at that byte; any two delays or neighbouring delays only (4 runs). The best is the setting of every pair arm; a setting other than (uniform, any two delays) is taken only if it is lower by 0.005 nats or more.
- The steps are chosen on the edge arm and not on the pair arms. This favours the edge arm if anything, and I state it as a limit.
- **A limit, stated now.** Seed 1 is both the development seed and one of the three reported seeds (1, 2, 3), as this line's brief fixes them, and the settings are chosen by the loss on the same last tenth that is reported.

## Hypothesis and what would refute it

After step 1 (pairs lost by 12 to 32 points on sums), **H2:** on the text too, at an equal number of strengths a cell, the pair arms do not reach a lower loss than the edge arm. **Refuted if** M64 or P64 has a mean loss over the three seeds lower than E64's by 0.01 nats or more with the lower loss on all three seeds, or the same holds for M32 or P32 against E32.

## Predictions (three seeds, mean, after 8 readings)

- Q1 (0.65): M64's loss is not below E64's by 0.01 nats or more.
- Q2 (0.6): P64's loss is above E64's by 0.05 nats or more.
- Q3 (0.6): M32's loss is not below E32's by 0.01 nats or more.
- Q4 (0.55): of the pairs held at the end in M64, at least half are at neighbouring delays (if stage C leaves any two delays).
- Q5 (0.6): E64 is 0.10 nats or more above every line read (seed 1).
- Q6 (0.7): E64 names more bytes right than the unsmoothed table of the last 2 bytes (38.1% in `nano/loop/PLAN_1.md`) and fewer than the table of the last 3 (47.2%).

## Cost (measured: about 75 to 80 s a reading with 1,024 cells, one core; 4 cores)

Stages A to C: 14 runs of 2 readings, about 2.6 minutes each: about 37 core-minutes. Claim runs: 18 runs of 8 readings and one reference, about 10.5 minutes each: about 200 core-minutes. About 4 core-hours, about one hour of wall time.

## What is not run

Other cell counts, other numbers of delays, more than 8 readings; the least-squares answering cell on the arms; pair-specific steps or turnover; the numpy program at length (3,000 bytes a case against the kernel); the references of `nano/ref` (another line's; quoted if present when this is reported).

## Entered on 20 November after stages A to C (seed 1, 2 readings), before any claim run of a pair arm was started

Loss on the last tenth after 2 readings, nats (`res/t2_dA_*.json`, `t2_dB_*.json`, `t2_dC_*.json`).

Stage A, the edge arm E64 (rows eta1, columns eta2):

| eta1 \ eta2 | 1/4 | 1 | 4 |
|---|---|---|---|
| 1/16 | | 2.0615 | |
| 1/4 | 2.1915 | **1.9899** | 3.3001 |
| 1 | 2.1455 | 2.0000 | 2.8703 |
| 4 | 2.1443 | 2.0109 | 3.2510 |

The best, eta1 = 1/4, lay on the edge of the grid in eta1, so eta1 = 1/16 was run as the plan says: worse. Stage B: return strengths born at 0.1 of the earlier size: 2.0069, worse. **Setting of every arm: eta1 = 1/4, eta2 = 1, return strengths born at the earlier size.**

Stage C, M64 at that setting:

| pairs tried at a turnover | any two delays | neighbouring delays |
|---|---|---|
| drawn uniformly | 2.0886 | 2.0897 |
| among the pairs of lines active at that byte | 2.0499 | **2.0332** |

**Setting of every pair arm: tried among the active pairs, neighbouring delays** (0.055 nats below the default). So on the text every pair-connection names two bytes that follow one another, at some distance back; Q4 has no object and is scored as not tested.

After 2 readings on this seed the edge arm is at 1.9899 and the best mixed arm at 2.0332. The claim runs (8 readings; seeds 1, 2, 3) were queued after these lines: `jobs_r2.txt`; the reference (every line read) and E64 on seed 1 had been started when stage A and B were done, since they do not depend on stage C.

## An addition, entered at 15:10 UTC on 20 November, when only the reference and E64 on seed 1 had finished and no pair arm's claim run had been read

`nano/loop` has since pushed its chosen settings for arm P with every line read (commit 1f7b4bf: eta1 = 1, eta2 = 1/4, return strengths born at nothing; chosen there on 256 cells). My stages A and B did not include return strengths born at nothing, and chose eta1 = 1/4, eta2 = 1, born at the earlier size, on the edge arm after 2 readings. So that it can be seen whether the comparison of the arms depends on this, three further runs on seed 1 at `nano/loop`'s setting (`rscale=0`), 8 readings: E64, M64, P64 (`jobs_r2x.txt`). No prediction is registered for them; they are reported as an addition and change no claim run. The reference "every line read" of this plan is at my setting and is therefore not `nano/loop`'s best figure for that system; theirs is quoted beside it when it exists.

## The claim runs (entered 20 November; table in `report2.txt`, made by `report2.py` from `res/t2_r_*.json`)

*A correction first:* the heading of the addition above gives 15:10 UTC; the machine's clock read 15:05 when it was committed. The rest of that entry stands.

The last tenth of the text, never read, after 8 readings of the stream; 1,024 cells; seeds 1, 2, 3 and their mean.

| Arm | nats: 1, 2, 3 | mean nats | bits | named right %: 1, 2, 3 | mean % |
|---|---|---|---|---|---|
| E64: 64 single lines | 1.8857, 1.9066, 1.8969 | **1.8964** | 2.736 | 46.65, 46.16, 46.66 | 46.49 |
| M64: 32 lines + 32 pairs | 1.9314, 1.9365, 1.9447 | **1.9375** | 2.795 | 45.53, 45.19, 45.22 | 45.31 |
| P64: 64 pairs | 2.2770, 2.2916, 2.2882 | **2.2856** | 3.297 | 37.09, 36.63, 37.44 | 37.05 |
| E32: 32 single lines | 1.9682, 1.9682, 1.9545 | **1.9637** | 2.833 | 44.84, 44.52, 45.14 | 44.83 |
| M32: 16 lines + 16 pairs | 2.0040, 1.9981, 2.0156 | **2.0059** | 2.894 | 43.78, 43.81, 43.54 | 43.71 |
| P32: 32 pairs | 2.3715, 2.4125, 2.4185 | **2.4008** | 3.464 | 35.37, 34.34, 34.14 | 34.61 |

Beside them. Every line read (650 strengths a cell, my setting, seed 1): 1.8373 nats, 48.23%. Smoothed tables of the last k bytes from `nano/ref` (branch `claude/nano-ref-8mmhfm`, commit 9378f61, `res/tables.json`, their Kneser-Ney figures): k = 2: 2.0381 nats, 38.08%; k = 3: 1.7191 nats, 47.22%; k = 6 (their best): 1.5165 nats, 54.80%.

**Outcome.** At an equal number of strengths a cell, no pair arm reaches a lower loss than its edge arm, on any seed. The mixed arms are 0.041 nats (64 strengths) and 0.042 nats (32 strengths) above their edge arms; the arms with pairs only are 0.39 and 0.44 nats above. **H2 is not refuted.**

**Predictions scored.**
- Q1 held: M64 is 0.041 above E64.
- Q2 held: P64 is 0.389 above E64.
- Q3 held: M32 is 0.042 above E32.
- Q4 not tested (stage C chose neighbouring delays, so every pair is at neighbouring delays).
- Q5 failed: E64 is 0.048 nats above every line read on seed 1, where I had said 0.10 or more. A cell that reads 64 of the 650 lines and rewires loses little here.
- Q6 held: E64 names 46.49% right, between the tables of the last 2 bytes (38.1%) and of the last 3 (47.2%).

**What I read from it (my readings; not tested here).**
1. The order of the arms is the same as on the sums: single lines, then half and half, then pairs alone, at both sizes. On the text the mixed arm loses little (0.04 nats, about 1.2 points of bytes named right) and the pairs alone lose much.
2. As on the sums, a pair-connection is active far less often than a line. In the mixed arm a little more than half of the withdrawals fall on pairs.
3. In every arm the cells end with more connections at the near delays than at the far ones (about twice as many at the present byte as nine bytes back), by rewiring alone.
4. The edge arm with 64 of 650 lines a cell (199,745 numbers) is within 0.05 nats of every line read (798,785 numbers) at the same setting.

**Limits.**
- The sending ends are input lines; only the receiving cell decides.
- The steps were chosen on the edge arm after 2 readings, on seed 1, which is also a reported seed, by the loss on the same last tenth; no step, period or withdrawal rule was chosen for the pair arms. The result says that pairs lose under the edge arm's settings.
- The setting is not `nano/loop`'s chosen one; see the addition above and its outcome below.
- Three seeds; the differences between seeds (about 0.02 nats within an arm) are smaller than the differences claimed between arms.
- Development level; repeated by no one; no checker gate, seal or review is claimed.

**Not run:** as listed in the plan; and any arm on more than three seeds.

## Outcome of the addition (entered 20 November; `res/t2_x_*.json`, last lines of `report2.txt`)

Seed 1, 8 readings, at `nano/loop`'s setting for every line read (eta1 = 1, eta2 = 1/4, return strengths born at nothing): E64 1.9499 nats (43.69%), M64 1.9759 (43.05%), P64 2.3674 (35.02%). At my setting the same seed gave 1.8857, 1.9314 and 2.2770. The order of the three arms is the same at both settings (single lines, then half and half, then pairs alone); the cells that read 64 lines do better at the setting chosen on them than at the one chosen for cells that read every line. One seed; no prediction was registered.
