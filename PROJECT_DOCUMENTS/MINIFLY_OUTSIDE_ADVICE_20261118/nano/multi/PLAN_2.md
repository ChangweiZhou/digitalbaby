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
