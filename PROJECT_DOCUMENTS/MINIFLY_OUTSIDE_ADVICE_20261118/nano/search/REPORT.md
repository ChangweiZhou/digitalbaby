# Line `nano/search`: can the kinds of cells be found by a search judged by learning, with one program and one setting on several tasks? (20 November 2026)

**Status.** Development level. One program of mine, one frozen setting, five reported seeds on sums and one run on each prose arm. No checker gate, seal or outside review is claimed. The task and the question are the coordinating session's, under Mr. Xu's steer as that session quoted it (head of each plan). The readings marked "mine" are this session's. Plans: `PLAN_1.md` (system, check, the choice of one setting on development material), `PLAN_2.md` (the tasks), `PLAN_3.md` (twice the kinds); each was committed before its runs and carries its outcome.

## 1. What was built

Cells that count, read by an answering cell that learns a trust for each kind and level of experience (`nano/table/trust.py`), with every choice of kind taken from the designer and given to a search:

- **Lines** a cell may be keyed on: the byte 1 to 16 places back, and the system's own guesses of the bytes 1 to 4 ahead, from two layers of readouts (24 lines). **A kind** is a set of 1 to 6 lines. Kinds are born at random with no trust; a cell exists for each combination of values that occurs.
- **Search**: 128 kinds alive. Every 8,192 bytes the 16 kinds that earned least (earnings = trust times the mismatch term, summed; no extra computation) are withdrawn and their cells freed, and 16 are born: half at random, half as a one-line change of a kind that earned.
- **One addition of mine** (`PLAN_1.md`, addition 1): the system keeps the last 131,072 bytes it read, and a newborn kind counts over them at once; its trust still starts at nothing and is earned on later bytes only.
- **Store**: 2^22 hashed cells for every task; a new cell displaces the least-counted cell of its bucket when the bucket is full.
- **One setting for every task**, chosen on development material (sums seed 1; 350,000 bytes of tiny Shakespeare's stream) and frozen: shrinking step, eta 0.03, e 2, no cap on a cell's weight. Only the number of byte values differs between texts (14 or 65).
- `kern_q.c` (compiled) and class `Ref` in `search.py` (plain Python) agree to 2e-15 on five cases, with stores, counts, written bytes and kinds equal (`check_output.txt`). The choice of which kinds die and are born is one piece of Python that both use.

Nothing in the program refers to a word end, the place of `=`, the length of a number or a list of delays. What I laid down, and what it costs in generality, is in `PLAN_1.md`, section 2, last paragraph (16 places back, 4 ahead, two layers, sizes to 6, the levels of experience, and the numbers above).

## 2. The table across tasks (frozen setting; files `res/c_*`, `res/c3_*`; full table `report_2.txt`)

Sums: never-shown pairs, the system writing its own answer bytes, whole answer right, %; five reported seeds 73129, 93683, 116089, 406364, 545353; mean of the last three readings, each seed; median; then the median of the last reading. Size and CPU are medians of a run.

| Task | Searched kinds (128 alive) | Control: the same 128 random kinds never changed | Hand-built kinds, other lines, as they report |
|---|---|---|---|
| Two-digit sums, 40,000 lines | 68.3, 47.4, 59.5, 43.9, 67.9; **59.5**; last 64.6 | 0.5, 45.7, 0.5, 18.9, 1.4; **1.4** | 77 to 81 (at most four of the last nine bytes); 91.4 (with look-ahead of depth three) |
| size | 1.69 M cells, 16.7 M counts above zero, 0.5 M cells displaced, 1,264 kinds tried, 60 CPU-s (with the 20,000 lines of subtraction) | 0.39 M cells, 4.9 M counts, 128 kinds, 21 CPU-s | 3.27 M cells, 34.6 M counts, 96 CPU-s (look-ahead, `nano/ahead2`) |
| Two-digit sums, 400,000 lines | 73.5, 28.3, 40.5, 56.7, 50.7; **50.7**; last 49.8. Median by lines: 54.4 at 10,000; 64.6 at 40,000; 72.4 at 100,000; 61.6 at 200,000; 32.8 at 300,000; 49.8 at 400,000 | 0.0, 15.5, 0.3, 20.7, 1.1; **1.1** | 75.2 (four delays with the cap, `nano/sparse`) |
| size | 1.70 M cells, 21.1 M counts, 11.0 M displaced, 7,904 tried, 435 CPU-s | 0.43 M cells, 7.6 M counts, 202 CPU-s | |
| Life: addition 40,000 lines, then subtraction 20,000 | addition before 64.6, after 43.4 (median; after as % of before: 60.4, 107.6, 105.6, 21.6, 53.3; **60.4**); subtraction, clean score 78.2, 70.8, 78.8, 69.2, 79.5; **78.2** | not run | look-ahead: kept 94.7%; subtraction clean 89.4 |
| Three-digit sums, 40,000 lines | 40.3, 65.6, 40.7, 72.3, 71.7; **65.6**; last 71.6 (by place 98, 82, 80, 100) | 0.3, 59.6, 29.5, 0.9, 0.0; **0.9** | none run by any line |
| size | 2.53 M cells, 27.1 M counts, 1.8 M displaced, 1,104 tried, 59 CPU-s | 0.56 M cells, 8.8 M counts, 32 CPU-s | |
| Three-digit sums, 400,000 lines | not run (started, stopped for time; no figure) | not run | |

Prose: tiny Shakespeare's stream read once, then the three books; each book's last tenth scored right after its stream, nats per byte.

| Arm | bryant | burgess | carroll | tiny Shakespeare's last tenth at the end (worn; no claim) | size at the end |
|---|---|---|---|---|---|
| searched, `nrep` 16 (the frozen setting) | 1.5699 | **1.3419** | 1.5519 | 2.0998 | 3.80 M cells, 35.8 M counts, 55.5 M displaced, 2,880 tried, 315 CPU-s |
| control: 128 random kinds never changed | 1.9225 | 1.7992 | 1.8822 | 1.9983 | 4.04 M cells, 63.1 M counts, 4.8 M displaced, 257 CPU-s |
| searched, `nrep` 8 | 1.5703 | 1.3368 | 1.4882 | 1.9843 | 3.89 M cells, 1,504 tried, 254 CPU-s |
| searched, `nrep` 32 | 1.6299 | 1.3524 | 1.4674 | 1.8979 | 3.71 M cells, 5,632 tried, 405 CPU-s |
| searched, 256 alive, `nrep` 32 (`PLAN_3.md`) | 1.6174 | 1.3546 | 1.6038 | 2.2031 | 4.10 M cells (the store is full), 37.4 M counts, 198 M displaced, 5,760 tried, 626 CPU-s |
| six hand-given runs of 1 to 6 bytes, in this program at the frozen numbers (orientation) | 1.5671 | 1.4370 | 1.5188 | 1.6003 | 0.58 M cells, 6.2 M counts, 18 CPU-s |
| T0 (`nano/table`, runs of six, its own numbers) | 1.5023 | 1.3645 | 1.4456 | 1.5306 | 0.58 M cells, 7.5 CPU-s |
| T2 (with word cells) | 1.4802 | 1.3162 | 1.4109 | 1.5067 | 2.42 M cells, 15 CPU-s |

The number tried, at 128 alive, two-digit sums at 40,000 lines (median of five; each seed in `report_2.txt`): `nrep` 8: 62.9 (496 tried); 16: 59.5 (1,264 with the life's 20,000 lines; 880 at 40,000); 32: 72.5 (1,600 tried). 256 alive with `nrep` 32: 78.7, 73.8, 85.4, 87.9, 57.3; **78.7**; last 75.0 (1,728 tried; 2.81 M cells, 27.9 M counts, 3.0 M displaced, 97 CPU-s); above 128 alive on four seeds of five.

## 3. The hypothesis and the predictions

The hypothesis (`PLAN_1.md`, section 4): with one frozen setting, at least 85% of never-shown two-digit sums at 40,000 lines AND below T0 on each book by 0.01 AND better than the same number of unsearched random kinds on both; and doubling the number tried improves both again.

**It does not hold as a whole.** Part by part:

| Part | Outcome |
|---|---|
| at least 85% of two-digit sums at 40,000 lines | **failed**: 59.5 (43.9 to 68.3); the development seeds had 88, 82, 64 |
| below T0 by 0.01 on each book | **failed**: below on one book (burgess, by 0.023), above on bryant by 0.068 and on carroll by 0.106 |
| better than the same number of unsearched random kinds, both texts | **held**, by a wide margin: sums 59.5 against 1.4 (above on every seed), each book 0.33 to 0.46 nats lower. So the hypothesis is **not refuted by its own criterion** (search does matter on both texts, with one setting); its levels are not reached |
| doubling the number tried improves both | **failed**: no order at 128 alive. Sums 62.9, 59.5, 72.5 for `nrep` 8, 16, 32; on prose each of the three `nrep` is best on one book. With twice the kinds alive as well (256, `PLAN_3.md`): sums better, 78.7 against 59.5; prose worse on all three books |

Predictions of `PLAN_2.md` (chance given): P1 at least 85% at 40,000 lines (0.15): failed. P2 search above control on every seed (0.90): held. P3 below T0 by 0.01 on each book (0.05): failed. P4 search below control on each book (0.95): held. P5 `nrep` 32 better than 16 on sums and all books (0.15): failed (better on sums, on carroll; worse on bryant, burgess). P6 16 better than 8 (0.40): failed. P7 400,000 lines at least 85% (0.50): failed (50.7). P8 three-digit at least 50% at 40,000 lines (0.25): **held** (65.6); at least 70% at 400,000 (0.30): not run. P9 addition kept at 85% of its level (0.50): failed (60.4); subtraction clean at least 60 (0.40): held (78.2). P10 a kind of digits with a guess among the five top earners on at least three seeds (0.60): held (five of five: for example [d2 d4 d7 d12 g0.2], [d3 d4 d7 g0.3], [d3 d4 d6 d7 g0.2], [d1 d4 d5 d7 g0.3], [d4 d5 d7 g0.3]). H, the whole (0.02): failed. Predictions of `PLAN_3.md`: Q1 sums median above 59.5 (0.65): held (78.7). Q2 all three books below those of 128 alive (0.45): failed (all three above, by 0.013 to 0.052). Q3 both (0.35): failed; the hypothesis of `PLAN_3.md` is refuted on prose. Q4 sums at least 85 (0.10): failed (two seeds of five are). Q5 below T0 on all books (0.05): failed.

## 4. What the search found (itself a result; `report_2.txt`, last section)

Lines: dN = the byte N places back; gR.H = the guess of layer R of the byte H ahead.

- **Two-digit sums.** On every seed the top earners include a kind of two or three digits of the two numbers together with a guess of a later answer byte: [d4 d5 d7 g0.3], [d3 d4 d6 d7 g0.2], [d4 d5 d7 g1.2] (development seed). At the tens digit d4 and d7 are the tens digits of the two numbers, d3 or d6 a units digit, and the guess two ahead is the units digit of the answer: this is the kind that the line `nano/sparse` laid down by hand ("a set of delays together with the guess"), and it carries the carry. The guess of the units digit is right on 100% (the place score of the units is 100 on four seeds of five). Also found: sets of the four digits themselves ([d3 d4 d6 d7], which serve shown pairs), and kinds that reach into the preceding line (d11 to d16). The preceding line is another sum, so I read those bytes as telling the place in the line (mine, not tested).
- **Three-digit sums.** The same family one digit further: [d5 d9 d11 g1.2] and [d1 d2 d5 d9 g1.2] on seed 406364 (d5 and d9 are the hundreds digits; the guess is of the tens digit of the answer, made by the layer that already uses the guess of the units).
- **Prose.** After tiny Shakespeare: [d1 d2 d3 d4 d5], [d1 d2 d3 g0.1], [d1 d2 d3 d5], [d1 d2 d3 d4 d5 d6], [d1 d2 d4]. Runs of bytes up to six are found unaided, with runs that skip a place; of the 20 top earners at the end 3 are keyed on a guess and none reaches further back than 10 places. No kind steps once a word: such a kind cannot be written in these lines, and T2's gain over T0 is of that sort.

## 5. Readings (mine)

1. **The kinds can be found by a general search judged by learning.** With no knowledge of either text, one program and one setting, the search arrives at runs of bytes on prose and at digits-with-a-guess on sums, and on both it is far above the same number of random kinds left alone. On three-digit sums, which no hand-built system of the programme has been run on, the same frozen program reaches 65.6% (median; 72 on three seeds) at 40,000 lines where nearly every line read is a pair not read before.
2. **It does not yet reach what a designer's kinds reach, on either text.** Sums 59.5 against 77 to 81 without look-ahead and 91 with; prose 0.07 to 0.11 above T0 on two books and 0.02 below on one. Against six hand-given runs inside this program at the same numbers the searched kinds are level on bryant, 0.095 better on burgess and 0.033 worse on carroll: on prose most of the distance to T0 is the single setting (step and e chosen for both texts) and not the kinds.
3. **More search did not keep making it better, except more kinds alive on sums.** With 256 kinds alive two-digit sums rise to 78.7 (the level of the hand-built kinds without look-ahead; two seeds at 85 and 88), at 1.7 times the cells; on prose the same doubling is worse on every book, with the store full and 198 M cells displaced: there the store, which I did not enlarge, is what binds. More kinds tried at the same number alive (`nrep`) gave no order, and neither did ten times the reading (two-digit sums at 400,000 lines are at 50.7 against 59.5 at 40,000, with a high point of 72.4 at 100,000). A run rises and falls again (seed 406364: 73.2 at 15,000 lines, 42.6 at 40,000). I see three reasons, none tested: (a) earnings are measured on the lines being read, where kinds that hold whole shown pairs earn most; over long reading the search selects for what serves shown lines ([d1 d2 d3 d4], [d1 d3 d4 d12] lead at 400,000 lines), and nothing in the judge speaks for never-shown pairs; (b) the frozen setting has no cap on a cell's weight, which `nano/sparse` found necessary over long reading, and my development material was 40,000 lines; (c) a kind that carries the answer is withdrawn like any other when its recent earnings are low, and nothing it knew is kept.
4. **Search as built here costs keeping.** Addition after subtraction is at 60% of its level (hand-built look-ahead: 95%), and tiny Shakespeare's last tenth is at 2.0 to 2.1 nats after the books against 1.60 for the six unchanging runs in the same program: kinds are withdrawn to make room for the present text and their cells are freed. A search that only adds while the store allows, or withdraws only what the present text does not contradict, is the obvious next form.
5. **Counting over kept bytes is what makes a search inside one reading possible** (44.9% to 54 to 88% on the development seed). It is also what one must be honest about: 131,072 bytes of the text are held outside the cells, and a kind born early in subtraction counts over the last lines of addition.
6. **The result of a run depends on what the search happens to find** (43.9 to 68.3 over seeds; 53.5 to 88.2 on one seed over `nrep`). The grid on one development seed chose in part a lucky run, and the development figures (88, 82, 64) overstate the reported seeds. Five seeds do not order `nrep` 8, 16 and 32.
7. **Size.** The searched system holds 4 to 7 times the cells of its own control on sums and is under the hand-built look-ahead system's 3.27 M; on prose it fills the store (3.8 M of 4.2 M places) and displaces 55 M cells in one life, six and a half times the cells of T0 for a worse score on two books.

## 6. What was not run, and what is owed

Not run: three-digit sums at 400,000 lines (started and stopped for time); the control on the life and on three-digit sums at 400,000 lines; a fifth task; a larger store; a cap on a cell's weight at 400,000 lines; the hand-built look-ahead system on three-digit sums (so "depth three fails there" is not shown by this line); more than five seeds. One development seed chose the setting. The prose arms are single runs (the search has a seed; its spread on prose is not measured). The "before reading it" figures of the books are in `report_2.txt`.

## 7. Commands

    python3 search.py check
    python3 search.py sums tag=NAME seed=73129 plan=add:40000,sub:20000 K=128 R=8192 prot=2 nrep=16 B=131072 step=0 eta=0.03 e=2 ncap=0     (nd=3 for three digits; nrep=0 the control)
    python3 search.py prose tag=NAME mode=life K=128 R=8192 prot=2 nrep=16 B=131072 step=0 eta=0.03 e=2 ncap=0                             (kinds=runs6 for the hand-given runs)
    python3 report.py > report_2.txt

Job lists: `jobs_g1s.txt`, `jobs_g1p.txt`, `jobs_g2.txt`, `jobs_g3.txt` (development), `jobs_c.txt`, `jobs_c3.txt` (claims). Raw lines: `out_*.txt`. The three books are read from the repository at commit 5d00eae (`minifly_v148_pretrain_probe_dev_20261025/data/`) into `data_local/`, which is not committed.
