# Line `nano/search`, first plan: can the kinds of cells be found by a search judged by learning, with one program and one setting on several tasks? (20 November 2026)

**Authorisation and steer.** Mr. Xu's steer, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc), exactly: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" The coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." The task and the question are the coordinating session's. **Source.** Branch `claude/nano-search-8mmhfm`, made from `claude/friendly-darwin-8mmhfm` at 5d00eae. Development level. No checker gate, seal or outside review is claimed. Written before any program of this line exists and before any run.

**Read before writing.** `AGENTS.md`; `nano/table/trust.py`, `life.py`, `score.py`; `nano/ahead2/REPORT.md` and `PLAN_1.md` (the prose statement of the look-ahead system of `nano/sparse`); `nano/sparse/PLAN_5.md` (shrinking step, cap on a cell's weight); `loop_small_case/cells2.py` (text, split, score of sums) and `ERRATA_20261120.md` (clean score of subtraction). The roadmap's section headings only; the other reports named to me are read as the work needs them. Copied: nothing yet; the sums text and split are re-written from `cells2.py`; the three books are read from the repository at `minifly_v148_pretrain_probe_dev_20261025/data/` (commit 5d00eae) into a folder that is not committed.

## 1. The question

Can the kinds of counting cells be found by a general search judged by learning, with one program and one setting of its numbers on different tasks, and does more search make it better? Everything a designer chose per text in the earlier lines (runs of bytes, word ends, which delays, a look-ahead chain of depth three) is to be replaced by random birth, learned trust and withdrawal.

## 2. The system (exactly what the first program is to do)

*Lines.* A cell may be keyed on: the byte d places back, d = 1..16 (16 lines); and the system's own guesses G[r][h], r = 0, 1 (the layer that made the guess), h = 1..4 (the guess of the byte h places ahead, h = 1 being the next byte): 8 lines. 24 lines in all.

*Kinds.* A kind is a set of 1 to 6 lines. Its layer is 0 if it has no guess line, otherwise one more than the highest layer among its guess lines (so 0, 1 or 2). A cell of a kind exists for each combination of values that has occurred at its lines. Each cell of a kind of layer 0 or 1 holds four sets of counts: of the byte that followed it (h = 1) and of the bytes 2, 3, 4 places ahead. A cell of a kind of layer 2 holds the first set only. A cell says, from a set of counts c with total n, s(v) = log((c'(v) + e) / (n' + S e)), with c' = c, n' = n while n is at most `ncap`, and c' = c ncap / n, n' = ncap beyond (the cap of `nano/sparse/PLAN_5.md`); it speaks from a set only once that set has counted. S is the number of byte values of the text (a matter of storage: 14 for sums, 65 for prose).

*Readouts and trust.* Readout (r, h), r = 0, 1, h = 1..4: branches = sum over the speaking cells of kinds of layer at most r of trust times s; the guess G[r][h] is the largest branch (ties to the lowest byte index). The answer is readout (2, 1) over all kinds; its softmax is the probability scored. One trust for each readout, kind and level of experience (40 levels as in `trust.py`, from that set's own n and number of different bytes). **Every trust is born at nothing.** When the byte a readout spoke about arrives, each trust it used changes by step times (s(arrived) minus the sum over branches of p times s), p and s as they were when the readout spoke. Then the cells count (the cells active h - 1 bytes ago count the arrived byte in their h-th set). Step: one of two forms, chosen once in development: eta / sqrt(1 + uses of that trust) (`trust.py`, shrink), or eta g / sqrt(sum of g squared so far for that trust) (a step that does not depend on the size of the mismatch, which differs between texts; mine, added because one number has to serve all texts).

*Store.* A fixed store of 2^22 cells (hashed, four places a bucket). A new cell takes an empty place, or the place of a cell of a withdrawn kind, or else of the cell in its bucket that has counted least. Counts are 16-bit; a set that reaches 65,535 in one count is halved.

*Search.* K kinds alive. At birth of the system K random kinds: size uniform in 1..4, lines uniform among the 24 without repeat. Every R bytes: each kind's earnings over the period are the sum, over the readouts it took part in and the bytes, of trust times mismatch term (the first-order gain in log likelihood from that trust not being nothing; no extra computation). Among kinds older than `prot` periods the `nrep` with the lowest earnings are withdrawn (their cells freed, their trusts set to nothing) and `nrep` kinds are born: with chance 1/2 a random kind as at birth, otherwise a change of a living kind drawn with chance proportional to its positive earnings: one line added, or one removed, or one replaced by a random line (each with chance 1/3; sizes stay in 1..6). A kind equal to a living one is not born. Control: the same K random kinds, `nrep` = 0.

*Scoring.* Nothing changes during scoring. Sums: as `cells2.py` (before the line stand line ends only; the bytes up to `=` are the line's; then the system writes its own bytes); never-shown pairs; whole answer right. Prose: nats per byte on the bytes scored, probability of every byte, nothing floored.

**Laid down by me, and what each costs in generality** (the one place): 16 places back and 4 ahead (a text whose regularities reach further is not served; the search cannot tell); two layers of guesses (a chain of reasoning longer than guess-of-guess is not served); sizes up to 6; the 40 levels of experience of `trust.py`; guesses as the single largest branch (not a distribution); the alphabet size S per text. None of these looks at a task, but all are numbers of mine, and `ncap`, `e`, eta, K, R, `prot`, `nrep` are chosen once in development (below) and then frozen for every task. Nothing refers to word ends, the place of `=`, the length of a number or a list of delays.

## 3. This step: build, check, choose one setting on development material

1. `search.py`: a plain Python reference of the whole system and a compiled kernel; `python3 search.py check` compares probabilities, trusts, counts and written bytes on a prefix, including a search event and a store small enough that cells are displaced. Nothing else runs before it passes.
2. Development material only: two-digit sums with **seed 1**, 40,000 lines (never-shown additions, mean of the last three readings of 30,000, 35,000, 40,000); tiny Shakespeare, the first 19/20 of the stream read once and the last twentieth of the stream scored (the last tenth of the file is not touched in this step; the three books are not touched in this step).
3. A small grid, one setting for both texts: step form (2) x eta (3) x e (0.1, 0.5, 2) at `ncap` = 16, K = 128, R = 8192, `prot` = 8, `nrep` = 8; then `ncap` in (16, 64, none) and R/`prot`/`nrep` variations around the best, as budget allows. The setting is chosen by rank: the one with the best sum of ranks on the two texts. Also run, for my own orientation (not claims): the hand-built kinds inside this program (runs of 1..6 bytes on prose; nothing for sums beyond the other lines' figures).
4. The claims (five reported seeds; the books; three-digit sums; the life with subtraction; doubling) are a later plan, written after the setting is frozen.

**Size and cost.** Store 2^22 cells: 2.3 GB at S = 65, 0.6 GB at S = 14. Expected 1 to 3 CPU-minutes a run; the grid about 40 runs on 4 cores, under 30 minutes.

## 4. Hypothesis of the line (the coordinating session's), and my chances before any run

With one frozen setting the searched system reaches at least 85% of never-shown two-digit sums at 40,000 lines AND is below T0 (1.5023, 1.3645, 1.4456) on each of the three books by at least 0.01 nats AND is better than the same number of unsearched random kinds on both; and doubling the number of kinds tried improves both again. Refuted if either task is not better than unsearched random kinds, or if no single setting does both.

| | Statement (mine, before any program) | Chance |
|---|---|---|
| H | the whole hypothesis | 0.10 |
| H1 | sums at least 85% at 40,000 lines (five seeds, median) | 0.20 |
| H2 | below T0 by 0.01 on each of the three books | 0.35 |
| H3 | search better than unsearched random kinds on both texts | 0.70 |
| H4 | doubling the kinds tried improves both | 0.35 |
| D1 | development: some single setting has sums seed 1 at least 70% and Shakespeare's development twentieth under 1.60 | 0.50 |
| D2 | development: the best setting for sums alone and the best for prose alone differ in e by a factor of 4 or more | 0.60 |
| D3 | sums at 400,000 lines at least 85% with the frozen setting (later step) | 0.40 |

Why H1 is low (mine): the kind that settles the tens digit holds four lines of ten values each, ten thousand cells; it can show its worth only after some ten thousand lines, so in 40,000 lines the search can judge a few generations of such kinds, and there are about 13,000 sets of at most four of the 24 lines. A search judged by learning pays for its generality in reading.

**Not run in this step:** any reported seed; the books; the last tenth of tiny Shakespeare; three-digit sums; subtraction; 400,000 lines.

## 5. Addition 1 (written after the check passed and after two first runs at a guessed setting, before the grid)

*The two first runs* (K = 128, R = 8192, `prot` = 8, `nrep` = 8, shrinking step, eta 0.1, e 0.5, `ncap` 16; development material; files not kept): sums seed 1 at 40,000 lines 44.9% (hundreds 90, tens 53, units 100), 448 kinds tried, 24 CPU-seconds; Shakespeare's development twentieth 1.893 nats, 992 kinds tried, 131 CPU-seconds. On prose the kinds that earned most were runs of two and three bytes ([d1 d2 d3], [d2 d3 d4], [d1 d2 d3 d5], [d1 d2]); on sums a kind keyed on three guesses and one byte. So the search finds runs unaided, and the setting was poor: with e = 0.5 a cell of 65 byte values cannot say more than a third for any byte under the cap of 16.

*The change, general and mine: the system keeps the last B bytes it read (with its guesses as they were made), and a kind that is born counts over them at once.* Its trust is still born at nothing and is earned only on bytes that arrive afterwards, so nothing is judged on what it was counted on. Reason: a kind born empty needs as much reading as its cells are many before it can earn anything, so the reading, and not computation, bounds how many kinds can be judged; with the bytes kept, a kind can be judged within a period or two and the number tried becomes a matter of computation. Cost in generality: none that looks at a task; a memory of B bytes is a number of mine (B = 131,072 to begin with) and is state to be reported beside the store. On the life with subtraction it means kinds born early in subtraction count over the last lines of addition still kept; that is said beside the figure.

*The grid now run* (development material only): step form (2) x eta (0.03, 0.1, 0.3) x e (0.02, 0.1, 0.5, 2) x `ncap` (16, none), at K = 128, R = 8192, `prot` = 2, `nrep` = 16, B = 131,072. Sums: seed 1, 40,000 lines. Prose, to keep the grid inside ten minutes: the first 300,000 bytes of the stream read, the next 50,000 scored (inside the stream; development only). The choice is by the sum of the two ranks. Then, around the chosen setting, `prot`, `nrep`, B and `ncap` 64 as budget allows, and the control without search.

## 6. Outcome of the first step (entered 20 November; development material only; files `res/g1_*`, `res/g2_*`, `res/g3_*`, raw lines `out_g1s.txt`, `out_g1p.txt`, `out_g2.txt`, `out_g3.txt`)

`check_output.txt`: five cases, compiled against plain Python, largest difference 2e-15, stores, counts, written bytes and kinds equal. The search itself (which kinds are withdrawn and born) is one piece of Python used by both, so the check covers the cells, the trusts, the guesses, the store and the counting over kept bytes, not the choice of kinds.

Grid (K = 128, R = 8192, `prot` = 2, `nrep` = 16, B = 131,072; sums seed 1, 40,000 lines, mean of the last three readings; prose 300,000 bytes read and the next 50,000 scored). A second small grid (eta 0.003 and 0.01) was added when eta = 0.03, the smallest of the first, came out best on prose. The settings at the top by the sum of the two ranks, and the best of each text alone:

| step | eta | e | `ncap` | sums seed 1, % | prose, nats | note |
|---|---|---|---|---|---|---|
| shrinking | 0.03 | 2 | none | **88.2** | **1.698** | rank 1 on sums, 5 on prose: **chosen** |
| shrinking | 0.03 | 0.5 | none | 83.9 | 1.705 | |
| shrinking | 0.1 | 0.1 | 16 | 87.0 | 1.918 | |
| shrinking | 0.003 | 0.1 | none | 64.1 | 1.634 | best on prose |
| shrinking | 0.01 | 0.1 | none | 69.9 | 1.648 | |
| summed squares | best of 24 | | | 60.7 | 1.767 | the step of mine is worse on both; dropped |

Beside them, in this program: six hand-given kinds (runs of 1 to 6 bytes), no search, on the same prose material: 1.704 at the chosen setting, 1.623 at eta 0.01, e 0.1. The same 128 random kinds never changed (chosen setting): sums 0.0, 26.1, 1.6 on seeds 1, 2, 3; prose 2.087. The chosen setting on two more development seeds of sums: 81.7 (seed 2), 64.0 (seed 3). `nrep` 8 and 32 at the chosen setting, sums seed 1: 53.5 and 54.9 (16: 88.2); prose: 1.739 and 1.699 (16: 1.698).

Predictions. D1 (some single setting has sums seed 1 at least 70% and the development twentieth under 1.60): **not scored as stated**, since the prose material was shortened to keep the grid short; on the shortened material the chosen setting has 88.2% and 1.698, and no setting is under 1.60 there (the six hand-given runs are not either). D2 (best e for sums alone and for prose alone differ by a factor of four or more): **held** (2 against 0.1).

Readings (mine).
1. **The kinds are found.** On sums the kinds that earn most at the end are, on seed 1, [d4 d5 d7 g1.2] and [d3 d4 d5 d7 g1.2]: at the tens digit these are the two tens digits of the sum's numbers, a units digit, and the system's own guess of the byte two ahead, which there is the units digit of the answer. That is the kind the line `nano/sparse` laid down by hand as "a set of delays together with the guess". On prose the kinds that earn most are runs of the last two, three and four bytes.
2. **Counting over kept bytes is what made the search work in 40,000 lines** (44.9% without, at a guessed setting, against 54 to 88 with).
3. **The result of one run depends on what the search happens to find**: 88, 82, 64 on three development seeds, and 54 to 88 on one seed as `nrep` changes. One development seed was a thin basis for the grid; the choice of setting is partly a choice of a lucky run.
4. **One setting for both texts costs each text something**: prose wants a smaller step and a smaller e than sums.
