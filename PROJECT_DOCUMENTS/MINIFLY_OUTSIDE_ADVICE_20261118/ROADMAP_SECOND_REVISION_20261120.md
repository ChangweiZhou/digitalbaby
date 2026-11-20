# The roadmap, second revision: towards a test at the level of nanoGPT, with connections decided jointly (proposed, 20 November 2026)

By the outside session. A proposal and a working order; it decides nothing for the project. It follows `ROADMAP_REVISION_PROPOSED_20261119.md` (whose steps R1 to R6 are kept and mapped below) and rests on `loop_small_case/REPORT.md`, which is development level and has been repeated by no one.

## 1. What Mr. Xu said on 20 November, and what I take from it

His words to this session, in full:

> The connection strength (ideally can involve many cells) should be jointly decided by the cells together. For example if one cell wants to drop out, then the connection strength to it should be zero no matter how willing other cells are. Obviously we can simplify the edge to only involve two cells, and perhaps refine even further to one sided edges, with inhibitory / excite signals respectively. But whether this is justified has to be answered by experiment. Similarly how the strength is fixed even in this case can be experimented, and you can try different rules. Also the network could have many different types of cells that adjusts around different time clocks, having different memory ability, ability to output signals (chemicals) etc. But we need to make sure not testing everything at once and leave a mess. My question is 1) where are we now? Are we ready for some nanogpt level experiment? 2) If not, can you revise the road map and keep doing some experiments? Spin off some agents to speed up the experiment process.

and, a moment later: "Note you can scale the network or do whatever you like." After this revision was sent to him: "Keep experimenting, we are not going to break anything."

What I take from it (these readings are mine):

1. **Three forms of a connection, from the general to the simple.** (i) A connection decided jointly by all the cells it involves, which may be more than two, any one of which can set it to zero. (ii) An edge between two cells, decided by both. (iii) A one-sided edge, carrying an excitatory or an inhibitory signal. The simpler forms are simplifications that an experiment has to justify. Everything in `loop_small_case` so far used form (iii): a strength owned and changed by the receiving cell alone, without a sign constraint, and once with it (one sign for each sending cell: 70.7% against 84.2%).
2. **How a strength is fixed is itself open to experiment**, in every one of the three forms.
3. **Kinds of cells** (different clocks, different memory, an output of signals that reach many cells) are allowed as variables, later and one at a time.
4. **One thing at a time, and no mess.** Section 5 is how I propose to keep to that.
5. **For the crossing of the report's section 12.4.** A strength that both ends decide is known at both ends, so a value passed over one connection that depends on that strength is not, on my reading, the "instruction computed separately for each cell and delivered to it" that the reviewer's page excludes. I take the crossing as admissible until told otherwise, and keep the form that needs only a common signal as the fallback. If this reading is wrong I would like to be told.

## 2. Where we are

Development level throughout; five seeds where a number is in bold.

| What | State on 20 November |
|---|---|
| The loop (a mismatch formed where the byte is named, relayed over learned return connections, each cell changing its own strengths) | learns; answers never-shown two-digit sums **84 to 88%** |
| The same, closer to what a cell has | reading 64 of 140 lines with rewiring **81%**; one sign for each cell **71%**; cells that are silent most of the time **about 70%**; the relay over two steps **84%**; yes or no alone 9% |
| Cells that hold bytes themselves | one digit across 0 to 8 letters **91% and 99%** with a signal passed back over one connection (28% and 45% without); two-digit sums with two delay lines in place of nine **66%**; two-digit numbers with letters between them 8%: not learned |
| Keeping | the first task is gone within a few thousand lines of the second; written into new cells with one old line in 32, **67%** is kept |
| Language | the next byte of prose never read: 46% and 52% on two books, with one layer of cells, ten bytes on lines and a least-squares answering cell: level with a table of the last two to three bytes, below a table of four |
| Speed | compiled kernels: 3.6 million bytes in 16 to 34 seconds with 256 cells; about 40 microseconds a byte with 1,024 cells |

## 3. Are we ready for an experiment at the level of nanoGPT?

**To set the test up: yes, and it is being done (line L0). To pass it: no.**

What that level is. nanoGPT's character-level example reads the tiny Shakespeare text (1.1 million bytes, 65 byte values; the first nine tenths for learning). Its network sees up to 256 bytes back through six layers and has about ten million numbers that learn; it is taught by back-propagation for some eighty readings of the text and reaches a loss of about 1.47 nats a byte (2.1 bits) on the last tenth. A small setting meant for a machine like ours (four layers, 128 wide, 64 bytes back) reaches about 1.88 nats (2.7 bits). These two figures are as I recall them from nanoGPT's own description; line L0 measures them here, on the same text and split, so that nothing below rests on my memory.

Where our system stands against that. On prose it is level with a table of the last two to three bytes. A well-made table of five to seven bytes is, on a text of this size, already near the small setting's level, and the full setting goes beyond tables by using what lies further back. Four things are missing, in the order in which I would test them, one at a time:

1. **An answering cell that gives probabilities.** Ours fits its branches by least squares; the loss in nats or bits, which is the measure of this test, needs branches that compete and sum to one. The mismatch keeps its form (the byte that arrived, less what was given to each branch).
2. **Scale.** 256 to 1,024 cells and eight readings so far; the test wants thousands of cells and tens of readings.
3. **Depth.** The relay works over two steps on sums; on language it is untried.
4. **What lies further back than the delay lines.** This is the cells that hold, and it is where we are weakest: two digits across a gap are not yet learned.

So the first three can be tried now, and I expect them to bring the loop to the neighbourhood of the small setting, which would be a result worth having; the fourth decides whether the full setting can be approached, and it depends on the memory line.

**Corrected the same day by the evidence.** The reference line measured the test (`claude/nano-ref-8mmhfm`): on the last tenth of the text a smoothed table of the last six bytes has a loss of 1.5165 nats a byte, and the small GPT after 2,000 steps has 1.906 (its own estimate 1.895, which agrees with the 1.88 I recalled). So on a text of this size the level of nanoGPT is the level of a good table of six bytes, give or take 0.05 nats, and my fourth point above was wrong for this text: what decides is estimating well from the last six or so bytes, not what lies further back. Following that, a table made of cells was built (`nano/table`: a cell for every run of bytes that occurs, at most seven active at a byte): 1.5225 nats when its strengths count, 1.5317 when they are changed by the mismatch alone, in five CPU-seconds. So a system of cells that learns one byte at a time from its own mismatch is at the level of this test today, as a table. What that does not show is the thing the loop was built for: a table answers nothing it has not met. The step that now decides, on this text, is whether cells taught by the relay add anything to the table's cells (L-TR below); and a text on which a table falls well short of the references is needed before the test says anything about memory or about new combinations.

## 4. The order I propose

Four tracks. Each step changes one thing and is scored beside what it changed from, and beside a comparator by back-propagation where one exists.

| | Step | Reached when | State |
|---|---|---|---|
| **L** | *Language at the level of nanoGPT* (`nano/`) | | |
| L0 | The test and its references: the text and split of nanoGPT; loss in nats and bits and the byte named right, on the last tenth read as one stream; tables of the last k bytes, properly smoothed; a small and a larger GPT taught by back-propagation here; the network of the loop taught by back-propagation | the figures exist and are reproducible | started 20 November (`nano/ref`) |
| L1 | An answering cell whose branches compete, against the least-squares one | lower loss at equal size | started 20 November (`nano/loop`) |
| L-T | *Added from the evidence:* a table made of cells (a cell for each run of bytes that occurs; strengths that count, or that are changed by the mismatch) | within 0.05 nats of the smoothed table | **reached at development level** (`nano/table`): 1.5225 counting, 1.5317 by the mismatch, against 1.5165 |
| L-TR | *Added from the evidence:* the table's cells and the relay's cells read by one answering cell | at least 0.02 nats below the table's cells alone | given to `nano/loop` as its next step |
| L2 | Scale: cells, readings, delay lines, each alone | the loss stops falling, or meets a table of five bytes | after L1 (`nano/loop`) |
| L3 | Depth: the relay over two and three steps | lower loss than one step at equal numbers of strengths | after L2 |
| L4 | Cells that hold, on language | lower loss than the delay lines alone | after M |
| L5 | Cost beside the references | stated | last |
| **C** | *Connections and cells* (`nano/joint`), on the small texts first, where a run takes seconds | | |
| C1 | An edge decided by both ends (each end holds its own part of the strength and can set the connection to zero) against the one-sided edge | the same score, or a difference stated with its size | **done at development level** (`claude/nano-joint-8mmhfm`): a product of two parts 84.0 and the smaller of two parts 84.2 against 84.2 one-sided; a product in which the sending end's part moves fast is worse (67 to 83) |
| C2 | The right to withdraw: either end, the receiving end only, the sending end only | stated | **done at development level**: no cost in score whoever holds it (83.5 to 84.0 against 84.2); with either end half of the connections go; by an end's own strengths 6 points better than the same number by chance |
| C3 | A connection of more than two cells (two sending cells jointly) against edges, at equal numbers of strengths | stated | started 20 November (`nano/multi`); first step on sums: worse than edges at equal numbers of strengths (48 against 81 at 64 strengths a cell). On the text of the test the table's cells are such connections and reach the test's level (L-T) |
| C4 | The sign: one-sided edges with excitatory and inhibitory senders (measured once: 70.7 against 84.2) together with C1 | stated | after C1 |
| C5 | Rules for fixing a strength, one against another | stated | open |
| C6 | Kinds of cells: clocks, memory, signals that reach many cells | stated | started 20 November (`nano/kinds`) |
| **M** | *Memory* (the earlier R3 and R4) | | |
| M1 | Two-digit numbers with letters between them: longer runs, then more cells, then a signal passed over two connections, one at a time | never-shown sums at every gap | two long runs started 20 November (`loop_small_case/PLAN_12.md`); the signal over two connections started 20 November (`nano/cross2`) |
| M2 | The first family's own items beside the record's reference | as R4 | after M1 |
| **K** | *Keeping* (the earlier R5) | | |
| K1 | Where a new task is written, settled by the cells and not by the program | the first task kept at the level the program's choice gives (67%) | started 20 November (`nano/keep`): cells silent most of the time keep less (5 to 8% against 16%); cells that stiffen with age keep 34% at a cost to both tasks; a third candidate is running. A diagnostic (`loop_small_case/PLAN_13.md`): after the second task the cells in between still carry most of the first (67% with the answering cell alone taught again); the loss is in the answering cell's reading of cells that both tasks share |

The earlier steps map as follows: R1 is done at development level; R2 continues as C; R3 and R4 as M; R5 as K; R6 as L5.

## 5. How this is kept from becoming a mess

1. **One line, one folder, one owner.** A line changes files only in its own folder; what it needs from elsewhere it copies and says so.
2. **One change at a time.** Each step has its own plan, committed before its runs: one question, one hypothesis with what would refute it, predictions with chances, sizes, cost, and what is not run. The outcome is entered in the plan and the predictions are scored, failures included.
3. **Every figure beside what it is compared with**, at equal size and length: the step before it, and the comparator.
4. **A development seed for choosing numbers, then the five reported seeds** (73129, 93683, 116089, 406364, 545353) for anything that is claimed.
5. **A numpy program is the reference for every compiled one**, and a check of the two is kept beside them.
6. **One ledger**, the table of section 4, updated when a step starts and when it ends. Nothing is called reached that another line has not yet repeated; what stands here is development level until then.

## 6. Lines started on 20 November

| Line | Folder | Who | Branch | Task |
|---|---|---|---|---|
| L0 | `nano/ref` | a separate agent session (`session_017dSEpXxn4bCLF4VwJ7tiNZ`) | `claude/nano-ref-8mmhfm` | the references of section 4, L0 |
| L1, then L2 | `nano/loop` | a separate agent session (`session_017dSEpWpbA9ALrDd5BncYTk`) | `claude/nano-loop-8mmhfm` | the loop on the test text |
| C1, then C2 | `nano/joint` | a separate agent session (`session_017dSEpU7vrz76xk4M2iSTnY`) | `claude/nano-joint-8mmhfm` | edges decided by both ends |
| M1 | `loop_small_case` | this session | `claude/friendly-darwin-8mmhfm` | two long runs of the eleventh plan's text (`PLAN_12.md`) |
| M1, the signal over two connections | `nano/cross2` | a separate agent session (`session_017dSEpWDotjRLQF7DQRsnSU`) | `claude/nano-cross2-8mmhfm` | the forward program for two crossings, checked against the instrument; then three texts |
| K1 | `nano/keep` | a separate agent session (`session_017dSEpXu5KdKMbirXHJwCXF`) | `claude/nano-keep-8mmhfm` | candidates by which the cells settle where the second task is written |
| C3 | `nano/multi` | a separate agent session (`session_017dSEpW3ZFJMFsWvpL1fwEs`) | `claude/nano-multi-8mmhfm` | connections of three ends against edges at equal numbers of strengths |
| C6 | `nano/kinds` | a separate agent session (`session_017dSEpWr5ht93HPhGeTZChy`) | `claude/nano-kinds-8mmhfm` | memory, clocks, a common signal: one at a time |
| M1, holding cells that read only lines | `nano/layered` | a separate agent session (`session_017dSEpXmJFSSep8BaKzCffH`) | `claude/nano-layered-8mmhfm` | two kinds of cells and the relay over two steps, so that no crossing is needed; from the crosswalk of the old lists, item 1 |
| K, a diagnostic | `loop_small_case` | this session | `claude/friendly-darwin-8mmhfm` | how much of the first task is still in the cells in between (`PLAN_13.md`); from the crosswalk, item 3 |

The first three agent sessions were started from this branch at fe397ca6 and the last four at 260c5d73, after Mr. Xu's word to keep experimenting, each with a written brief: one line, one folder, a plan committed before each step's runs, settings chosen on the development seed, a numpy reference for every compiled program, and a short report to this session after each step. 
Their results are merged into this branch and entered in the ledger as they come.

Added the same day: the agent of `nano/joint` finished C1 and C2 within the first hour and was given three further steps (a whole cell dropping out; rules for fixing a strength; the sign with the right to withdraw). The old lists of mechanisms on the default branch were set against the loop at Mr. Xu's request (`OLD_LISTS_AGAINST_THE_LOOP_20261120.md`); two of its items became the last two lines of the table above.
