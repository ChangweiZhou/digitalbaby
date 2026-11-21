# Step 4: gates drawn at random from the system's own lines and kept by what they earn (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source.** `claude/nano-layers-8mmhfm` at the commit that carries this file. `layers.py` gains the drawing and the review (plain Python and numpy around the checked kernel; the kernel's text is unchanged, and `layers.py check` passes again). Written before any run of this step. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

Steps 1 to 3 used gates given by hand: the bytes at the system's delay lines, a pair of them, the first cell's sureness. `nano/search/REPORT.md` drew kinds of cells at random and kept them by the trust they earn. The same is asked here of gates.

## 2. The one change

A further cell holds M slots. Each slot is a family of parts indexed by (kind of sender, gate value), born at nothing, whose gate is **drawn at random** from three sorts: the byte at one delay from 1 to 16 back; the two bytes at two such delays together; the level of counts of the sender of one kind (silent: its own value). **Review:** whenever the bytes read have doubled since the last review, from 10,000 on (the schedule on which `nano/merge` finds its classes; no new number), each slot's earnings are measured as the mean size of its parts weighted by their uses; the half of the slots with the smallest earnings is withdrawn and as many gates are drawn anew, born at nothing. Nothing names a delay, a byte or a text; a new gate changes no answer until it has learned.

*Arms*, in one run on each text (sums seed 1; the life with the three worn books; draws from one seed of the generator):

| Arm | Slots | Sorts drawn |
|---|---|---|
| second-nl (hand-given: the bytes at delays 1 to D, D = 9 on sums and 6 on prose) | | |
| drawn:bytes | D | one byte (the same number of parts as second-nl) |
| drawn:bytes, 2D | 2D | one byte |
| drawn:all | D | all three |
| drawn:all, 2D | 2D | all three |
| nl + drawn:all | the hand-given delays and D drawn slots | all three |
| drawn:bytes, never withdrawn (the control: the first draws stay) | D | one byte |
| second-nl with all 16 delays by hand | 16 | |

## 3. Hypothesis, what refutes it, predictions (mine, before the runs)

*Hypothesis.* At an equal number of parts, gates drawn and kept by what they earn reach at least three quarters of the hand-given gates' gain over the first cell on both texts (drawn:bytes against second-nl: 4.8 points on sums seed 1, 0.0292 nats on prose). *Refuted if* under half on either text. The size of the gap is the result either way.

| | Statement | Chance |
|---|---|---|
| R1 | sums: drawn:bytes within 1 point of second-nl (five of the sixteen delays tell the place) | 0.60 |
| R2 | prose: drawn:bytes gives at least half of second-nl's gain | 0.50 |
| R3 | prose: drawn:bytes is at least 0.003 nats better than the same never withdrawn | 0.55 |
| R4 | prose: drawn:all with 2D slots reaches second-nl's gain | 0.30 |
| R5 | prose: drawn gates beside the hand-given add at least 0.003 nats; sums: within 1 point | 0.50 |
| R6 | prose: the slots of drawn:bytes at the end of the life include the bytes at delays 1 and 2 | 0.55 |
| R7 | prose: all 16 delays by hand add at least 0.002 nats over delays 1 to 6 | 0.40 |
| R8 | the hypothesis holds | 0.40 |

## 4. Size, cost, what is not run

A one-byte slot holds K1 times S parts (4,032 on sums, 650 on prose), a pair K1 times S times S (56,448 and 42,250), a level slot K1 times 41. Numbers used are reported. About four CPU-minutes on sums, one on prose. *Not run:* other seeds of the draws (one more on prose if time allows, entered as an addition); other review schedules or shares withdrawn; gates drawn from a cell's sureness or from classes; drawn gates in cells on cells.

## 5. Outcome (entered 20 November after the runs; `res/sums_armsstep4_seed1.json`, `res/life_armsstep4_booksworn.json`, `report_4.txt` with the gates held at the end and the history of the reviews)

*A fault of the plan, repaired before the sums run was read:* the arm "drawn:bytes, 2D" asked for eighteen distinct one-byte gates on sums, where there are sixteen delays; the first start of the sums run drew without end and was stopped after 14 CPU-minutes with nothing read. The arm has min(2D, 16) slots (sixteen on sums: every delay, so there it is the hand-given arm with all sixteen delays under another name plus reviews that redraw what they withdraw).

Sums: seed 1, with classes, 40,000 lines, 1,231,004 cells, % after a read line, mean of the last three readings (the empty past beside it); 215 CPU-seconds. Prose: T2, 2,419,727 cells, mean gain over the first cell on three worn books; 23 CPU-seconds. First cell: 92.7 (90.8 with the empty past) and 0.

| Arm | Sums, % (empty past) | parts used | Prose, gain | parts used | Gates held at the end on prose |
|---|---|---|---|---|---|
| second-nl (hand-given, delays 1 to D) | 97.4 (96.5) | 33,547 | 0.0292 | 3,857 | bytes at 1 to 6 |
| drawn:bytes, D slots | 97.1 (40.2) | 33,520 | 0.0202 | 3,776 | bytes at 12, 11, 4, 1, 7, 8 |
| drawn:bytes, 2D slots (16 on sums) | 96.9 (95.7) | 59,647 | 0.0288 | 7,534 | bytes at 16, 1, 10, 3, 4, 2, 13, 5, 8, 14, 7, 6 |
| drawn:all, D slots | 96.6 (96.3) | 155,183 | 0.0125 | 46,256 | levels of kinds 0, 1, 2; pairs (8, 13), (1, 10); byte at 9 |
| drawn:all, 2D slots | 96.4 (87.4) | 207,268 | **0.0323** | 4,716 | levels of kinds 4, 8, 1, 7, 2, 9, 5, 0; bytes at 7, 2, 15, 9 |
| nl + drawn:all, D slots | 97.0 (95.3) | 181,716 | **0.0372** | 6,424 | bytes at 1 to 6 by hand; levels of kinds 9, 5, 0; bytes at 12, 10, 6 |
| drawn:bytes, never withdrawn | 97.4 (15.1) | 33,560 | 0.0233 | 3,865 | bytes at 7, 15, 5, 1, 4, 8 |
| second-nl with all 16 delays by hand | 97.5 (44.5) | 59,653 | 0.0309 | 10,321 | |

*Predictions.* R1 held (97.1 against 97.4). R2 held (0.69 of the hand-given gain). R3 failed, in the wrong direction: the drawn gates with reviews are 0.003 nats **worse** than the first draws left alone. R4 held (0.0323 against 0.0292). R5 held (drawn gates beside the hand-given add 0.0080 nats on prose; on sums 0.4 points less, within the noise). R6 failed (delay 1 is held, delay 2 was drawn at 40,000 bytes and withdrawn at 80,000). R7 failed narrowly (sixteen delays by hand add 0.0017). R8, **the hypothesis: not shown.** At an equal number of parts the drawn gates reach 0.92 of the hand-given gain on sums and 0.69 on prose: above the half that would refute, under the three quarters asked.

Readings (mine).
1. **The review as I wrote it does not search.** Earnings were measured as the size a slot's parts have reached, and a slot drawn late is compared with slots that have learned since the start: the three slots that survived the first review (the bytes at delays 4, 1 and 7 on prose) were never withdrawn again, and every later draw, the byte at delay 2 among them, was withdrawn at the next review with half the earnings of the old ones. The withdrawing took teaching away and gave nothing: the first draws left alone did better. A fair review compares gates of one age, or the rate at which the parts grow; that is owed and not run here.
2. **What the drawing did find is a sort of gate that I had not given by hand: the level of counts of the sender of one kind.** Where all three sorts are drawn, the levels survive the first review far more often than their share of the draws (eight of twelve slots at the end on prose) and the arm with twelve such slots passes the hand-given bytes (0.0323 against 0.0292); beside the hand-given bytes six drawn slots add 0.0080. On prose the level of a kind says whether a run of that length has been met and how often: which senders are present is a fact of the moment that the bytes do not tell. It is `nano/table/PLAN_14.md`'s gate in a richer form, found here by the draw.
3. **On sums gates that reach beyond the line fail with the empty past** (40.2, 15.1, 44.5 against 96.5): a byte at a delay of ten or more is the line before, and the empty past puts line ends where the system never met them. This is the instrument's fault of `loop_small_case/ERRATA_20261120.md`, section 3, made larger by gates that read far back; after a read line all arms are within a point of one another.
4. Pairs of far bytes drawn at random cost parts and earn little (the arm with six slots of all sorts held two pairs and gave 0.0125).
