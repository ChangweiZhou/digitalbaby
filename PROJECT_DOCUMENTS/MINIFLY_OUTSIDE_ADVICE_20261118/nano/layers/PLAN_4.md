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
