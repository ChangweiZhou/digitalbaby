# nano/layers: further answering cells and their gates (20 November 2026)

Branch `claude/nano-layers-8mmhfm`, folder `nano/layers`; source `claude/friendly-darwin-8mmhfm` at 7c653ed05, whose `nano/merge/merge.py` and `nano/table/all.py` are imported unchanged. Work under Mr. Xu's words of 20 November 2026 as quoted in each plan ("我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。"). Development level throughout; no checker gate, seal or review is claimed; an independent repeat has not been made. Every plan was committed before its runs (one start was made too early and stopped unread, `PLAN_5.md`); every prediction is scored in its plan. Readings are mine.

## The question

The coordinator's hypothesis: the reading is where the learning should grow. If one further answering cell with one family of gates gains on sums and on prose, then other facts of the moment used as gates, cells on cells, and gates the system draws for itself should each add, on both kinds of text, with one setting.

## What was built

`layers.py`: one call inserted in the text of each base kernel after the first cell has answered and before anything is learned; a list of further answering cells runs there. A further cell hears the same senders and the branches of one cell below; its parts of the trust are born at nothing, indexed by the kind of sender and a gate value (with or without the sender's level of counts), and change by the system's own step times the cell's own mismatch. No signal is passed back. Checked against a plain numpy reference on both kernels (4e-14); the first cell's trusts and counts equal the base kernels' exactly; the coordinator's single second cell is repeated to the digit on sums and to twelve digits on prose (`check_output.txt`, `PLAN_1.md`).

## Results

Sums: `nano/merge`'s system with classes, 40,000 lines, 1.23 million cells, never-shown sums after a read line. Prose: T2, 2.4 to 5.0 million cells. Steps 1 to 4 and the choice are on sums seed 1 and three worn books; the claims are on five reported seeds and on a book no line had opened.

| Step | Finding | Sums seed 1, % | Prose, gain over the first cell on three worn books, nats |
|---|---|---|---|
| | first cell; the single second cell | 92.7; 96.7 | 0; 0.0239 |
| 1 | **the level of counts in the parts' index costs more than it gives**: parts by (kind, byte at each delay), 2.5 to 5% of the parts | 97.4 | 0.0292 |
| 1 | on sums one delay that tells the place holds the whole gain; on prose the six delays add nearly as a sum | 96.6 to 96.9 for delay 2, 5 or 8 alone | 0.0022 to 0.0099 each |
| 1 | parts shared by all kinds: harmful on sums, two thirds of the gain on prose; the control gives nothing | 86.2; 90.8 | 0.0164; -0.0002 |
| 2 | three more gates (the pair at delays 1 and 2; a trust the sending cell owns; the first cell's sureness) each add on prose beside the delays' parts and add nearly independently; on sums none adds a point (**the step's hypothesis refuted as stated**) | 97.6 to 97.8 | 0.0321 to 0.0329 each; 0.0390 together |
| 2 | the first cell's sureness alone tells the place on sums; the class as a gate fails | 96.3; 93.3 | 0.0056; - |
| 2 | coarse and fine parts of the delays together | 96.8 | 0.0364 |
| 3 | **cells on cells do not add beyond one wide cell with the same families (refuted)**; a reading cell with a trust for each of four cells returns its best input | chain 98.0, wide 97.8, reader 97.4 | chain 0.0366, wide 0.0390, reader 0.0295 |
| 3 | the same parts twice in depth add, and pay in keeping | 97.6 | 0.0433 (wide: 0.0390) |
| 4 | gates drawn at random at equal parts reach 0.92 of the hand-given gain on sums and 0.69 on prose (**not shown**); the review as written does not search; **the draw found the level of counts of another sender as a gate that earns** | 97.1 (hand 97.4) | 0.0202 (hand 0.0292); 0.0372 for hand-given plus six drawn slots |
| 5 | the frozen setting by a rule fixed beforehand: one wide cell with coarse and fine delays' parts, pair, own, sure, and 2D drawn slots | 97.1 | 0.0522 |

**The claims** (`PLAN_5.md`, section 6; one frozen setting, nothing chosen on this material):

| | First cell | Single second cell | nl | Frozen cell |
|---|---|---|---|---|
| Sums, five seeds, median (mean of the last three readings, after a read line), % | 92.9 | 97.1 | 97.8 | 97.1 |
| The same with the empty past | 91.8 | 96.8 | 96.7 | 97.4 |
| `edgeworth-parents` (opened once), nats | 1.3424 | 1.3209 | 1.3165 | **1.2949** |
| chesterton-ball, austen-emma, melville-moby_dick | 1.6093, 1.2056, 1.4387 | 1.5900, 1.1861, 1.4177 | 1.5880, 1.1814, 1.4134 | 1.5704, 1.1624, 1.3926 |
| Test text's last tenth, nothing changing (worn) | 1.4973 | 1.4820 | 1.4761 | 1.4549 |
| Kept: the same after `edgeworth-parents`; after the three books | 1.5026; 1.5061 | 1.5049; 1.5072 | 1.5059; 1.5066 | 1.4988; 1.4901 |
| Parts used (sums; prose life with `edgeworth-parents`) | 7,779; 327 | 708,574; 85,706 | 33,553; 3,857 | 2.2 million; 828,155 |
| CPU | | | | 155 s a seed of sums (three further cells); 21 s the life |
| For the record, the larger prose system (164 kinds), test text's last tenth (worn) | 1.4740 | 1.4444 | 1.4421 | 1.4884 |

**The hypothesis of the line is reached on prose (on T2) and not on sums.** The frozen cell is 0.0260 nats below the single second cell on the unopened book (0.010 asked) and level with it on sums at the median (0.0; 1 point asked). It is not refuted (that needed neither), and it does not hold for both kinds of text.

**And the frozen setting does not transfer to a larger prose system** (`PLAN_5.md`, section 7; a reading, on worn books, not a claim). On `all.py`'s system with all kinds of cells but the relay system (164 kinds, 5.21 million cells, 306 CPU-seconds) the frozen cell is *worse than the first cell*: 1.4884 against 1.4740 on the test text's last tenth, and 1.4617, 1.3255, 1.3817 against 1.4546, 1.2413, 1.3418 on the three worn books. The single second cell (1.4444) and nl (1.4421) gain there as on T2. Of everything in this line only nl (the delays' parts by kind and byte) gained on all three systems.

## Readings (mine)

1. On prose with ten kinds of sender (T2) the reading had about as much again to give as the single second cell took (0.0215, then 0.0260 more), from gates that name nothing of the text: more bytes, a pair, the cell's own trust, the first cell's sureness, the levels of other senders. With 164 kinds the same cell overshoots and loses; the gain is not yet a property of the form, only of the form at that size with that step.
2. On sums at 40,000 lines the single second cell had taken what the reading gives: every further cell that knows the place stands at 97 to 98 with both digits that can err near 98%. The smallest cell is the best there (nl above the single second cell on five seeds of five). I read the rest as held in the cells; this line did not test that.
3. The parts are short of teaching, not of detail: fewer, coarser parts gain more; coarse plus fine gain more still; the same parts twice gain like a larger step. The step was the first cell's and was never changed here.
4. Depth in this additive form adds nothing: every further cell adds to one sum of trust times message. Depth would need new senders made from the cells below, which this line did not build.
5. Search: drawing gates is cheap and found one family I had not given; keeping them by the size their parts have reached is not a search, because late draws never catch up.
6. Keeping is unchanged as a weak point: the further cells' gains on an old text are mostly lost after another book (the frozen cell keeps 0.004 of 0.042 after one book; it is the only cell that ends below the first cell).

## Owed, not run

The control with the step doubled (step 3); a review that compares gates of one age (step 4); other seeds of the draws; sums without classes, over 400,000 lines, three digits, and the life of two tasks; why the frozen cell fails on the system with 164 kinds (a step that knows how many parts move at once is the first thing to try; not run); the frozen cell with the relay system; an independent repeat of anything here.

## Faults of mine in this line

One arm of step 4 asked for eighteen distinct gates of sixteen and drew without end (stopped unread, repaired, said in `PLAN_4.md`). The prose run of step 5 was first started before its plan was committed (stopped unread, files removed, said in `PLAN_5.md`). The class gate restarts at every finding of classes and was given little chance (step 2).
