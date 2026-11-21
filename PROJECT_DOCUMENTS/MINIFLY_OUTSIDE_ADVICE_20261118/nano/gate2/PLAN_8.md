# Line `nano/gate2`, eighth plan: the cells of this line on prose (20 November 2026)

**Authorisation**, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。"; and, in two later messages of the coordinating session that reached this session only after its seventh plan had ended: "继续大胆尝试，小心求证。" and "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (the coordinating session's translations: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas."; "Go on trying boldly."; "Go on trying boldly, and verify with care."; "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."). The coordinating session asks that no further run be started whose only material is sums, and that the transfer of the fifth plan's cell to prose come next. **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; this branch at the commit that carries this file. Development level; no gate, seal or review is claimed. Written before any run of this line on prose. I know the coordinating session's reported figures for the second cell on prose (0.020 to 0.023 nats lower on three unopened books; the control nothing) and that, by its word, the gain on an old text is gone after other books are read; I have read `../table/gate.py` (opened after step 1) and have not opened its prose result files.

## 1. The question

Everything this line has shown is on sums. Two things can be asked of prose with the program that exists: (a) does the second cell of `PLAN_1.md`, in my program, give on prose what the coordinating session reports; (b) what does the cell of `PLAN_5.md` (each sender heard with probability one half while learning), refuted on sums, do on prose: on new text, and on what is kept of an old text?

## 2. The system, the life and the measures

`gate3.py`'s cells (first, second, control, the cell of the fifth plan and its control, from one run) on `merge.py`'s prose system without classes, as `PLAN_15.md`, section 4, gives it: 65 byte values, six delay lines, seven kinds (runs of 0 to 6 bytes), e = 0.02, the constant step 0.00005, trusts born 0.25; the second cells' parts born at nothing, the same step. The life: the first nine tenths of the test text (`../data/tinyshakespeare.txt`), then `chesterton-ball`, `austen-emma`, `melville-moby_dick` (read from the repository's tree at this commit into the uncommitted build folder; bytes not among the 65 values become a space, carriage returns are removed), the first nine tenths of each read and each last tenth scored right after its stream with nothing changing; the test text's last tenth scored once right after its own stream and once at the end of the life (what is kept). Measure: nats a byte (`../merge/score.py`), every byte of each last tenth. Program: `prose3.py` (the life; the cells are `gate3.py`'s compiled loop, already checked against numpy for 65 values, six delays and a constant step).

*Size.* Each second cell 7 x 40 x 6 x 65 = 109,200 parts. *Cost.* A few CPU-minutes; one run (the cell of the fifth plan draws from a random stream; one seed, 1).

## 3. Hypothesis, what refutes it, the prediction (mine, before the run)

*Hypothesis.* The cell that hears half its senders while it learns does not transfer to prose either: its loss is above the plain second cell's on each of the three books. *Refuted if* it is lower than the second cell's on two books or more.

*The prediction:* above the second cell's on all three books, chance 0.70. Beside it: (i) my second cell's gain over the first is 0.015 to 0.028 nats on each of the three books, that is, the coordinating session's figure repeats (0.75); (ii) on the test text's last tenth at the end of the life the cell of the fifth plan is lower than the plain second cell, that is, it keeps better (0.40); (iii) it is above the first cell on at least two books (0.50).

## 4. Not run

Prose with classes; the worn books; other seeds of the random stream; anything further on sums (by the coordinating session's request: the open stream, the diagnostic at the score, and an arm with classes found once, which its messages asked of this line and which reached it after the runs they concerned, are NOT RUN).

## 5. Outcome (20 November; `res/prose_life_new_seed1.json`, `out/prose_life_new_seed1.txt`)

One run; 928,030 cells that count at the end; each second cell 109,200 parts, about 63,000 used (the controls 1,370); 26.5 CPU-seconds. Nats a byte on each last tenth, scored right after its stream (lower is better):

| Text | First | Second | Control | Cell of the fifth plan (half heard) | Its control |
|---|---|---|---|---|---|
| the test text, after its stream | 1.5226 | **1.5058** | 1.5267 | 1.5217 | 1.5347 |
| `chesterton-ball` | 1.6316 | **1.6116** | 1.6368 | 1.6160 | 1.6262 |
| `austen-emma` | 1.2416 | **1.2203** | 1.2407 | 1.2430 | 1.2557 |
| `melville-moby_dick` | 1.4687 | **1.4453** | 1.4690 | 1.4670 | 1.4783 |
| the test text, at the end of the life | 1.5370 | 1.5351 | 1.5371 | 1.5458 | 1.5487 |

**(a) The second cell repeats on prose, exactly.** In my program it is 0.0200, 0.0213 and 0.0234 nats below the first cell on the three books and 0.0168 on the test text; the control gives nothing (+0.005, -0.001, +0.000). After this run I read the coordinating session's two result files for the same life from its branch at bfba02b (`gate_life_booksnew_gate0_modenone.json` and `..._gate1_...`): all five scores of the first cell, the second cell and the control are the same to the sixth decimal place. On the old text at the end of the life the second cell's gain is 0.002: nearly all of it is gone, as the coordinating session said.

**(b) The cell that hears half its senders while it learns does not transfer to prose.** Its loss is above the plain second cell's on all three books (by 0.004, 0.023, 0.022) and on the test text; against the first cell it is 0.016 lower on one book, 0.002 lower on another and 0.001 higher on the third. It keeps the old text worse, not better (1.5458 against 1.5351 for the second cell and 1.5370 for the first). So the one thing it did better on sums, standing through a change of task, does not show on prose in this life. The hypothesis of this plan holds; the mechanism of the fifth plan is entered as refuted on both kinds of text. One seed of its random stream.

**Predictions.** The prediction held (above the second cell's on three books of three). (i) held (0.0200, 0.0213, 0.0234). (ii) failed (it keeps the old text worse). (iii), as I wrote it, asked that its loss be above the first cell's on at least two books: failed (one book).
