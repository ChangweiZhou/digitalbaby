# Line `nano/aheadprose`, second plan: a guess of the next word's first letter (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session and given there as its translation from the Chinese (the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." and then "Good, keep trying." The step is the coordinating session's (its opening message: "guesses of the next word's first letter made at the end of a word (the place where the loss is)", to be taken only if step 1 gains); the exact system, the hypothesis' figures and the predictions are this session's. **Source.** This branch at 8fda8ad (`ahead.py` of `PLAN_1.md`). Written before the program is changed and before any run. Development level; one text and three books; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

`PLAN_1.md`, section 6: the look-ahead kinds gain 0.0045, 0.0067 and 0.0077 nats on the three books (0.0029 to 0.0041 of it from the present guess); the hypothesis did not hold and was not refuted. The gain is spread evenly over the places; the loss at the first letter of a word (37 to 41% of the loss on the books, 2.7 to 2.9 nats at such a place) is as it was. A guess of the byte two ahead is right at 14 to 19% of first letters: a guess at a fixed distance seldom looks at the next word at all.

## 2. The question and the one change

**If the system states to itself the first letter of the word that begins next, wherever it stands, and cells may be keyed on that statement, does the loss fall, and does it fall at the first letter of a word?**

One change from the system of `PLAN_1.md` (step of the added trusts 0.00005; nothing chosen here).

* *A fourth set of counts in T2's cells.* Each of T2's cells also counts the first letter of the word that begins next at or after the byte the cell is about to see named (a word is a run of letters, as in `words.py`; if the next byte is itself the first letter of a word, it is that byte). The text supplies it when that letter arrives. At most the last 32 places wait for it; a place further back than that when the letter arrives gets no count.
* *A third guess, w,* formed before g3 at every place: T2's cells' fourth counts, read with their own trusts (a fourth table of trusts for the ten kinds, born at 0.25), give w, the byte of the largest branch. Those trusts change by the usual rule when the letter arrives.
* *Three added kinds of cell, born with no trust,* counting the next byte: (f) (the last byte, w); (g) (the last two bytes, w); (h) (the last byte, g2, w). They join the answer beside all the cells of `PLAN_1.md`. The guesses g3 and g2 are formed as before and do not use w.

*The control.* The same, with the cells of (f), (g), (h) keyed on the w of 64 bytes earlier (byte value 0 for the first 64 bytes of a stretch); g2 and g3 stay the present ones.

What I expect w to be able to do, said beforehand: at the place just before a word begins, w is a second statement about the next byte from the same cells, so it brings little there. Inside a word and at its end, w says what kind of word is expected after this one, and a cell of (the last two bytes, w) can then weigh the ways the present word may end. Whether that is worth anything I do not know.

## 3. The measure

As `PLAN_1.md`, section 3: the life A, B, each book's last tenth scored right after its own stream, nothing changing. The comparison is with the system of `PLAN_1.md` (1.4757, 1.3095, 1.4032). The last tenth of the test text is **not scored** for this system unless the hypothesis holds, in which case it is scored once as the final system's. One look, at which nothing is chosen: the last twentieth of the stream of the test text (the system of `PLAN_1.md` gave 1.5347 there, T2 1.5334).

## 4. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The system with w is at least 0.003 nats below the system of `PLAN_1.md` on each of the three books. *Refuted if* it is not below it by 0.001 on at least two. *Control:* if the control gains as much, the gain is not the guess.

| | Statement | Chance |
|---|---|---|
| W1 | at least 0.003 below the system of `PLAN_1.md` on each book (the hypothesis) | 0.15 |
| W2 | at least 0.001 below on at least two books (not refuted) | 0.35 |
| W3 | not worse by more than 0.002 on any book | 0.80 |
| W4 | on the books' last tenths together, w is right at 18% of all places or more | 0.60 |
| W5 | at places where the next byte is the first letter of a word, w is right at least as often as the answer's own largest branch less 2 points | 0.70 |
| W6 | the loss at first letters of words falls by at least 0.01 nats (at those places) on at least two books | 0.20 |
| W7 | the control's gain is at least half of the system's on at least two books, or both are under 0.001 | 0.50 |

## 5. Size, cost, what is not run

T2's cells get a fourth set of 65 counts and a fourth table of 400 trusts; the three kinds add at most 65 x 65 and twice 65 x 65 x 65 cells (those that have counted are reported) and 120 trusts. A life about 30 CPU-seconds, four gigabytes. The numpy reference is extended and checked against the compiled loop before any figure is used; the system of `PLAN_1.md` must come out unchanged to the last digit.

If this gains nothing (refuted), the line stops with a report. Not run: w used in forming g2 and g3; a guess of the whole next word; the two further steps the coordinating session listed (a guess passed on as its two most likely bytes; the kinds of `nano/table/PLAN_10.md` beside these), which wait on this outcome.

## 6. Outcome (entered 20 November after the runs; `res/life_la2_scoreA0.json`, `res/life_la2_lagw64_scoreA0.json`, `res/dev_la2.json`, `res/life_la1_scoreA0.json`, `check_output.txt`)

The plan was committed at 28d8b29 before the program was changed. The extended loop equals the extended numpy reference to 4e-15 on seven cases with every guess equal; the system of `PLAN_1.md`, run again through the changed program, gives its three figures on the books unchanged to twelve places. The last tenth of the test text was not scored for any system of this plan.

The life A, B, each book's last tenth right after its own stream, nothing changing; nats a byte.

| | Bryant | Burgess | Carroll |
|---|---|---|---|
| the system of `PLAN_1.md` | 1.4757 | 1.3095 | 1.4032 |
| with w and the kinds (f), (g), (h) | 1.4756 | 1.3093 | 1.4025 |
| gain | **0.0001** | **0.0002** | **0.0007** |
| control: (f), (g), (h) keyed on the w of 64 bytes earlier | 1.4749 | 1.3082 | 1.4029 |
| the control's gain | 0.0008 | 0.0013 | 0.0003 |
| loss at the first letter of a word: before; with w; control | 2.8971; 2.8965; 2.8981 | 2.6842; 2.6823; 2.6834 | 2.8085; 2.8058; 2.8079 |

The look at the last twentieth of the stream of the test text: 1.5355 (the system of `PLAN_1.md` 1.5347, T2 1.5334).

*What w was worth as a guess* (share of places where w is the first letter of the word that begins next, %): Bryant 18.2, Burgess 22.5, Carroll 20.7; together 19.7. Naming always the most common first letter of that tenth would give 16.1, 14.8, 16.7. Where the next byte is itself the first letter of a word: 19.8, 23.8, 24.0, against 20.3, 25.3, 25.1 for the answer's own largest branch at those places. Inside a word: 17.6, 21.8, 19.6.

*Size and cost.* The fourth set has 4.68 million counts above zero at the end of the life; cells that have counted: (f) 2,147, (g) 17,108, (h) 16,228; the largest trust they reached 0.04, 0.03, 0.05. A life took 26 CPU-seconds.

**The hypothesis is refuted**: the gain is under 0.001 on all three books, and the control, whose w says nothing about the present place, gains as much or more on two of them.

Predictions: W1 failed. W2 failed. W3 held (no book worse). W4 held (19.7%). W5 held (w is within 0.5 to 1.5 points of the answer's largest branch at the places before a word). W6 failed (the loss at first letters falls by 0.0006, 0.0019, 0.0027). W7 held (the control's gain is more than half of the system's on Bryant and Burgess).

Readings (mine).
1. **A guess of the next word's first letter is worth nothing as a line here.** Nothing I can measure separates it from three more kinds of cell keyed on a byte unrelated to the place.
2. **The guess itself is weak, and that is the whole of it.** It is right at a fifth of places, 3 to 8 points above always naming the commonest first letter, and at the place before a word it is no better than the answer the system already gives. The cells do not know which word comes next; stating to themselves what they do not know, and keying cells on the statement, adds nothing. On sums the guess three ahead was right on every never-shown sum.
3. So the loss at the first letter of a word is not reached by look-ahead of this sort. What would reach it is knowledge the cells do not have (what the words before mean, or who speaks), and that is another line's question.
4. The line stops here, as the plan said. The two further steps (a guess passed on as its two most likely bytes; these kinds beside those of `nano/table/PLAN_10.md`) are **not run**.
