# Line `nano/aheadprose`: look-ahead guesses used as lines, on prose (20 November 2026)

**Status.** Development level. One text (tiny Shakespeare) and three books (Bryant, Burgess, Carroll); one life each; one program of mine. No checker gate, seal or outside review is claimed. The task was the coordinating session's, under Mr. Xu's words of 20 November as that session quoted them in its translation (`PLAN_1.md`, head). Readings marked "mine" are this session's.

**The question.** On two-digit sums, cells that also count two and three bytes ahead, with guesses the cells can be keyed on, took never-shown sums from 77 to 91% (`nano/sparse`, repeated by `nano/ahead2`). Does the same mechanism lower the loss on prose?

**The answer, in brief (my reading).** A little, and not where the loss is. Beside T2 the look-ahead kinds gain 0.0045, 0.0067 and 0.0077 nats a byte on three books on which nothing was chosen (0.3 to 0.5% of the loss); about two fifths of that is also given by a control whose guesses say nothing about the present place. A further guess, of the next word's first letter, gains nothing (under 0.001 on every book; its control gains as much). The hypothesis of step 1 did not hold and was not refuted; that of step 2 is refuted. The line stopped there.

## 1. What was run

`PLAN_1.md` (committed at f8c8ea1 before the program existed) and `PLAN_2.md` (committed at 28d8b29 before the program was changed) hold the exact systems, the hypotheses, the predictions and the full outcomes. `ahead.py` holds the compiled loop, a numpy reference of the whole loop with the added cells in dictionaries, and the check (`check_output.txt`: seven cases, largest difference 4e-15, every guess equal). `score.py`, `table.py`, `words.py` are unchanged copies from `nano/table` at 91f62217.

*Step 1.* T2 (`nano/table/PLAN_3.md`, arm 2, trust's step 0.00005, e = 0.02) with one change: each of its cells also counts the bytes two and three ahead; before each byte the answering cell forms a guess three ahead (g3), then two ahead (g2), each the byte of the largest branch, with their own trusts; five added kinds born with no trust: (last byte, g3), (last two bytes, g3), (last byte, g2), (last two bytes, g2), (last byte, g2, g3). One number was chosen, on the last twentieth of the stream of the test text: the step of the added trusts (0.00005, the smallest of three).

*Step 2.* One change from step 1: each of T2's cells also counts the first letter of the word that begins next; a guess w from those counts; three more kinds born with no trust: (last byte, w), (last two bytes, w), (last byte, g2, w). Nothing chosen.

*Controls.* The same systems with the cells keyed on the guess made 64 bytes earlier.

## 2. The table

The life A, B of `nano/table/PLAN_10.md`, addition 2: the stream of the test text, then the three books in turn; each last tenth scored right after its own stream, nothing changing; nats a byte. The last tenth of the test text was scored once for T2 (the check of the program: 1.4973, as `nano/table` has it) and once for the system of step 1, and for nothing else.

| | test text, last tenth | Bryant | Burgess | Carroll |
|---|---|---|---|---|
| T2 (in this program; equal to `nano/table`) | 1.4973 | 1.4802 | 1.3162 | 1.4109 |
| step 1: with look-ahead | 1.4938 | 1.4757 | 1.3095 | 1.4032 |
| gain over T2 | 0.0034 | 0.0045 | 0.0067 | 0.0077 |
| step 1's control | not scored | 1.4786 | 1.3133 | 1.4073 |
| the control's gain over T2 | | 0.0016 | 0.0029 | 0.0036 |
| step 2: with the guess of the next word's first letter | not scored | 1.4756 | 1.3093 | 1.4025 |
| gain over step 1 | | 0.0001 | 0.0002 | 0.0007 |
| step 2's control | not scored | 1.4749 | 1.3082 | 1.4029 |
| the control's gain over step 1 | | 0.0008 | 0.0013 | 0.0003 |

On the last twentieth of the stream of the test text (where the one number was chosen): T2 1.5334, step 1 1.5347, step 2 1.5355. There both systems with look-ahead are slightly worse than T2.

The guesses as guesses, three books together (share of places right, %): the answer's largest branch 58.3; the guess two ahead 41.4; three ahead 30.0; the next word's first letter 19.7 (always naming the commonest first letter: 15 to 17). Where the guessed byte is the first letter of a word: the answer 20 to 25; two ahead 14 to 19; three ahead 8 to 14.

## 3. Hypotheses and predictions

Step 1 (the coordinating session's hypothesis): at least 0.005 below T2 on each book; refuted if not 0.002 below on at least two. Result: 0.0045, 0.0067, 0.0077. **Not held (one book short by 0.0005), not refuted.** Of my nine predictions six held and three failed (Q1, the hypothesis; Q4, that the control would be level with T2; Q6, that a larger step would be chosen).

Step 2 (hypothesis mine): at least 0.003 below step 1 on each book; refuted if not 0.001 below on at least two. Result: 0.0001, 0.0002, 0.0007. **Refuted.** Of seven predictions four held and three failed (W1, W2, W6).

## 4. Readings (mine)

1. **The mechanism that carried the carry on sums carries little on prose.** There the guess three ahead was a fact (the units digit, right every time) that no cell at the place of the tens digit held; a cell keyed on it could do what no other cell could. Here a guess is right at 30 to 41% of places and is a summary of what the long cells already say; a cell keyed on it mostly says it again.
2. **What there is of a gain is only partly the guess**: 0.003 to 0.004 nats a book from the present guess, 0.002 to 0.004 from having the added kinds at all, whatever byte they are keyed on. I did not find out what the second part is.
3. **It is not where the loss is.** The loss at the first letter of a word (37 to 41% of the loss on the books) moved by 0.002 to 0.009 nats at such a place in step 1 and by under 0.003 in step 2. The system's guess of the next word's first letter is right at a fifth of places, barely above naming the commonest letter. The cells do not know which word comes next, and stating a thing to oneself does not make it known.
4. **Kinds born with no trust did no harm anywhere on the books** (the control included), a fourth instance of the rule of `nano/table/PLAN_8.md`; on the development twentieth of the test text they cost 0.001 to 0.002.
5. The cost: three times the counts of T2 (four times in step 2) and three times its CPU, for 0.3 to 0.5% of the loss.

## 5. What this does not show

Guesses further ahead, or of a whole word; a guess passed on as its two most likely bytes, or as a spread; these kinds beside those of `nano/table/PLAN_10.md` (the two further steps the coordinating session listed were not run, since step 2 gained nothing); a longer life or other orders of the books; the measure in which everything goes on changing; any text beyond these four. Each figure is one run: the system has no seed, so the figures repeat exactly, and the spread across the three books is the only spread there is.

## 6. Commands

    python3 ahead.py check
    python3 ahead.py dev la=0;  python3 ahead.py dev la=1 etan=0.00005 (also 0.0002, 0.0005);  python3 ahead.py dev la=2
    python3 ahead.py life la=0
    python3 ahead.py life la=1 etan=0.00005
    python3 ahead.py life la=1 lag=64 etan=0.00005 scoreA=0
    python3 ahead.py life la=2 scoreA=0;  python3 ahead.py life la=2 lagw=64 scoreA=0
    python3 ahead.py life la=1 scoreA=0        (step 1 again through the changed program: equal)

`scoreA=1` (the default) scores the last tenth of the test text; it was used for the two runs named in section 2 and should not be used again for new arms.
