# English + addition life, design 2 (data only, 25 Nov 2026)

`make_life.py SEED CYCLES` writes `data_seed{SEED}_c{CYCLES}/`. It writes data only; no network is run. The exact rules are in the script's docstring.

Design 1 was replaced the same day, because six things varied together in it:
- problem stage and delay;
- problem size, memory distance and carry;
- exposure and stage;
- a control whose answer frequencies differed from the real life's;
- data amount and content;
- the single held-out split.

## Lines

```text
seventy one plus fifty, as every child in school knows, is one hundred twenty one - in equation 71+50=121
```

- **Verb:** "is" or "equal to".
- **Filler:** one of "", ", we say,", ", as everyone knows," and ", as every child in school knows,".
- **E2 test only:** "a and b, … make s". This form never appears in the life.
- **Numbers:** 0–1000, with sums of at most 1000.

## Problem types (cells)

There are 8 cells, given by two things:
- the digits of the larger operand: 1, 2 or 3;
- the number of carries in column addition: 0, 1, or 2 or more.

The cells are d1c0, d1c1, d2c0, d2c1, d2c2, d3c0, d3c1 and d3c2. Each cell has 16 taught units. A unit is an unordered pair, and each presentation draws the order.

## How each confound is removed

| Confound in design 1 | Design 2 |
|---|---|
| Stage tied to delay | There are no stages. Every arithmetic block contains every taught unit of every cell once, shuffled. The mean byte position of each cell's lines is the same, 82–83k in a 165k life. |
| Exposure tied to stage | Every taught unit is shown exactly CYCLES times, in every cell. |
| Size tied to carry | Size and carry are crossed in the cells, which are balanced: 16 units each. |
| Size tied to memory distance | Within each cell and block, the 8 (verb, filler) combinations are used equally and rotate across blocks. |
| Control mismatched in answer frequency | Within each cell and block, the control permutes the sums among the cell's lines. No line keeps its own sum. Each cell's answer words and frequencies are therefore identical to the real life's, and only the operand → sum relation is broken. |
| Data amount tied to content | Data scaling is CYCLES alone: the same pools and proportions, with more presentations and more new English. The held-out sets do not depend on CYCLES. |
| One split | The split is drawn by the data seed. Seeds 1 and 2 are generated; use at least two. |

**What the fillers achieve:** the distance from the second operand to the answer is 4–43 bytes, identical in every cell. The distance from the first operand still grows with the word length of the second operand, which is inherent to word forms. Its median rises from 31 bytes (d1) to 52 (d3), but the ranges overlap (13–73), so it can be entered as a covariate in a regression.

**What is held out (by the seed):**
- 10% of the numbers 20–999 never appear in any arithmetic line.
- Per cell, 16 held-out units, or 8 in d1c0 and d1c1. Those two cells have only 30 and 25 pairs. Units are unordered, so a table that sorts the operands cannot pass.

## Life

The life is Eng(shak), then CYCLES × [arithmetic block, English block], with the English blocks alternating Emma and Shakespeare.
- English blocks are 32k bytes.
- Arithmetic blocks are 128 lines, plus about 10% English lines.
- There are no review lines; rehearsal is its own later test.
- English is read from the start of each stream and stays far from the held-out middle stretch.

| Data | Bytes | Arithmetic lines | Shakespeare read | Emma read |
|---|---|---|---|---|
| seed 1, CYCLES 3 | 165,077 | 384 | 65k of 558k | 65k of 444k |
| seed 1, CYCLES 6 | 298,166 | 768 | — | — |

## Tests

Files are named `test_{kind}.jsonl`. Every item carries:
- the byte spans `word` (the addition answer), `word_first` (its first word) and `digit` (translation, because it comes after the word sum);
- the cell, the lag from each operand, the verb and filler, and the level.

| Kind | What it is | Level | Items (seed 1) |
|---|---|---|---|
| taught | the 16 taught units per cell | E1 | 128 |
| e2 | taught units in the untaught template | E2 | 128 |
| heldpair | held-out units | E3 | 112 |
| heldop | a held number as operand, sum not held (d2/d3 cells) | E3 | 96 |
| heldsum | a held number as sum, operands not held (d2/d3 cells) | E3 | 96 |

Verb and filler are balanced within each test and cell. E3 claims compare the real life against `control.txt`.

## Checks in `exposure.json` (seeds 1 and 2, CYCLES 3)

All pass:
- held numbers in the life: 0;
- held pairs in the life: 0;
- held test lines in the life: 0;
- E2 lines in the life: 0;
- control sums per cell are the same multiset as the real sums;
- control lines keeping their true sum: 0;
- exposure is equal across cells.
