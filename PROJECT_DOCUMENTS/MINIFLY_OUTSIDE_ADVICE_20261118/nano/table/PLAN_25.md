# How the loss and the store move with the bytes read: one life of six million bytes of prose (roadmap second revision, sections 15 and 20; a measurement) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (my translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."). Earlier the same day he asked how the system's cost compares with the usual methods in order of growth, and what changes if the text is finite. **Source.** `claude/friendly-darwin-8mmhfm` at f4e1a4ba8; `gate.py` as it stands. Written before the program of this plan (`curve.py`) exists and before any run. Development level; no checker gate, seal or review is claimed.

## 1. Why

The bitter lesson is a statement about what happens as data and computation grow. Every figure of this programme is at one size: about a million bytes. Nothing says whether the system's loss keeps falling with more text, how fast, or what the store costs as it does; `growth.py` counted cells along one text and scored nothing. The measure is now the loss while going on (section 20), which can be read at every point of a life.

## 2. What is measured

One life: all the English prose in the repository that this line has used or that is unopened, read once in this order: the test text (tiny Shakespeare, all of it), `bryant-stories`, `burgess-busterbrown`, `carroll-alice`, `chesterton-thursday`, `chesterton-brown`, `austen-persuasion`, `chesterton-ball`, `austen-emma`, `melville-moby_dick`, `edgeworth-parents` (the last is opened here for the first time by this line; `nano/layers` opened it once): about 6.3 million bytes, mapped to the test text's 65 byte values as in `life.py`.

The system: runs of 0 to 6 bytes (nothing given by hand), the two answering cells, with no number set for a text: the rule of the step with 0.03 (`PLAN_22.md`, addition 1), every trust born at nothing, and the learned room for unmet followers (`PLAN_23.md`). The stream is read in blocks of 10,000 bytes; each block is answered with the system as it stands before the block and then learned (so the loss is that of going on, a block at a time).

Reported: the loss of each cell over every stretch of the life in which the bytes read double (100,000 to 200,000, and so on), and over each text; the cells that have counted and the counts above zero at each doubling.

## 3. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| Z1 | on the stretches after the first 400,000 bytes the second cell's loss falls with every doubling of the bytes read, save where a new author begins | 0.50 |
| Z2 | from the stretch 400,000 to 800,000 to the stretch 3.2 to 6.3 million the second cell's loss falls by at least 0.10 nats | 0.45 |
| Z3 | the cells that have counted grow as the bytes read to a power between 0.55 and 0.75 over the last three doublings | 0.65 |
| Z4 | the second cell's gain over the first is larger on the last stretch than on the stretch 400,000 to 800,000 | 0.60 |

These are expectations of a measurement; there is no mechanism to hold or fail. The texts differ in difficulty (the books' own figures in `PLAN_15.md` run from 1.22 to 1.63), so the curve mixes more reading with other authors; the losses by text are given so that the two can be told apart as far as they can.

## 4. Cost

About five CPU-minutes and 2 GB. Not run: the same life in another order; a life of one author; streams of other kinds (`nano/suite` holds them, at one to two million bytes each).
