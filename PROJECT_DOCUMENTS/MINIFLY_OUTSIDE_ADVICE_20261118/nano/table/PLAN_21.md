# A cell that recalls: the last place where the present run of bytes stood, and what followed there (roadmap second revision, section 15; a general sender, first tried on prose) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (my translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."). **Source.** `claude/friendly-darwin-8mmhfm` at f68c445eb; `gate_all.py` as it stands (its place for one more sender whose message is given). Written before the program of this plan (`recall.py`) exists and before any run. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. The kinds of cell of this line hold contexts of at most six bytes in a row, or two or three of the last eight. Nothing in the system can use a repeat longer than that.
2. Three cells in four have been met once (`growth.py`); the system at its best holds some 18 million numbers for a million bytes (`PLAN_16.md`, section 7). It stores its stream several times over in pieces, and the one thing the stream itself would give, what followed the last time this happened, it cannot say.
3. General-purpose compressors have had such a part for decades (a model that follows the last match), and it is of no kind of text.

## 2. The one change

One more sender. At each byte it looks for the **last earlier place in what it has read where the same eight bytes stood** as stand before the present byte, and says the byte that followed there. Its message is that of a cell that has counted once: log((1 + e)/(1 + 65 e)) for that byte and log(e/(1 + 65 e)) for the others, e = 0.02 as for every cell; where there is no such place it says nothing (a flat message). Its trust is indexed by how far back the two runs go on agreeing (ten levels: none; 8 to 9 bytes; 10 to 11; 12 to 15; 16 to 19; 20 to 23; 24 to 31; 32 to 47; 48 to 63; 64 or more) and is **born at nothing**; in the second answering cell it has parts like any sender. Eight is the first length no kind of this system holds as a run; it is stated here, not chosen by a run.

With nothing changing it may recall only places in the stream it has read (the first nine tenths); with everything going on, any earlier place.

*The system it is added to:* the bare system of `PLAN_20.md` (runs of 0 to 6 bytes, the timing cells, the pool; no cells that step once a word, no companions, no relay system), with the second answering cell whose parts are indexed by kind and byte. *The control:* the same sender recalling the byte that followed a place drawn by chance, with the same levels.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The recalling cell lowers the loss on the last tenth of the test text by at least 0.005 nats with nothing changing (second cell), and by more with everything going on. *Refuted if* it lowers neither by 0.002.

| | Statement | Chance |
|---|---|---|
| Q1 | nothing changing: at least 0.005 lower (second cell; `PLAN_20.md` gives the bare system's figure) | 0.40 |
| Q2 | everything going on: at least 0.005 lower | 0.50 |
| Q3 | the gain going on is larger than the gain with nothing changing | 0.70 |
| Q4 | its trust rises with the length of the agreement, from under 0.3 at 8 to 9 bytes to over 1 at 32 or more | 0.60 |
| Q5 | the control gives under a fifth of the gain | 0.85 |
| Q6 | a recalled byte is the true byte at more than 60% of the bytes where the agreement is 16 or longer | 0.60 |

I expect little on this text, which has few long repeats beyond the speakers' names; the idea is meant for streams with more of them, and the line `nano/suite` will try it there. A small gain here would not be evidence against it.

## 4. Cost, what is not run

Two runs of about three CPU-minutes and 3 GB. Not run: other lengths than eight; several places recalled; the lives with other books; streams of other kinds (with `nano/suite`).
