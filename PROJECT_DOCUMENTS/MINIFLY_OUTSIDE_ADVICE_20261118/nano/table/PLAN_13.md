# The relay system through the life of two texts (roadmap second revision, track K2; the comparator that was still owed) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese; my translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." **Source.** `claude/friendly-darwin-8mmhfm` at ed2b3cd76; the relay system is `nano/loop/loop.py` with its kernels, unchanged (copied from `claude/nano-loop-8mmhfm`); texts and segments are those of `life.py` (`PLAN_5.md`). Written before any run. Development level; no checker gate.

## 1. Why

`PLAN_5.md` measured what cells that count keep of a first text after reading three other books (a rise of 0.009 nats), and the reference line measured a small GPT through the same life (a rise of 0.49 to 0.77). The relay system, which is the system this project began with, has not been taken through it: it was asked of the line `nano/loop`, which had finished its brief before the request reached it and rightly did not change its brief on a message.

## 2. The life and the system

The relay system of the loop line's first plan: 1,024 cells, ten delay lines, branches that compete, eta2 = 1/4, fading 1e-5, and its two forms: return strengths born at nothing with eta1 = 1 (the form its development seed chose), and born at the earlier size (the relay proper). It reads the stream of A eight times, as in that plan; then the first nine tenths of the three books, in turn, the three together eight times; no byte of A again. A second system reads the books only (eight times). Scored with nothing changing, on the bytes of `life.py`: the last tenth of A after A and after the books; the books' held-out tenths after A, after both, and for the system that read the books only. Seed 1 for both forms; seed 73129 as well for the form born at nothing.

## 3. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| G1 | after the books, the loss on A's last tenth has risen by at least 0.2 nats, in both forms | 0.75 |
| G2 | the rise is larger than the cells that count showed (0.009) by a factor of ten or more | 0.90 |
| G3 | having read A first helps on the books: at least 0.03 nats below the system that read the books only | 0.60 |

There is no hypothesis to refute here beyond these: the run fills a row that the ledger lacks. A reading eight times over is not the life the cells that count had (one reading); that difference is stated beside the figures, as the reference line stated it for the GPT.

## 4. Cost

About ten CPU-minutes a life (43 seconds a reading of A at this size).

## 5. Outcome (entered 20 November after the runs; `res/life_relay_*.json`)

Loss in nats a byte, nothing changing during a scoring. "The books" are the three held-out tenths together, weighted by length. Beside the relay system: the cells that count (`PLAN_5.md`, T2; one reading of each text) and the small GPT (`nano/ref`; each text read 7.65 times in random blocks).

| System | A after A | A after A and the books | rise | The books after A and the books | The books, system that read them only | what A gave |
|---|---|---|---|---|---|---|
| relay system born at nothing, seed 1 | 1.7674 | 2.2141 | **0.447** | 1.7289 | 1.7583 | 0.029 |
| the same, seed 73129 | 1.7689 | 2.2175 | **0.449** | 1.7250 | 1.7584 | 0.033 |
| the relay proper (return strengths born at a size), seed 1 | 1.8395 | 2.2437 | **0.404** | 1.7942 | 1.8445 | 0.050 |
| cells that count, read with trust (T2) | 1.4973 | 1.5067 | 0.009 | 1.4152 | 1.4300 | 0.015 |
| small GPT by back-propagation | 1.537 | 2.030 to 2.302 | 0.49 to 0.77 | 1.403 to 1.414 | 1.584 | 0.18 |

G1 held (a rise of 0.40 to 0.45 in both forms). G2 held (about fifty times the rise of the cells that count). G3 held on two lives of three and missed by 0.001 on the third (0.029, 0.033, 0.050). A life took 11 to 15 CPU-minutes on one thread.

Readings (mine).
1. **On prose the relay system forgets as a network taught by back-propagation does**: 0.40 to 0.45 nats lost on the first text after the three books, against 0.49 to 0.77 for the small GPT and 0.009 for the cells that count. After the books it is nearly where a system that never read the first text stands (2.21 against 2.38).
2. Having read the first text helps the relay system a little on the books (0.03 to 0.05), more than it helps the cells that count (0.015) and much less than it helps the GPT (0.18).
3. The three learners did not live the same life: the relay system and the GPT read each text about eight times, the cells that count once. The comparison of what is kept does not depend on that (more readings of the second text can only take more of the first); the comparison of levels does.
4. So on prose as on sums: the cells taught by the relay generalise or fit, and lose what they had; the cells that count keep. This run adds nothing new to that; it fills the row that was missing.
