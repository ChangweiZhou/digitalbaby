# Why the senders of recency harm with nothing changing, as a hypothesis with a falsifier: trust by the age of what is said (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). This step is mine, under "verify with care": it tests the explanation I gave for a harm. **Source.** This branch at the commit that holds this file. Written before any run with `age=1` on any stream (it has run only inside the check). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

System Z (`PLAN_6.md`; its six runs are in `res/*__Z.json`, the outcome is entered there): going on it is below xz on all six streams; **with nothing changing it is above the base on five of six** (1.547 against 1.523 on tiny Shakespeare; 2.240 against 2.122 on c; 2.014 against 1.757 on table). Each of the senders that say what was just read showed this in its own step (`PLAN_3.md`: c harmed by 0.116; the last-byte senders: all six streams harmed by 0.02 to 0.08). My explanation so far, untested: while reading, what such a sender says is fresh, and the answering cell learns a trust that fits fresh sayings; with nothing changing the same sayings grow old and the trust stays.

## 2. The change, exactly (`age=1`)

If that is the cause, the answering cell lacks one thing it could know locally: **how old the saying is**. So the trusts of the senders of recency are indexed by age in place of their present levels; nothing else changes and no cell or sender is added.

* A last-byte sender: forty levels = (different bytes the run's cell has met: 1, 2, 3 or 4, more) x (bytes since that cell last counted: 1 to 3, 4 to 15, 16 to 63, ... four-fold, ten steps). Each cell keeps the time it last counted (one number more a cell). The count's size is no longer in the index.
* A recalling sender: forty levels = (bytes of agreement beyond eight: 0, 1, 2 to 3, 4 to 7, ..., 64 and more; eight steps) x (bytes back to the recalled place: under 64, under 1,024, under 16,384, under 262,144, more).

While reading, old sayings occur too (a cell not met for a long time; a place far back), so the trusts for great ages are learned from the system's own life. With nothing changing the time goes on (as `nano/table/PLAN_16.md`, correction 3, asks) while cells and places stand still, so sayings age and fall into the levels learned for old ones. The steps of the two scales are laid down here and not tuned.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With the trusts indexed by age, on the system Z: with nothing changing the loss is lower than Z's by at least 0.02 on at least four of the six streams; going on it is not above Z's by more than 0.005 on any stream. **Refuted if** the loss with nothing changing is lower by less than 0.02 on three or more streams (then age is not what the answering cell lacked), or going on is harmed by more than 0.005 somewhere (then the age index costs what the count's size gave).

| | Statement (second cell) | Chance |
|---|---|---|
| G1 | the hypothesis holds | 0.45 |
| G2 | with nothing changing the system with age is not above the base (first cell, `PLAN_1.md`) by more than 0.005 on at least four streams | 0.30 |
| G3 | going on, the system with age is below Z by at least 0.001 on at least three streams | 0.45 |
| G4 | the trust of the last-byte sender of the run of two bytes falls with age (mean over the three youngest steps above the mean over the oldest three used, on at least five streams) | 0.75 |

## 4. Runs, what is not run

The two prose streams first (there is nothing to choose), then the four others once; 8 CPU-seconds each. Not run: other scales of age; age for the cells themselves (counts that fade); an index of both the count's size and the age (more levels than the forty every sender has).
