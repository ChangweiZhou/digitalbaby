# A pool of cells keyed on a few chosen bytes, on prose, each kind born with no trust (roadmap second revision, tracks L and C3) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at 5dfb117dc with `PLAN_8.md`'s outcome; `place.py`, `trust.py`, `words.py` as they stand. Written before any run. Development level; one text; no checker gate.

## 1. The evidence this follows

`PLAN_8.md`: many kinds of cells can be given at once if the trust in each is born at nothing; the answering cell then finds which kinds matter. Mr. Xu, 20 November: "The connection strength (ideally can involve many cells) should be jointly decided by the cells together." The line `nano/sparse` asks on sums whether cells keyed on a few chosen bytes (connections of few ends) can answer what a cell of a whole run cannot. The same cells can be asked for on prose at once, where a run takes a minute: a cell of a whole run of six bytes has seldom occurred; a cell keyed on two or three of the last eight bytes, with gaps between them, has occurred often.

## 2. The question

**Do cells keyed on two or three of the last eight bytes, with gaps, add to the table's cells and the word cells on prose, when all such kinds are offered together and each must earn its trust?**

*The cells.* For every set of two or of three places among the last eight bytes (the present byte is place 0), except the two sets that the table's cells already are (places 0, 1 and places 0, 1, 2): 27 kinds of two places and 55 of three, 82 kinds. A cell for each combination of byte values that occurs at its places; it counts what follows; read with trust beside T2 (e = 0.02, the trust's step 0.00005, one reading). The trust in each of the 82 kinds is born at nothing.

*Arms.* (0) T2 (1.4973). (P) T2 with the pool of 82 kinds. (P2) T2 with the 27 kinds of two places only. (Pc) a control: the pool with its places taken eight bytes further back (places 8 to 15), where single bytes should carry little.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* Arm P is at least 0.01 nats below arm 0. *Refuted if* it gains less than 0.01.

| | Statement | Chance |
|---|---|---|
| W1 | arm P gains at least 0.01 | 0.45 |
| W2 | arm P gains at least 0.003 | 0.75 |
| W3 | arm P2 gains at least half of what arm P gains | 0.50 |
| W4 | arm Pc gains less than a third of what arm P gains | 0.75 |
| W5 | among the ten most trusted kinds of arm P, at least seven include the present byte | 0.70 |

## 4. Size, cost, what is not run

A kind of two places has at most 4,225 cells and a kind of three places as many as occur (reported); 3,280 more trusts. A run takes one to three CPU-minutes and about two gigabytes of memory. Not run here: sets of four places; places beyond eight back; the pool on sums (the line `nano/sparse`).
