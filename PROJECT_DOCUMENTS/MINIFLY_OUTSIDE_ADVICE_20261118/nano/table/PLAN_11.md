# Companions born with no trust (roadmap second revision, tracks K and L; one change from `PLAN_6.md`) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at a774aef89; `two.py` and the lives of `PLAN_5.md` and `PLAN_6.md`. Written before any run. Development level; no checker gate.

## 1. The evidence this follows

`PLAN_10.md`, section 7: in a system that has read little, the fading companions do harm (on the first of three books read from nothing: 1.5962 against 1.5598 without them), while the kinds of cell that were born with no trust did no harm at any point. `PLAN_8.md`: a new kind of sender should be born with no trust. In `PLAN_6.md` the companions were born with the trust of an old kind (0.25), because that plan came before the rule.

## 2. The one change

The answering cell's trust in every kind of companion is born at nothing. Everything else as `PLAN_6.md` with a half-life of 10,000 bytes (T2; e = 0.02; the trust's step 0.00005).

*Measured* (as `PLAN_6.md`, addition 3): each text's last tenth right after its own stream, nothing changing, in the life A, B and in the life of B only; A again at the end of the life A, B. Beside the figures of `PLAN_6.md` for companions born at 0.25 and for no companions.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* Born with no trust, the companions no longer harm a young system and still help an older one: on the first book of the life of B only the loss is within 0.005 of the system without companions (1.5598), and on the second and third books of the life A, B it is at least 0.02 below the system without companions (1.3162, 1.4109). *Refuted if* either fails.

| | Statement | Chance |
|---|---|---|
| N1 | first book, life of B only: within 0.005 of 1.5598 | 0.65 |
| N2 | second and third books, life A, B: at least 0.02 below the system without companions | 0.55 |
| N3 | A after A: at most 1.4930 (half of the companions' gain of 0.008 kept) | 0.45 |
| N4 | A at the end of the life A, B rises less than with companions born at 0.25 (which rose by 0.120) | 0.60 |

## 4. Cost, what is not run

Under a minute of CPU. Not run: other half-lives; the going-on measure (one further run if the hypothesis holds).

## 5. Outcome (entered 20 November after the run; `res/two_T2_H10000_moments_birth0.json`)

Loss in nats a byte, nothing changing during a scoring; each text's last tenth right after its own stream.

| | A after A | Bryant, Burgess, Carroll in the life A, B | Bryant, Burgess, Carroll in the life of B only | A at the end of the life A, B (rise) |
|---|---|---|---|---|
| T2, no companions | 1.4973 | 1.4802, 1.3162, 1.4109 | 1.5598, 1.2769, 1.3836 | 1.5067 (0.009) |
| companions born at 0.25 (`PLAN_6.md`) | 1.4892 | 1.4720, 1.2558, 1.3630 | 1.5962, 1.2805, 1.3808 | 1.6093 (0.120) |
| companions born with no trust | **1.4869** | 1.4675, 1.2592, 1.3667 | **1.5632, 1.2735, 1.3758** | 1.5837 (0.097) |

**The hypothesis holds.** N1 held (the first book read from nothing: 1.5632, 0.003 from the system without companions, where companions born at 0.25 cost 0.036). N2 held (0.057 and 0.044 below the system without companions on the second and third books of the life A, B). N3 held (A after A is 1.4869, better than with companions born at 0.25). N4 held (the rise on A after B is 0.097 against 0.120).

Readings (mine).
1. **Born with no trust, the companions no longer harm a young system and help everywhere else**: in the life of B only they are now level on the first book and better than no companions on the second and third (by 0.003 and 0.008); in the life A, B they keep nearly all of what companions born at 0.25 gave. The rule of `PLAN_8.md` holds a third time, on a kind of cell it was not found on.
2. The cost of `PLAN_6.md` is smaller and still there: after B, scored with nothing changing, the loss on A rises by 0.097. That cost is of the trusts being set for the latest text, and a birth at nothing does not remove it.
3. Not run: the system of `PLAN_10.md` with its companions born with no trust (I leave the last tenth of the test text unscored for it today; it has been scored often enough); the going-on measure.
