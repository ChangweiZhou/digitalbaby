# Line `nano/sparse`, seventh plan (J1): the relay system's trust also by the place in the line (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one; `sparse3.py` gains the option `xplace=1` (no change to the kernel: the level given for the relay system's message now carries the place as well; check as before). The step is the coordinating session's proposal (its message after my second report), with its criterion. Written before any run with the option. Development level; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

`PLAN_6.md`, development seed 1 (the five seeds are running as this is written): with the relay system read as one more sender, trusted by its confidence only, the whole has 75.2% of never-shown sums at 400,000 lines where the relay system alone has 83.2% in the same run and m = 3 alone about 58%. The tens digit of the whole is 75.2; the relay system's is 91.2. `PLAN_3.md`, E4a: a trust by place lifts m = 3 alone from 56.8 to 77.8.

## 2. The question and the one change

**If the answering cell can trust the relay system differently at each place of the line, is the whole at least as good as the relay system alone?** One change from `PLAN_6.md`: the relay system's trust is one number for each of ten levels of its confidence and each count of bytes since the line end (0 to 9), born at nothing. The counting cells' trusts stay as they are (by kind and experience, not by place). The counter of bytes since the line end is laid down by the program, as in E4a; this is a concession and is said. Life: `plan=add:400000`, no second task. Five reported seeds (the development seed is not run first: nothing is chosen).

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis (the coordinating session's criterion).* At 400,000 lines the whole is at least the relay system alone on the same run, at the median of five seeds (mean of the last three readings). *Refuted if* it stays more than 3 points under.

Reading (mine) before the run: I do not expect this to be enough. At the tens digit the relay system's trust can rise, but the counting cells that say "no carry" with the weight of some forty cells are heard at every place with one trust for kind and experience, and the relay system's levels of confidence at the tens digit already had the largest trusts on seed 1 (5 to 8). What the relay system lacks against them is not a trust by place but their silence.

| | Statement | Chance |
|---|---|---|
| J1a | the whole is at least the relay system alone | 0.25 |
| J1b | the whole is within 3 points of the relay system alone | 0.40 |
| J1c | the whole is above `PLAN_6.md`'s whole (same seeds) by at least 3 points | 0.50 |

## 4. Size, cost, what is not run

As `PLAN_6.md` with 100 trusts for the relay system in place of 10. About 5 CPU-minutes a run. Not run: the counting cells' trusts by place together with this (it would be two changes; it is the next arm if this one fails); the second task (J2 of the coordinating session's message, which needs the replay of `nano/replay`).

## 5. Outcome (entered 20 November after the runs; `res/s_rlp_seed*.json`; table in `report_e7.txt`)

Never-shown additions, whole answer, %, seeds 73129, 93683, 116089, 406364, 545353.

| | add 40,000 | add 200,000 | add 400,000 (last) | mean of the last three: each seed; median |
|---|---|---|---|---|
| the whole, the relay system's trust by confidence and place | 57.6 | 67.4 | 74.0 | 74.8, 71.3, 71.1, 73.3, 73.1; **73.1** |
| the whole, trust by confidence only (`PLAN_6.md`) | 58.0 | 72.0 | 77.6 | 77.5, 75.3, 76.5, 76.6, 76.8; 76.6 |
| relay system alone, same runs | 66.2 | 82.8 | 84.8 | 84.2, 82.7, 86.4, 81.8, 86.7; 84.2 |

The relay system's trust by its ten levels of confidence at the end (seed 73129): where the hundreds digit is named 0, 0, 0, 0, 0, 2.1, 2.8, 1.4, 0.3, 0.2; where the tens digit is named 0, -0.1, 0.6, 2.5, 5.7, 8.3, 7.9, 5.6, 3.9, 2.5; where the units digit is named all under 0.1.

**The hypothesis is refuted**: the whole is 11.1 points under the relay system alone (more than 3 under refutes), on every seed. J1a, J1b and J1c failed: with the trust by place the whole is 3.5 points below the whole without it.

Readings (mine).
1. The answering cell does find where the relay system matters: its trust is large where the tens digit is named, small where the hundreds digit is named and nothing where the units digit is named. That did not help, as I expected: the trust at the tens digit was already the largest without the place, and splitting each trust over ten places makes each grow from nothing on a tenth of the bytes.
2. To the coordinating session's question, whether the relay system should then simply be the answer where it is confident: on this evidence the relay system alone is better than either joining at the tens digit (89.8 against 77.6 and 74.0), so on this text a whole that took the relay system's answer for the tens digit and the cells' for the rest would be at about the relay system's own level, no higher, and would lose addition with it. The tenth plan (`PLAN_10.md`) now supplies the carry from the counting cells themselves, which changes what the relay system is needed for.
