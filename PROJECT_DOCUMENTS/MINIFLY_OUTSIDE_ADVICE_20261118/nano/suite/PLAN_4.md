# One rule for the trust's step, on every stream and on two-digit sums as one stream among others (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). The step was set by the coordinating session. **Source.** This branch at 154b89e. Written before any run with the rule on any stream (the kernel's option `rule=2` has been run only inside `suite.py check`, 2,500 bytes of the book against the reference). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

The programme has carried two settings of the trust's step since its first lines: on prose 0.00005, constant; on sums 0.1 divided by the root of (1 + the times the trust was used). They differ 2,000-fold at first use. With them go two smoothings (e = 0.02 and 2) and two trusts at birth (0.25 and 0.02). `PLAN_3.md`, outcome: with the prose step a recalling sender's trusts stay small and are smallest at the levels used least (prediction C8 failed), and on the two streams where the system is still above xz what it lacks is, on my reading, speed.

## 2. The rule, exactly (`rule=2` in the kernel)

Each trust (and each part of the second cell) keeps the sum A of the squares of its own driver g (g = the message at the byte that came less the expected message, the quantity its change is proportional to under any of the rules). Its change at a use is **eta x g / sqrt(A)**, A including the present g. This is the per-coordinate rule known as AdaGrad, with nothing added. It is local: a trust needs only its own past drivers. Its first change is of size eta whatever the scale of the messages; a trust used often and pulled both ways slows as the root of its uses; a trust used rarely still moves by a visible amount.

It has one number, eta. **eta is chosen on the two prose streams only** (tiny Shakespeare and the book) from the grid 0.001, 0.003, 0.01, 0.03, 0.1, by the sum over the two streams of the losses with nothing changing and going on, second cell; then frozen. Everything else stays as it is: e = 0.02, trusts of the runs born at 0.25, every other trust and part born at nothing.

**The system it is tried on:** the best general system so far that costs no cells: runs of 0 to 6 bytes, the second answering cell with parts by kind and byte, one recalling sender (`kinds=runs gate=1 recall=1`); 0.3 to 0.56 million cells, 320 trusts, at most 6,144 parts. Compared with the same system at the constant step 0.00005 (`res/*__recall1.json`, already there).

**Two-digit sums as one stream among others** (`suite.py sums`): the stream, split and scoring of `nano/merge/merge.py` and `nano/table/gate.py` (lines "ab+cd=efg" drawn from 8,000 shown pairs; 500 pairs never shown, answered digit by digit from the system's own digits, each after a read line, five draws of the line before, with nothing changing; 40,000 lines; kinds: every set of at most four of the last nine delays, 256 kinds, as the programme's sums system has; no recalling sender). Check, run before this plan: with the per-text setting (e = 2, step 0.1 shrinking, born 0.02) on the development seed 1 the first cell has 79.4 and 79.3 (last reading; mean of the last three), `gate.py`'s stored 79.44 and 79.32, with the same 1,213,119 cells. Arms on sums, five reported seeds (73129, 93683, 116089, 406364, 545353), each once:

| arm | e | born | step |
|---|---|---|---|
| per-text (the programme's) | 2 | 0.02 | 0.1 / sqrt(1 + uses) |
| the one setting | 0.02 | 0.25 | the rule, eta as chosen on prose |
| only the step replaced | 2 | 0.02 | the rule, the same eta |
| the prose setting as it was | 0.02 | 0.25 | 0.00005 constant |

## 3. Question, hypothesis, what refutes it

**Question.** Can one rule with one number replace both settings of the step without loss anywhere?

**Hypothesis.** With eta chosen on prose: (a) going on, the rule lowers the loss on at least three of the five kinds of stream and harms no stream by more than 0.005 (the rule of `PLAN_2.md`); (b) on two-digit sums the one setting is not more than 2 points under the per-text setting at the median of five seeds, first cell (mean of the last three readings). **Refuted if** (a) fails, or the one setting is more than 2 points under on sums. If only the arm "only the step replaced" holds on sums, then the step's rule is general and the smoothing is the per-text number that remains; I will say that and not call the hypothesis held.

| | Statement | Chance |
|---|---|---|
| D1 | the chosen eta lowers the going-on loss on both prose streams by at least 0.005 against the constant step | 0.50 |
| D2 | going on, the rule gains at least 0.03 on c and on table | 0.55 |
| D3 | (a) holds | 0.60 |
| D4 | with nothing changing the rule harms at least one stream by more than 0.005 | 0.50 |
| D5 | sums, the one setting: within 2 points of the per-text setting or above (so (b) holds) | 0.30 |
| D6 | sums, only the step replaced: within 2 points of the per-text setting or above | 0.50 |
| D7 | sums, the prose setting as it was: more than 10 points under the per-text setting | 0.70 |
| D8 | under the rule the recalling sender's trust rises with the length of agreement (as C8 of `PLAN_3.md`) on at least five streams | 0.55 |
| D9 | with the rule, going on, the system is below xz on c or on table | 0.30 |

## 4. Runs, cost, what is not run

Development: 5 values of eta on 2 streams, 5 CPU-seconds each. Claim: the four other streams once at the chosen eta; sums 4 arms x 5 seeds, about 50 CPU-seconds each (1.2 million cells). Not run: other rules for the step; a rule for e; eta chosen on sums or on any stream but the two prose ones; sums with classes; 400,000 lines.
