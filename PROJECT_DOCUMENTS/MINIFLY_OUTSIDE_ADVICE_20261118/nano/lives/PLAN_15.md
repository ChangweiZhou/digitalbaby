# How little rehearsal keeps a rule whole: a small store of episodes, read again one byte in a hundred (21 November 2026)

**Authorisation and steer.** As `PLAN_11.md`, where Mr. Xu's words are quoted in full; the one that governs this step: "这些你自己决定，不需要问我。一切从第一原理和事实出发。" (my translation: "Decide these yourself, you need not ask me. Everything from first principles and from facts."). The choice and design of this step are mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Development level; no checker gate, seal or review is claimed.

**What I had seen when I wrote this.** `PLAN_12.md`'s reported seeds for its two main forms; `PLAN_13.md`'s development seed and three of its ten reported runs; `PLAN_14.md`'s development seed (stiffening). No run of this plan's program other than its check (`check15_output.txt`, passed).

## 1. The facts this follows

* **The rule is not what is lost.** After English the relay network's cells still hold the rule of addition (`PLAN_11.md`: 95.7 of 96.6 with the reading put back).
* **With the four local changes the reading among the digits is kept too** (`PLAN_12.md`, five seeds: 92.9% of never-shown sums named when only the digit branches may answer, 98.8% before), and the English is kept when the sums return (46.6% to 42.5% against 40.9% to 19.0%).
* **What is still lost is the balance between the two texts' branches in the answering cell**: straight after the English a branch of another byte is louder than the right digit, so whole answers stand at 0.2%. A hundred lines of sums restore 85%.
* **More returns help but do not cure** (`PLAN_13.md`, development seed: 0.7%, 40.2%, 43.3% straight after the three pieces of English). The sums use 254 of the 256 cells and every cell of the English is one of them, so every piece of English works on strengths the sums need, and every return on strengths the English needs.
* **Slower strengths do not cure either** (`PLAN_14.md`, development seed): the English asks the digit branches to be silent for 400,000 bytes, and a smaller step only makes that take longer and the return slower.

So with cells that both texts use, the balance in the answering cell is held only while both texts go on being read. **What a nervous system has for exactly this (my reading of textbook knowledge; I cite no paper):** a second, small and fast store of single episodes, and the replaying of them into the slow, shared network while other things are learned. This programme has used replay before (`nano/replay`, as that line reports: addition kept at 93.8% through subtraction against 1.8%), at one replayed line for every line read or for every second line, from a store that held every line shown, on the network as it then was. The question here is the dose: **how little rehearsal does the network need once its strengths change only while their own cell speaks?** A learner that must read everything twice to keep it is not a persistent learner; one that reads one byte in a hundred again may be.

## 2. What is run (`relay15.py`)

**The store.** As the stream is read it is cut into consecutive episodes of 40 bytes. The system keeps 1,000 of them (40,000 bytes), chosen so that every episode read so far has the same chance of being among them: when the store is full, the n-th episode takes the place of a kept one, drawn evenly, with chance 1,000 / n. No kind of text is named, no boundary is given, no line is aligned. The store is bounded, and it is filled in one pass.

**Rehearsal.** After every 30 / rho bytes of the stream, one kept episode, drawn evenly, is read again with learning on: its first ten bytes fill the delay line without learning, its thirty others are learned as any byte is; then the delay line of the stream is put back. rho is the number of bytes rehearsed for each byte read: **1/100** and **1/20**.

**The life, the learner and the marks** are `PLAN_13.md`'s: sums, then three pieces of English with a return of the sums after each; the network as it is (R) and with the four changes of `PLAN_12.md` (Rgtqr, theta 4), nothing in either changed. The runs without rehearsal are `PLAN_13.md`'s and are not run again (with rho at nought the reader gives the plain reading bit for bit: checked). Added: how many of the rehearsed episodes came from the sums and how many from the English.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With the four changes, one byte in a hundred rehearsed from a store of 1,000 episodes keeps the sums whole through the English: straight after each of the three pieces of English, with nothing changing, the network names three quarters or more of the never-shown sums it named before that piece (median of five seeds). The network as it is does not reach that at the same dose.

**Refuted if** Rgtqr at 1/100 keeps under a quarter straight after the second English: then even with local changes the answering cell needs the old text in bulk, and rehearsal at a price a persistent learner can pay is not the answer for this network.

| | Statement (five reported seeds; median; "kept" = whole answers straight after an English over those straight before it) | Chance |
|---|---|---|
| Q1 | Rgtqr at 1/100 keeps three quarters or more at all three returns (the hypothesis) | 0.50 |
| Q2 | Rgtqr at 1/20 keeps three quarters or more at all three returns | 0.75 |
| Q3 | R at 1/100 keeps under a quarter at all three returns | 0.55 |
| Q4 | R at 1/20 keeps three quarters or more at all three returns | 0.40 |
| Q5 | Rgtqr at 1/100 keeps more than R at 1/20 at the second return: the changes are worth more than five times the rehearsal | 0.50 |
| Q6 | Rehearsal costs the English nothing: Rgtqr's held-out English at the end of each piece is within 1 point of the same form without rehearsal, at 1/100 | 0.70 |
| Q7 | It also keeps the English through the returns: at 1/100 Rgtqr's held-out English falls by less than 1 point over each return of the sums | 0.50 |
| Q8 | It does not harm the learning of the sums: at 1/100 Rgtqr ends the first phase within 1 point of the same form without rehearsal | 0.80 |

## 4. Runs, size, cost, what is not run

Development seed 1, the four arms (two forms, two doses): to see that the program runs through the life; nothing is chosen, and the store's size, the episode's length and the two doses are fixed here. Reported: the seeds 73129, 93683, 116089, 406364, 545353, the four arms. Size: 256 cells, about 230,000 strengths; the store, 40,000 bytes. Cost: about 3 CPU-minutes a run, 24 runs.

**Not run, and what this plan does not settle.** (1) *Episodes made by the system's counting cells in place of kept bytes* (`nano/replay`'s way). An earlier criterion of this programme asked that no line be shown again by the program; a store of 1,000 raw episodes is a part of the system as built here, but whether kept bytes count as the system's own memory is for Mr. Xu and Mr. Zhou to say. (2) A smaller store, or a dose under 1/100. (3) Subtraction after addition, where the bytes are the same. (4) A life long enough that an old text holds a vanishing share of the store. (5) Revising what has become false: rehearsal of old episodes works against it, and nothing here measures that.

## 5. Outcome

(to be entered after the runs)
