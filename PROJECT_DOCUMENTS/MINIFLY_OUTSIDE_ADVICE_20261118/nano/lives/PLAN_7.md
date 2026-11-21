# A cell's own belief in its memories: old against new weighed inside the cell, not in the shared trusts (21 November 2026)

**Authorisation and steer.** As `PLAN_1.md` and `PLAN_6.md` (Mr. Xu's words quoted there). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written after the development lives of `PLAN_5.md` and four of its six singles, while its reported lives were running and before any of them was read; before any run of the program of this plan. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows, and the hypothesis in one sentence

`PLAN_1.md`: about three quarters of what a life costs on the streams read earlier sits in the trusts, which are shared by all contexts and follow the stream being read. `PLAN_5.md` (development): giving a short context two more memories that hold what followed lately lowers the loss on every stream, by 1 to 13%; there the three memories of a context are three senders, and how far each is believed is again a trust, one number for each kind and level, shared by all contexts. So the weighing of old evidence against new, the thing that must change when the kind of text changes, was put where the system forgets.

**Hypothesis in one sentence:** that weighing belongs inside the cell. A context knows from its own record which of its memories has foretold its own followers; nothing shared need be told.

## 2. The mechanism, exactly (`cellmix.py`)

For a run of 0 to 3 bytes, a context has the three cells of `PLAN_5.md`: one that counts for life, one that halves its counts at 512, one at 32. New:

* **The cell holds a share for each of its three memories** (summing to one; equal at birth).
* **A sender for the cell's mixture.** A memory's opinion of a value v is (its count of v + e) / (its counts + 256 e), with the system's e = 0.02. The cell's opinion is the sum over its memories of share x opinion. Its message for a value its life memory has met is the log of its opinion of v less the log of its opinion of a value never met. The trust for this sender is by the level of the life memory, born at nothing, with its parts in the second answering cell, like every sender.
* **The shares move by the cell's own record.** After a byte b: each share becomes ((1 - s) x share + s / 3) x that memory's opinion of b, and the shares are made to sum to one; s = 1/32, the smaller bound (one event in 32 of a context may change which memory is right).
* The three memories count as before.

**Three systems** on the small system of `PLAN_1.md`:

* **M** (`PLAN_5.md`'s chosen form): the three memories speak as three senders.
* **C**: the two bounded memories are silent; the cell's mixture speaks beside the life memory.
* **MC**: all of them speak.

One of C and MC is chosen on the development order by the mean over the six tests of the going-on loss at the end of the life, then frozen; M is the comparison.

**Check before any run** (`python3 cellmix.py check`): the compiled kernel against a reference written out in full (the shares, the mixture's message, silent kinds), under both rules of the step; with no mixture sender and no silent kind the losses are those of `mem.py`'s kernel; no answer moves when its own byte and all later ones are replaced.

## 3. What refutes it, predictions (mine)

**The hypothesis is supported if**, on the five reported orders, the chosen system (a) ends a life lower than M on at least five of the six tests in at least four orders, and (b) loses less than M on the streams read earlier: its going-on excess at the end over the stream's own stage is smaller than M's in at least two thirds of the cases. **Refuted if** (b) fails: then weighing inside the cell does not take the weighing out of the trusts, whatever it gains otherwise.

| | Statement (chosen system against M; second answering cell; going on; reported orders unless said) | Chance |
|---|---|---|
| C1 | (a) holds | 0.45 |
| C2 | (b) holds | 0.40 |
| C3 | Read alone, the chosen system is below M by 0.001 or more on at least four of the six streams | 0.50 |
| C4 | Read alone, it harms no stream against M by more than 0.005 | 0.65 |
| C5 | The cost of a life (mean over six tests at the end less the mean of its own singles) is lower than M's in at least four of five orders | 0.40 |
| C6 | On the development order C alone (bounded memories silent) is at least as low as M at the end of the life | 0.40 |
| C7 | With nothing changing, the chosen system is lower than M on at least four of six singles (the cell's shares stop moving, but they were set by the cell's own record) | 0.45 |

## 4. Runs, size, cost, what is not run

Development: two lives (C, MC). Claim: six singles and five lives of the chosen system. Size: as M (2.47 million cells at the end of the development life) and three shares for a cell. Cost: as M, about a quarter more. **Not run:** shares for runs of 4 to 6 bytes; more than three memories; with remembered trusts; the large system.

## 4a. The choice (entered 21 November, 01:06 UTC, before any single or reported life of C or MC; `res/cellmix__C_dev.json`, `res/cellmix__MC_dev.json`)

Development order, mean over the six tests of the going-on loss at the end of the life: the plain system 0.9184; M 0.8656; C (the bounded memories silent, the cell's mixture speaking) 0.8671; **MC (all speak) 0.8549. MC is taken and frozen.** C6 is scored here, since it is a statement about the development order: **failed** (0.8671 against M's 0.8656; C is lower than M on the two prose streams and Python and higher on C headers, the table and markup).

## 5. Outcome (entered 21 November, after the five reported orders; `report_7.txt`, `res/cellmix__MC_*.json`, `out/`, `check7_output.txt`)

**Read alone, going on** (second answering cell, nats a byte; Shakespeare, book, Python, C, table, XML): M 1.3266, 1.3931, 0.9213, 0.6319, 0.2977, 0.2603 (mean 0.8052); **MC 1.3234, 1.3904, 0.9171, 0.6244, 0.2911, 0.2567 (mean 0.8005)**; MC less M: -0.0033, -0.0027, -0.0043, -0.0075, -0.0066, -0.0037. With nothing changing, MC less M: -0.0014, -0.0029, -0.0043, +0.0303, +0.0190, +0.0056.

**Lives** (five reported orders; cost = mean over six tests at the end less the mean of the system's own singles).

| Order | Mean at the end: plain, M, MC | Cost of the life: plain, M, MC | Tests on which MC is below M |
|---|---|---|---|
| 73129 | 0.8814, 0.8327, 0.8264 | +0.0457, +0.0275, +0.0259 | 6 of 6 |
| 93683 | 0.8814, 0.8337, 0.8273 | +0.0457, +0.0285, +0.0268 | 6 of 6 |
| 116089 | 0.9291, 0.8781, 0.8669 | +0.0934, +0.0730, +0.0664 | 6 of 6 |
| 406364 | 0.8834, 0.8335, 0.8265 | +0.0477, +0.0283, +0.0261 | 6 of 6 |
| 545353 | 0.8861, 0.8401, 0.8332 | +0.0504, +0.0349, +0.0327 | 6 of 6 |

| | Outcome |
|---|---|
| C1 | **Held.** MC ends lower than M on six of six tests in five of five orders |
| C2 | **Held, and small.** MC's excess at the end over the stream's own stage is smaller than M's in 25 of 25 cases; of M's excess it removes 6% at the median (2.5% to 9.5% over the 21 cases in which M's excess is above 0.02) |
| C3 | **Held.** Read alone, MC is below M by 0.001 or more on six of six streams |
| C4 | **Held.** Read alone and going on, it harms no stream |
| C5 | **Held.** The cost of a life is lower than M's in five of five orders (by 6, 6, 9, 8 and 6%) |
| C6 | **Failed** (section 4a: C alone 0.8671 against M's 0.8656 on the development order) |
| C7 | **Failed.** With nothing changing MC is lower than M on three of six singles, and worse by 0.030 on C and 0.019 on the table |

**By its own criterion the hypothesis is supported** ((a) and (b) hold). **What that is worth (my reading):** a cell that weighs its own memories is a gain on every stream and in every life measured (0.2 to 1.2% of the nats on a single stream, 6 to 9% of the cost of a life), and it is paid for with nothing changing on the two most regular streams. It does not do what section 1 hoped for: nine tenths and more of what a stream loses over a life are still lost with it. The weighing that is forgotten stays in the trusts.

**Not run:** as section 4 lists.
