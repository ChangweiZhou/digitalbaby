# A woven life: the six streams in turn, 65,536 bytes at a time (21 November 2026)

**Authorisation and steer.** As `PLAN_1.md` and `PLAN_6.md` (Mr. Xu's words quoted there). The test itself is proposed in the report Mr. Xu gave on 21 November, "Outside literature for the twenty research prompts", under prompt 1, "Experiments for one day": "Run the pool on one stream alone, and on the six streams interleaved in 64 KB blocks", with, as the outcome that would count against it, "The pool loses more than 0.002 nats/byte on the single stream, or gains nothing when the streams are interleaved, which would mean that the gain came from the protocol's one-shot returns". The design below is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written before any woven stream was built or read. Development level; no checker gate, seal or review is claimed.

## 1. Why

In every life of this folder each stream is read whole and returns once, as its last tenth. The report's strongest argument against remembered trusts is that this protocol "may overstate the value of a memory over time". A life in which the kinds of text alternate all the time is the opposite case: every change is a return, there are hundreds of them, and a system has no long stretch in which to settle. It also asks more of the counts in two memories, which were shaped on lives with five changes.

The first half of the report's test is already in `PLAN_3.md` (read alone, the mixture is 0.0000 to 0.0005 below the plain answer going on, on all six streams: no loss).

## 2. What is run (`weave.py`)

The six bodies (first nine tenths) of `nano/suite`'s streams are cut into blocks of 65,536 bytes; the blocks are read in rounds, one block of each stream in an order drawn once for the life by the seed, a stream that is used up dropping out. One system reads the whole woven stream once. Measured: the loss while reading, for each stream (all its blocks; and the second half of its blocks, after the system has met every kind many times); then, in forked copies, each stream's test with nothing changing and going on. The reference for each stream is the same system reading that stream alone (its body in one piece, then its test).

Systems, each with the numbers it has: the plain small system; with remembered trusts at the pace 65,536 (`PLAN_3.md`); with bounded kinds and the cell's mixture (MC of `PLAN_7.md`); MC with remembered trusts; and each of the two with remembered trusts once more with the waking of `PLAN_8.md`. All six are run on one kernel (`awake.py`'s, with the parts switched off that a system does not use).

Orders: development, the order of `nano/suite`'s table; reported, the first three reported seeds. Nothing is chosen.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** The value of remembered trusts is not an artefact of streams that return once: in a woven life, while reading the second half of each stream's blocks, the mixture is below the plain answer of the same run by 0.005 nats a byte or more on at least four of the six streams in every reported order.

**Refuted if** it is so on fewer than two streams in two or more orders (then the gain of `PLAN_3.md` came from the one-shot returns, as the report's counter-argument has it).

| | Statement (second answering cell; reported orders; "reading" means while reading the second half of a stream's blocks) | Chance |
|---|---|---|
| V1 | The hypothesis holds | 0.50 |
| V2 | Weaving costs the plain system: its reading loss is above the single's over the same bytes by 0.02 or more on at least four of six streams in every order | 0.65 |
| V3 | The mixture is nowhere worse than the plain answer by more than 0.002 (reading and tests, every order) | 0.80 |
| V4 | Counts in two memories help more in a woven life than in a life of whole streams: MC's reading loss is below the plain system's by more than its gain on the singles (0.035 at the mean), at the mean over the six streams, in every order | 0.50 |
| V5 | With MC and remembered trusts together the cost of weaving (reading loss less the same system's single over the same bytes, mean over six streams) is under half of the plain system's cost in every order | 0.40 |
| V6 | The waking of `PLAN_8.md` lowers the mixture's reading loss further by 0.003 or more at the mean in at least two of three orders | 0.40 |
| V7 | On the tests at the end, going on, the plain woven system is within 0.02 of its singles on at least four of six streams (a life whose kinds return all the time is kept better than one in which they return once) | 0.45 |

## 4. Runs, size, cost, what is not run

Twenty-four woven lives (six systems, four orders); a woven life has one stage, so it costs about a third of a life of `PLAN_1.md` (those with remembered trusts about 15 CPU-minutes on a free machine). The singles exist except for the reading loss over the second half of a body, which is taken from the stored singles' runs again where needed. **Not run:** blocks of other lengths; streams woven with those that are not text; the large system.

## 5. Outcome

(to be entered after the runs)
