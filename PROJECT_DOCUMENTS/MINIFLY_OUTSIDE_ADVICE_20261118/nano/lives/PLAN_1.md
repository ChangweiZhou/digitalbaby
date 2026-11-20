# A life over six streams of different kinds: what the general system keeps (20 November 2026)

**Authorisation and steer.** Mr. Xu's words, in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (my translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas.") and "继续大胆尝试，小心求证。" (my translation: "Go on trying boldly, and verify with care."). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file; the program and the six streams of `nano/suite` as taken in at 96f797e. Written before any run of a life. Development level; no checker gate, seal or review is claimed.

## 1. Why this step

The programme's first criterion is keeping: what was learned of A must stand through the learning of B and be there when A returns. On two-digit sums it was met late on 20 November by the relay system with replay (93.8% of never-shown addition kept through a second task). On streams of general kinds no line has measured it: `nano/suite` judged each idea on six streams, **each read by a system of its own**, and lists "a life over several streams in turn (keeping)" as not run. A learner that is meant to persist reads one thing after another with one store and one set of trusts. This step measures what that costs, before anything is built against it.

## 2. The system, exactly

`lives.py` holds the kernel of `nano/suite/suite.py` with five changes and no other (each is a replacement in that program's text, asserted to apply exactly once):

1. **Raw bytes.** A stream is its bytes, 256 values, nothing mapped (in `nano/suite` each stream was mapped to its own table of at most 128 values, which cannot serve one system that reads several streams). A context key holds 8 bits a byte instead of 7.
2. **A timing kind is silent until its byte has been met.** In `nano/suite` the kind "bytes since v last came" said "long ago" for a value never met. With 256 kinds of which some sixty to a hundred and ninety have a byte in the stream, the others would be so many copies of the kind keyed on the last byte alone. Here such a kind neither speaks nor counts until v has been met once.
3. **The system's own time.** Ages (of the byte a cell counted last, of a recalled place) and the timing clocks were differences of places in the one array a system read. Here they are differences of the system's own time, the number of bytes it has read so far, so that a test which lies elsewhere in the array is read as what comes next. Read in one piece from the start of one array, this is the same as before.
4. The sums of squared drivers can be read and written from outside (for the diagnostics of section 3).
5. **A guard against drivers at the size of the machine's rounding** (found while this program's check was written, before any life was run). Under the rule of the step, `nano/suite`'s kernel changes a trust as soon as the sum of its squared drivers is above zero, by the rule's full number in the direction of the driver. Where a first driver is a difference of nearly equal numbers, its direction is decided by rounding: on raw bytes with the timing kinds, the compiled kernel and the reference written out in full parted ways by up to 0.2 in a single probability. Here a trust does not change while that sum is at or below 1e-20 (a driver below 1e-10). With the guard at zero the kernel is `nano/suite`'s bit for bit. What the guard changes in that line's figures is measured on the singles (section 3) and reported.

Two systems, with the one setting of `nano/suite` (smoothing 0.02; trusts of the runs born at 0.25, all others at nothing; the rule of the step with the number 0.03; three recalling senders of the last 8 bytes; a sender for the byte each run's cell counted last; trusts of those senders by age; a second answering cell with parts by kind and byte at each of the last six places):

* **ZA**: runs of 0 to 6 bytes. 17 kinds. No more cells than the plain system of runs. `kinds=runs+last recall=3 gate=1 rule=2 eta=0.03 age=1`.
* **ZPA scaled**: the same with the timing cells and the pool, and the step's number scaled by the senders that speak. 355 kinds, of which those of bytes never met stay silent. `kinds=runs+last+since+pool recall=3 gate=1 rule=2 eta=0.03 age=1 spow=0.5`.

Nothing is chosen in this plan: no number of either system is touched.

**Check before any run** (`python3 lives.py check`, `check_output.txt`; 24 tests, all passed): the compiled kernel against a plain-Python reference written out in full on raw bytes with 256 values (18 cases over text, markup, a table, machine code, random bytes and repeats, with every kind and option used here; agreement to 2e-14 under the constant step, to 3e-9 under the step by uses, and under the rule of the step to 8e-7 with a dozen senders and 5e-5 with some two hundred, where relative rounding of small drivers is carried into the trusts); on three of `nano/suite`'s mapped streams, with the guard at zero, the same losses bit for bit as that line's own kernel; a test placed elsewhere in the array behind a copy of the last 4,096 bytes read gives the same losses bit for bit as one that follows directly; a measurement in a forked copy leaves the system as it was; no answer moves when its own byte and all later ones are replaced.

## 3. The life and what is measured

A stream's first nine tenths are its **body**, its last tenth its **test** (the cut of `nano/suite`). A life: one system reads the six bodies one after another, once each, in a given order. After each body (a **stage**), in copies of the system that are thrown away afterwards (forked processes), so that the life itself is never changed by a measurement:

* every stream's test **with nothing changing** (both answering cells);
* every stream's test **going on** (each byte answered before it is read, cells and trusts moving), each from the same state of the stage; with the loss over the first 1,000, 4,000, 16,000 and 64,000 bytes of the test, to see how fast the system comes back;
* the loss while the body was read (whole, first tenth, last tenth), the cells and the counts.

**Singles**: each stream read by a system of its own, the same measurements (this is `nano/suite`'s table on raw bytes). On the mapped streams of `nano/suite`, each single is also run once with the guard at zero and once with it, to say what the guard changes.

For a stream j in a life, with "own stage" the stage at which its body has just been read:

* **kept, going on**: the going-on loss on its test at the end of the life less that at its own stage;
* **kept, nothing changing**: the same with nothing changing;
* **what the earlier reading gave or cost**: its loss at its own stage less the single's; and the loss while its body was read less the single's.

**Two diagnostics at the end of a life**, for every stream but the last read, to say where a loss with nothing changing sits: (D1) the system of the end with the trusts and parts it had at that stream's own stage put back (the counts of the end, the trusts of then); (D2) the single of that stream with the trusts and parts of the life's end put in (the counts of that stream alone, the trusts of the end).

**Orders.** Development (to see that the program does what is written, nothing is chosen): the order of `nano/suite`'s table (tiny Shakespeare, book, Python, C, table, markup). Reported: the five orders drawn by `numpy.random.default_rng(seed).permutation(6)` for the seeds 73129, 93683, 116089, 406364, 545353. The systems are deterministic; the orders are the only draws.

## 4. Hypothesis, what refutes it, predictions (mine, written before any life was run)

**Hypothesis.** What the general system has learned of a stream is kept through the reading of up to five streams of other kinds, in the sense that matters for a learner that goes on: at the end of the life, going on, each stream's test costs at most 0.02 nats a byte more than at its own stage, for every stream in every reported order, for both systems. What is lost with nothing changing sits in the trusts, which follow the stream being read, and not in the counts.

**Refuted if** the going-on loss is more than 0.02 above the own stage's for any stream in three or more of the five reported orders (either system); the second sentence is refuted if putting back the trusts of the own stage (D1) removes less than half of the excess with nothing changing, at the median over the cases where that excess is above 0.02.

| | Statement (second answering cell unless said; reported orders) | Chance |
|---|---|---|
| K1 | ZA: kept going on within 0.02 for every stream in every order | 0.60 |
| K2 | ZPA scaled: the same | 0.55 |
| K3 | With nothing changing, the excess at the end over the own stage is above 0.05 in at least half of the cases (a stream not read last, an order), ZPA scaled | 0.70 |
| K4 | D1 removes at least two thirds of that excess at the median (ZPA scaled) | 0.55 |
| K5 | D2 (the single's counts with the life's last trusts) shows at least half of that excess at the median | 0.50 |
| K6 | Going on, the excess over the own stage within the first 4,000 bytes of the test is at least three times the excess over the whole test (the system comes back within thousands of bytes), at the median over cases with an excess above 0.005 | 0.55 |
| K7 | The two prose streams help each other: the later of the two, at its own stage and going on, is below its single by at least 0.01 in at least four of five orders (ZPA scaled) | 0.75 |
| K8 | A stream of another kind read before does not harm: at its own stage, going on, no stream is above its single by more than 0.01 in any order (ZPA scaled) | 0.50 |
| K9 | The store adds up: cells at the end of a life are between 90% and 100% of the sum of the six singles' cells (both systems) | 0.70 |
| K10 | The first answering cell keeps better than the second with nothing changing: its excess is smaller in at least two thirds of the cases (ZPA scaled) | 0.65 |

## 5. Runs, size, cost, what is not run

* **Size.** Singles on raw bytes: as `nano/suite` (0.3 to 0.56 million cells for ZA; 1.8 to 4 million for ZPA scaled, perhaps more with 256 timing kinds). A life: the sum, expected 2.7 million cells for ZA and 15 to 25 million for ZPA scaled (3 to 6 GB of memory with a forked copy alive). No network, no gradient passed back.
* **Cost.** ZA: about 10 CPU-seconds a stream, a life with its 72 scorings under 5 CPU-minutes; seven lives and six singles under an hour. ZPA scaled: about 2 to 4 CPU-minutes a megabyte; a life with its scorings about one CPU-hour, run one at a time because of memory; six lives and six singles about 7 CPU-hours. If the development life shows that this is beyond the machine, the ZPA lives are cut to the first three reported orders and that is entered here as it is.
* **Not run.** Any remedy (a set of trusts for each kind of stream, chosen by how well it has foretold lately, is the first candidate if the loss is in the trusts; it is not built here). A bound on the store. Lives in which a stream returns more than once, or longer than six streams. Streams that are not text (the next plan). The systems with classes or made kinds (another program).

## 5a. Addition 1 (written after the development life of ZA and the six singles, before any reported order was run)

The development life (`out/ZA_dev.txt`, `res/life__ZA_dev.json`) already speaks against the hypothesis for ZA: going on, tiny Shakespeare's test costs 1.3434 at its own stage and 1.5035 at the end of the life; the book 1.4022 and 1.5591; Python 0.9626 and 1.0603; and the excess does not go within thousands of bytes (tiny Shakespeare, first 1,000 bytes of the test: 1.225 and 1.612; first 64,000: 1.324 and 1.519). Each later stream is also read worse than by a system of its own (the last tenth of the markup's body: 0.3531 against 0.3260). The reported orders are run as planned and the predictions are scored on them. Two things are added, so that the reported runs also say where the loss sits:

* **D3, going on from the end of the life** (in forked copies, for every stream but the last read): (a) with the trusts, the parts and their sums (uses, squared drivers) of the stream's own stage put back; (b) with the trusts and parts of the end, and only the sums of the own stage put back. (a) says how much of the going-on loss is in the trusts and how much in the counts; (b) says whether the rule of the step, whose steps only become smaller as the sums grow, is what keeps the system from coming back.
* **An arm with the constant step**: ZA0 = ZA with the step 0.00005 of the prose setting in place of the rule (`rule=0 eta=0.00005`), the same orders and singles. `nano/suite` found the rule lower on all six streams read alone; the question is whether that order holds in a life.

Added predictions (mine): **A1** D3(a) removes at least half of the going-on excess at the median over the cases with an excess above 0.02 (0.60). **A2** D3(b) removes at least a third of it at the median (0.50). **A3** ZA0 keeps better than ZA: its going-on excess at the end is smaller in at least two thirds of the cases (a stream not read last, a reported order) (0.60). **A4** but ZA0 is not below ZA at the end of the life going on in more than a third of those cases (the rule's better level outweighs what it loses) (0.50).

## 6. Outcome

(to be entered after the runs)
