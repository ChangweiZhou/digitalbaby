# nano/skip: a clock for the delay chain (20 November 2026)

**By** the session that holds this line of the outside work. **Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Status.** Development level throughout: one program, small texts, one development seed for choosing numbers and the five reported seeds (73129, 93683, 116089, 406364, 545353) where a table says so, fifteen seeds in one place. Nobody else has repeated any of it. No checker gate, seal or outside review is claimed. Every plan (`PLAN_1.md` to `PLAN_9.md`) was committed before its runs and carries its outcome and the score of every prediction, failures included. Files were changed only in this folder; what was copied from `loop_small_case` is named at the head of `skip.py`.

## 1. The question

The delay-chain system of `loop_small_case` learns two-digit sums `aa+bb=ccc` that it was never shown. With 0 to 8 random letters between the `+` and the second number the numbers are no longer at fixed delays, and neither that system nor the gated holding cells learn the sums (8 to 10% after 2,400,000 lines). The coordinator's idea: the chain need not step at every byte. If the letters never enter it, the text is the plain one again. Can a local rule, which names no byte and is not told which bytes are letters, find which bytes to pass over?

## 2. The system

`cells2.py`'s plain system: a line for each byte value at delay 0 (the present byte) and at nine chain delays; 256 logistic cells in between; one answering cell; the mismatch relayed back over return strengths; 55,570 numbers that learn with 18 byte values. **The one change:** when a byte arrives, the byte that was present either enters the chain or is passed over. The compiled kernel (`kern_s.c`) agrees with the numpy program to rounding on 23 cases (`check_skip_output.txt`).

## 3. What was found

Never-shown pairs, whole answer right, %, mean of the last three readings; 1,200,000 lines unless said; medians of the five reported seeds unless said. Where the letters are out of the chain the gaps of 0, 4 and 8 letters are the same lines to the chain and give one figure.

| Text | No clock (gaps 0 / 4 / 8) | Letters passed over by the program (a check) | Trial clock | Plan |
|---|---|---|---|---|
| `g2`: two-digit sums, 0 to 8 letters after the `+` | 52.2 / 1.3 / 0.7 | 93.7 (seed 1, 400,000 lines) | **96.7** (fifteen seeds: 93.4 to 97.4) | 2 |
| the same with 26 letter values | 45.6 / 1.6 / 1.0 | 94.1 (seed 1, 400,000) | **94.8** | 5 |
| `g3`: the letters anywhere before the `=` | 50.4 / 10.3 / 1.9 | 93.7 (seed 1, 400,000) | **95.9** | 6 |
| `t3`: three-digit sums, letters after the `+` | 13.3 / 0.1 / 0.1 (seed 1) | 91.8 | **94.3** | 8 |
| `m1`: one digit, sum modulo 10, 1,000,000 lines, sums shown | 98.8 / 94.2 / 10.8 | 100 | **100** | 4 |
| `d2`: no letters, 400,000 lines | 95.5 | | 95.3 | 4 |

Controls on `g2`: the same closing by chance with nothing learned 5.2 / 0.8 / 0.7; four byte values closed by chance 3.3 / 2.2 / 1.0.

1. **The sum of the sizes of the strengths held from a byte value's chain lines does not separate the letters** (`PLAN_1.md`; the coordinator's first proposal). In the chain without a clock the letters one and two bytes back are the most listened-to lines of all (5.7 and 3.9 times birth, digits 2.8 and 3.3), because they tell the cells where the line is. A strength that is of no use still wanders to about twice birth, and a fading strong enough to pull it back removes the learning first.
2. **A trial clock finds the letters** (`PLAN_2.md`). Each byte value has an entry to the chain. Every 256 bytes each entry is closed by chance, with its own tendency, or open. The answering cell sends out one number, the same to all entries: the size of its mismatch. An entry keeps the mean of that number over its closed periods and over its open ones and leans to whichever is lower, in units of its own uncertainty. It names no byte, reads no strength, and is told nothing about letters. Exactly the letters are closed, usually within the first 100,000 to 200,000 lines, and no digit at the end of any run.
3. **The evidence it uses is in the answer** (`PLAN_3.md`): at 30,000 lines the whole difference in mismatch between a chain with and without the letters lies in the answer's digits. An aligned chain starts learning the answer within 10,000 lines, which is why the rule is early.
4. **It holds where I expected it to struggle**: with 26 letter values to find (all closed by 400,000 to 500,000 lines; the time grew 2.5 times for 6.5 times the values, because the letters close in parallel), with the letters inside the numbers, on a text where one closing alone earns nothing (`m1`), and on three-digit numbers. With nothing to pass over it costs 0.2 points.
5. **The coordinator's second rule, which reads no mismatch, is not the same thing** (`PLAN_7.md`, `PLAN_8.md`). It passes over a byte that leaves unchanged what a table of counting cells expects of the next two bytes. On `g2` it reaches 95.5% (seed 1), at least as early as the trial clock. On three-digit numbers it passes over the first digit of every line and stays at 9.7%, where the trial clock is at 94.3%: it finds bytes that are locally without consequence, and a number's first digit is one. With a look-ahead of one byte the same happens already on two-digit numbers (8.1%).

6. **Where a clock by byte value cannot serve, entries for a byte in its context can, with one more local fact** (`PLAN_9.md`). Text `h2`: `aa+<0 to 4 fillers>bb=ccc`, a filler being a letter and a digit drawn by chance, so the filler digits are the same byte values as the digits that matter. 2,400,000 lines, never-shown pairs with 0 / 2 / 4 fillers, medians of five seeds: no clock 59.9 / 1.5 / 0.8; fillers passed over by the program 95.4; the trial rule with an entry for each byte value after each byte value (236 occur), each entry listening to the answering cell's number for 16 bytes after its own byte: 53.3 / 3.3 / 1.1; the same with each entry also knowing whether the byte before it entered the chain: **96.2 / 96.2 / 96.2** (each seed: 96.2, 97.5, 55.7 to 91.2, 54.5 to 59.9, 97.2). Without that knowledge closing half a filler is punished and the entries, each trying alone, mostly never close. With it every filler is closed as a whole on every seed. On two seeds of five an entry of the answer's first digit is closed by chance early and never reopens, which costs up to 40 points on one of them. The expectation rule would propose none of the filler bytes on this text (each changes the expectation; only the filler as a whole does not).

## 4. What is not shown, and the loose ends

- **Entries closed early stay closed.** An entry closed by chance before the cells have come to use its byte costs nothing to leave closed, and is not reopened (`m1` with entries born half closed; the answer's first digit on two seeds of `h2`). With hundreds of entries this is the rule's weakest point as it stands.
- The listening time of plan 9 (16 bytes) is fitted to these lines. Where the consequence of a byte comes later than an entry listens, the rule sees nothing.

- A clock for each byte value cannot pass over a byte in one place and keep the same byte in another; plan 9 takes the smallest such text, where one byte of context tells the two apart. Longer contexts, and fillers that cannot be told from the byte before, are untested.
- **The cost in locality, in one place.** An entry of the trial clock reads: its own state (closed or open) and its own two means; and one number that the answering cell sends out at every byte to all entries alike, the sum over its branches of its squared mismatch. In the period form (plans 2 to 8) all entries also draw their states at the same moments, every 256 bytes: a common slow clock. In the listening form (plan 9) there is no common period: an entry draws when its own byte comes and attends to the number for 16 bytes after. Nothing else crosses from cell to cell: no strength is read, no byte is named, no signal is computed for one entry in particular.
- The numbers set by hand (period 256, starting chance 0.2, floor 0.01, fading of the evidence 1/3,000, alpha 2) were chosen on `m1`, seed 1, and not retuned; moved one at a time they all still find the letters, up to about twice as late.
- **The loose end with `=`.** In 2 of the about 80 runs of the rule on these texts the `=` is closed, in both from the first reading to the last: `g2`, seed 1008 (96.4% of never-shown pairs, its arm's median being 95.4%) and `d2`, seed 116089 (96.4%, the same seed without a clock 95.5%). It cost nothing measurable in either. The reason it can happen: the `=` is still on its delay-0 line when it is the present byte, and the answer does not need its place in the chain, so its evidence is near nothing either way. On a text where such a byte matters in some lines and not in others this could cost, and that is untested. Entries born half closed close digits too (`m1`, seed 1): a digit closed half the time from birth is never used, so closing it costs nothing.
- The scores are taken with every entry in its likelier state and no drawing; the stream is read with the drawing throughout.
- Nothing here is on prose, and nothing has been compared with a reference taught by back-propagation.

## 5. Corrections of my own statements

- Before `PLAN_2.md` I expected the trial clock to be too late (chance 0.30 for the line's hypothesis). It was early.
- In the outcome of `PLAN_2.md`, B, I wrote that a chain without the letters must make the line's other bytes easier to name. `PLAN_3.md` showed it does not; the evidence is in the answer.
- Before `PLAN_5.md` I expected the time to grow in proportion to the number of byte values (chance 0.30 at 26). It did not.
- In `PLAN_9.md`, addition 1, I wrote that the trial rule "does not find two bytes that pay only together". The reported seeds show it does on two seeds of five, in part; it finds them unreliably.
- In `PLAN_7.md` I predicted that the one-byte expectation rule would pass over the first digit of the line. On the texts as read here it does not, because nothing is named after the end mark and the table has no count there; with the table counting across the end mark it does.

## 6. Predictions

Of 63 predictions written before their runs, 42 held and 21 failed: plan 1: 4 of 6; plan 2: 5 of 9; plan 3: 2 of 3; plan 4: 3 of 5; plan 5: 4 of 5; plan 6: 4 of 4; plan 7: 6 of 9; plan 8: 6 of 6 (after the rule's behaviour was known from plan 7); plan 9: 8 of 16; and the line's own hypothesis, which I had put at 0.30, held.

## 7. Commands and cost

    python3 check_skip.py                                                                  # compiled kernel against numpy
    python3 skip.py <label> text=g2 seed=73129 lines=1200000 every=100000 fast=1 clock=trial form=evid alpha=2 p0=0.2
    python3 skip.py <label> text=g2 ... clock=none | clock=oracle | clock=values:4 | clock=trial alpha=0 p0=0.2
    python3 expect.py run <label> text=t3 seed=73129 theta=0.2 two=1 across=1                # the expectation rule
    python3 ctx.py <label> seed=73129 clock=trial key=pair listen=16 bit=1                  # plan 9: entries for a byte after a byte
    python3 look_early.py 1                                                                # plan 3's measurement
    python3 table.py p2C evid,grad,none,chance,values4 73129,93683,116089,406364,545353    # tables from res/
    bash runC.sh ... runG.sh                                                               # the queues as run

A run of 1,200,000 lines takes 80 to 210 CPU-seconds; everything here together about six CPU-hours.
