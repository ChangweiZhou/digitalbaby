# Step 1 of the roadmap: the tissue reads the stream and stays informative (brief, revision 2; written from 00:35 UTC on 19 November 2026 by the clock of this machine)

The coordinator gives this brief its experiment number when it releases it. Until then a worker starts nothing. The brief has two stages in this one file. **Stage A** is a trial of under two CPU-hours on one seed (section 11). **Stage B** is the full runs (section 12); they begin only after the coordinator has read stage A's report and released stage B. The brief carries everything the worker needs; where it seems not to, the worker stops that part and asks (section 14). Every number in it that can be computed without running the tissue is computed by `check_step1_brief.py` beside this file (Python standard library only, `python3 -I`, about 25 seconds); its output is section 19.

Revision 2 takes in the check of revision 1 (`$C/CHECK_OF_TASK_STEP_1_BRIEF_20261119.md`, "the check"), the reviewer's decisions on what the check left open, and the roadmap's step 1 as changed at 00:31 UTC on 19 November. Section 21 says how each finding was settled.

## 0. Sources, shell names and host

- `C = PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117` on the reviewer's branch `V = origin/claude/vibrant-cerf-sje726`.
- The governing words are Mr. Xu's, in `$C/WHAT_A_NEURON_HAS_20261118.md`, sections 7 to 16.
- The roadmap is `$C/ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md` (draft 3, with the change of 00:31 UTC on 19 November). Step 1 and the conditions of its section 5 bind this brief.
- The tissue is `$C/THE_TISSUE_DEFINED_REVISION_5_20261119.md` ("the definition"), which folds in errata E1 to E10. Formulas are cited by their numbers, which revisions 4 and 5 share. Where this brief cites "sentence n", n is the number in revision 4, and the worker reads the sentence of revision 5 that carries the bracket "(r4 n)". Where the two revisions differ, revision 5 governs; the decisions of section 5.8 and the measurement of sections 7, 11 and 12 are this brief's. The program is this definition, with the decisions of section 5.8, and nothing else.
- The errata E1 to E9, risks and undecided items are those of `$C/FIFTH_READING_OF_THE_TISSUE_DEFINED_20261118.md` ("the fifth reading"). Erratum E10 is the reviewer's, from finding 1 of the check. Revision 5 of the definition carries E1 to E10 (its section 17); the table of section 5.2 records how each is applied here.
- The design block is that of `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md` ("the contract").
- The host of the worker is Linux on x86-64 with four cores and Python 3.13. The program uses Python 3.13 and numpy only (no torch); every receipt records both versions.

## 1. Terms

All terms of the definition (its section 0) are used in its sense. In addition:

- *Tissue*: one population of 2,048 cells born from one seed and run on one stream, never copied, never reset.
- *First-form tissue*: a tissue in which the strength of every connection between cells changes by formula 7 (the first form of choice C15).
- *Coincidence tissue*: a tissue in which the strengths of the connections into inner cells change by the coincidence rule of section 5.5 (the second form of C15, erratum E1); everything else is as in the first-form tissue.
- *Fixed tissue*: a tissue born from the same seed as the two above, in which no strength changes and no connection is given up or made (section 5.6). It is "a tissue whose strengths never change" of the roadmap.
- *Arm*: one of the two rules of strength (first form, coincidence). Each arm is run as one tissue per seed.
- *Formula 11 as written*: sentence 54 of the definition, in which a connection at least P_s steps old is given up when its strength is below 1/128 *or* its sending side's running average ū_ij is below 1/128. *Formula 11 with E10*: the same without the clause on ū_ij (section 5.2).
- *Position* t: the t-th byte of the stream, counted from 0. The window of byte t is steps 32t to 32t + 31.
- *Reading step of byte t*: step 32t + 31 (φ = 31).
- *Recorded output* of cell i at byte t: x_i at the reading step of byte t, after operation 4 of sentence 118, stored as float16 with every value below 0.01 stored as 0.
- *Lag* L: the byte at position t − L, for L = 0 (the present byte), 1, 2, 4 and 8.
- *Recording window*: a stretch of consecutive positions at which outputs are recorded. Stage A has two windows of 4,096 positions: A1 (positions 2,048 to 6,143) and A2 (12,288 to 16,383). Stage B has four windows of 8,192 positions: W0 (0 to 8,191), W1 (24,576 to 32,767), W2 (122,880 to 131,071) and W3 (253,952 to 262,143). W0 is *the start of life*; W3 is *late in the stream*.
- *Block*: 128 consecutive positions of a window, starting at the window's first position (32 blocks in a stage A window, 64 in a stage B window).
- *Distance of a cell at a step*: its distance from the input lines (definition, section 0) along the connections present at that step, where a cell is at distance 1 if it has at least one connection from the line of a byte value present in the window.
- *Distance d of a cell in a window*: the most frequent of its distances at the first step of each block of the window (ties to the smaller distance). The receipt records, for each cell, the share of blocks at that distance.
- *Silent*, *constant*, *score S*, *baseline B*, *countermodels C0 and C<L*: defined in section 7.
- *Comparator*: the network trained by back-propagation that the roadmap (section 5, condition 5) names as positive control. It has no part in this brief (section 3, item 4).

## 2. The question, and what reaching step 1 buys

**Question.** When the tissue of the definition reads, with no correction, a long stream of the first family's items one byte after another, do its cells one, two and four connections from the input lines stay neither silent nor constant, and can the present byte and the bytes one and two positions back be told from their recorded outputs late in the stream better than at the start of life, better than in a tissue whose strengths never change, and better than the present byte alone would allow? The bytes four and eight positions back are read and reported; holding over that span is asked at step 3 (roadmap, step 1 as changed on 19 November).

**What reaching it buys for the goal.** Every later step corrects cells from what they carry (roadmap, section 6, step 1: "the material every later correction works on"). Capability a (roadmap, section 3) is the dependency of all others: if cells away from the input lines fall silent or constant, or if learning by formula 7 removes from them what they carried of the bytes (risk H1), then no feedback at step 2 and no relay at step 3 has anything to work on. With erratum E10 in force, step 1 also shows which of the two rules of strength keeps that material; without it, the check showed that both arms would lose almost all their connections at byte 4 in the same way, and the rules could not be told apart (check, finding 1). Because the steps are phases of one life (roadmap, decision D5), the tissues of stage B are saved at their last step. Only the first form of step 2 can continue them: the first opening of step 2 (128 feedback cells, E2) and form (b) of C52 need tissues born with more feedback cells, which need step-1 phases of their own.

## 3. The design and report block of the contract

1. **Claim and level.** Level E0 (access): "Under the stated conditions, cells at distances 1, 2 and 4 from the input lines of the [first-form | coincidence] tissue are neither silent nor constant late in a stream of 262,144 bytes, and the bytes at lags 0, 1 and 2 can be told from their recorded outputs better than at the start of life, better than in the fixed tissue, and, at lags 1 and 2, better than the present byte alone allows." It is a mechanistic diagnostic, not a promotion test. No learned value, answer or association is tested, and none may be claimed.
2. **Exposure record.** The stream (section 4) shows, as bytes and without any feedback, the items of the taught half of the record's split: 3,960 pairs of each of the four operations, 15,840 pairs in all, each with its answer and end mark. In 262,144 bytes the stream begins 16,355 to 16,379 items (section 19, A.4), so every one of the 15,840 pairs is shown at least once and about 520 are shown twice. The other half, 15,844 pairs (3,961 per operation), never appears; the reserve of never-taught instances of step 5 is to be drawn from it (roadmap, section 5, condition 4). Any control tissue of step 5 (a tissue without the prior teaching) must have a step-1 phase with matched exposure, or the comparison is not matched. No feedback is given: the yes and no lines are 0 at every step (section 5.8, item 1). The tissue emits a byte or no byte at every position (sentence 69); the emissions are recorded and not scored. There is no first scored choice.
3. **Countermodels.** No answer is scored, so the exact-key table, the canonical-key table and the part and frequency shortcuts of the contract have nothing to score. What applies is the structure of the stream: the bytes at different lags are not independent (an "=" is followed by a digit; an end mark by a task byte). The countermodels are the same classifier fitted to codes of the stream instead of to cells: C0(L), a one-hot code of the present byte alone, and C<L(L), a one-hot code of the bytes at lags 0 to L − 1 (section 7.5). C0 lies 0.03 to 0.07 above chance at lags 1, 2 and 4 (section 19, part E), so a tissue that carries only the present byte could pass a rule that ignored it (check, finding 2). The rule of success therefore requires S ≥ C0(L) + 0.02 at lags 1 and 2 (section 7.6). The fixed tissue enters the rule as part of the baseline B.
4. **State and controls.** All state persists from birth to the end of the stream; nothing is reset (sentences 120, 121; roadmap conditions 1 and 2). The controls are second tissues born from the same seed, never copies (condition 3): the fixed tissue for both arms; the two arms are each other's comparison for the rule of strength. No tissue is known to stay informative under the step's conditions, so there is no positive control of the tissue; stage A supplies the nearest substitute, a form of the tissue that does keep cells informative, if one does. The positive control of the measure is the analysis applied to a code that holds the bytes at lags 0 to 8 exactly (check 9); it must score at least 0.99 at every lag. The comparator (condition 5) solves arithmetic answers, not the measure of this step, and is not run; its release, if the coordinator wants one for this stream, is the coordinator's. The information and the feedback are equal across the compared tissues: the same stream, no feedback.
5. **Rule of success, cost and stopping.** The cheap preflight before the costly run is stage A (section 11), after checks 1 to 10 (section 8). The rule of success is section 7.6; budget, stopping and the order of opening are sections 10 and 12; cost is section 9.

## 4. The stream

**The layout on record.** The first family's items are written by the record's function `render` of `exp11_cell.py` on the default branch. That source is not on this branch; its text is reproduced, "written again from exp11_cell.py and exp45.py", in `$C/check_e61_brief.py`, part 3 (lines 263 to 297 of that file), and as `render` "exp11_cell.render" in `$C/check_e51_brief.py`, line 19. Quoted from `$C/check_e61_brief.py`:

```
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EOS, BOS, BATCH = 256, 257, 64

def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt, list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]

def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]); rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]

def stream(tasks_pairs, seed, kmax):
    rng = Random(seed + 77)
    flat = [(t, pr) for t, pairs in tasks_pairs for pr in pairs]
    while True:
        order = list(range(len(flat))); rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            k = rng.randint(0, kmax)
            yield render(t, a, b, k, rng) + (k,)
```

with `apply` giving (a + b) mod 89, (a − b) mod 89, max(a, b) and min(a, b), and the record's stream `stream([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)`.

**The stream of step 1 (defined here; it differs from the record's in two points only).**

1. The record's end of an answer is the symbol 256 (`EOS`), which is not a byte. Here the end mark is **the byte 10** (line feed), placed where `EOS` was. It is a byte like the others: it enters on its input line and nothing else marks it.
2. The record begins every item, at the input of its network, with the start symbol 257 (`BOS`), and resets the network's state for each batch. Here there is no start symbol and no reset: the items' bytes are concatenated in the order in which the record's `stream` yields them, and the tissue reads them one after another.

Everything else is the record's, item for item: the split (half the pairs, `frac = 0.5`), the generator `Random(seed + 77)`, the order, the number k of letters (uniform on 0 to 12) and the letters. A stage B tissue reads the first **262,144 bytes** of the stream of its own seed; a stage A tissue reads the first 16,384 bytes of the stream of seed 73129 (the first 512 for the tissue of section 11.1, item 4). An item reads, for example, `w58<VSQQZ88=58\n`: the task byte (`p` add, `q` sub, `v` max, `w` min), the first operand in two digits, the sign (`+ - > <`), k capital letters, the second operand in two digits, `=`, the answer in two digits and the end mark. An item has 10 + k bytes (mean 16). The stream uses 46 byte values, and all 46 are present in every recording window of every seed (section 19, A.6b). Its digests for the three seeds are in section 19, A.4, and check 3 asserts them.

## 5. The tissue

### 5.1 By reference

The tissue is the first form of the definition: sentences 1 to 146, formulas 1 to 12 in the order of operations of sentence 118, the numbers of section 16, the first forms of the choices of section 14 (except C15 in the coincidence tissue), changed only by the errata of section 5.2 and decided where the definition leaves a choice to a programmer by section 5.8. The space, lines and drivers are sentence 6; kinds sentences 16 and 87; starting sources sentence 28 (with E4); formulas 1 to 12 sentences 43 to 55 (formula 11 with E10); the input end sentences 60 and 63; the output end and the emission sentences 69 and 72; the feedback route sentences 85 to 90 (with E4); the order of operations sentence 118; the start sentence 121.

### 5.2 Errata, and how this brief applies them

| No. | Change | Applied here |
|---|---|---|
| E1 | Fifth reading: move C15 from third place to first in the order of opening for step 1; run the first form beside a tissue with C15's second form at inner cells from the start; read each distance and each lag separately | The coincidence tissue is run from the start beside the first-form tissue (sections 5.5, 6). Every measure is read for d = 1, 2, 4 and for each lag separately (section 7). In the order of opening, C15 is first (section 12.3) |
| E2 | Sentence 93 reworded; 128 feedback cells first at step 2; step 2 reports output-end cells reached from yes-cells and no-cells | No feedback at step 1. The receipts give, at birth and at each window, the number of output-end cells taking a yes-cell and a no-cell as a source |
| E3 | Whenever C15 takes a form in which a rise in arrival strengthens active excitatory inputs, the kinds of feedback cells are reversed | **Not applied.** In the coincidence tissue the output-end cells keep formula 7, so the kinds of sentence 87 (no-cells excitatory, yes-cells inhibitory) are the corrective ones at the cells that emit, and are kept in all tissues. Whether inner cells that receive from feedback cells need the reversed kinds is for the brief of step 2 (check, finding 6) |
| E4 | A feedback cell has its feedback line only; formulas 7, 9 and 12 do not apply to it, and its threshold stays 0 | Applied: feedback cells draw no sources, never make a connection, keep θ = 0, and their strengths do not change. Formulas 1 to 6 and 10 are computed for them; formula 11 never removes the feedback line |
| E5 | Term "encoder cell": a cell with at least one input line in its neighbourhood; a feedback cell within reach of input lines may draw them | **Not applied** (the reviewer's decision on finding 9 of the check). Reason: by Mr. Xu's words in section 16 ("可以认为有专门的细胞收到反馈信息之后再传送给输入端"), dedicated cells receive the feedback and pass it on to the input end; E4 keeps those cells dedicated, and a feedback cell that drew input lines would carry the byte, which E4 removes (risk H3). The definition's term "encoder cell" stands. This is a decision on Mr. Xu's words, taken by the reviewer, and binds step 2 as one life |
| E6 | Sentences 50 and 139: the receiving cell's mismatch and activity are "taken to reach the synapse by the spread of potential (unsettled)"; new item in section 18 | A wording; no effect on the program. The report repeats the wording when it describes formula 7 |
| E7 | Sentence 91: "at the steps at which the feedback cell's output arrives; the net change over the window includes the background change" | A wording; no feedback at step 1 |
| E8 | Section 12: give a wider range for encoder cells at one connection from a feedback cell | The counts of this brief's draws are given: 27, 30 and 33 encoder cells for the three seeds (section 19, B.1) |
| E9 | The brief of step 1 states whether feedback cells are present with silent lines | Present, with silent lines (section 5.8, item 1) |
| E10 | The reviewer's, from finding 1 of the check. C17, first form: "A connection at least P_s steps old is given up when its strength is below 1/128." The test on the sending side's running average ū_ij becomes a further form of C17 | **Applied in every tissue of stage B and in the three main tissues of stage A.** Formula 11 as written is run only in the 512-byte tissue of stage A (section 11.1, item 4). Either end still decides whether a connection ends, because its strength is set by both ends (sentence 42). Consequences: a connection from an input line is never given up, since its strength stays 1/8 (sentence 63); a connection from a silent feedback cell keeps its strength, since its trace is 0 and formula 7 does not move it; a cell below its target activity keeps adding connections until its activity reaches a* or its neighbourhood is exhausted, about 89 makings or 355 bytes on average (section 19, F.5b), so the cost depends on the number reached (section 9) |

### 5.3 The birth, exactly (choices C30, C53, C54, C56, C3, C7, C51, and the undecided item 4)

All draws of the birth use one generator `rb = random.Random(seed)` of the Python standard library, in this order. Positions are pairs (x, y) of floats.

1. Coordinates: for i = 0, …, 2,047: x_i = `rb.random()`, then y_i = `rb.random()`.
2. Input lines: for v = 0, …, 255, the line of byte value v is at (0, `rb.random()`).
3. Output lines: for v = 0, …, 255, the line of byte value v is at (1, `rb.random()`).
4. Feedback cells: `FB = rb.sample(range(2048), 64)`. The first 32 indices of `FB` (in the order returned) receive the yes line; the last 32 receive the no line. This is done before the output-end cells are assigned (undecided item 4).
5. Kinds: for i = 0, …, 2,047, `u_i = rb.random()` is drawn for every cell.
6. Drivers of the output lines: for v = 0, …, 255 in turn, the cell that drives line v is, among the cells that are neither feedback cells nor already drivers, the one with the smallest (1 − x_i)² + (y_i − y_v)², where y_v is the y of output line v; ties go to the smaller index (sentence 6, C54). No draw is used.
7. Kinds: a cell is inhibitory if u_i < 1/4 and it is neither an output-end cell nor a feedback cell. Output-end cells are excitatory. No-cells are excitatory and yes-cells inhibitory (sentence 87), in every tissue. Lines have σ = +1.
8. Neighbourhoods (sentence 6, read as "within distance 1/8" = "at squared distance at most 1/64"): for every cell i that is not a feedback cell, its members are the cells j ≠ i with (x_j − x_i)² + (y_j − y_i)² ≤ 1/64, in increasing index, followed by the input lines v with x_i² + (y_i − y_v)² ≤ 1/64, in increasing v. A line is numbered 2,048 + v in the program. Feedback cells have no neighbourhood in the first form.
9. Starting sources: for i = 0, …, 2,047 in turn, skipping feedback cells, `src_i = rb.sample(members_i, min(8, len(members_i)))`. Every starting connection has strength 1/8 (number 6 of section 16), traces e = 0, running average ū = 0, and was made at step 0.
10. A cell is an *encoder cell* if it is not a feedback cell and its neighbourhood holds at least one input line (the definition's term); an *output-end cell* if it drives a line; an *inner cell* otherwise (feedback cells excluded). These kinds are fixed for life, since neighbourhoods are fixed.

All draws during life (only the sources of new connections, formula 12) use a second generator `rl = random.Random(seed + 500000)`. The birth is therefore identical, array for array, in every tissue of a seed.

### 5.4 One step, exactly

The order is sentence 118. All arithmetic is in float64. z_j(k) is the value of line j at step k, or x_j(k − 1) for a cell (with x_j(−1) = 0). Quantities named "before the step" are those held at the end of step k − 1.

1. φ = k mod 32. If φ = 0, the input line of the byte at position k // 32 is set to 1 and all other input lines to 0; they keep these values for the whole window. **Both feedback lines are 0 at every step of step 1** (section 5.8, item 1); the world compares nothing.
2. Every cell computes A_i(k) = Σ_j σ_j w_ij z_j(k) over its present sources (formula 1). A feedback cell's only source is its feedback line, with w = 1 and σ = +1.
3. m_i(k) = A_i(k) − p_i(k) (formula 5, with p from before the step).
4. v_i(k) = ½ v_i(k − 1) + ½ A_i(k); x_i(k) = min(1, max(0, v_i(k) − θ_i(k))) (formulas 2, 3).
5. Strength changes, from traces held before the step, x_i(k − 1) and ϑ_i(k), for every connection from a cell into a cell that is not a feedback cell (connections from lines are never changed, sentence 63): in the first-form tissue formula 7 at every such connection; in the coincidence tissue formula 7 at connections into encoder and output-end cells, and the coincidence rule of section 5.5 at connections into inner cells. Then formula 8 (bounds 0 and 1). In the fixed tissue nothing.
6. e_ij(k + 1) = (31/32) e_ij(k) + (1/32) z_j(k) (formula 6); p_i(k + 1) = (31/32) p_i(k) + (1/32) A_i(k) (formula 4); θ_i(k + 1) = max(0, θ_i(k) + (1/128)(x_i(k) − 1/8)) for every cell except feedback cells, whose θ stays 0 (formula 9, E4); ā_i(k + 1) = (127/128) ā_i(k) + (1/128) x_i(k) (formula 10). In the 512-byte tissue of stage A only, also ū_ij(k + 1) = (127/128) ū_ij(k) + (1/128) z_j(k), which revision 5 withdraws (its sentence 56) and which is kept there for formula 11 as written; no other tissue holds ū. In the coincidence tissue, inner cells also update ϑ_i (section 5.5).
7. Giving up, then making (formulas 11 and 12), except in the fixed tissue. Giving up (formula 11 with E10): every connection with k − k₀ ≥ 128 (k₀ its step of making; 0 at birth) and w_ij < 1/128 (after operation 5) is removed; a feedback cell's feedback line is never removed. In the 512-byte tissue of stage A only, formula 11 as written: the condition is w_ij < 1/128 or ū_ij < 1/128 (ū after operation 6). Making: then, for i = 0, …, 2,047 in increasing index, every cell that is not a feedback cell and has θ_i(k + 1) = 0, ā_i(k + 1) < 1/8 and k − (step of its last making, or 0) ≥ 128 makes one connection: the candidates are its neighbourhood members that are not its present sources, in the order of section 5.3, item 8, and the source is `rl.choice(candidates)`. If there is no candidate, nothing is made and the step of last making is not changed. The new connection has w = 1/8 (from a line or a cell alike, sentence 63 and formula 12), e = 0, ū = 0, k₀ = k. The cell's step of last making becomes k.
8. If φ = 31, the emission is read (sentence 72): the byte value of the most active output-end cell if its x exceeds 1/2, ties to the lowest value; otherwise none. It is recorded (section 7.1).

### 5.5 The coincidence rule (C15, second form; erratum E1; corrected by finding 5 of the check)

The definition names the form ("coincidence rule with a sliding threshold") and gives no formula. This brief fixes it. Both parts are the brief writer's choices.

- **From an excitatory cell into an inner cell:** Δw_ij = η · e_ij(k) · x_i(k − 1) · (x_i(k − 1) − ϑ_i(k)), with η = 1/8.
- **The sliding threshold:** ϑ_i(k + 1) = (1 − 1/4096) · ϑ_i(k) + (1/4096) · x_i(k)² / a*, with a* = 1/8 and ϑ_i(0) = a*, held by the receiving cell and updated in operation 6. The time constant is 4,096 steps (128 bytes, a power of two), a new number of this brief; ϑ starts at a* and not at 0 (sentence 121), so that it does not begin below every activity.
- **From an inhibitory cell into an inner cell:** Δw_ij = η · e_ij(k) · (x_i(k − 1) − a*): inhibition is strengthened when the receiving cell is above its target and weakened when it is below.

The sending side contributes its trace e_ij; the receiving side its activity at the previous step, and its own sliding threshold or the target a*. The rule is a function of the two ends of the connection (sentence 42). For an excitatory input it is the rule of Bienenstock, Cooper and Munro (1982), whose threshold averages over many stimuli (here over 128 bytes, against one byte in revision 1, which the check showed made the rule strengthen every active input to 1). For an inhibitory input it is the rule of Vogels and colleagues (2011), which changes an inhibitory strength in proportion to presynaptic activity times the postsynaptic rate minus a target rate. Its living counterparts are NMDA-receptor-dependent potentiation and depression with a threshold that slides with the cell's recent activity (textbook; the mechanism of the sliding is *unsettled*), and homeostatic inhibitory plasticity.

The coincidence rule does not use the cell's prediction. In the coincidence tissue the inner cells therefore do not do what sentence 36 ([S], section 8: "predicting the information newly arriving from other cells") describes for every cell; the definition admits the rule as a further form of C15, and a favourable result for this arm would have to be reconciled with section 8 before it is carried forward. The report says so.

### 5.6 The fixed tissue

Born as in section 5.3 from the same seed; it reads the same stream. Operations 1 to 4, 6 (formula 9 included) and 8 are carried out; operation 5 and operation 7 are not, so that every strength and every connection stays as at birth. The threshold control is kept because it is the cell's regulation of its own activity, not a change of a connection; without it the fixed tissue would differ from the other two in its activity as well as in learning. A lifelong block of all synaptic change has no exact living counterpart; the fixed tissue is a model control that acts on no learner. One fixed tissue per seed serves both arms: the arms differ only in operation 5.

### 5.7 The numbers of the first form, with their values

| No. | Number | Value | Choice | Source |
|---|---|---|---|---|
| 1 | Cells | 2,048 | C23 | definition, section 16 |
| 2 | Feedback cells | 64 (32 yes, 32 no) | C56 | section 16 |
| 3 | Radius of a neighbourhood | 1/8 | C30 | section 16 |
| 4 | Share of inhibitory cells | 1/4 | C3 | section 16 |
| 5 | Starting number of connections | 8 (then the cell's own) | C6 | section 16 |
| 6 | Starting strength | 1/8 | C5, C18 | section 16 |
| 7 | τ_v | 2 steps | C11 | section 16 |
| 8 | τ (prediction, traces) | 32 steps | C13, C16 | section 16 |
| 9 | η | 1/8 (both rules) | C15 | section 16; section 5.5 |
| 10 | η_θ | 1/128 | C29 | section 16 |
| 11 | a* | 1/8 (also the target of ϑ and of the inhibitory coincidence rule) | C29 | section 16; section 5.5 |
| 12 | P_s | 128 steps | C17, C18 | section 16 |
| 13 | Giving-up level | 1/128 (on the strength only, E10) | C17 | section 16; E10 |
| 14 | Steps per byte | 32 | C24 | section 16 |
| 15 | Emission level | 1/2 | C33 | section 16 |
| 16 | Feedback cells in form (b) | none at step 1 (section 5.8, item 2) | C52 | section 16 |
| — | Time constant of ϑ (coincidence tissue only) | 4,096 steps; ϑ(0) = a* | C15, second form | this brief, section 5.5 |

Values fixed by decision, convention or unit (definition, section 16): 256 input lines and 256 output lines; two feedback lines; the unit square with the faces x = 0 and x = 1; feedback-line strength 1; maximum strength, activity, slope and line value 1; half yes and half no; starting values 0 (except ϑ); the seed.

Numbers of this brief that belong to the experimenter's reading and not to the tissue: the stream's length (262,144 bytes in stage B, 16,384 in stage A); the windows (four of 8,192 positions in stage B, two of 4,096 in stage A); blocks of 128 positions; the lags 0, 1, 2, 4, 8; the distances 1, 2, 4; the level 0.01 (silent, constant, and the floor below which a recorded output is stored as 0) and the share 1/128 of the definition of silent; the share 1/2 of cells; a sample of at most 128 cells at a distance; four folds and a gap of 16 positions; the margin 0.02, the level 0.95 and the allowance 0.01 of the rule of success; the level 0.1146 (the score of the code of the pair (byte one back, position within the item), plus 0.02) of the outcomes of stage A. The life generator's offset 500,000 and the sampling generator's offset 900,000 are conventions.

The form of every choice used: C1 one part; C2 no gate; C3 1/4 inhibitory, output-end cells excitatory, feedback cells by sentence 87; C5 graded 0 to 1; C6 8, then the cell's own; C7 8 at random within the neighbourhood; C8 encoder cells receive from cells; C10 formula 3; C11 τ_v = 2; C12 one prediction; C13 running average, τ = 32; C14 signed difference; C15 formula 7 (first-form tissue) or the coincidence rule of section 5.5 at inner cells (coincidence tissue); C16 exponential, τ = 32; C17 age ≥ P_s and w < 1/128 (E10; formula 11 as written in the 512-byte tissue of stage A); C18 θ = 0, ā < a*, P_s since the last making or the start, random source, w = 1/8; C19 strength by both ends, either end ends it, the receiving side starts it; C20 passages formed by traces; C21 one line for each value; C22 encoder cells predict; C23 2,048; C24 32, all at once; C25 all 0; C26 to C28 no chemical signal; C29 formula 9 at all cells except feedback cells (E4); C30 unit square, radius 1/8; C31 one component; C32 256 lines, active for the whole window; C33 most active output-end cell above 1/2, ties lowest; C34 every position at φ = 31; C37 yes and no lines, here silent; C38 no emission counts as "no" (no feedback is given at step 1); C39 φ = 0, one step (unused); C40 through feedback cells only (unused); C41 m_i · x_i(k − 1); C42 no relay by activity; C43 as the neighbourhood gives; C44 not opened; C45 bytes only; C46 only by the cells' own changes; C47 the grid; C48 no electrical synapses; C49 sentence 82; C50 unused; C51 one recorded seed per tissue (two generators, section 5.3); C52 form (a); C53 random points on x = 0 and x = 1; C54 nearest eligible cell, ties by index; C55 the reading step; C56 64 feedback cells with the feedback line only (E4), kinds by sentence 87.

### 5.8 Decisions on the items still undecided for a programmer (fifth reading, section 6) and on others found while writing this brief

1. **Feedback cells at step 1 (item 1; E9).** The 64 feedback cells are present from birth, with both feedback lines 0 at every step. Reason: the roadmap makes the steps phases of one life (D5), and cells added by an agent that knows when the task changes do not exist (`WHAT_A_NEURON_HAS`, item 4.8); a tissue without feedback cells could not go on to step 2. With E10 the connections from feedback cells keep their starting strength through step 1 (a silent sender gives a trace of 0, so formula 7 does not change them), and the 423 to 465 that exist at birth remain, together with any made later.
2. **Form (b): emission lines after giving up (item 2).** Step 1 has no form (b) cells. With E10 an emission-line connection, whose strength is fixed, is never given up, so the question does not arise. A form (b) tissue must have its 64 further feedback cells from birth, so it needs a step-1 phase of its own; whether step 2 runs one is for the brief of step 2.
3. **The kind of a magnitude feedback cell (item 3).** Unused at steps 1 and 2 (decision D3). Revision 5 (its sentence 113) makes it excitatory; by the reasoning of its sentence 89, a line that is larger when the answer is nearer plays the part of "yes" and would need an inhibitory cell under formula 7. The brief of the step that first uses it settles this with the reviewer.
4. **How the feedback cells are chosen (item 4).** At random, by `rb.sample`, after the coordinates and lines are drawn and before the output-end cells are assigned; the first 32 drawn are yes-cells (section 5.3, item 4).
5. **E4 against E5.** Decided by the reviewer: E4 holds and E5 is not applied (section 5.2).
6. **"Within distance 1/8"** is read as squared distance at most 1/64.
7. **Ages and the first making.** Starting connections count as made at step 0, so the first giving up and the first making are possible at step 128, byte 4 of life (sentence 57).
8. **Order inside operation 7.** All giving up first, then the makings in increasing index of the cell, each drawing from the members that are not its sources after the giving up; a cell with no candidate makes nothing and keeps its step of last making.
9. **Line connections.** With E10 a connection from an input line is never given up (its strength stays 1/8). Encoder cells may draw lines of byte values that never occur (210 of the 256); such connections carry nothing and remain.
10. **Emission at step 1.** The emission is read and recorded at every position (sentence 69); nothing is compared and no feedback line is set.
11. **Recording.** At the reading step (C55), after operation 4; values below 0.01 are stored as 0 (check, finding 4).

## 6. The arms, the seeds and the tissues

The seeds are the project's three: 73129, 93683 and 116089. From a seed are drawn, by the generator `Random(seed)` (section 5.3): the coordinates, the positions of the lines, the feedback cells and their lines, the kinds and the starting sources; by `Random(seed + 500000)`: the sources of new connections; by `Random(seed + 900000)`: the samples of cells for the classifier (section 7.3); and, by the record's generators `Random(seed * 1000 + 50 + 89 · 7 + code)` and `Random(seed + 77)`, the split and the stream (section 4). All tissues of one seed have the same birth and read the same stream.

| Tissue | Rule of strength | Formula 11 | Command code |
|---|---|---|---|
| First-form tissue | Formula 7 at all connections between cells | with E10 | `first` |
| Coincidence tissue | Formula 7 into encoder and output-end cells; section 5.5 into inner cells | with E10 | `coincidence` |
| Fixed tissue | None; no giving up, no making | not applied | `fixed` |
| First-form tissue, formula 11 as written (stage A only, 512 bytes) | Formula 7 | as written | `first --formula11-as-written` |

Stage A: the first three on seed 73129 for 16,384 bytes, and the fourth for 512 bytes. Stage B round 1: the first three on each of the three seeds for 262,144 bytes, nine tissues.

## 7. What is recorded and how it is read

The tissue is not touched by any measurement: outputs are recorded, and the analysis below is the experimenter's, carried out on the records after the window. No part of it acts on the tissue (`WHAT_A_NEURON_HAS`, item 4.9).

### 7.1 Records

For each tissue:

1. At the first step of each block of each recording window: the distance of every cell (breadth-first along the present connections, from the lines of the byte values present in the window). From these, at the end of the window, the distance d of every cell in the window and the share of blocks at that distance.
2. At the reading step of each position of the window: the recorded output of **every** cell (float16, values below 0.01 stored as 0), the position, and the present byte. Recording all cells is required because the cells at a distance are known only at the end of the window.
3. For every cell, over each window: the number of positions with recorded output above 0.01, the mean and the standard deviation (population) of the recorded output, the mean θ.
4. The connections, at the steps listed for each stage (sections 11.2 and 12.1): the number by kind of receiving cell (encoder, inner, output-end) and kind of source (input line of a byte value in use, input line of a value not in use, excitatory cell, inhibitory cell, feedback cell), with their mean strength; the number of starting connections remaining; the number given up and made since the previous count; the mean, largest and smallest number of connections of a cell; the number of cells whose neighbourhood is exhausted; the number of distinct byte values among the input lines that have at least one connection; the output-end cells taking a yes-cell and a no-cell as a source (E2).
5. At every position of the stream: the emission (int16, −1 for none).
6. At the end: the full state (section 13.3).

### 7.2 Silent and constant

In a window, a cell is **silent** if its recorded output exceeds 0.01 at fewer than 1/128 of the window's positions (fewer than 32 of 4,096 in stage A, 64 of 8,192 in stage B). It is **constant** if it is not silent and the standard deviation of its recorded output over the window is below 0.01 (the mark used in `CELLS_IN_USE_IN_LIMITED_NETWORKS_20261117.md`). For each d in {1, 2, 4} the measure is the share of the cells at distance d in that window (all of them, not a sample) that are neither silent nor constant; the report gives the number of such cells beside every share.

### 7.3 The classifier

For one tissue, one window, one distance d and one lag L:

- *Cells*: a sample of 128 of the cells at distance d in the window (all of them if fewer), drawn by `Random(seed + 900000).sample(sorted(cells), 128)`, a new generator for each (window, d); the same sample serves all lags. The number of cells sampled is reported.
- *Samples*: the positions t of the window with t − L ≥ 0. The features of t are the recorded outputs at t of the sampled cells, as float64, with every value below 0.01 set to 0 (as stored). The label of t is the byte at position t − L, taken from the stream.
- *Folds*: the window is cut into four consecutive stretches of equal length. For each stretch in turn, it is the test set, and the fitting set is every other sample of the window except those within 16 positions of the test stretch (positions from its start − 16 to its end + 15 are left out of the fitting set).
- *Fitting*: on the fitting set, the mean and the population standard deviation of each feature; features with a standard deviation below 10⁻⁹ are dropped; each sample is standardised. For each byte value that occurs among the fitting labels, its centroid is the mean of the standardised fitting samples with that label.
- *Test*: each test sample, standardised with the fitting set's means and deviations, is assigned the byte value of the nearest centroid in squared Euclidean distance; ties go to the smaller byte value. A test label that has no centroid counts as wrong.
- *Score S(d, L, window)*: the predictions of the four folds are pooled; for each byte value present among the test labels, the share of its samples assigned rightly; S is the mean of these shares over the byte values present (balanced accuracy). Its standard error is the standard deviation of the four folds' balanced accuracies divided by 2.

**Chance.** A classifier whose assignments do not depend on the label has an expected balanced accuracy of exactly 1/K, where K is the number of byte values present; K = 46 in every window of every seed (section 19, A.6b; the smallest class has 88 positions in a stage B window and is printed for every window), so chance is 1/46 = 0.0217. The majority rate (always the commonest byte, "=" or the end mark) is 0.0616 to 0.0634 in plain accuracy and also 1/46 in balanced accuracy.

### 7.4 The fixed tissue

The same records and the same classifier give S_fixed(d, L, window) and the shares of section 7.2 for the fixed tissue of the same seed, with its own distances (which do not change) and its own sample.

### 7.5 The countermodels of the stream

- **C0(L)**: the classifier of section 7.3 fitted, in the same window and folds, to a one-hot code of the present byte (46 features, one per byte value of the window) in place of the cells' outputs. Its values are in section 19, part E: 1.0000 at L = 0; in stage B's W0 and W3, 0.074 to 0.087 at L = 1, 0.050 to 0.059 at L = 2, 0.076 to 0.080 at L = 4 and 0.034 to 0.043 at L = 8; in stage A's A1 and A2, 0.070 to 0.086, 0.050 to 0.056, 0.070 to 0.073 and 0.034 to 0.035.
- **C<L(L)**, for L ≥ 1: the same with a one-hot code of the bytes at lags 0 to L − 1 (46 · L features). It is what a population would give that held every nearer byte exactly and nothing else. The worker computes it in every window (check 9 (d)); it is not computed here.

### 7.6 The rule of success of stage B (fixed before any run)

The baseline is **B(d, L) = max(S(d, L, W0), S_fixed(d, L, W3))**: the plastic tissue must beat both its own early life and its birth wiring read in the same late window.

For one arm and one seed, step 1 is **reached** when all of the following hold:

1. For each d in {1, 2, 4}: in W3, the share of cells at distance d that are neither silent nor constant is at least 1/2.
2. For each d in {1, 2, 4}, at L = 0: S(d, 0, W3) − B(d, 0) ≥ 0.02, or S(d, 0, W3) ≥ 0.95 and S(d, 0, W3) ≥ B(d, 0) − 0.01.
3. For each d in {1, 2, 4} and each L in {1, 2}: S(d, L, W3) − B(d, L) ≥ 0.02 and S(d, L, W3) ≥ C0(L, W3) + 0.02.

There are 12 conditions per arm and seed (3 of condition 1, 3 of condition 2, 6 of condition 3). The second clause of condition 2 is for scores that are at their ceiling (as the present byte at d = 1 is expected to be); "better" cannot be shown there, and not falling is required instead. The margin 0.02 is about three standard errors of a difference of two scores of this size (the standard error of one score is about 0.005 when 46 classes have about 100 to 500 test samples each).

**Step 1 is reached by an arm** when it is reached on all three seeds. **Step 1 is reached** when at least one arm reaches it. The report counts, per arm and seed, how many of the 12 conditions hold.

### 7.7 Lags of four and eight bytes, and readings beside the rule

For L = 4 and L = 8 the report gives, for each arm, seed and d, the values of condition 3 (S − B and S − C0) and states "the byte L back is told beyond the baseline and the present byte" when both are at least 0.02, otherwise "not told beyond the baseline and the present byte". They are not part of the rule: holding a byte over that span is capability e, which the roadmap asks at step 3 (check, finding 7).

For each arm, seed, d and L the report also states, in these words:

- "Learning added to what the wiring gives" when S(d, L, W3) − S_fixed(d, L, W3) ≥ 0.02; "learning removed from what the wiring gives" when it is ≤ −0.02; otherwise "no difference from the fixed tissue was found".
- For L ≥ 1: "beyond every nearer byte" when S(d, L, W3) ≥ C<L(L) + 0.02, otherwise "not beyond what the nearer bytes give".

## 8. Checks before any run

Each check writes `TISSUE_STEP1/qual/check<n>.json` with an entry `pass`. If a check fails, the worker stops, reports the first difference and asks. An interrupted check is made again from its beginning. Checks 1 to 10 are made once, before stage A; stage B uses the same program and does not repeat them unless the program has changed.

1. **The worked example.** The program's own functions for formulas 1 to 12 and sentence 118, applied to the four cells F, E, I and O of section 13 of the definition with the starting values given there (outside sources held at the stated outputs, u at 0.5 for steps 0 to 15 and 0.125 from step 16, the no line at 1 at step 0 only, ages 0), reproduce, to four decimals, (a) every row of the table of section 13 and the final strengths O: u 0.75099, f 0.24949, F 0.25014, I: E 0.5003, q 0.2510, g 0.2495, with F as revision 4 writes it; (b) the rows and final strengths of section 19, C.2, with F as E4 makes it (feedback line only, θ = 0, no formula 7 or 9 at F); (c) those of section 19, C.3, as the coincidence tissue makes it (E4; F excitatory, since E3 is not applied; the coincidence rule of section 5.5 at the inner cell I, ϑ_I starting at a*); (d) every row of the table of section 13, part 1, of revision 5 and its final strengths (O: u 0.7504, f 0.2498, F 0.2502; E: h 0.2554, c 0.4973; I: E 0.5003, q 0.2510, g 0.2495), with F as revision 5 writes it (line only, v = 0, θ = 0); (e) for the whole first-form tissue of seed 73129 with E10 over its first 1,024 steps, the values of section 13, part 2, of revision 5 for cells 497 and 333 at steps 0, 32, 64, 96, 127, 128, 129 and 160, and the counts of its sentences 150 to 152 and of its section 12 (no connection given up; 1,984 connections made at step 128; 13,875 made by step 1,024; smallest starting strength between cells 0.0455). Variant (a) catches a wrong order of operations (traces or p updated before formula 7 is applied; x(k) in place of x(k − 1)); variant (c) catches a wrong sign of either part of the coincidence rule (I's inhibitory input g falls from 0.25 to 0.21127 because I is below its target).
2. **The births.** For each seed, the birth of section 5.3 gives the digests and counts of section 19, B.1: the digest of the coordinates (`sha256(json.dumps(P))[:16]` of the list of [x, y] pairs), of the sources (`sha256(json.dumps([sorted(src_i) for i in range(2048)]))[:16]`, lines numbered 2,048 + v), and of `[FB, drivers, inhibitory flags]`; the numbers of cells of each kind; the neighbourhood sizes and their total; the 15,872 starting connections and those from lines; and the counts of cells at each distance from 1 to 11 and not reached (64, the feedback cells). The births of the tissues of a seed are equal in every array.
3. **The stream.** For each seed, the SHA-256 of the first 262,144 bytes equals that of section 19, A.4; its first 48 bytes are those printed there; it holds 46 byte values, all present in each window of sections 1 and 7; every item parses as task byte, two digits, sign, 0 to 12 capitals, two digits, "=", two digits and the byte 10, and its answer is right; every pair is in the taught half of the split.
4. **The sign of the rules.** The program's rule functions give the values of section 19, part D, to six decimals: formula 7 gives −0.000750 for σ = +1, m = +0.1, x(k − 1) = 0.3, e = 0.2, and the three other signs as printed; the excitatory coincidence rule gives +0.001500 for ϑ = 0.1 and −0.001500 for ϑ = 0.5; the inhibitory coincidence rule gives +0.004375 for x(k − 1) = 0.3 and −0.001875 for 0.05; every rule gives 0 for e = 0.
5. **No reset, and nothing carried that should not be.** Two first-form tissues are born from seed 73129 and run on two streams that differ only in the byte at position 5, which is "S" and is replaced by the next byte value in use in increasing order of value, "T" (section 19, A.10). At the reading step of position 40, at least one cell's v differs between them by more than 10⁻¹²; so the state carries information across item boundaries. In addition the report states that the program contains no assignment that sets a cell's or a connection's state after birth other than by the formulas.
6. **Silent feedback, E4 and E10.** Over the first 2,048 bytes of a first-form tissue (seed 73129): both feedback lines are 0 at every step; every feedback cell has v = x = θ = 0 and exactly one source (its line); no feedback cell ever makes a connection; no connection from a line changes strength; no connection from a line or from a feedback cell is ever given up; every connection given up had strength below 1/128.
7. **The fixed tissue.** After 2,048 bytes of the fixed tissue (seed 73129), the list of connections and every strength are bit-equal to the birth.
8. **Saving and resuming.** A first-form tissue on seed 73129 run for 4,096 bytes without interruption, and the same run with `--ckpt-bytes 1024 --die-after-byte 1700` (exit code 9) followed by the same command with `--resume` in place of `--die-after-byte 1700`, give bit-equal final states (every array, both generators' states, the stream position) and equal records.
9. **The analysis.** (a) Applied to a one-hot code of the present byte in A1 and A2 of seed 73129 and in W0 and W3 of each seed, the analysis program gives C0(L) equal to section 19, part E, to four decimals. (b) Positive control: applied to a one-hot code of the bytes at lags 0 to 8 together (414 features) it gives S ≥ 0.99 at every lag. (c) Negative control: applied to 128 features drawn from `numpy.random.default_rng(1)` as the absolute values of standard normal numbers, independent of the stream, with values below 0.01 set to 0, it gives S within 0.0217 ± 0.01 at every lag. (d) It computes C<L(L) for L = 1, 2, 4, 8 in the same windows and writes the values into the check's receipt. (e) In A2 of seed 73129 it scores the present byte from a one-hot code of the pair (byte one back, number of bytes since the last byte 10 before it, 0 for the byte that follows it) and gives 0.0946 to four decimals; the check script prints the same value.
10. **Cost.** Seconds of CPU per byte for each of the three main tissues on seed 73129 over bytes 256 to 1,279 (the first 256 bytes not counted; these bytes lie after the first giving up and making at byte 4, so the cost of making is included), with the mean number of connections of a cell at the end. The forecast for stage A is 16,384 bytes times the measured cost per byte of each tissue, times min(2, 93.8 / c) for the two plastic tissues, where c is the mean number of connections of a cell at the end of the timed bytes (under E10 connections grow at most to 93.8 a cell; section 9), plus the 512-byte tissue. If the forecast exceeds 2.5 CPU-hours, the worker asks before starting stage A. The check passes if every timing is finite and checks 1 to 9 passed.

## 9. Size and cost

**Size.** 2,048 cells, 32 steps a byte, 15,872 starting connections (7.75 a cell on average; feedback cells have none), neighbourhoods of 26 to 166 members (mean about 96 over the cells that have one). Under E10 a cell's number of connections can grow to at most its neighbourhood's size: 189,118 to 192,179 connections in all, 92.3 to 93.8 a cell over the 2,048 (section 19, B.1). A cell that stays below its target reaches that in about 355 bytes (F.5b). A stage B stream is 262,144 bytes, 8,388,608 steps, 65,536 periods of P_s; a stage A stream is 16,384 bytes, 524,288 steps.

**How the cost was estimated.** The program holds the connections as flat arrays (one entry per connection: sending index, receiving index, w, e, ū, step of making) in float64 and int64, computes arrivals with `numpy.bincount(receiver, weights=σ[sender] · w · z[sender], minlength=2048)`, and removes and appends connections only at steps when one is given up or made. An operation here is one arithmetic operation or comparison on one element. Counted from formulas 1 to 12: about 20 operations a connection a step in a plastic tissue (arrival 3, formula 7 or the coincidence rule 5, bounds 2, trace 3, ū 3, giving-up test 3, gathering 1) and about 4 in the fixed tissue; about 25 operations a cell a step. The time is taken as about 80 numpy calls a step at about 3 microseconds each, plus about 1.5 nanoseconds an operation (40 calls in the fixed tissue). The making of connections is done in Python: about 16 cells a step are eligible while cells grow (2,048 cells, one each per 128 steps), at about 30 microseconds a making (building the list of candidates from about 96 members, then `rl.choice`), adding up to about 0.5 ms a step (check, finding 12). These rates are the brief writer's general knowledge of numpy and Python on one core; they were not measured here (numpy is not used by the check script), and check 10 and stage A measure the actual cost.

| Mean connections a cell | Operations a byte (plastic) | Time a byte (plastic; with 16 makings a step) | CPU-hours a plastic tissue, stage B | CPU-hours a plastic tissue, stage A | Time a byte (fixed) |
|---|---|---|---|---|---|
| 8 (birth) | 12.1 million | 26 ms (41) | 1.9 (3.0) | 0.12 (0.19) | 9 ms |
| 12 | 17.4 million | 34 ms (49) | 2.5 (3.6) | 0.15 (0.22) | 11 ms |
| 32 | 43.6 million | 73 ms (88) | 5.3 (6.4) | 0.33 (0.40) | 19 ms |
| 93.8 (every neighbourhood full) | 124.6 million | 195 ms (210) | 14.2 (15.3) | 0.89 (0.96) | 43 ms |

**Stage A**: about 0.7 CPU-hours with the checks if cells hold near 12 connections; 2.1 CPU-hours if every neighbourhood filled at once (an upper bound, section 19, G.2). **Stage B round 1** (six plastic and three fixed tissues): 23.5 CPU-hours at 12 connections with makings throughout, 40.7 at 32, and 87.0 if every neighbourhood is full (G.3). **The ceiling of round 1 stays 40 CPU-hours.** Under E10 the number of connections that cells reach is not known in advance; stage A measures it and the cost per byte at the end of its stream, when growth has settled. The forecast of stage B is made from stage A's cost per byte over its last 4,096 bytes (section 11.4). If that forecast exceeds 40 CPU-hours, the coordinator decides, at the release of stage B, on a higher ceiling or a shorter stream; the worker does not decide.

**Files saved, and their size.** Free disk on the worker is small; the worker checks it before each stage (`df`) and asks if less than 2 GB is free.

| What | Size |
|---|---|
| Records of all 2,048 cells (float16, `numpy.savez_compressed`, one file a window) | stage A: 16.8 MB a window, 101 MB for its three recorded tissues before compression; stage B: 33.6 MB a window; W0 and W3 kept, 67 MB a tissue, 604 MB for round 1 before compression; W1 and W2 deleted once their measures are in the receipt |
| Per-cell summaries and emissions | about 7 MB for stage B round 1 |
| Final state and saved state of a tissue | 1.3 MB at 12 connections a cell, 8.7 MB at 93.8 |
| Receipts and logs | under 1 MB a tissue |

If the compressed records of stage B round 1 exceed 400 MB, only the measures and the records of W3 are committed, and the W0 records are deleted after their measures are in the receipts; the report says so.

## 10. Budget and rule for stopping

- Stage A: checks 1 to 10 and four tissues, expected 0.7 CPU-hours, at most about 2.1; the worker asks if a forecast exceeds 2.5 CPU-hours (check 10, and again at byte 4,096 of each plastic tissue from its own rate).
- Stage B: round 1, at most 40 CPU-hours (or the ceiling the coordinator sets at its release); opening 2 (section 12.3), if run, at most 20 CPU-hours. Nothing else is run under this brief.
- Any check fails: stop and ask.
- A tissue is never stopped early because of its measures; every tissue reads its full stream.

## 11. Stage A: the trial

### 11.1 What is run

Seed 73129 only, after checks 1 to 10:

1. The first-form tissue with E10, 16,384 bytes.
2. The coincidence tissue with E10 and the rule of section 5.5, 16,384 bytes.
3. The fixed tissue, 16,384 bytes.
4. A first-form tissue with formula 11 as written, 512 bytes, to record the loss of connections at step 128 that the check derives (finding 1: at least 15,630 of the 15,872 starting connections of seed 73129 given up; of the 754 from input lines, only 11 can be kept, those of the bytes "w", "5", "8" and "<" read in positions 0 to 3; section 19, B.1).

The three tissues of items 1 to 3 run at the same time; item 4 runs after them or beside them, within three processes.

### 11.2 What is read (the check's task 6)

1. **Connections** (section 7.1, item 4): for tissue 4, after operation 7 of steps 127 and 129, and at the end of bytes 128, 256, 384 and 511; for tissues 1 to 3, after operation 7 of steps 127, 129 and 1,024, and at the end of every 128th byte (bytes 127, 255, …, 16,383).
2. **Activity**: for tissues 1 to 3, every 1,024 bytes (16 stretches of 1,024 positions, each with 8 blocks of 128), the distances by the mode of section 1 within the stretch, and, for the cells at d = 1, 2 and 4, the shares of section 7.2 (a stretch of 1,024 positions; silent means above 0.01 at fewer than 8 of them) and the mean and standard deviation of their recorded outputs. Outputs of all cells are recorded at every position for this purpose; only windows A1 and A2 are kept on disk.
3. **Scores**: S(d, L) for d = 1, 2, 4 and L = 0, 1, 2, 4, 8, by section 7.3, in window A1 (bytes 2,048 to 6,143) and window A2 (bytes 12,288 to 16,383), for tissues 1 to 3, beside C0(L) and C<L(L) of the same windows.
4. **Cost**: CPU seconds per byte of each tissue over each stretch of 1,024 bytes, with the mean number of connections of a cell.
5. **Emissions**: the share of positions with a byte emitted, per tissue and stretch.

### 11.3 The three outcomes, fixed before the trial, and what each leads to

A plastic tissue's cells at a distance d are said to **stay informative** in A2 when the share of cells at distance d that are neither silent nor constant is at least 1/2 and S(d, 0, A2) ≥ 0.1146: the score for the present byte of a one-hot code of the pair (byte one back, position of the byte since the last byte 10 before it), which holds the stream's structure but not the present byte (0.0946 in A2 of seed 73129; check 9 (e)), plus the margin 0.02.

1. **Cells at two and four connections stay informative**: in at least one of tissues 1 and 2, the cells at d = 2 and the cells at d = 4 stay informative in A2. Then stage B is released by the coordinator with nothing further, with E10 in both arms and with the cost forecast of section 11.4.
2. **They are silent or constant**: in both tissues 1 and 2, the share of cells neither silent nor constant is below 1/2 at d = 2 or at d = 4 in A2. Then stage B is not run, and the named choices go back to the reviewer: the starting strength (1/8, number 6 of section 16 of the definition), the fixed strength of line connections (1/8, sentence 63), and the target a* (1/8, C29; the check's candidate is 1/32, finding 10), since neither C15 nor C29's further forms address that state.
3. **Anything else** (for example, shares at least 1/2 but S(d, 0, A2) below 0.1146, or one distance informative and the other not in different tissues): reported as found, with all measures of section 11.2; the coordinator decides.

Independently of these outcomes, tissue 4 is read against finding 1: if more than 300 of the 15,872 starting connections remain after step 129, the report says that the check's derivation is not borne out, and gives the counts by kind of source.

### 11.4 Stage A's report

`TISSUE_STEP1/STAGE_A_REPORT.md`: the outcome (1, 2 or 3) in the words of section 11.3; the tables of section 11.2; the outcome of the registered expectations of section 17 that stage A can score (marked "stage A"); the measured cost per byte of each tissue over its last 4,096 bytes, its mean number of connections there, and the forecast for stage B round 1: 262,144 bytes times that cost for each of the six plastic and three fixed tissues, the plastic tissues' cost taken as the larger of the first-form and coincidence tissues' and multiplied by 1.25. The worker sends the message of section 15 and stops; stage B begins only after the coordinator has read this report and released stage B.

## 12. Stage B: the full runs

### 12.1 Round 1

Nine tissues (section 6): the first-form tissue, the coincidence tissue (both with E10) and the fixed tissue, on each of the three seeds, for 262,144 bytes each, with three processes at a time. Records of section 7.1 in the four windows W0 to W3. Connections (section 7.1, item 4) at the first step of each window and at the end of every 8,192nd byte. Each plastic tissue writes, at the end of W1 (position 32,767), the CPU time so far and its forecast for the rest (its own rate of the last 8,192 bytes times the remaining bytes, times 1.25). If the sum of the forecasts of round 1 exceeds the ceiling, the worker lets the running tissues reach their next saved state, stops them, and asks.

### 12.2 After round 1

If at least one arm reaches step 1 (section 7.6), the work under this brief ends; the report is written. If neither does, opening 2 is run, and then the work ends, whatever its outcome.

### 12.3 The order of opening (definition, section 15, step 1, as changed by erratum E1)

Each opening is one stated change from round 1's tissue, run on all three seeds, in this order:

| Order | Choice | The one change | Under this brief |
|---|---|---|---|
| 1 | C15, second form | The coincidence rule of section 5.5 at inner cells | Already run from the start, as the coincidence tissue (E1) |
| 2 | C29 | The target a* = 1/32 in place of 1/8 wherever a* appears (formulas 9 and 12, and section 5.5), the first further form of C29 in revision 5 (its section 15, step 1) | Run if round 1 does not reach step 1, on the arm that met more of the 36 conditions (12 per seed) in round 1, the first-form arm on a tie. Its comparison is a fixed tissue with a* = 1/32 on each seed (three more fixed tissues, about 2.4 CPU-hours); the coordinator confirms the limit of opening 2 at the release of stage B |
| 3 | C17 and C18 | Under E10 the further form of C17 is the test on the sending side's running average (formula 11 as written), which stage A tests; C18's further form, making toward coincident activity through a chemical signal, needs C26 to C28, which are opened at step 3 | Not run; needs an exact form from the reviewer and a new release |
| 4 | C11 with C24 | τ_v = 1 with s = 16 (the form "16 with τ_v = 1" of C24) | Not run under this brief |
| 5 | C30 | One dimension | Not run |
| 6 | C16 | Traces with τ = 64 | Not run |
| 7 | C3 | 1/8 inhibitory | Not run |
| 8 | C6 | Starting number of connections 4, then 16 | Not run |
| 9 | C1 with C13 | Two parts, one predicting the other | Not run |
| 10 | C2 | A retention gate | Not run |

Whenever an opening changes the sign with which a rise in arrival acts on active excitatory inputs at the cells that emit, the kinds of the feedback cells are reversed with it (E3).

### 12.4 What is reported when the step is not reached

The sentence "Step 1 was not reached within its budget" with the arms and openings run; for every tissue the 12 conditions with their numbers and which failed, by d and L; the readings of section 7.7, including lags 4 and 8; the shares of cells silent and constant at every window and d; the counts and mean strengths of connections by kind at every window (in particular the mean strength of excitatory connections into inner cells, which bears on H1, and the number of cells whose neighbourhood is exhausted); and the comparison with the registered expectations of section 17. Mr. Xu decides whether the route continues, changes or stops there (roadmap, section 5, condition 8).

## 13. The program, saving and resuming, receipts and files

### 13.1 Files

All under `TISSUE_STEP1/` on the worker's own branch:

- `tissue_step1.py`: the tissue, the stream and the runs. `python3 tissue_step1.py run <first|coincidence|fixed> <seed> [--bytes N] [--stage A|B] [--formula11-as-written] [--out DIR] [--ckpt-bytes N] [--resume] [--die-after-byte N] [--opening 2]` and `python3 tissue_step1.py check <1..10>`. It is committed before the first check is run.
- `step1_read.py`: the analysis of section 7 on the records of a tissue (also used by check 9); it writes the measures into the tissue's receipt.
- `step1_tables.py`: makes every table of the reports from the receipts.
- `runs/<stage>_<tissue>_<seed>/`: `receipt.json`, the window records (`A1.npz`, `A2.npz` or `W0.npz` to `W3.npz`), `summary.npz`, `emissions.npy`, `final_state.npz`, `log.txt`.
- `qual/check<n>.json` and their logs.
- `STAGE_A_REPORT.md` and `STEP1_REPORT.md`.

### 13.2 Receipts

`receipt.json` of a tissue records: the brief's file name and the commit of `V` from which it was read; the SHA-256 of `tissue_step1.py` and `step1_read.py`; the versions of Python and numpy; the stage, the tissue, the seed, the form of formula 11 and the opening if any; the stream's SHA-256 and length; the birth's three digests; the records of section 7.1, items 1 and 4; every measure of section 7 (or 11.2); the CPU and wall time by stretch or window; the entry `beside` (the number of other runs in progress, set by the worker at the start); the list `resumes` (byte and time of every resumption).

### 13.3 Saving and resuming

A saved state is written every 16,384 bytes in stage B and every 4,096 bytes in stage A (`--ckpt-bytes`; 0 for none), and at the end of each window, at the end of a byte's window (after operation 8 of φ = 31): every array of the cells and connections, the step, the stream position, the states of the three generators (`getstate()`), the step of each cell's last making, the records so far and the CPU counters. It is written under another name and renamed with `os.replace`. With `--resume` and a saved state present, the run loads it, asserts that the stream's SHA-256 of the bytes read so far equals the saved one, and goes on from the next byte, appending to `resumes`; with no saved state it starts from birth. The saved state is deleted when `final_state.npz` and the receipt are written. `--die-after-byte N` ends the process with `os._exit(9)` after byte N (check 8 only). A run interrupted before its first saved state is started again after its folder has been moved to `TISSUE_STEP1/runs/interrupted/`.

### 13.4 What is committed

The program, the analysis and table scripts, every receipt, the records (within the limits of section 9), the summaries and emissions, the final states, the logs, the check receipts and the two reports. The final states of stage B are what the first form of step 2 continues from.

## 14. Rules for the worker

- Start a branch of your own from the head of the default branch `origin/codex/minifly-consolidated-20261023` at the time of release, and record its commit in the reports. Read this brief from the reviewer's branch, fetched by `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'`, with `git show`. The program imports nothing from the repository.
- Install numpy in a virtual environment under Python 3.13. Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). If a script of the worker starts runs from Python, each child is started with an environment from which every variable beginning with `E11_`, `E16_` or `EXP45_` is removed (decision 37).
- This host stops a machine about five minutes after its session's turn ends; the disk is kept and detached processes are terminated. While runs proceed, do not end your turn: issue a command that waits for the processes and repeat it until they have finished; commit and push the receipts and logs between two such commands. Start every run with `--resume` from the first time on, with its output appended to its own log. The reports say which runs were resumed, from which byte, and which were started again.
- Push only to your own branch; never to the default branch or any other; open no pull request; read other branches with `git show`.
- Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, no name of a language model appears in a committed file or commit message.
- The reports are in standard academic English: full sentences, every term defined at first use, every figure with the receipt or command that gives it, the committed files listed from `git ls-tree`.
- Nothing may be asked of the tissue that a living cell cannot do, and no operation may act on the tissue for a measurement; only outputs are recorded. If the brief cannot be followed at some point, stop that part, say what you found, and ask; go on with whatever does not depend on the answer.
- Stage B is not started without the coordinator's release after stage A's report.

## 15. Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed), with the forecast of check 10; stage A finished, with its outcome (1, 2 or 3), the counts of tissue 4 at steps 127 and 129, and the forecast for stage B; after release, all nine tissues of round 1 have passed W1, with the re-forecast; round 1 finished, with the 12 conditions per arm and seed; opening 2 finished (if run); the report committed. Each message gives your branch, the commit and the main figures in at most ten lines.

## 16. What the reports must state

1. The claim at the level actually earned (E0), in the words of section 3, item 1, or the sentence of section 12.4; for stage A, its outcome in the words of section 11.3.
2. The numbers fixed by us (section 5.7, both tables), the numbers of the reading, and the form of every choice used (the list in section 5.7), with any opening named.
3. The errata table of section 5.2 (E1 to E10, with E3 and E5 not applied and why) and the decisions of section 5.8, as applied.
4. For every tissue: the counts at birth (check 2); the records of section 7.1, item 4; the shares of section 7.2 for d = 1, 2, 4 with the numbers of cells at every window; the numbers of cells sampled; S(d, L, window) with its standard error for every d, L and window; C0(L) and C<L(L) for every window; B(d, L); the 12 conditions and whether each holds; the readings of section 7.7, including lags 4 and 8.
5. One table per seed with a row for each (d, L) and columns S at W0 and W3 for the two arms and the fixed tissue, B, C0, C<L and the conditions; one table of the shares of cells neither silent nor constant.
6. The number of connections and the number of cells with exhausted neighbourhoods at every count; the number of distinct byte values of the stream with at least one input-line connection; the connections from feedback cells.
7. The emissions: the share of positions with a byte emitted and the five commonest emitted bytes, per tissue and window. They are not scored.
8. The outcome of each registered expectation of section 17.
9. CPU time per tissue, the forecast of check 10, stage A's measured cost and its forecast for stage B, and the re-forecast at W1, against the estimate of section 9.
10. That the coincidence rule does not use the cell's prediction (section 5.5), and that a favourable result for that arm would have to be reconciled with section 8 of Mr. Xu's words.
11. That the result is of three seeds (stage A: one seed) and one stream, that no feedback was given, and the limits of section 18.
12. The list of committed files and the commit of the default branch the work started from.

## 17. Registered expectations (before any run)

The fifth reading states, for step 1 (its section 5 and risk H1): encoder cells stay informative through their unchanging lines; "for the inner cells read at two and four connections, and for the lags of four and eight bytes, the first form is likely to fail"; "it is not impossible on paper"; the reader's confidence is moderate. The mechanism (H1): at an active cell, formula 7 weakens excitatory inputs in proportion to the covariance of their traces with the cell's activity, so that inner cells, which have no unchanging input, lose the variation that carries the byte; and the bytes four and eight back lie 128 and 256 steps back, where the traces keep 0.0172 and 0.0003 (section 19, F.1), so they could be held only by sustained recurrent activity, which the same rule cancels. The check simulated a single cell two connections from the input lines: without giving up, formula 7 halved its excitatory strengths in 100,000 bytes at the definition's activity level and reduced them from 0.125 to 0.011 within 3,000 bytes when the senders were fully active; its output stayed a function of the present byte while its amplitude fell towards the levels called silent and constant (check, section 5.3). The expectations below are the brief writer's reading of H1, of the check and of the definition; each is scored in the reports as held or not held.

| No. | Stage; tissue | Measure | Expectation | Chance it holds |
|---|---|---|---|---|
| X0 | A; formula 11 as written | Starting connections remaining after operation 7 of step 129 | Fewer than 300 of the 15,872 (check, finding 1: at least 15,630 given up for seed 73129) | 0.95 |
| X1 | A and B; first-form and coincidence (E10) | Input-line connections | None is given up; the number of distinct byte values in use with at least one input-line connection never falls below its value at birth | 0.95 |
| X2 | A and B; all | Connections from feedback cells | The 423 starting ones (seed 73129; 460 and 465 for the others) remain at every count, with their strength 1/8 | 0.9 |
| X3 | A; first-form | Number of connections | The mean number of connections of a cell at byte 16,383 is above 32 (under E10, cells below their target keep adding connections; check, finding 10) | 0.6 |
| X4 | B; first-form | Mean strength of excitatory connections into inner cells, W3 against birth (1/8) | Lower (H1) | 0.7 |
| X5 | B; coincidence | The same | Not lower | 0.5 |
| X6 | A; first-form and coincidence | Outcome of section 11.3 | Outcome 1 | 0.4 |
| X7 | A | The same | Outcome 2 | 0.35 |
| X8 | B; first-form | Neither silent nor constant, d = 2 and 4, W3 | Share ≥ 1/2 at both | 0.4 |
| X9 | B; coincidence | The same | Share ≥ 1/2 at both | 0.5 |
| X10 | B; first-form | S(d, 0, W3) − S_fixed(d, 0, W3) at d = 2 and 4 | ≤ −0.02 on at least two seeds ("learning removed", H1) | 0.5 |
| X11 | B; both arms | Lags 1 and 2, all d, W3 | S below C0(L) + 0.02 in at least 4 of the 6 pairs (d, L), per arm and seed | 0.6 |
| X12 | B; both arms | Lags 4 and 8 at d = 2 and 4 | Not told beyond the baseline and the present byte | 0.8 |
| X13 | B; first-form | Step 1 reached on all three seeds | Reached | 0.05 |
| X14 | B; coincidence | The same | Reached | 0.07 |
| X15 | B; fixed | The fixed tissue meets conditions 1 and 2 of section 7.6 with W0 against W3 of itself, on all three seeds | Met. The stream is stationary and the fixed tissue's thresholds settle within about ten bytes, so its W0 and W3 are equal in distribution, and its cells at distance 4 have activity of order 1/8⁴ of a line's effect and are silent by section 7.2 (check, finding 11) | 0.01 |
| X16 | — | Checks 1 to 10 pass at the first attempt | Pass | 0.5 |
| X17 | A; coincidence | Connections from inhibitory cells into inner cells at byte 16,383, against birth | Fewer than half remain: below its target an inner cell's inhibitory connections are weakened at every step (by η · e · a* when it is silent) and given up below 1/128 under E10 | 0.8 |` followed on the next line by `| X17 | A; coincidence | Connections from inhibitory cells into inner cells at byte 16,383, against birth | Fewer than half remain: below its target an inner cell's inhibitory connections are weakened at every step (by η · e · a* when it is silent) and given up below 1/128 under E10 | 0.8 |

## 18. What step 1 cannot show

- Anything about answers, correction, keeping or generalisation: no feedback is given and nothing is scored.
- Whether the cells carry the bytes in a form that later steps can use: the classifier is the experimenter's and is fitted to many records; a living cell has no such classifier.
- Anything beyond three seeds (stage A: one), one stream and the first form (or the one change of an opening).
- Holding a byte over four or eight positions as a requirement: that is capability e, asked at step 3.
- How the feedback cells' connections fare when the lines are active (step 2).
- Whether the present byte is carried by the tissue's learning or by its wiring, except through the fixed tissue, which keeps the threshold control.
- How fast giving up should act in a living tissue: E10 removes the test on the sending side's running average, whose time scale (four bytes) has no living counterpart (check, finding 1); the slower living process is not modelled.

## 19. Output of `check_step1_brief.py`

`python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_step1_brief.py`, run at 00:39 UTC on 19 November 2026 (25 seconds):

```
==============================================================================================================
A. THE STREAM
A.1 task add (task byte 112, sign byte 43): pairs 7921, in the stream 3960, never in it 3961
A.1 task sub (task byte 113, sign byte 45): pairs 7921, in the stream 3960, never in it 3961
A.1 task max (task byte 118, sign byte 62): pairs 7921, in the stream 3960, never in it 3961
A.1 task min (task byte 119, sign byte 60): pairs 7921, in the stream 3960, never in it 3961
A.2 pairs in the stream: 15840 of 31684; never shown: 15844
A.3 an item has 10 + k bytes, k uniform on 0..12: mean length 16.0 bytes; shortest 10, longest 22
A.4 seed 73129: SHA-256 of the 262144 bytes 7772341c01ab095437347063d75110bf0a585b73372668f4dd3187e8f95e8775
    items begun: 16362 (passes through the taught pairs begun: 2); distinct byte values: 46; first 48 bytes: b'w58<VSQQZ88=58\nv05>LZOKMVKCQH12=12\np02+MS61=63\nq'
    the first 2,000 items are all taught pairs: True; mean item length in the stream 16.02 bytes
A.4 seed 93683: SHA-256 of the 262144 bytes a6363a43f56c1cdca66e6c264d8faa774d24338eaa453643b49ee9a922ca6f4f
    items begun: 16355 (passes through the taught pairs begun: 2); distinct byte values: 46; first 48 bytes: b'p11+NLZVXQ64=75\nq66-A54=12\nw51<EGDUQTUT52=51\np29'
    the first 2,000 items are all taught pairs: True; mean item length in the stream 16.03 bytes
A.4 seed 116089: SHA-256 of the 262144 bytes 0473b978c0a8db0db2cd4bde07fb65515fb84fc8f71eef9601625335e4448bbe
    items begun: 16379 (passes through the taught pairs begun: 2); distinct byte values: 46; first 48 bytes: b'v57>YNWWSESPEM83=83\nq28-06=22\np42+ADIPW20=62\nw41'
    the first 2,000 items are all taught pairs: True; mean item length in the stream 16.00 bytes
A.5 the byte values of the stream: b'\n+-0123456789<=>ABCDEFGHIJKLMNOPQRSTUVWXYZpqvw'
A.6 seed 73129 window W0 (bytes 0 to 8191): distinct values 46; balanced-accuracy chance 1/46 = 0.0217; most frequent byte b'=' share 0.0634 (the majority rate); rarest b'Y' count 100
A.6 seed 73129 window W1 (bytes 24576 to 32767): distinct values 46; balanced-accuracy chance 1/46 = 0.0217; most frequent byte b'=' share 0.0631 (the majority rate); rarest b'L' count 104
A.6 seed 73129 window W2 (bytes 122880 to 131071): distinct values 46; balanced-accuracy chance 1/46 = 0.0217; most frequent byte b'=' share 0.0616 (the majority rate); rarest b'E' count 105
A.6 seed 73129 window W3 (bytes 253952 to 262143): distinct values 46; balanced-accuracy chance 1/46 = 0.0217; most frequent byte b'\n' share 0.0625 (the majority rate); rarest b'H' count 97
A.6b seed 73129: byte values present by window: A1 46 (smallest class 44); A2 46 (smallest class 46); W0 46 (smallest class 100); W1 46 (smallest class 104); W2 46 (smallest class 105); W3 46 (smallest class 97)
A.6b seed 93683: byte values present by window: A1 46 (smallest class 42); A2 46 (smallest class 37); W0 46 (smallest class 99); W1 46 (smallest class 96); W2 46 (smallest class 97); W3 46 (smallest class 101)
A.6b seed 116089: byte values present by window: A1 46 (smallest class 46); A2 46 (smallest class 37); W0 46 (smallest class 88); W1 46 (smallest class 101); W2 46 (smallest class 92); W3 46 (smallest class 89)
A.7 shares of the byte values over the whole stream (seed 73129), rarest five and commonest five: b'Z' 0.0139, b'L' 0.0141, b'O' 0.0142, b'W' 0.0142, b'Y' 0.0142 ... b'1' 0.0399, b'6' 0.0404, b'2' 0.0406, b'=' 0.0624, b'\n' 0.0624
A.8 stage B recording windows: W0 bytes 0 to 8191; W1 bytes 24576 to 32767; W2 bytes 122880 to 131071; W3 bytes 253952 to 262143; steps of the stream 8,388,608
A.9 stage A: 16384 bytes (524,288 steps, 4,096 periods of P_s); windows A1 bytes 2048 to 6143; A2 bytes 12288 to 16383
A.10 check 5: the byte at position 5 of seed 73129 is b'S'; it is replaced by the next byte value in use, b'T'
==============================================================================================================
B. THE BIRTH (first form, 64 feedback cells with no sources, erratum E4)
B.1 seed 73129: digest of coordinates adb76cb771d012de; of the sources c0638c096619aecb; of feedback cells, drivers and kinds 66952e5b022b0422
    cells: output-end 256, feedback 64 (yes 32, no 32), encoder 240, inner 1488; inhibitory 397
    neighbourhood members (cells and lines) smallest 32, mean 96.9, largest 154; cells only: mean 90.9; cells with fewer than 8 members: 0
    all neighbourhood members of all cells (the most connections E10 allows): 192,179, 93.8 a cell on average over the 2,048 cells
    formula 11 as written, step 128: line connections kept 11 of 754 (only lines of the first four bytes b'w58<' can be kept; u-bar at step 128 of a line active for one window: b'5' 0.1344, b'8' 0.1727, b'<' 0.2220, b'w' 0.1045)
    starting connections 15872; from lines 754, share of lines in the encoder cells' draws 0.393; smallest x of an output-end cell 0.8432
    cells by distance from the input lines (d: count): 1: 232, 2: 189, 3: 199, 4: 193, 5: 201, 6: 185, 7: 179, 8: 171, 9: 198, 10: 196, 11: 41; not reached: 64
    cells recorded (d = 1, 2, 4): 232 + 189 + 193 = 614
    output-end cells at distance 8 to 11 from the input lines (not reached: 0)
    cells taking a feedback cell as a source at birth: 383; of them output-end 60 (taking a yes-cell 47, a no-cell 16), encoder 27; connections from feedback cells 423
B.1 seed 93683: digest of coordinates 3b7f27e7070b57ff; of the sources 8e946fac980830fc; of feedback cells, drivers and kinds 988ddd6833af5c1a
    cells: output-end 256, feedback 64 (yes 32, no 32), encoder 244, inner 1484; inhibitory 437
    neighbourhood members (cells and lines) smallest 27, mean 95.3, largest 162; cells only: mean 89.4; cells with fewer than 8 members: 0
    all neighbourhood members of all cells (the most connections E10 allows): 189,118, 92.3 a cell on average over the 2,048 cells
    formula 11 as written, step 128: line connections kept 10 of 833 (only lines of the first four bytes b'p11+' can be kept; u-bar at step 128 of a line active for one window: b'+' 0.2220, b'1' 0.3071, b'p' 0.1045)
    starting connections 15872; from lines 833, share of lines in the encoder cells' draws 0.427; smallest x of an output-end cell 0.8387
    cells by distance from the input lines (d: count): 1: 238, 2: 175, 3: 176, 4: 206, 5: 179, 6: 191, 7: 197, 8: 196, 9: 217, 10: 200, 11: 9; not reached: 64
    cells recorded (d = 1, 2, 4): 238 + 175 + 206 = 619
    output-end cells at distance 8 to 11 from the input lines (not reached: 0)
    cells taking a feedback cell as a source at birth: 419; of them output-end 53 (taking a yes-cell 28, a no-cell 27), encoder 30; connections from feedback cells 460
B.1 seed 116089: digest of coordinates 0121fb1fafc58c37; of the sources 503c1f17014882ab; of feedback cells, drivers and kinds 655770c0dec995ca
    cells: output-end 256, feedback 64 (yes 32, no 32), encoder 262, inner 1466; inhibitory 405
    neighbourhood members (cells and lines) smallest 26, mean 96.7, largest 166; cells only: mean 90.2; cells with fewer than 8 members: 0
    all neighbourhood members of all cells (the most connections E10 allows): 191,803, 93.7 a cell on average over the 2,048 cells
    formula 11 as written, step 128: line connections kept 10 of 791 (only lines of the first four bytes b'v57>' can be kept; u-bar at step 128 of a line active for one window: b'5' 0.1344, b'7' 0.1727, b'>' 0.2220, b'v' 0.1045)
    starting connections 15872; from lines 791, share of lines in the encoder cells' draws 0.377; smallest x of an output-end cell 0.8451
    cells by distance from the input lines (d: count): 1: 247, 2: 191, 3: 190, 4: 196, 5: 176, 6: 187, 7: 198, 8: 212, 9: 197, 10: 181, 11: 9; not reached: 64
    cells recorded (d = 1, 2, 4): 247 + 191 + 196 = 634
    output-end cells at distance 9 to 11 from the input lines (not reached: 0)
    cells taking a feedback cell as a source at birth: 422; of them output-end 55 (taking a yes-cell 34, a no-cell 26), encoder 33; connections from feedback cells 465
==============================================================================================================
C. THE WORKED EXAMPLE OF SECTION 13 (formulas 1 to 12 in the order of sentence 118; ages start at 0, so nothing is given up or made)
C.1 variant r4: step | F x | E A, m, x | I A, m, x | O A, m, x
     0 | 0.4725 0.1250 0.0750 0.0625 0.0462 -0.0737 0.0231 0.3125 0.0000 0.4062
     1 | 0.2210 0.2431 0.1908 0.1433 0.0625 -0.0552 0.0186 0.4306 0.1181 0.3662
     2 | 0.0959 0.1801 0.1218 0.1518 0.1030 -0.0130 0.0370 0.3659 0.0497 0.3131
     3 | 0.0340 0.1487 0.0866 0.1402 0.1072 -0.0083 0.0486 0.3339 0.0161 0.2700
     8 | 0.0000 0.1243 0.0521 0.1061 0.0851 -0.0275 0.0427 0.3099 -0.0076 0.2026
    16 | 0.0000 0.1239 0.0401 0.1055 0.0842 -0.0221 0.0446 0.0304 -0.2855 0.0561
    24 | 0.0000 0.1236 0.0308 0.1064 0.0847 -0.0167 0.0499 0.0315 -0.2205 0.0000
    31 | 0.0000 0.1234 0.0245 0.1072 0.0851 -0.0129 0.0544 0.0315 -0.1766 0.0000
    final strengths: {"F": {"s1": 0.25027}, "E": {"h": 0.25523, "c": 0.49739, "F": 0.24966}, "I": {"E": 0.5003, "q": 0.25102, "g": 0.24949}, "O": {"u": 0.75099, "f": 0.24949, "F": 0.25014}}
    largest difference from the published table after rounding to four decimals: 0.0000 (0.0001 is a rounding of the last place)
C.2 variant E4: step | F x | E A, m, x | I A, m, x | O A, m, x
     0 | 0.5100 0.1250 0.0750 0.0625 0.0462 -0.0737 0.0231 0.3125 0.0000 0.4062
     1 | 0.2550 0.2525 0.2001 0.1480 0.0625 -0.0552 0.0186 0.4400 0.1275 0.3709
     2 | 0.1275 0.1886 0.1300 0.1584 0.1053 -0.0107 0.0382 0.3742 0.0577 0.3196
     3 | 0.0638 0.1565 0.0939 0.1473 0.1105 -0.0051 0.0508 0.3415 0.0232 0.2770
     8 | 0.0020 0.1252 0.0517 0.1081 0.0868 -0.0264 0.0450 0.3104 -0.0083 0.2042
    16 | 0.0000 0.1238 0.0390 0.1051 0.0840 -0.0228 0.0443 0.0302 -0.2865 0.0555
    24 | 0.0000 0.1235 0.0300 0.1060 0.0845 -0.0172 0.0496 0.0313 -0.2214 0.0000
    31 | 0.0000 0.1233 0.0238 0.1068 0.0849 -0.0134 0.0541 0.0313 -0.1772 0.0000
    final strengths: {"F": {}, "E": {"h": 0.2554, "c": 0.4973, "F": 0.24957}, "I": {"E": 0.50031, "q": 0.25102, "g": 0.24949}, "O": {"u": 0.75031, "f": 0.24983, "F": 0.25017}}
C.3 variant K: step | F x | E A, m, x | I A, m, x | O A, m, x
     0 | 0.5100 0.1250 0.0750 0.0625 0.0462 -0.0737 0.0231 0.3125 0.0000 0.4062
     1 | 0.2550 0.2525 0.2001 0.1480 0.0626 -0.0551 0.0187 0.4400 0.1275 0.3709
     2 | 0.1275 0.1886 0.1300 0.1584 0.1055 -0.0105 0.0383 0.3742 0.0577 0.3196
     3 | 0.0638 0.1565 0.0939 0.1473 0.1109 -0.0048 0.0511 0.3415 0.0232 0.2770
     8 | 0.0020 0.1252 0.0517 0.1081 0.0877 -0.0255 0.0458 0.3104 -0.0083 0.2042
    16 | 0.0000 0.1238 0.0390 0.1051 0.0858 -0.0213 0.0459 0.0302 -0.2865 0.0555
    24 | 0.0000 0.1235 0.0300 0.1060 0.0872 -0.0153 0.0520 0.0313 -0.2214 0.0000
    31 | 0.0000 0.1233 0.0238 0.1068 0.0883 -0.0112 0.0570 0.0313 -0.1772 0.0000
    final strengths: {"F": {}, "E": {"h": 0.2554, "c": 0.4973, "F": 0.24957}, "I": {"E": 0.49887, "q": 0.24641, "g": 0.21127}, "O": {"u": 0.75031, "f": 0.24983, "F": 0.25017}}
==============================================================================================================
D. ONE-STEP TESTS OF THE SIGN OF THE TWO STRENGTH RULES (w = 0.5, e = 0.2, x(k-1) = 0.3)
D.1 formula 7: sigma +1, m +0.1: delta w = -0.000750
D.1 formula 7: sigma +1, m -0.1: delta w = +0.000750
D.1 formula 7: sigma -1, m +0.1: delta w = +0.000750
D.1 formula 7: sigma -1, m -0.1: delta w = -0.000750
D.2 coincidence rule, excitatory sender: theta-bar 0.1: delta w = +0.001500
D.2 coincidence rule, excitatory sender: theta-bar 0.5: delta w = -0.001500
D.2b coincidence rule, inhibitory sender: x(k-1) 0.3: delta w = +0.004375 (a* = 1/8)
D.2b coincidence rule, inhibitory sender: x(k-1) 0.05: delta w = -0.001875 (a* = 1/8)
D.3 with e = 0 every rule gives 0; with x(k-1) = 0 formula 7 and the excitatory coincidence rule give 0, and the inhibitory coincidence rule gives -0.003125 (inhibition weakened below the target)
==============================================================================================================
E. THE COUNTERMODEL C0: THE CLASSIFIER OF SECTION 7 FITTED TO A ONE-HOT CODE OF THE PRESENT BYTE ALONE
E.0 seed 73129 A1: C0 by lag L=0: 1.0000, L=1: 0.0858, L=2: 0.0497, L=4: 0.0695, L=8: 0.0335; classes 46, chance 0.0217
E.0 seed 73129 A2: C0 by lag L=0: 1.0000, L=1: 0.0703, L=2: 0.0561, L=4: 0.0729, L=8: 0.0354; classes 46, chance 0.0217
E.1 seed 73129 W0: C0 by lag L=0: 1.0000, L=1: 0.0870, L=2: 0.0590, L=4: 0.0756, L=8: 0.0340; classes 46, chance 0.0217
E.1 seed 73129 W3: C0 by lag L=0: 1.0000, L=1: 0.0792, L=2: 0.0504, L=4: 0.0767, L=8: 0.0381; classes 46, chance 0.0217
E.1 seed 93683 W0: C0 by lag L=0: 1.0000, L=1: 0.0778, L=2: 0.0531, L=4: 0.0784, L=8: 0.0406; classes 46, chance 0.0217
E.1 seed 93683 W3: C0 by lag L=0: 1.0000, L=1: 0.0818, L=2: 0.0508, L=4: 0.0803, L=8: 0.0372; classes 46, chance 0.0217
E.1 seed 116089 W0: C0 by lag L=0: 1.0000, L=1: 0.0807, L=2: 0.0529, L=4: 0.0772, L=8: 0.0342; classes 46, chance 0.0217
E.1 seed 116089 W3: C0 by lag L=0: 1.0000, L=1: 0.0744, L=2: 0.0515, L=4: 0.0804, L=8: 0.0430; classes 46, chance 0.0217
==============================================================================================================
F. TIME CONSTANTS AND SHARES
F.1 (31/32)^9 = 0.7515
F.1 (31/32)^32 = 0.3621
F.1 (31/32)^64 = 0.1311
F.1 (31/32)^128 = 0.0172
F.1 (31/32)^256 = 0.0003
F.2 lags in steps: L=0: 0 steps, L=1: 32 steps, L=2: 64 steps, L=4: 128 steps, L=8: 256 steps; state v keeps (1/2)^32 = 2.33e-10 of a byte one window back
F.3 running average u-bar of a line after one window of 32 steps on, from 0: 0.2220; steps of silence until it falls below 1/128: 427 (13.3 bytes)
F.4 from a running average of 1/16 (a line active once in 16 bytes, at its peak somewhat higher): 266 steps (8.3 bytes) of silence
F.5 byte b'A': share 0.0146, mean gap 68.4 bytes, share of gaps longer than 14 bytes 0.834
F.5 byte b'p': share 0.0156, mean gap 63.9 bytes, share of gaps longer than 14 bytes 0.899
F.5 byte b'9': share 0.0171, mean gap 58.5 bytes, share of gaps longer than 14 bytes 0.720
F.5 byte b'\n': share 0.0624, mean gap 16.0 bytes, share of gaps longer than 14 bytes 0.619
F.5b under E10, a cell that stays below its target adds one connection every 128 steps until its neighbourhood is full: about 89 makings (seed 73129, mean), 11,375 steps, 355 bytes
F.6 first possible making and giving up at step P_s = 128 = byte 4 of life; P_s periods in the stream: 65,536
==============================================================================================================
G. COST AND FILES
G.1 mean connections 8: plastic 12,124,160 operations a byte, 25.9 ms a byte (41.2 with 16 makings a step); stage B stream 1.88 CPU-hours (3.00); stage A stream 0.118 (0.188); fixed tissue 9.4 ms a byte
G.1 mean connections 12: plastic 17,367,040 operations a byte, 33.7 ms a byte (49.1 with 16 makings a step); stage B stream 2.46 CPU-hours (3.57); stage A stream 0.154 (0.223); fixed tissue 11.0 ms a byte
G.1 mean connections 32: plastic 43,581,440 operations a byte, 73.1 ms a byte (88.4 with 16 makings a step); stage B stream 5.32 CPU-hours (6.44); stage A stream 0.332 (0.402); fixed tissue 18.9 ms a byte
G.1 mean connections 93.8: plastic 124,583,936 operations a byte, 194.6 ms a byte (209.9 with 16 makings a step); stage B stream 14.17 CPU-hours (15.29); stage A stream 0.885 (0.955); fixed tissue 43.2 ms a byte
G.2 stage A at a mean of 12 connections (16 makings a step; two plastic tissues, the fixed tissue, the 512-byte tissue): 0.50 CPU-hours; checks 5 to 8 and 10 (about 7,000 bytes of plastic tissue at 12 connections and the 1,024 timed bytes of check 10 at this mean) 0.20; in all 0.69
G.2 stage A at a mean of 93.8 connections (every neighbourhood full, so no making; two plastic tissues, the fixed tissue, the 512-byte tissue): 1.82 CPU-hours; checks 5 to 8 and 10 (about 7,000 bytes of plastic tissue at 12 connections and the 1,024 timed bytes of check 10 at this mean) 0.28; in all 2.10
G.3 stage B round 1 at a mean of 12 connections (with 16 makings a step throughout): 23.5 CPU-hours; ceiling 40
G.3 stage B round 1 at a mean of 32 connections (with 16 makings a step throughout): 40.7 CPU-hours; ceiling 40
G.3 stage B round 1 at a mean of 93.8 connections (every neighbourhood full, no making): 87.0 CPU-hours; ceiling 40
G.4 records of all 2,048 cells in float16: stage A 16.8 MB a window, 33.6 MB a tissue, 101 MB for its three recorded tissues; stage B 33.6 MB a window, W0 and W3 kept: 67.1 MB a tissue, 604 MB for round 1 before compression
G.5 per-cell summaries (4 numbers a cell a window, float64) and emissions (int16 a byte): 7.1 MB for round 1
G.6 saved state at a mean of 12 connections: 24,576 connections, 1.28 MB a tissue
G.6 saved state at a mean of 93.8 connections: 192,102 connections, 8.65 MB a tissue
```

## 20. References

- Bienenstock, Cooper and Munro (1982), Journal of Neuroscience 2: 32-48.
- Vogels, Sprekeler, Zenke, Clopath and Gerstner (2011), Science 334: 1569-1573.
- As cited in the definition. Textbook facts (NMDA-receptor-dependent plasticity, homeostatic plasticity) are not referenced. Page numbers are from memory and should be checked before citation.

## 21. Settlement of the check of revision 1

| Finding of the check | Settled by |
|---|---|
| 1. Formula 11's sending-side test removes nearly every connection at byte 4 | Erratum E10 adopted by the reviewer (section 5.2); stage A's 512-byte tissue records the loss at steps 127 and 129 (section 11.1, item 4); expectation X0 |
| 2. The rule can be met by a tissue carrying only the present byte | Condition 3 of section 7.6 requires S ≥ C0(L) + 0.02 at lags 1 and 2; section 3, item 3 |
| 3. The start of life measured on the wrong cells and population | Baseline B = max(S(W0), S_fixed(W3)) (section 7.6); distance by the mode over blocks (section 1); a sample of 128 cells at each distance by `Random(seed + 900000)` (section 7.3); all cells recorded (section 7.1) |
| 4. The classifier ignores amplitude; float16 decides small signals | Values below 0.01 set to 0 in storage and in the features, also in check 9 (sections 1, 7.3) |
| 5. The coincidence rule's threshold follows each byte; its inhibitory part is not the cited rule | ϑ with time constant 4,096 steps and ϑ(0) = a*; inhibitory inputs by η · e · (x − a*); both marked as the brief writer's; the departure from sentence 36 stated (section 5.5); check 1(c) and check 4 recomputed |
| 6. The reversal of the feedback cells' kinds is wrong for the cells that emit | E3 not applied; kinds of sentence 87 in every tissue (sections 5.2, 5.3) |
| 7. Lags 4 and 8 require capability e of step 3 | The reviewer's decision: lags 0 to 2 required, 4 and 8 read and reported (sections 2, 7.6, 7.7), as the roadmap now states |
| 8. The fallback cannot address the cause | E10 is adopted before release, so opening 2 is kept (section 12.3); opening 3 restated under E10 |
| 9. E4 against E5 | The reviewer's decision: E4 kept, E5 not applied, with the reason from section 16 (section 5.2) |
| 10. The threshold control and a* make byte-selective cells keep growing | Registered (X3); a* = 1/32 named in outcome 2 of stage A (section 11.3); growth and its cost measured in stage A (sections 9, 11) |
| 11. Registered expectations | X15 (formerly X14) set to 0.01 with the check's reason; new X0 on starting connections; X1 and X2 rewritten for E10; X14 (coincidence) brought near X13 |
| 12. Cost of making | Making cost added to the estimate (section 9); check 10 measures bytes after the first making; stage A measures the cost before stage B |
| 13. Smaller points | (1) no candidate: nothing made, step of last making unchanged (section 5.4, operation 7); (2) K = 46 printed for every window of every seed (A.6b); (3) d = 1 only by lines of byte values present in the window (section 1); (4) check 5's replacement byte fixed, "S" by "T"; (5) only the first form of step 2 can continue these tissues (section 2); (6) section 2's sentence on telling the rules apart made conditional on E10 |
| 14. The contract's design block (check, section 7) | C<L computed by check 9 (d); no positive control of the tissue, stated (section 3, item 4); stage A is the cheap preflight (section 3, item 5); matched step-1 exposure for step 5's control tissues (section 3, item 2); the fixed tissue enters the rule through B, and revision 1's separate sentence on it is withdrawn |

## History

Revision 2 (19 November 2026, from 00:35 UTC): from the check of revision 1 (`CHECK_OF_TASK_STEP_1_BRIEF_20261119.md`, findings 1 to 13 and its section 7) and the reviewer's decisions (E10 adopted; lags 0 to 2 required; E4 kept and E5 not applied); restructured into stage A (trial) and stage B (full runs); the check script revised and run again.

Revision 1 (19 November 2026, from 00:03 UTC): written from roadmap draft 3, revision 4 of the definition and the fifth reading. Checked by another agent; not released.
