# Step 1 of the roadmap: the tissue reads the stream and stays informative (brief, revision 1; written from 00:03 UTC on 19 November 2026 by the clock of this machine)

The coordinator gives this brief its experiment number when it releases it. Until then a worker starts nothing. The brief carries everything the worker needs; where it seems not to, the worker stops that part and asks (section 12). Every number in it that can be computed without running the tissue is computed by `check_step1_brief.py` beside this file (Python standard library only, `python3 -I`, about 25 seconds); its output is section 17.

## 0. Sources, shell names and host

- `C = PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117` on the reviewer's branch `V = origin/claude/vibrant-cerf-sje726`.
- The governing words are Mr. Xu's, in `$C/WHAT_A_NEURON_HAS_20261118.md`, sections 7 to 16.
- The roadmap is `$C/ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md` (draft 3). Step 1 and the conditions of its section 5 bind this brief.
- The tissue is `$C/THE_TISSUE_DEFINED_REVISION_4_20261118.md` ("the definition"). Sentences are cited as "sentence n" and formulas as "formula n" of the definition. The program is this definition, with the errata of section 5.2 and the decisions of section 5.8, and nothing else.
- The errata, risks and undecided items are those of `$C/FIFTH_READING_OF_THE_TISSUE_DEFINED_20261118.md` ("the fifth reading").
- The design block is that of `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md` ("the contract").
- The host of the worker is Linux on x86-64 with four cores and Python 3.13. The program uses Python 3.13 and numpy only (no torch); every receipt records both versions.

## 1. Terms

All terms of the definition (its section 0) are used in its sense. In addition:

- *Tissue*: one population of 2,048 cells born from one seed and run on one stream, never copied, never reset.
- *First-form tissue*: a tissue in which the strength of every connection between cells changes by formula 7 (the first form of choice C15).
- *Coincidence tissue*: a tissue in which the strengths of the connections into inner cells change by the coincidence rule of section 5.5 (the second form of C15, erratum E1), with the kinds of the feedback cells reversed (erratum E3); everything else is as in the first-form tissue.
- *Fixed tissue*: a tissue born from the same seed as the two above, in which no strength changes and no connection is given up or made (section 5.6). It shows what the wiring gives without learning.
- *Arm*: one of the two rules of strength (first form, coincidence). Each arm is run as one tissue per seed.
- *Position* t: the t-th byte of the stream, counted from 0. The window of byte t is steps 32t to 32t + 31.
- *Reading step of byte t*: step 32t + 31 (φ = 31).
- *Recorded output* of cell i at byte t: x_i at the reading step of byte t, after operation 4 of sentence 118.
- *Lag* L: the byte at position t − L, for L = 0 (the present byte), 1, 2, 4 and 8.
- *Recording window*: a stretch of 8,192 consecutive positions at which outputs are recorded. There are four: W0 (positions 0 to 8,191), W1 (24,576 to 32,767), W2 (122,880 to 131,071) and W3 (253,952 to 262,143). W0 is *the start of life*; W3 is *late in the stream*.
- *Distance d of a cell*: its distance from the input lines (definition, section 0) along the connections present at the first step of a recording window. A cell with at least one input line among its sources is at d = 1.
- *Silent*, *constant*, *score S*, *countermodels C0 and C<L*: defined in section 7.
- *Comparator*: the network trained by back-propagation that the roadmap (section 5, condition 5) names as positive control. It has no part in this brief (section 3, item 4).

## 2. The question, and what reaching step 1 buys

**Question.** When the tissue of the definition reads, with no correction, a long stream of the first family's items one byte after another, do its cells one, two and four connections from the input lines stay neither silent nor constant, and can the present byte and the bytes one, two, four and eight positions back be told from their recorded outputs better late in the stream than at the start of life?

**What reaching it buys for the goal.** Every later step corrects cells from what they carry (roadmap, section 6, step 1: "the material every later correction works on"). Capability a (roadmap, section 3) is the dependency of all others: if cells away from the input lines fall silent or constant, or if learning by formula 7 removes from them what they carried of the bytes (risk H1), then no feedback at step 2 and no relay at step 3 has anything to work on. Step 1 also settles, before any feedback exists, which of the two rules of strength keeps that material, so that step 2 starts from a tissue known to stay informative. Because the steps are phases of one life (roadmap, decision D5), the tissues of step 1 are saved at their last step so that step 2 can continue them.

## 3. The design and report block of the contract

1. **Claim and level.** Level E0 (access): "Under the stated conditions, cells at distances 1, 2 and 4 from the input lines of the [first-form | coincidence] tissue are neither silent nor constant late in a stream of 262,144 bytes, and the bytes at lags 0, 1, 2, 4 and 8 can be told from their recorded outputs better than at the start of life." It is a mechanistic diagnostic, not a promotion test. No learned value, answer or association is tested, and none may be claimed.
2. **Exposure record.** The stream (section 4) shows, as bytes and without any feedback, the items of the taught half of the record's split: 3,960 pairs of each of the four operations, 15,840 pairs in all, each with its answer and end mark. In 262,144 bytes the stream begins 16,355 to 16,379 items (section 17, A.4), so every one of the 15,840 pairs is shown at least once and about 520 are shown twice. The other half, 15,844 pairs (3,961 per operation), never appears; the reserve of never-taught instances of step 5 is to be drawn from it (roadmap, section 5, condition 4). No feedback is given: the yes and no lines are 0 at every step (section 5.8, item 1). The tissue emits a byte or no byte at every position (sentence 69); the emissions are recorded and not scored. There is no first scored choice.
3. **Countermodels.** No answer is scored, so the exact-key table, the canonical-key table and the part and frequency shortcuts of the contract have nothing to score. What applies is the structure of the stream itself: the bytes at different lags are not independent (an "=" is followed by a digit; an end mark by a task byte). The countermodels are therefore the same classifier fitted to codes of the stream instead of to cells: C0(L), a one-hot code of the present byte alone, and C<L(L), a one-hot code of the bytes at lags 0 to L − 1 (section 7.5). A score at lag L at or below C0(L) is consistent with cells that carry the present byte only. The fixed tissue (item 4) is the countermodel for "learning added nothing".
4. **State and controls.** All state persists from birth to the end of the stream; nothing is reset (sentences 120, 121; conditions 1 and 2 of the roadmap). The controls are second tissues born from the same seed, never copies (condition 3): the fixed tissue (strengths and connections never change) for both arms; the two arms are each other's comparison for the rule of strength. The positive control of the measure is the analysis applied to a code that holds the bytes at lags 0 to 8 exactly (check 9); it must score at least 0.99 at every lag. The comparator (condition 5) solves arithmetic answers, not the measure of this step, and is not run; its release, if the coordinator wants one for this stream, is the coordinator's. The information and the feedback are equal across the compared tissues: the same stream, no feedback.
5. **Rule of success, budget and stopping.** Section 7.6 (rule of success), section 10 (budget, stopping rule, order of opening, what is reported when the step is not reached), section 9 (cost and preflight).

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

Everything else is the record's, item for item: the split (half the pairs, `frac = 0.5`), the generator `Random(seed + 77)`, the order, the number k of letters (uniform on 0 to 12) and the letters. Each tissue reads the first **262,144 bytes** of the stream of its own seed. An item therefore reads, for example, `w58<VSQQZ88=58\n`: the task byte (`p` add, `q` sub, `v` max, `w` min), the first operand in two digits, the sign (`+ - > <`), k capital letters, the second operand in two digits, `=`, the answer in two digits and the end mark. An item has 10 + k bytes (mean 16). The stream uses 46 byte values (section 17, A.5). Its digests for the three seeds are in section 17, A.4, and check 3 asserts them.

## 5. The tissue

### 5.1 By reference

The tissue is the first form of the definition: sentences 1 to 146, formulas 1 to 12 in the order of operations of sentence 118, the numbers of section 16, the first forms of the choices of section 14 (except C15 in the coincidence tissue), changed only by the errata of section 5.2 and decided where the definition leaves a choice to a programmer by section 5.8. The space, lines and drivers are sentence 6; kinds sentence 16 and 87 (with E3); starting sources sentence 28 (with E4); formulas 1 to 12 sentences 43 to 55; the input end sentences 60 and 63; the output end and the emission sentences 69 and 72; the feedback route sentences 85 to 90 (with E4); the order of operations sentence 118; the start sentence 121.

### 5.2 Errata of the fifth reading, and how this brief applies them

| No. | Change (fifth reading, section 7) | Applied here |
|---|---|---|
| E1 | Move C15 from third place to first in the order of opening for step 1; run the first form beside a tissue with C15's second form at inner cells from the start (with E3); read each distance and each lag separately | The coincidence tissue is run from the start beside the first-form tissue (sections 5.5, 6). Every measure is read for d = 1, 2, 4 and for each lag separately (section 7). In the order of opening, C15 is first (section 10.3) |
| E2 | Sentence 93 reworded (in expectation over many positions); 128 feedback cells first at step 2; step 2 reports output-end cells reached from yes-cells and no-cells | No feedback at step 1. The receipts give, at birth and at each window, the number of output-end cells taking a yes-cell and a no-cell as a source, so that step 2 starts from known counts |
| E3 | Whenever C15 takes a form in which a rise in arrival strengthens active excitatory inputs, the kinds of feedback cells are reversed | In the coincidence tissue yes-cells are excitatory and no-cells inhibitory. With silent lines this has no effect at step 1 (a feedback cell's output is 0 throughout); it is set so that the same tissue can go on to step 2 |
| E4 | A feedback cell has its feedback line only; formulas 7, 9 and 12 do not apply to it, and its threshold stays 0 | Applied: feedback cells draw no sources, never make a connection, keep θ = 0, and their strengths do not change. Formulas 1 to 6 and 10 are computed for them (formula 11 never removes the feedback line) |
| E5 | Term "encoder cell": a cell with at least one input line in its neighbourhood; a feedback cell within reach of input lines may draw them | The term is changed. Under E4 a feedback cell draws nothing in the first form, so no feedback cell draws an input line at step 1. A feedback cell that draws input lines is a further form of C56, for the brief of step 2 (section 5.8, item 5) |
| E6 | Sentences 50 and 139: the receiving cell's mismatch and activity are "taken to reach the synapse by the spread of potential (unsettled)"; new item in section 18 | A wording; no effect on the program. The report repeats the wording when it describes formula 7 |
| E7 | Sentence 91: "at the steps at which the feedback cell's output arrives; the net change over the window includes the background change" | A wording; no feedback at step 1 |
| E8 | Section 12: give a wider range for encoder cells at one connection from a feedback cell | The counts of this brief's draws are given: 27, 30 and 33 encoder cells for the three seeds (section 17, B.1) |
| E9 | The brief of step 1 states whether feedback cells are present with silent lines | Present, with silent lines (section 5.8, item 1) |

### 5.3 The birth, exactly (choices C30, C53, C54, C56, C3, C7, C51, and the undecided item 4)

All draws of the birth use one generator `rb = random.Random(seed)` of the Python standard library, in this order. Positions are pairs (x, y) of floats.

1. Coordinates: for i = 0, …, 2,047: x_i = `rb.random()`, then y_i = `rb.random()`.
2. Input lines: for v = 0, …, 255, the line of byte value v is at (0, `rb.random()`).
3. Output lines: for v = 0, …, 255, the line of byte value v is at (1, `rb.random()`).
4. Feedback cells: `FB = rb.sample(range(2048), 64)`. The first 32 indices of `FB` (in the order returned) receive the yes line; the last 32 receive the no line. This is done before the output-end cells are assigned (undecided item 4).
5. Kinds: for i = 0, …, 2,047, `u_i = rb.random()` is drawn for every cell.
6. Drivers of the output lines: for v = 0, …, 255 in turn, the cell that drives line v is, among the cells that are neither feedback cells nor already drivers, the one with the smallest (1 − x_i)² + (y_i − y_v)², ties going to the smaller index (sentence 6, C54). No draw is used.
7. Kinds: a cell is inhibitory if u_i < 1/4 and it is neither an output-end cell nor a feedback cell. Output-end cells are excitatory. Feedback cells: in the first-form tissue no-cells are excitatory and yes-cells inhibitory (sentence 87); in the coincidence tissue the reverse (E3). Lines have σ = +1.
8. Neighbourhoods (sentence 6, read as "within distance 1/8" = "at squared distance at most 1/64"): for every cell i that is not a feedback cell, its members are the cells j ≠ i with (x_j − x_i)² + (y_j − y_i)² ≤ 1/64, in increasing index, followed by the input lines v with x_i² + (y_i − y_v)² ≤ 1/64, in increasing v. A line is numbered 2,048 + v in the program. Feedback cells have no neighbourhood in the first form.
9. Starting sources: for i = 0, …, 2,047 in turn, skipping feedback cells, `src_i = rb.sample(members_i, min(8, len(members_i)))`. Every starting connection has strength 1/8 (number 6 of section 16), traces e = 0, running average ū = 0, and was made at step 0.
10. A cell is an *encoder cell* if it is not a feedback cell and its neighbourhood holds at least one input line (E5); an *output-end cell* if it drives a line; an *inner cell* otherwise (feedback cells excluded). These kinds are fixed for life, since neighbourhoods are fixed.

All draws during life (only the sources of new connections, formula 12) use a second generator `rl = random.Random(seed + 500000)`. The birth is therefore identical in the three tissues of a seed, except for the kinds of the 64 feedback cells in the coincidence tissue.

### 5.4 One step, exactly

The order is sentence 118. All arithmetic is in float64. z_j(k) is the value of line j at step k, or x_j(k − 1) for a cell (with x_j(−1) = 0). Quantities named "before the step" are those held at the end of step k − 1.

1. φ = k mod 32. If φ = 0, the input line of the byte at position k // 32 is set to 1 and all other input lines to 0; they keep these values for the whole window. **Both feedback lines are 0 at every step of step 1** (section 5.8, item 1); the world compares nothing.
2. Every cell computes A_i(k) = Σ_j σ_j w_ij z_j(k) over its present sources (formula 1). A feedback cell's only source is its feedback line, with w = 1 and σ = +1.
3. m_i(k) = A_i(k) − p_i(k) (formula 5, with p from before the step).
4. v_i(k) = ½ v_i(k − 1) + ½ A_i(k); x_i(k) = min(1, max(0, v_i(k) − θ_i(k))) (formulas 2, 3).
5. Strength changes, from traces held before the step and x_i(k − 1), for every connection from a cell into a cell that is not a feedback cell (connections from lines are never changed, sentence 63): in the first-form tissue formula 7 at every such connection; in the coincidence tissue formula 7 at connections into encoder and output-end cells, and the coincidence rule of section 5.5 at connections into inner cells. Then formula 8 (bounds 0 and 1). In the fixed tissue nothing.
6. e_ij(k + 1) = (31/32) e_ij(k) + (1/32) z_j(k) (formula 6); p_i(k + 1) = (31/32) p_i(k) + (1/32) A_i(k) (formula 4); θ_i(k + 1) = max(0, θ_i(k) + (1/128)(x_i(k) − 1/8)) for every cell except feedback cells, whose θ stays 0 (formula 9, E4); ā_i(k + 1) = (127/128) ā_i(k) + (1/128) x_i(k) and ū_ij(k + 1) = (127/128) ū_ij(k) + (1/128) z_j(k) (formula 10). In the coincidence tissue, inner cells also update ϑ_i (section 5.5).
7. Giving up, then making (formulas 11 and 12), except in the fixed tissue. Giving up: every connection with k − k₀ ≥ 128 (k₀ its step of making; 0 at birth) and with w_ij < 1/128 or ū_ij < 1/128 (values after operation 6) is removed; a feedback cell's feedback line is never removed. Making: then, for i = 0, …, 2,047 in increasing index, every cell that is not a feedback cell and has θ_i(k + 1) = 0, ā_i(k + 1) < 1/8 and k − (step of its last making, or 0) ≥ 128 makes one connection: the candidates are its neighbourhood members that are not its present sources, in the order of section 5.3, item 8, and the source is `rl.choice(candidates)` (nothing if there are none). The new connection has w = 1/8 (from a line or a cell alike, sentence 63 and formula 12), e = 0, ū = 0, k₀ = k. The cell's step of last making becomes k.
8. If φ = 31, the emission is read (sentence 72): the byte value of the most active output-end cell if its x exceeds 1/2, ties to the lowest value; otherwise none. It is recorded (section 7.1).

### 5.5 The coincidence rule (C15, second form; erratum E1)

The definition names the form ("coincidence rule with a sliding threshold") and gives no formula. This brief fixes it, with no number that the definition does not already have:

- Δw_ij = η · e_ij(k) · x_i(k − 1) · (x_i(k − 1) − ϑ_i(k)), with η = 1/8, for every connection from a cell into an inner cell, whatever the kind of the sending cell.
- ϑ_i(k + 1) = (31/32) ϑ_i(k) + (1/32) · x_i(k)² / a*, with a* = 1/8 and ϑ_i(0) = 0 (sentence 121), held by the receiving cell; updated in operation 6.

The sending side contributes its trace e_ij; the receiving side its activity at the previous step and its own sliding threshold. The rule is a function of the two ends of the connection (sentence 42). For an excitatory input it is the rule of Bienenstock, Cooper and Munro (1982): a rise of the receiving cell's activity above its threshold strengthens active inputs, which is the property that E3 names. For an inhibitory input the same sign strengthens inhibition onto a cell active above its threshold, which is the form of inhibitory plasticity that holds a cell's activity near a target (Vogels and colleagues, 2011). The time constant of ϑ is τ = 32 and its target is a*, the definition's own numbers. Its living counterpart is NMDA-receptor-dependent potentiation and depression with a threshold that slides with the cell's recent activity (textbook; the sliding is *unsettled* in its mechanism).

### 5.6 The fixed tissue

Born as in section 5.3 from the same seed (feedback kinds as in the first-form tissue; they have no effect at step 1). It reads the same stream. Operations 1 to 4, 6 (formula 9 included) and 8 are carried out; operation 5 and operation 7 are not, so that every strength and every connection stays as at birth. The threshold control is kept because it is the cell's regulation of its own activity, not a change of a connection; without it the fixed tissue would differ from the other two in its activity as well as in learning. One fixed tissue per seed serves both arms: the arms differ only in operation 5 and in the kinds of feedback cells, whose output is 0 throughout.

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
| 8 | τ (prediction, traces; ϑ in the coincidence tissue) | 32 steps | C13, C16 | section 16; section 5.5 |
| 9 | η | 1/8 (both rules) | C15 | section 16; section 5.5 |
| 10 | η_θ | 1/128 | C29 | section 16 |
| 11 | a* | 1/8 (also the target of ϑ) | C29 | section 16; section 5.5 |
| 12 | P_s | 128 steps | C17, C18 | section 16 |
| 13 | Giving-up level | 1/128 | C17 | section 16 |
| 14 | Steps per byte | 32 | C24 | section 16 |
| 15 | Emission level | 1/2 | C33 | section 16 |
| 16 | Feedback cells in form (b) | none at step 1 (section 5.8, item 2) | C52 | section 16 |

Values fixed by decision, convention or unit (definition, section 16): 256 input lines and 256 output lines; two feedback lines; the unit square with the faces x = 0 and x = 1; feedback-line strength 1; maximum strength, activity, slope and line value 1; half yes and half no; starting values 0; the seed.

Numbers of this brief that belong to the experimenter's reading and not to the tissue: the stream's length 262,144 bytes; four recording windows of 8,192 bytes; the lags 0, 1, 2, 4, 8; the distances 1, 2, 4; the level 0.01 and the share 1/128 of the definitions of silent and constant; the share 1/2 of cells; four folds and a gap of 16 positions; the margin 0.02, the level 0.95 and the allowance 0.01 of the rule of success. The life generator's offset 500,000 is a convention.

The form of every choice used: C1 one part; C2 no gate; C3 1/4 inhibitory, output-end cells excitatory, feedback cells by sentence 87 (reversed by E3 in the coincidence tissue); C5 graded 0 to 1; C6 8, then the cell's own; C7 8 at random within the neighbourhood; C8 encoder cells receive from cells; C10 formula 3; C11 τ_v = 2; C12 one prediction; C13 running average, τ = 32; C14 signed difference; C15 formula 7 (first-form tissue) or the coincidence rule at inner cells (coincidence tissue); C16 exponential, τ = 32; C17 age ≥ P_s and (w or ū < 1/128); C18 θ = 0, ā < a*, P_s since the last making or the start, random source, w = 1/8; C19 strength by both ends, either end ends it, the receiving side starts it; C20 passages formed by traces; C21 one line for each value; C22 encoder cells predict; C23 2,048; C24 32, all at once; C25 all 0; C26 to C28 no chemical signal; C29 formula 9 at all cells except feedback cells (E4); C30 unit square, radius 1/8; C31 one component; C32 256 lines, active for the whole window; C33 most active output-end cell above 1/2, ties lowest; C34 every position at φ = 31; C37 yes and no lines, here silent; C38 no emission counts as "no" (no feedback is given at step 1); C39 φ = 0, one step (unused); C40 through feedback cells only (unused); C41 m_i · x_i(k − 1); C42 no relay by activity; C43 as the neighbourhood gives; C44 not opened; C45 bytes only; C46 only by the cells' own changes; C47 the grid; C48 no electrical synapses; C49 sentence 82; C50 unused; C51 one recorded seed per tissue (two generators, section 5.3); C52 form (a); C53 random points on x = 0 and x = 1; C54 nearest eligible cell, ties by index; C55 the reading step; C56 64 feedback cells with the feedback line only (E4).

### 5.8 Decisions on the items still undecided for a programmer (fifth reading, section 6) and on others found while writing this brief

1. **Feedback cells at step 1 (item 1; E9).** The 64 feedback cells are present from birth, with both feedback lines 0 at every step. Reason: the roadmap makes the steps phases of one life (D5), and cells added by an agent that knows when the task changes do not exist (`WHAT_A_NEURON_HAS`, item 4.8); a tissue without feedback cells could not go on to step 2. Consequence, registered here and reported: a silent feedback cell's output is 0, so every connection from it has ū = 0 and is given up by formula 11 when it is 128 steps old; connections from feedback cells exist only transiently after step 128 (section 15, expectation X2). This is the definition's rule acting, and the brief of step 2 must take it into account.
2. **Form (b): emission lines after giving up (item 2).** Step 1 has no form (b) cells. For a tissue of form (b): under E4 a feedback cell does not make connections, so the question would be whether its emission lines can be given up; the decision is that a feedback cell's emission lines, like its feedback line, are never given up. A form (b) tissue must have its 64 further feedback cells from birth, so it needs a step-1 phase of its own; whether step 2 runs one is for the brief of step 2.
3. **The kind of a magnitude feedback cell (item 3).** The magnitude line carries 1 − d/2, larger when the answer is nearer; it plays the part of "yes". A magnitude feedback cell therefore takes the kind of a yes-cell: inhibitory with formula 7, excitatory where E3 reverses the kinds. Unused at steps 1 and 2 (decision D3).
4. **How the feedback cells are chosen (item 4).** At random, by `rb.sample`, after the coordinates and lines are drawn and before the output-end cells are assigned; the first 32 drawn are yes-cells (section 5.3, item 4).
5. **E4 against E5.** E4 gives a feedback cell its feedback line only; E5 allows a feedback cell near the input lines to draw them. With both, a feedback cell would carry the byte, which is what E4 removes (H3). Decision: E4 holds in the first form; E5 changes the term only; a feedback cell drawing input lines is a further form of C56.
6. **"Within distance 1/8"** is read as squared distance at most 1/64.
7. **Ages and the first making.** Starting connections count as made at step 0, so the first giving up and the first making are possible at step 128, byte 4 of life (sentence 57).
8. **Order inside operation 7.** All giving up first, then the makings in increasing index of the cell, each drawing from the members that are not its sources after the giving up.
9. **Line connections and giving up.** Formula 11 applies to connections from input lines: their strength stays 1/8, and they are given up when their ū falls below 1/128. A line of a byte value that never occurs (210 of the 256) has ū = 0, so every connection from it is given up 128 steps after it was made (section 15, expectation X1).
10. **Emission at step 1.** The emission is read and recorded at every position (sentence 69); nothing is compared and no feedback line is set.
11. **Recording.** At the reading step (C55), after operation 4.

## 6. The arms, the seeds and the tissues

The seeds are the project's three: 73129, 93683 and 116089. From a seed are drawn, by the generator `Random(seed)` (section 5.3): the coordinates, the positions of the lines, the feedback cells and their lines, the kinds and the starting sources; by `Random(seed + 500000)`: the sources of new connections; and, by the record's generators `Random(seed * 1000 + 50 + 89 · 7 + code)` and `Random(seed + 77)`, the split and the stream (section 4). The three tissues of one seed therefore have the same birth and read the same stream.

| Tissue | Rule of strength | Per seed | Command code |
|---|---|---|---|
| First-form tissue | Formula 7 at all connections between cells | 1 | `first` |
| Coincidence tissue | Formula 7 into encoder and output-end cells; section 5.5 into inner cells; feedback kinds reversed (E3) | 1 | `coincidence` |
| Fixed tissue | None; no giving up, no making | 1 | `fixed` |

Round 1 is nine tissues: three per seed.

## 7. What is recorded and how it is read

The tissue is not touched by any measurement: outputs are recorded, and the analysis below is the experimenter's, carried out on the records after the run. No part of it acts on the tissue (`WHAT_A_NEURON_HAS`, item 4.9).

### 7.1 Records

For each tissue:

1. At the first step of each recording window: the distance d of every cell (breadth-first along the present connections, from the input lines), and the list of cells at d = 1, 2 and 4.
2. At the reading step of each position of the window: the recorded output of every cell in that list (float16), the position, and the present byte.
3. For every cell, over each window: the number of positions with recorded output above 0.01, the mean and the standard deviation (population) of the recorded output, the mean θ.
4. At each window's first step: the number of connections by kind of receiving cell (encoder, inner, output-end) and kind of source (input line, excitatory cell, inhibitory cell, feedback cell), with their mean strength; the number of connections given up and made since the previous window; the mean number of connections of a cell; the number of distinct byte values among the input lines that have at least one connection; the output-end cells taking a yes-cell and a no-cell as a source (E2).
5. At every position of the stream: the emission (int16, −1 for none).
6. At the end: the full state (section 11.3).

### 7.2 Silent and constant

In a window, a cell is **silent** if its recorded output exceeds 0.01 at fewer than 64 of the 8,192 positions (a share below 1/128). It is **constant** if it is not silent and the standard deviation of its recorded output over the window is below 0.01 (the mark used in `CELLS_IN_USE_IN_LIMITED_NETWORKS_20261117.md`). For each d in {1, 2, 4} the measure is the share of the cells at distance d that are neither silent nor constant.

### 7.3 The classifier

For one tissue, one window, one distance d and one lag L:

- *Samples*: the positions t of the window with t − L ≥ 0 (in W0 the first L positions are left out). The features of t are the recorded outputs at t of all cells at distance d (all of them, whether silent or not), as float64. The label of t is the byte at position t − L, taken from the stream.
- *Folds*: the window is cut into four consecutive stretches of 2,048 positions. For each stretch in turn, it is the test set, and the fitting set is every other sample of the window except those within 16 positions of the test stretch (positions from its start − 16 to its end + 15 are left out of the fitting set).
- *Fitting*: on the fitting set, the mean and the population standard deviation of each feature; features with a standard deviation below 10⁻⁹ are dropped; each sample is standardised. For each byte value that occurs among the fitting labels, its centroid is the mean of the standardised fitting samples with that label.
- *Test*: each test sample, standardised with the fitting set's means and deviations, is assigned the byte value of the nearest centroid in squared Euclidean distance; ties go to the smaller byte value. A test label that has no centroid counts as wrong.
- *Score S(d, L, window)*: the predictions of the four folds are pooled; for each byte value present among the test labels, the share of its samples assigned rightly; S is the mean of these shares over the byte values present (balanced accuracy). Its standard error is the standard deviation of the four folds' balanced accuracies divided by 2.

**Chance.** A classifier whose assignments do not depend on the label has an expected balanced accuracy of exactly 1/K, where K is the number of byte values present; K = 46 in every window of every seed (section 17, A.6), so chance is 1/46 = 0.0217. The majority rate (always the commonest byte, "=" or the end mark) is 0.0616 to 0.0634 in plain accuracy and also 1/46 in balanced accuracy.

### 7.4 The fixed tissue

The same records and the same classifier give S_fixed(d, L, window) and the shares of section 7.2 for the fixed tissue of the same seed. *What the wiring gives* is S_fixed(d, L, W3); *what learning adds* is S(d, L, W3) − S_fixed(d, L, W3).

### 7.5 The countermodels of the stream

- **C0(L)**: the classifier of section 7.3 fitted, in the same window and folds, to a one-hot code of the present byte (46 features, one per byte value of the window) in place of the cells' outputs. Its values, computed in section 17, part E, are 1.0000 at L = 0, 0.074 to 0.087 at L = 1, 0.050 to 0.059 at L = 2, 0.076 to 0.080 at L = 4 and 0.034 to 0.043 at L = 8.
- **C<L(L)**, for L ≥ 1: the same with a one-hot code of the bytes at lags 0 to L − 1 (46 · L features). It is what a population would give that held every nearer byte exactly and nothing else. The worker computes it; it is not computed here.

### 7.6 The rule of success (fixed before any run)

For one arm and one seed, step 1 is **reached** when all of the following hold:

1. For each d in {1, 2, 4}: in W3, the share of cells at distance d that are neither silent nor constant is at least 1/2.
2. For each d in {1, 2, 4} and each L in {0, 1, 2, 4, 8} (15 pairs): S(d, L, W3) − S(d, L, W0) ≥ 0.02, or S(d, L, W3) ≥ 0.95 and S(d, L, W3) ≥ S(d, L, W0) − 0.01.

The second clause of condition 2 is for scores that are at their ceiling at the start (as the present byte at d = 1 is expected to be); "better" cannot be shown there, and not falling is required instead. The margin 0.02 is about three standard errors of a difference of two scores of this size (the standard error of one score is about 0.005 when 46 classes have about 100 to 500 test samples each).

**Step 1 is reached by an arm** when it is reached on all three seeds. **Step 1 is reached** when at least one arm reaches it. The report counts, per arm and seed, how many of the 18 conditions (3 of condition 1, 15 of condition 2) hold.

### 7.7 Readings stated beside the rule (not part of it)

For each arm, seed, d and L the report states, in these words:

- "Learning added to what the wiring gives" when S(d, L, W3) − S_fixed(d, L, W3) ≥ 0.02; "learning removed from what the wiring gives" when it is ≤ −0.02; otherwise "no difference from the fixed tissue was found".
- For L ≥ 1: "beyond the present byte" when S(d, L, W3) ≥ C0(L) + 0.02, otherwise "not beyond what the present byte gives"; and "beyond every nearer byte" when S(d, L, W3) ≥ C<L(L) + 0.02.
- If step 1 is reached by an arm but the fixed tissue meets the same 18 conditions on the same seeds, the sentence is: "The [arm] reached step 1; the fixed tissue of the same seeds met the same conditions, so the conditions do not show what learning adds."

## 8. Checks before any run

Each check writes `TISSUE_STEP1/qual/check<n>.json` with an entry `pass`. If a check fails, the worker stops, reports the first difference and asks. An interrupted check is made again from its beginning.

1. **The worked example.** The program's own functions for formulas 1 to 12 and sentence 118, applied to the four cells F, E, I and O of section 13 of the definition with the starting values given there (outside sources held at the stated outputs, u at 0.5 for steps 0 to 15 and 0.125 from step 16, the no line at 1 at step 0 only, ages 0), reproduce, to four decimals, (a) every row of the table of section 13 and the final strengths O: u 0.75099, f 0.24949, F 0.25014, I: E 0.5003, q 0.2510, g 0.2495, with F as revision 4 writes it; (b) the rows and final strengths of section 17, C.2, with F as E4 makes it (feedback line only, θ = 0, no formula 7 or 9 at F); (c) those of section 17, C.3, as the coincidence tissue makes it (E4; F inhibitory by E3; the coincidence rule at the inner cell I, ϑ_I starting at 0). Variant (a) catches a wrong order of operations (traces or p updated before formula 7 is applied; x(k) in place of x(k − 1)); variant (c) catches a wrong sign of the coincidence rule and of E3.
2. **The births.** For each seed, the birth of section 5.3 gives the digests and counts of section 17, B.1: the digest of the coordinates (`sha256(json.dumps(P))[:16]` of the list of [x, y] pairs), of the sources (`sha256(json.dumps([sorted(src_i) for i in range(2048)]))[:16]`, lines numbered 2,048 + v), and of `[FB, drivers, inhibitory flags]`; the numbers of cells of each kind; the neighbourhood sizes; the 15,872 starting connections and those from lines; and the counts of cells at each distance from 1 to 11 and not reached (64, the feedback cells). The births of the three tissues of a seed are equal in every array except the kinds of the 64 feedback cells in the coincidence tissue.
3. **The stream.** For each seed, the SHA-256 of the first 262,144 bytes equals that of section 17, A.4; its first 48 bytes are those printed there; it holds 46 byte values; every item parses as task byte, two digits, sign, 0 to 12 capitals, two digits, "=", two digits and the byte 10, and its answer is right; every pair is in the taught half of the split.
4. **The sign of the rules.** The program's rule functions give the values of section 17, part D, to six decimals: formula 7 gives −0.000750 for σ = +1, m = +0.1, x(k − 1) = 0.3, e = 0.2, and the three other signs as printed; the coincidence rule gives +0.001500 for ϑ = 0.1 and −0.001500 for ϑ = 0.5; both give 0 for x(k − 1) = 0 or e = 0.
5. **No reset, and nothing carried that should not be.** Two first-form tissues are born from seed 73129 and run on two streams that differ only in the byte at position 5 (a letter or digit of the first item replaced by another byte value in use). At the reading step of position 40, at least one cell's v differs between them by more than 10⁻¹²; so the state carries information across item boundaries. In addition the report states that the program contains no assignment that sets a cell's or a connection's state after birth other than by the formulas.
6. **Silent feedback and E4.** Over the first 2,048 bytes of a first-form tissue (seed 73129): both feedback lines are 0 at every step; every feedback cell has v = x = θ = 0 and exactly one source (its line); no feedback cell ever makes a connection; no connection from a line changes strength.
7. **The fixed tissue.** After 2,048 bytes of the fixed tissue (seed 73129), the list of connections and every strength are bit-equal to the birth.
8. **Saving and resuming.** A first-form tissue on seed 73129 run for 4,096 bytes without interruption, and the same run with `--ckpt-bytes 1024 --die-after-byte 1700` (exit code 9) followed by the same command with `--resume` in place of `--die-after-byte 1700`, give bit-equal final states (every array, both generators' states, the stream position) and equal records.
9. **The analysis.** (a) Applied to a one-hot code of the present byte in W0 and W3 of each seed, the analysis program gives C0(L) equal to section 17, part E, to four decimals. (b) Positive control: applied to a one-hot code of the bytes at lags 0 to 8 together (414 features) it gives S ≥ 0.99 at every lag. (c) Negative control: applied to 200 features drawn from `numpy.random.default_rng(1)` standard normal, independent of the stream, it gives S within 0.0217 ± 0.01 at every lag.
10. **Cost.** Seconds of CPU per byte for each of the three tissues on seed 73129 over bytes 256 to 2,303 (the first 256 bytes not counted), with the mean number of connections of a cell at the end. The forecast for round 1 is 262,144 bytes times the measured cost per byte of each tissue, times 2 for the first-form and coincidence tissues (connections may grow; section 9), summed over the nine tissues. If the forecast exceeds 40 CPU-hours, the worker asks before starting. The check passes if every timing is finite and checks 1 to 9 passed.

## 9. Size and cost

**Size.** 2,048 cells, 32 steps a byte, 15,872 starting connections (7.75 a cell on average; feedback cells have none), neighbourhoods of 26 to 166 members (mean about 96), so that a cell's number of connections can grow to at most its neighbourhood's size. The stream of a tissue is 262,144 bytes, 8,388,608 steps, 65,536 periods of P_s.

**How the cost was estimated.** The program holds the connections as flat arrays (one entry per connection: sending index, receiving index, w, e, ū, step of making) in float64 and int64, computes arrivals with `numpy.bincount(receiver, weights=σ[sender] · w · z[sender], minlength=2048)`, and removes and appends connections only at steps when one is given up or made. An operation here is one arithmetic operation or comparison on one element. Counted from formulas 1 to 12: about 20 operations a connection a step in a plastic tissue (arrival 3, formula 7 or the coincidence rule 5, bounds 2, trace 3, ū 3, giving-up test 3, gathering 1) and about 4 in the fixed tissue; about 25 operations a cell a step. The time is taken as about 80 numpy calls a step at about 3 microseconds each, plus about 1.5 nanoseconds an operation (40 calls in the fixed tissue). These two rates are the brief writer's general knowledge of numpy on one core for arrays of 10⁴ to 10⁵ elements; they were not measured here (numpy is not used by the check), and check 10 measures the actual cost before any run.

| Mean connections a cell | Operations a byte (plastic) | Time a byte (plastic) | CPU-hours a plastic tissue | CPU-hours a fixed tissue |
|---|---|---|---|---|
| 8 (birth) | 12.1 million | 26 ms | 1.9 | 0.7 |
| 12 | 17.4 million | 34 ms | 2.5 | 0.8 |
| 32 | 43.6 million | 73 ms | 5.3 | 1.4 |
| 90 (every neighbourhood full) | 119.6 million | 187 ms | 13.6 | 3.0 |

Round 1 (six plastic and three fixed tissues) is 17.1 CPU-hours at a mean of 12 connections and 36.0 at 32; about 3.1 · 10¹³ operations at 12. **Ceiling: 40 CPU-hours for round 1.** With three processes at a time, round 1 takes about 6 to 12 hours of wall time. The number of connections of the plastic tissues is not known in advance (cells below their target activity add one connection every 128 steps; section 15, X3), so the run itself re-forecasts at the end of W1 (section 10.2).

**Files saved, and their size.** Free disk on the worker is small; the worker checks it before the runs (`df`) and asks if less than 2 GB is free.

| What | Size |
|---|---|
| Records of section 7.1, item 2 (float16, `numpy.savez_compressed`, one file a window) | about 10 MB a window, 42 MB a tissue, 374 MB for round 1 before compression if the counts of cells at d = 1, 2, 4 stay near birth's (614 to 634); upper bound 1,208 MB if every cell were at one of those distances |
| Per-cell summaries and emissions (items 3 to 5) | about 7 MB for round 1 |
| Final state and saved state of a tissue | 1.3 MB at 12 connections a cell, 8.3 MB at 90 |
| Receipts and logs | under 1 MB a tissue |

If the compressed records of round 1 exceed 400 MB, the records of W1 and W2 are deleted after their measures are in the receipts, and only those of W0 and W3 are committed; the report says so.

## 10. Budget, stopping, order of opening, and the report when the step is not reached

### 10.1 Budget

Round 1: the nine tissues of section 6, at most 40 CPU-hours. Opening 2 (section 10.3), if run: three tissues, at most 20 CPU-hours. In all at most 60 CPU-hours. Nothing else is run under this brief.

### 10.2 Rule for stopping

1. Any check fails: stop and ask.
2. The forecast of check 10 exceeds 40 CPU-hours: ask before starting.
3. At the end of W1 (position 32,767), each plastic tissue writes the CPU time so far and its forecast for the rest (its own rate of the last 8,192 bytes times the remaining bytes, times 1.5). If the sum of the forecasts of round 1 exceeds 40 CPU-hours, the worker lets the running tissues reach their next saved state, stops them, and asks.
4. After round 1: if at least one arm reaches step 1 (section 7.6), the work under this brief ends; the report is written. If neither does, opening 2 is run, and then the work ends, whatever its outcome.
5. A tissue is never stopped early because of its measures; every tissue reads its full stream.

### 10.3 The order of opening (definition, section 15, step 1, as changed by erratum E1)

Each opening is one stated change from round 1's tissue, run on all three seeds, in this order:

| Order | Choice | The one change | Under this brief |
|---|---|---|---|
| 1 | C15, second form | The coincidence rule at inner cells, with E3 | Already run from the start, as the coincidence tissue (E1) |
| 2 | C29 | Synaptic scaling in place of the threshold control: θ stays 0 for every cell (formula 9 is not applied), and after operation 5 of every step every strength from a cell into cell i (not a feedback cell) is multiplied by 1 + (1/128)(1/8 − x_i(k)), then formula 8. The first further form of C29, "exempt output-end cells", is passed over because step 1 reads no output-end cell | Run if round 1 does not reach step 1, on the arm that met more of the 54 conditions (18 per seed) in round 1, the first-form arm on a tie. Its comparison is round 1's fixed tissue (scaling is a change of strength) |
| 3 | C17 and C18 | Giving up by the product of trace and mismatch near 0; making toward coincident activity through a chemical signal | Not run: the definition gives no number for the first and the second needs C26 to C28, which are opened at step 3. Needs an exact form from the reviewer and a new release |
| 4 | C11 with C24 | τ_v = 1 with s = 16 (the form "16 with τ_v = 1" of C24); the lags of four and eight bytes then lie 64 and 128 steps back | Not run under this brief |
| 5 | C30 | One dimension | Not run |
| 6 | C16 | Traces with τ = 64 | Not run |
| 7 | C3 | 1/8 inhibitory | Not run |
| 8 | C6 | Starting number of connections 4, then 16 | Not run |
| 9 | C1 with C13 | Two parts, one predicting the other | Not run |
| 10 | C2 | A retention gate | Not run |

Whenever an opening changes the sign with which a rise in arrival acts on active excitatory inputs, the kinds of the feedback cells are reversed with it (E3).

### 10.4 What is reported when the step is not reached

The sentence "Step 1 was not reached within its budget" with the arms and openings run; for every tissue the 18 conditions with their numbers and which failed, by d and L; the readings of section 7.7; the shares of cells silent and constant at every window and d; the counts and mean strengths of connections by kind at every window (in particular the mean strength of excitatory connections into inner cells, which bears on H1); the number of distinct byte values with an input-line connection at every window; and the comparison with the registered expectations of section 15. Mr. Xu decides whether the route continues, changes or stops there (roadmap, section 5, condition 8).

## 11. The program, saving and resuming, receipts and files

### 11.1 Files

All under `TISSUE_STEP1/` on the worker's own branch:

- `tissue_step1.py`: the tissue, the stream and the runs. `python3 tissue_step1.py run <first|coincidence|fixed> <seed> [--bytes N] [--out DIR] [--ckpt-bytes N] [--resume] [--die-after-byte N] [--opening 2]` and `python3 tissue_step1.py check <1..10>`. It is committed before the first check is run.
- `step1_read.py`: the analysis of section 7 on the records of a tissue (also used by check 9); it writes the measures into the tissue's receipt.
- `step1_tables.py`: makes every table of the report from the receipts.
- `runs/<tissue>_<seed>/`: `receipt.json`, `W0.npz` to `W3.npz`, `summary.npz`, `emissions.npy`, `final_state.npz`, `log.txt`.
- `qual/check<n>.json` and their logs.
- `STEP1_REPORT.md`.

### 11.2 Receipts

`receipt.json` of a tissue records: the brief's file name and the commit of `V` from which it was read; the SHA-256 of `tissue_step1.py` and `step1_read.py`; the versions of Python and numpy; the tissue, the seed and the opening if any; the stream's SHA-256 and length; the birth's three digests; the records of section 7.1, items 1 and 4; every measure of section 7; the CPU and wall time by window; the entry `beside` (the number of other runs in progress, set by the worker at the start); the list `resumes` (byte and time of every resumption).

### 11.3 Saving and resuming

A saved state is written every 16,384 bytes (`--ckpt-bytes`, default 16,384; 0 for none) and at the end of each window, at the end of a byte's window (after operation 8 of φ = 31): every array of the cells and connections, the step, the stream position, the states of both generators (`getstate()`), the step of each cell's last making, the records so far and the CPU counters. It is written under another name and renamed with `os.replace`. With `--resume` and a saved state present, the run loads it, asserts that the stream's SHA-256 of the bytes read so far equals the saved one, and goes on from the next byte, appending to `resumes`; with no saved state it starts from birth. The saved state is deleted when `final_state.npz` and the receipt are written. `--die-after-byte N` ends the process with `os._exit(9)` after byte N (check 8 only). A run interrupted before its first saved state is started again after its folder has been moved to `TISSUE_STEP1/runs/interrupted/`.

### 11.4 What is committed

The program, the analysis and table scripts, every receipt, the records (within the limit of section 9), the summaries and emissions, the final states, the logs, the check receipts and the report. The final states are what step 2 continues from.

## 12. Rules for the worker

- Start a branch of your own from the head of the default branch `origin/codex/minifly-consolidated-20261023` at the time of release, and record its commit in the report. Read this brief from the reviewer's branch, fetched by `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'`, with `git show`. The program imports nothing from the repository.
- Install numpy in a virtual environment under Python 3.13. Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). If a script of the worker starts runs from Python, each child is started with an environment from which every variable beginning with `E11_`, `E16_` or `EXP45_` is removed (decision 37).
- This host stops a machine about five minutes after its session's turn ends; the disk is kept and detached processes are terminated. While runs proceed, do not end your turn: issue a command that waits for the processes and repeat it until they have finished; commit and push the receipts and logs between two such commands. Start every run with `--resume` from the first time on, with its output appended to its own log. The report says which runs were resumed, from which byte, and which were started again.
- Push only to your own branch; never to the default branch or any other; open no pull request; read other branches with `git show`.
- Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, no name of a language model appears in a committed file or commit message.
- The report is in standard academic English: full sentences, every term defined at first use, every figure with the receipt or command that gives it, the committed files listed from `git ls-tree`.
- Nothing may be asked of the tissue that a living cell cannot do, and no operation may act on the tissue for a measurement; only outputs are recorded. If the brief cannot be followed at some point, stop that part, say what you found, and ask; go on with whatever does not depend on the answer.

## 13. Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed), with the forecast of check 10; all nine tissues have passed W1, with the re-forecast; round 1 finished, with the 18 conditions per arm and seed; opening 2 finished (if run); the report committed. Each message gives your branch, the commit and the main figures in at most ten lines.

## 14. What the report must state

1. The claim at the level actually earned (E0), in the words of section 3, item 1, or the sentence of section 10.4.
2. The numbers fixed by us (section 5.7, both tables), the numbers of the reading, and the form of every choice used (the list in section 5.7), with any opening named.
3. The errata table of section 5.2 and the decisions of section 5.8, as applied.
4. For every tissue: the counts at birth (check 2); the records of section 7.1, item 4, at every window; the shares of section 7.2 for d = 1, 2, 4 at every window; S(d, L, window) with its standard error for every d, L and window; C0(L) and C<L(L) for every window; the 18 conditions and whether each holds; the readings of section 7.7.
5. One table per seed with a row for each (d, L) and columns S at W0 and W3 for the two arms and the fixed tissue, C0, C<L and the conditions; one table of the shares of cells neither silent nor constant.
6. The number of distinct byte values of the stream with at least one input-line connection, and the connections from feedback cells, at every window (expectations X1, X2).
7. The emissions: the share of positions with a byte emitted and the five commonest emitted bytes, per tissue and window. They are not scored.
8. The outcome of each registered expectation of section 15.
9. CPU time per tissue, the forecast of check 10 and the re-forecast at W1, against the estimate of section 9.
10. That the result is of three seeds and one stream, that no feedback was given, and the limits of section 16.
11. The list of committed files and the commit of the default branch the work started from.

## 15. Registered expectations (before any run)

The fifth reading states, for step 1 (its section 5 and risk H1): encoder cells stay informative through their unchanging lines; "for the inner cells read at two and four connections, and for the lags of four and eight bytes, the first form is likely to fail"; "it is not impossible on paper"; the reader's confidence is moderate. The mechanism (H1): at an active cell, formula 7 weakens excitatory inputs in proportion to the covariance of their traces with the cell's activity, so that inner cells, which have no unchanging input, lose the variation that carries the byte; and the bytes four and eight back lie 128 and 256 steps back, where the traces keep 0.0172 and 0.0003 (section 17, F.1), so they could be held only by sustained recurrent activity, which the same rule cancels. The fifth reading gives no paper analysis of the coincidence form. The expectations below are the brief writer's reading of H1 and of the definition; each is scored in the report as held or not held.

| No. | Tissue | Measure | Expectation | Chance it holds |
|---|---|---|---|---|
| X1 | First-form and coincidence | Input-line connections | Connections from the lines of the 210 byte values that never occur are all given up at step 128 and every 128 steps after being re-made; connections from the lines of rare bytes (letters, task bytes, signs) are given up after about 13 bytes without that byte (section 17, F.3), which is the case for 72 to 90 per cent of the gaps between occurrences (F.5). At each window's start, fewer than half of the 46 byte values in use have an input-line connection that is more than 128 steps old | 0.7 |
| X2 | All three | Connections from feedback cells | At the start of W1, W2 and W3 no connection from a feedback cell is older than 128 steps (silent lines give ū = 0) in the plastic tissues; in the fixed tissue the 423 to 465 starting ones remain | 0.95 |
| X3 | First-form | Number of connections | The mean number of connections of a cell at W3 is above 12 (cells pushed below their target activity by H1 keep adding connections) | 0.6 |
| X4 | First-form | Mean strength of excitatory connections into inner cells, W3 against birth (1/8) | Lower (H1) | 0.75 |
| X5 | Coincidence | The same | Not lower | 0.5 |
| X6 | First-form | Neither silent nor constant, d = 2 and 4, W3 | Share ≥ 1/2 at both | 0.5 |
| X7 | Coincidence | The same | Share ≥ 1/2 at both | 0.7 |
| X8 | All | S(1, 0, W3) | ≥ 0.95 in the fixed tissue; below 0.95 in at least one plastic tissue of the six, through the giving up of line connections (X1) | 0.5 |
| X9 | First-form | S(d, 0, W3) − S_fixed(d, 0, W3) at d = 2 and 4 | ≤ −0.02 on at least two seeds ("learning removed", H1) | 0.55 |
| X10 | Both arms | S(d, L, W3) at L ≥ 1, all d | Not beyond the present byte (≤ C0(L) + 0.02) in at least 9 of the 12 pairs (d, L ≥ 1), per arm and seed | 0.6 |
| X11 | Both arms | S(d, 4, W3) and S(d, 8, W3) at d = 2, 4 | Below S(d, ·, W0) + 0.02, so condition 2 fails at the lags of four and eight bytes (H1) | 0.7 |
| X12 | First-form | Step 1 reached (section 7.6) on all three seeds | Reached | 0.05 |
| X13 | Coincidence | The same | Reached | 0.10 |
| X14 | Fixed | The 18 conditions on all three seeds | Met (W0 starts with θ = 0, before the threshold control has settled, so a late window may score higher without learning) | 0.15 |
| X15 | — | Checks 1 to 10 pass at the first attempt | Pass | 0.5 |

## 16. What step 1 cannot show

- Anything about answers, correction, keeping or generalisation: no feedback is given and nothing is scored.
- Whether the cells carry the bytes in a form that later steps can use: the classifier is the experimenter's and is fitted to many records; a living cell has no such classifier.
- Anything beyond three seeds, one stream and the first form (or the one change of an opening).
- How the feedback cells' connections fare when the lines are active (step 2); at step 1 they are given up (section 5.8, item 1).
- Whether the present byte is carried by the tissue's learning or by its wiring, except through the fixed tissue, which keeps the threshold control.

## 17. Output of `check_step1_brief.py`

`python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_step1_brief.py`, run at 00:08 UTC on 19 November 2026 (23 seconds):

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
A.7 shares of the byte values over the whole stream (seed 73129), rarest five and commonest five: b'Z' 0.0139, b'L' 0.0141, b'O' 0.0142, b'W' 0.0142, b'Y' 0.0142 ... b'1' 0.0399, b'6' 0.0404, b'2' 0.0406, b'=' 0.0624, b'\n' 0.0624
A.8 recording windows: W0 bytes 0 to 8191; W1 bytes 24576 to 32767; W2 bytes 122880 to 131071; W3 bytes 253952 to 262143; steps of the stream 8,388,608
==============================================================================================================
B. THE BIRTH (first form, 64 feedback cells with no sources, erratum E4)
B.1 seed 73129: digest of coordinates adb76cb771d012de; of the sources c0638c096619aecb; of feedback cells, drivers and kinds 66952e5b022b0422
    cells: output-end 256, feedback 64 (yes 32, no 32), encoder 240, inner 1488; inhibitory 397
    neighbourhood members (cells and lines) smallest 32, mean 96.9, largest 154; cells only: mean 90.9; cells with fewer than 8 members: 0
    starting connections 15872; from lines 754, share of lines in the encoder cells' draws 0.393; smallest x of an output-end cell 0.8432
    cells by distance from the input lines (d: count): 1: 232, 2: 189, 3: 199, 4: 193, 5: 201, 6: 185, 7: 179, 8: 171, 9: 198, 10: 196, 11: 41; not reached: 64
    cells recorded (d = 1, 2, 4): 232 + 189 + 193 = 614
    output-end cells at distance 8 to 11 from the input lines (not reached: 0)
    cells taking a feedback cell as a source at birth: 383; of them output-end 60 (taking a yes-cell 47, a no-cell 16), encoder 27; connections from feedback cells 423
B.1 seed 93683: digest of coordinates 3b7f27e7070b57ff; of the sources 8e946fac980830fc; of feedback cells, drivers and kinds 988ddd6833af5c1a
    cells: output-end 256, feedback 64 (yes 32, no 32), encoder 244, inner 1484; inhibitory 437
    neighbourhood members (cells and lines) smallest 27, mean 95.3, largest 162; cells only: mean 89.4; cells with fewer than 8 members: 0
    starting connections 15872; from lines 833, share of lines in the encoder cells' draws 0.427; smallest x of an output-end cell 0.8387
    cells by distance from the input lines (d: count): 1: 238, 2: 175, 3: 176, 4: 206, 5: 179, 6: 191, 7: 197, 8: 196, 9: 217, 10: 200, 11: 9; not reached: 64
    cells recorded (d = 1, 2, 4): 238 + 175 + 206 = 619
    output-end cells at distance 8 to 11 from the input lines (not reached: 0)
    cells taking a feedback cell as a source at birth: 419; of them output-end 53 (taking a yes-cell 28, a no-cell 27), encoder 30; connections from feedback cells 460
B.1 seed 116089: digest of coordinates 0121fb1fafc58c37; of the sources 503c1f17014882ab; of feedback cells, drivers and kinds 655770c0dec995ca
    cells: output-end 256, feedback 64 (yes 32, no 32), encoder 262, inner 1466; inhibitory 405
    neighbourhood members (cells and lines) smallest 26, mean 96.7, largest 166; cells only: mean 90.2; cells with fewer than 8 members: 0
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
     1 | 0.2550 -0.0025 -0.0549 0.0205 0.0625 -0.0552 0.0186 0.1850 -0.1275 0.2434
     2 | 0.1275 0.0613 0.0106 0.0319 0.0415 -0.0745 0.0063 0.2508 -0.0577 0.1951
     3 | 0.0638 0.0931 0.0422 0.0539 0.0472 -0.0664 0.0035 0.2832 -0.0235 0.1870
     8 | 0.0020 0.1238 0.0637 0.1044 0.0818 -0.0248 0.0345 0.3142 0.0080 0.2055
    16 | 0.0000 0.1243 0.0498 0.1091 0.0859 -0.0158 0.0477 0.0321 -0.2760 0.0620
    24 | 0.0000 0.1239 0.0383 0.1098 0.0863 -0.0119 0.0529 0.0333 -0.2132 0.0000
    31 | 0.0000 0.1237 0.0304 0.1103 0.0867 -0.0092 0.0572 0.0333 -0.1707 0.0000
    final strengths: {"F": {}, "E": {"h": 0.25432, "c": 0.49784, "F": 0.25037}, "I": {"E": 0.50051, "q": 0.25172, "g": 0.25086}, "O": {"u": 0.75804, "f": 0.24597, "F": 0.24981}}
==============================================================================================================
D. ONE-STEP TESTS OF THE SIGN OF THE TWO STRENGTH RULES (w = 0.5, e = 0.2, x(k-1) = 0.3)
D.1 formula 7: sigma +1, m +0.1: delta w = -0.000750
D.1 formula 7: sigma +1, m -0.1: delta w = +0.000750
D.1 formula 7: sigma -1, m +0.1: delta w = +0.000750
D.1 formula 7: sigma -1, m -0.1: delta w = -0.000750
D.2 coincidence rule: theta-bar 0.1: delta w = +0.001500 (the same for both kinds of sending cell)
D.2 coincidence rule: theta-bar 0.5: delta w = -0.001500 (the same for both kinds of sending cell)
D.3 with x(k-1) = 0 both rules give 0; with e = 0 both give 0
==============================================================================================================
E. THE COUNTERMODEL C0: THE CLASSIFIER OF SECTION 7 FITTED TO A ONE-HOT CODE OF THE PRESENT BYTE ALONE
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
F.6 first possible making and giving up at step P_s = 128 = byte 4 of life; P_s periods in the stream: 65,536
==============================================================================================================
G. COST AND FILES
G.1 mean connections   8: plastic tissue 378,880 operations a step, 12,124,160 a byte, 0.81 ms a step, 25.9 ms a byte, 1.88 CPU-hours for the stream; frozen tissue 3,735,552 a byte, 9.4 ms a byte, 0.69 CPU-hours
G.1 mean connections  12: plastic tissue 542,720 operations a step, 17,367,040 a byte, 1.05 ms a step, 33.7 ms a byte, 2.46 CPU-hours for the stream; frozen tissue 4,784,128 a byte, 11.0 ms a byte, 0.80 CPU-hours
G.1 mean connections  32: plastic tissue 1,361,920 operations a step, 43,581,440 a byte, 2.28 ms a step, 73.1 ms a byte, 5.32 CPU-hours for the stream; frozen tissue 10,027,008 a byte, 18.9 ms a byte, 1.37 CPU-hours
G.1 mean connections  90: plastic tissue 3,737,600 operations a step, 119,603,200 a byte, 5.85 ms a step, 187.1 ms a byte, 13.62 CPU-hours for the stream; frozen tissue 25,231,360 a byte, 41.7 ms a byte, 3.04 CPU-hours
G.2 round 1 (six plastic and three frozen tissues): 17.1 CPU-hours at a mean of 12 connections, 36.0 at 32; ceiling 40 CPU-hours
G.3 total operations of round 1 at a mean of 12 connections: 3.108e+13
G.4 records: 4 windows x 8192 bytes x the cells at d = 1, 2, 4 at the window's start (634 at birth in the largest of the three draws) x 2 bytes (float16): 10.4 MB a window, 41.5 MB a tissue, 374 MB for round 1 before compression, if the counts stay as at birth; upper bound if every cell were at one of these distances 1208 MB
G.5 per-cell summaries (4 numbers a cell a window, float64) and emissions (int16 a byte): 7.1 MB for round 1
G.6 saved state at a mean of 12 connections: 24,576 connections, 1.28 MB a tissue
G.6 saved state at a mean of 90 connections: 184,320 connections, 8.31 MB a tissue
```

## References

- Bienenstock, Cooper and Munro (1982), Journal of Neuroscience 2: 32-48.
- Vogels, Sprekeler, Zenke, Clopath and Gerstner (2011), Science 334: 1569-1573.
- As cited in the definition. Textbook facts (NMDA-receptor-dependent plasticity, homeostatic plasticity) are not referenced. Page numbers are from memory and should be checked before citation.

## History

Revision 1 (19 November 2026, from 00:03 UTC): written from roadmap draft 3, revision 4 of the definition and the fifth reading, for the coordinator's release. Not yet read by a second agent.
