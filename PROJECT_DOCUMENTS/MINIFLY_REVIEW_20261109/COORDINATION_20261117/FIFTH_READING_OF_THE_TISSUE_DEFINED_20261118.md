# Fifth reading of "The tissue defined, revision 4" (18 November 2026, written from 23:50 UTC by the clock of this machine)

## Summary

**Verdict: passes after the listed changes.** All the remaining changes are rewordings or single-sentence changes to the first form. The briefs of steps 1 and 2 can carry them as the errata table in section 7, without a revision 5.

1. **Earlier findings.** Of the fourth reading's findings, G1, G3 and G5 to G9 are settled. G2, G4 and F6 are partly settled: they are restated and named, but their substance remains (findings H1 and H2). F12 and all nine undecided items are settled.
2. **Checks by my own arithmetic.** The worked example agrees to four decimals at every row given, for the feedback cell and all three other cells. The mark counts agree: 146 sentences (28 [S], 58 [F], 60 [C]) and 52 choices. My counts of end cells reached from feedback cells agree within the spread of draws.
3. **Formula 7.** It is a function of both ends of the connection, as section 16 requires. It is net anti-Hebbian: on average it weakens an active cell's excitatory inputs in proportion to how far their traces vary with its activity. At inner cells, all of whose inputs are plastic, this removes the variation that lasts a window, and byte information is exactly that. This is the writer's stated risk, and on paper it is real for the cells at two and four connections that step 1 reads.
4. **Feedback cells after a time.** Under the activity control (formula 9) and the growth rule (formula 12), the feedback cells grow ordinary connections until their output is mostly context. This changes what reaches the ends.
5. **The pairing of kinds.** "No-cells excitatory, yes-cells inhibitory" is correctly marked as our choice. It presupposes the cancelling sign of formula 7, so it must be reversed whenever C15 is opened to its second form. The order of opening does not say so.
6. **Reach.** At the start only 20 to 32 of the 256 output-end cells take a yes-cell as a source, so for most byte values the right cell cannot be strengthened. Learning can start from a tissue that is nearly always wrong, but mainly by elimination.
7. **Section 16's own wording.** "Only some cells of the input end receive the yes or no" is met only in a narrower sense: no encoder cell receives it, because feedback cells are excluded from the input end by definition.

## 1. Terms

- *Anti-Hebbian*: a strength change of the opposite sign to the product of sending and receiving activity.
- *Elimination*: learning in which the cells that emitted wrongly become less driven in that context, while nothing makes the right cell more driven.
- *Background change*: the change formula 7 makes at a connection in a window apart from the steps at which a feedback cell's output arrives.
- *No-cell* and *yes-cell*: a feedback cell that receives the no line or the yes line.

All other terms are those of revision 4.

## 2. Findings in order of seriousness

### H1. Formula 7 removes from inner cells the variation that carries the byte (sentences 41, 49, 93; C15; step 1)

At an active cell, formula 7 changes w_ij by −η · σ_j · m_i · x_i(k − 1) · e_ij. Here m_i is the arrival minus its running average (τ = 32), and x_i follows the arrival.

- An excitatory input that turns on with a byte raises the arrival for the whole window of 32 steps. The prediction catches up only over that time, so m stays above 0 while the input's trace rises. The input is weakened.
- When the input turns off, m falls below 0 while the trace decays, and the input is strengthened.
- Without the factor x the two halves nearly balance: over one window on and the following decay, the sums are about +6.3Δ and −6.6Δ for a rise Δ.
- The factor x_i(k − 1) gives more weight to the half in which the cell is more active, which is the "on" half. The net change is therefore a weakening, in proportion to the covariance of the input's trace with the cell's activity.

**The resulting state.** Formula 7 stops changing a connection on average when the cell's activity-weighted mismatch is uncorrelated with that input's trace. Excitatory inputs that drive the cell are pushed toward zero and are then given up (formula 11). They are replaced only at random, when the cell becomes under-active (formula 12).

**Where this matters.**

- Encoder cells keep their line inputs, which do not change (sentence 63). This is the arrangement of the cerebellum-like counterpart, in which unchanging afferents carry the signal.
- Inner cells have no unchanging input, so the byte information they carry is what the rule removes.
- Bytes four and eight back lie 128 and 256 steps back. The traces there keep (31/32)^128 ≈ 0.017 and (31/32)^256 ≈ 0.0003 of their weight, and the state keeps nothing. So those bytes could be held only by sustained recurrent activity, which is again the part the rule cancels.

**Consequence for step 1.** On paper, step 1 is likely to fail at two and four connections from the input lines, and at the lags of four and eight bytes. It is not trivially passed. That it fails is not certain, because thresholds, the bound at zero and inhibitory inputs complicate the balance. The reader's confidence is moderate.

**Proposed wording, in the brief of step 1 (errata E1).** "Step 1 runs the first form and, beside it from the start, a tissue in which C15 takes its second form (coincidence with a sliding threshold) at inner cells, with the kinds of C56 reversed (E3). Each result is read for cells at one, two and four connections, and for each lag separately." Section 15, step 1: move C15 from third place to first.

### H2. The feedback route reaches few output-end cells, and learning begins mainly by elimination (sentences 85 to 93; C40, C56)

**Starting counts**, recomputed over three draws with 64 feedback cells:

| | Revision 4's table | My draws |
|---|---|---|
| Output-end cells taking a feedback cell as a source | 43 to 63 | 44 to 66 |
| Output-end cells at two connections | 160 to 173 | 144 to 161 |
| Output-end cells within three connections | all 256 | all 256 |
| Encoder cells taking a feedback cell as a source | 25 to 38 | 22 to 46 |

In my draws, the output-end cells were split as follows:

| Feedback cells | Output-end cells that take a yes-cell | Output-end cells that take a no-cell | Output-end cells that take both |
|---|---|---|---|
| 64 | 20 to 32 | 15 to 39 | 1 to 5 |
| 128 | 44 to 81 | 30 to 77 | 13 to 17 |

**What a yes or no moves, by hand.**

- **The emitting output-end cell.** A "no" from a no-cell connected to it weakens its context inputs at φ = 1 to 3. A "yes" from a yes-cell strengthens them. Both directions are right.
- **Active output-end cells that did not emit but receive the same feedback.** They move in the same direction. This is wrong if the cell's byte was the right one, as sentence 92 says.
- **Inactive cells.** They do not move.
- **Cells at two or more connections.** They receive the change only mixed with context (sentence 101).

**Size against the background change.**

- In the example, the "no" weakens O's connection from u by 0.0030 + 0.0011.
- The background change later in the same window, driven by the next byte's context acting on the same trace, more than reverses this. O ends the window with u at 0.75099, a net change of +0.0010 against the direction of the correction.
- The feedback's effect therefore exists only in expectation over many positions, and it is of the same size as the background change in one window.
- Sentence 93's "an output-end cell becomes less driven in contexts that are followed by a 'no'" holds in expectation, not at each position.

**Can learning start from a tissue that is nearly always wrong?** Yes, but mainly by elimination, and only at the 15 to 39 output-end cells that take a no-cell. The right cell can be strengthened only if it emitted and takes a yes-cell, which at the start is true for 20 to 32 cells. If the stream uses some twenty byte values, most of their cells have no route for a "yes".

New connections are made only by cells that are under-active with θ = 0 (formula 12). So an output-end cell at its target activity does not acquire a feedback cell. "How it is passed on is learned" then rests on rare random draws.

**Proposed wording (errata E2).**

- Sentence 93: "In expectation over many positions, …; at a single position the feedback's change is of the size of the background change of one window (section 13)."
- Section 15, step 2: keep 128 feedback cells first.
- The brief of step 2 reports the number of output-end cells reached from yes-cells and from no-cells at the start and at the end of life.

### H3. Feedback cells become context cells (sentences 52, 55, 85; C29, C18, C56)

Feedback cells follow formulas 1 to 12 (sentence 85).

- A no-cell is active for about four steps of each 32 when the emission is wrong. In the example its activity is 0.47, 0.22, 0.10 and 0.03, so its mean activity is about 0.026.
- A yes-cell is almost never active early in life.
- Both are therefore below a* = 1/8. Their thresholds fall to 0, and they make a new connection every P_s = 128 steps until their context drive raises their mean activity to 1/8.
- At that point most of a feedback cell's output is context, and the yes or no is a small part of it. In the same way, formula 7 at the feedback cell changes its ordinary inputs with the large mismatch that the feedback line causes (m ≈ 1 at φ = 0).

So what the feedback cells pass on degrades over life, for reasons visible on paper. Mr. Xu's words do not require this. They ask for dedicated cells that receive the feedback.

**Proposed wording for sentence 85 (errata E4).** "…Each receives one feedback line at fixed strength 1 and, in the first form, no ordinary sources. Formulas 7, 9 and 12 do not apply to it, and its threshold stays 0. Feedback cells with ordinary sources are a further form of C56."

The starting counts of section 12 are unchanged, because they count connections from feedback cells.

### H4. The pairing of kinds in C56 is tied to the sign of C15, and the order of opening ignores this (sentences 87, 91; C15, C56; section 15)

Sentence 87 marks the pairing as our choice, gives its reason (sentence 91), and does not claim its living counterparts as findings. All of this is correct.

The pairing presupposes three things:

- the cancelling sign of formula 7: a rise in arrival weakens active excitatory inputs;
- that the receiving cell was active just after the reading step;
- that the receiving cell's own activity drove the emission positively. This holds for the output-end cell that emitted, and for no other cell, because path signs are unknown (sentence 103).

Under C15's second form (coincidence with a sliding threshold), a rise strengthens active inputs, so the pairing must be reversed. Section 15 opens C15's second form at step 1 (third) and at step 2 (fifth) without reversing C56.

**Proposed wording, added to C56 (errata E3).** "Whenever C15 takes a form in which a rise in arrival strengthens active excitatory inputs, the kinds are reversed: yes-cells excitatory, no-cells inhibitory."

### H5. "Only some cells of the input end receive the yes or no" is met only in a narrower sense (sentences 81, 85, 89; terms)

"我们的输入端不是所有的细胞都收到yes/no，只有某些收到" says that at the input end, not all cells receive the yes or no; some do. The dedicated cells of the next sentence ("可以认为", "it can be taken") are one way of meeting this.

The definition's terms make an encoder cell "a cell, not a feedback cell", and a feedback cell draws no input lines. So no cell of the input end receives the yes or no; only cells of another kind do.

Two further points:

- Sentence 89 extends the same route to the output end and correctly marks this as our choice. Section 16 speaks only of the input end, and section 13 lets cells receive feedback directly.
- Nothing lays down which cells of the ends the feedback cells reach, which is right. The reviewer's "What follows" in section 16 ("not given alike to all cells of an end") generalises Mr. Xu's words from the input end to both ends. The definition follows the reviewer's generalisation and says so in sentence 89.

**Proposed wording (errata E5).** Term "Encoder cell": "a cell with at least one input line in its neighbourhood". Sentence 85: "A feedback cell within reach of input lines may also draw them, and is then also an encoder cell. It is one of the 'some' cells of section 16."

### H6. Formula 7 and section 16: what each side holds (sentences 48 to 50, 139; C15, C19)

- The sending side holds e_ij (the trace of its output at the connection), σ_j and ū_ij. A living presynaptic terminal has residual calcium and a history of release; ū_ij also has a counterpart. Note that in living synapses the presynaptic history used by Hebbian plasticity is usually read on the receiving side, through glutamate bound to NMDA receptors. Holding it on either side meets section 16.
- The receiving side holds m_i · x_i(k − 1). Both are whole-cell quantities. They are present at a synapse only through the spread of potential into the dendrite, including back-propagating action potentials. Whether a signed difference between the arrival and its 32-step average is available at a synapse is unsettled. The reader is unsure.
- Sentence 50 ("a function of quantities present at the connection only") should therefore read "of quantities present at the connection, where the receiving cell's mismatch and activity are taken to reach the synapse by the spread of potential (unsettled)".
- Revision 3's formula 8 was already a function of both ends. The change in revision 4 is the factor x_i(k − 1), the rate 1/8 and the withdrawal of formula 14.
- Whether Mr. Xu meant more by "共同决定" ("determined jointly"), for example that the sending cell's own state, not only its output, enters the change, the reader is unsure. This can be put to him in one line.

**Proposed wording (errata E6).** Sentence 50 as above. Section 18, new item: "Whether the sending side's share in section 16 is more than its output trace."

### H7. Marks and stated facts

| Sentence | Present mark | Finding | Proposed |
|---|---|---|---|
| 50 | [F] | Doubtful | H6 |
| 91 | [F] | Holds for φ = 1 to 3 only. The same traces are changed by the rest of the window, which in the example reverses the net change | Add "at the steps at which the feedback cell's output arrives; the net change over the window includes the background change (sentence 93)" |
| 93 | [F] | "The cell's context drive foretells the feedback arrival" does not follow, since the prediction is a running average, not a context-driven quantity. The rest holds only in expectation | H2 |
| 139 | [F] | Same as sentence 50 | "…holds or that are taken to reach its connections (sentence 50)" |
| Section 12, encoder cells at one connection | — | 25 to 38; my draws 22 to 46 | Give a range from more draws, or "about 10 to 20 per cent" |
| 116, table | [F] | Agrees (my value for n = 9 is 0.996 against 0.997, a rounding difference) | None |

All other arithmetic I checked holds:

- (31/32)^32 = 0.362, (31/32)^9 = 0.751, (31/32)^98 = 0.045, (1/2)^31, and 3 · 32 + 2 = 98;
- the first makings at step 128;
- the shares of input lines among an encoder cell's draws.

No sentence asks for something a living cell cannot do.

## 3. Task (1): status of the fourth reading's findings and of the undecided items

| Item | State | Place |
|---|---|---|
| G1 Emission blind to the last byte | Settled | Sentences 115, 116, 119; reach of 0.975 to 0.999 confirmed |
| G2 Two rules undoing each other | Partly settled | One rule (sentences 42, 49, 93). The cancelling part still opposes a rise in context drive, and it outweighs a single feedback in the example (H1, H2) |
| G3 "No" equals silence | Settled | Sentences 86, 109, 111 |
| G4 A shared factor | Partly settled | Stated (sentence 92) and confined by x_i(k − 1). The property remains, and reach is small (H2) |
| G5 Lines changed | Settled | Sentence 63 |
| G6 Two readings of section 15 | Settled | Sentences 94 to 96; C52 |
| G7 Growth at the start | Settled | Sentences 55, 57 |
| G8 Conversion | Settled | Sentence 73; section 18, item 5 |
| G9 Marks and facts | Settled | Sentences 4, 5, 7, 57, 66, 139; section 12 |
| F6 | Partly settled | As G2 and G4 |
| F12 | Settled | Section 15; E3 corrects one entry |
| Undecided items 1 to 9 | Settled | Sentences 86, 63, 55, 86/118, 112, 109, 111, 6, 117 |

## 4. Task (3): recomputation

- **Worked example.** I ran formulas 1 to 12 in the order of sentence 118, from the stated starting values, for F, E, I and O over all 32 steps. Every row given agrees to four decimals (steps 0, 1, 2, 3, 8, 16, 24 and 31), and so do the final strengths: O's u 0.75099, f 0.24949 and F 0.25014; I's E 0.5003, q 0.2510 and g 0.2495.
- **By hand at step 1, cell O.** p = 0.3125, A = 0.4306, m = 0.1181, receiving factor 0.1181 · 0.4063 = 0.0480, Δw_u = −0.0030, Δw_f = +0.0015. At step 1 the connection from F does not change, because its trace is still 0 (it receives F's step-0 output only at step 2).
- **Counts of end cells:** section 2, H2.

## 5. Task (4): steps 1 and 2 by hand

**Step 1.** It is not trivially passed. Cells exist at one, two and four connections. Encoder cells stay informative through their unchanging lines. For the inner cells read at two and four connections, and for the lags of four and eight bytes, the first form is likely to fail (H1). It is not impossible on paper.

**Step 2.**

- No trivial pass exists: no cell has the coming byte at the reading step (sentence 70), and form (b) gives the emitted byte only after the reading.
- A "no" weakens the connections that drove a wrong emission at the emitting cell, if that cell takes a no-cell. A "yes" strengthens those that drove a right one, if it takes a yes-cell.
- Cells that were active but did not emit move in the same direction, which is wrong for the cell of the right byte.
- Learning can start, mainly by elimination, at 15 to 39 output-end cells. Strengthening is possible at 20 to 32 cells (H2).
- Feedback cells drift toward carrying context (H3).
- Under these conditions step 2 is possible on paper but slow. With E2 to E4 (128 feedback cells first, kinds tied to C15, feedback cells without ordinary sources) it starts with about twice the reach and with a feedback signal that does not degrade.

## 6. Tasks (5) and (6): what is still undecided, and new faults

Undecided for a programmer, all of which a brief can settle:

1. Whether step 1's tissue contains the feedback cells with silent lines, or no feedback cells at all.
2. In form (b), from which emission lines a cell draws a new connection after giving one up.
3. The kind of a magnitude feedback cell (sentence 111).
4. How the 64 feedback cells are chosen among the 2,048 cells: before the output-end cells are assigned, and at random.

New faults are those of findings H1 to H7. None asks for something a living cell cannot do.

## 7. Errata that can be attached to revision 4

These are rewordings and changes of one sentence, and a brief can carry them.

| No. | Place | Change |
|---|---|---|
| E1 | Section 15, step 1; brief of step 1 | Move C15 from third place to first. Run the first form beside a tissue with C15's second form at inner cells from the start (with E3). Read each distance and each lag separately (H1) |
| E2 | Sentence 93; section 15, step 2 | "In expectation over many positions …; at a single position the feedback's change is of the size of the background change of one window." Keep 128 feedback cells as the first form opened at step 2. The step 2 brief reports the numbers of output-end cells reached from yes-cells and from no-cells (H2) |
| E3 | C56; section 15 | "Whenever C15 takes a form in which a rise in arrival strengthens active excitatory inputs, the kinds of feedback cells are reversed." (H4) |
| E4 | Sentence 85; C56 | First form: a feedback cell has its feedback line only. Formulas 7, 9 and 12 do not apply to it, and its threshold stays 0. Ordinary sources are a further form (H3) |
| E5 | Term "encoder cell"; sentence 85 | A feedback cell within reach of input lines may draw them and is then also an encoder cell, so that "some cells of the input end" receive the yes or no (H5) |
| E6 | Sentences 50, 139; section 18 | "…taken to reach the synapse by the spread of potential (unsettled)". New item: whether the sending side's share in section 16 is more than its output trace (H6) |
| E7 | Sentence 91 | Add "at the steps at which the feedback cell's output arrives; the net change over the window includes the background change" (H7) |
| E8 | Section 12 | Encoder cells at one connection from a feedback cell: give a wider range or a share (H7) |
| E9 | Brief of step 1 | State whether the feedback cells are present with silent lines (section 6, item 1) |

## References

As cited in revision 4. The remaining facts are textbook: NMDA receptor coincidence detection, back-propagating action potentials, and cerebellum-like cancellation with unchanging afferents.
