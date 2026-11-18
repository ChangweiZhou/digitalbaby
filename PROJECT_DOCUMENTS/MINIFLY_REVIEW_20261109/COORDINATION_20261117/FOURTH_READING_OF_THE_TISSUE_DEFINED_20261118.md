# Fourth reading of "The tissue defined, revision 3" (18 November 2026, written from 23:31 UTC by the clock of this machine)

## Summary

**Verdict: passes after the listed changes.** No sentence asks for something a living cell cannot do. Every mark holds except those listed in finding G9. A programmer could write the tissue after the items in section 5 are decided. Before the briefs of steps 1 and 2 are written, three faults must be removed, because each is visible on paper and would make step 2 fail.

1. **The emission does not see the byte just read** (sentence 112 is false). The path from the input lines to an output-end cell is 8 to 11 connections. Each connection adds one step, and each cell's state follows its arrival with τ_v = 4. So at the reading step φ = 15, an output-end cell nine or ten connections in has received only 0.75 or 0.16 per cent of the eventual effect of the byte just read. The emission therefore rests mainly on bytes two and three positions back.
2. **At output-end cells, the cancelling strength rule works against the feedback rule.** Formula 8 weakens exactly the context inputs that formula 14 strengthens. In the definition's own example, over one window formula 8 is about 5 times larger than formula 14 on a "yes" and about 340 times larger on a "no".
3. **"No" cannot be told from silence.** The first form gives "no" as the value 0 on a feedback line, which is also the line's value when no feedback is present. A cell cannot tell a "no" from the absence of feedback without knowing the window phase φ, and no cell has φ. Formula 14 and the running average r̄ need that distinction.

Further points:

- My twelve findings of the third reading: ten are settled and two are partly settled (F6 and F12).
- The worked example is right to four decimals at all sixteen steps for all three cells. The counts of cells at each distance are confirmed by four independent random draws. The mark counts (29 [S], 66 [F], 82 [C], 177 sentences) and the 51 choices are confirmed.
- Sentence 70 (line strengths stay at 1/8) conflicts with sentence 173 and with the example, in which formula 14 changes line strengths.
- A yes or no given alike to every cell of an end points in the right direction only for the cell that was emitted. At the input end it points mostly to the wrong byte, and it weakens the lines of bytes that come before hard positions.
- The first form takes one of two readings of section 15. The other reading, in which the input end receives the emitted byte, is at least as close to Mr. Xu's words.
- The richer feedback of section 14 has a place without a rewrite. Only two details are missing: the coding of a value of zero and of a negative value, and the rule when two kinds of feedback arrive at the same step.

## 1. Terms

- *Reading step* is φ = s − 1, the step at which the emission is read (sentence 76).
- *Path length* of a cell is its distance from the input lines.
- *Low-pass delay* is the delay that the state of formula 2 adds at each cell. A change in a cell's arrival enters its state with weight 1/τ_v per step, so its state reaches a new level only over several steps.
- *Credit* to a cell is the change that formula 14 makes in its strengths after a feedback.
- *Cancelling rule* is formula 8 with its sign as fixed in sentence 55.

All other terms are those of revision 3.

## 2. Findings in order of seriousness

### G1. The emission is nearly blind to the byte just read (sentences 112, 114, 174; C11, C24)

**What I computed.** The path length of the output-end cells, recomputed with `python3 -I` over four draws, is 8 to 11 connections. The counts were:

| Path length | 8 | 9 | 10 | 11 |
|---|---|---|---|---|
| Output-end cells, four draws | 0 to 2 | 39 to 116 | 136 to 188 | 3 to 36 |

Each connection adds one step. Each cell's state takes in its arrival with weight 1/4 per step (τ_v = 4). For a chain of n cells with unit gain, no threshold, and a line held at 1 from φ = 0, the share of the eventual response reached is as follows.

| Cells in the chain (n) | Reached at φ = 15 | One window later (φ = 31) | Two windows later (φ = 47) |
|---|---|---|---|
| 4 | 0.595 | 0.975 | 0.999 |
| 8 | 0.027 | 0.568 | 0.939 |
| 9 | 0.0075 | 0.406 | 0.881 |
| 10 | 0.0016 | 0.263 | 0.795 |
| 11 | 0.0003 | 0.154 | 0.684 |

Thresholds and strengths below 1 only lower these shares.

**Consequences.**

- Sentence 112's reason ("sixteen steps exceed the path of about ten connections, so the emission can reflect the byte just read") counts only the step delays. It ignores the low-pass delay, which is about 3 more steps per cell.
- Sentence 114 ("the emission at φ = 15 reflects the byte") fails for the same reason.
- Sentence 174's timing of the input end's contribution is also wrong. The activity that shaped the emission lies about 30 to 45 steps before the feedback, not 11 to 27.
- Step 2 asks that the coming byte be given wherever it can be known from the last few bytes. In the first form the last byte is almost absent from the emission. This holds whatever the cells learn, because the delay is a property of the formulas and the geometry.

**Proposed wording.**

- Sentence 112, [C]: "There are s steps per byte, chosen so that an output-end cell has received most of the effect of the byte just read by the reading step. With τ_v = 4 and a path of about ten connections, s = 64 reaches about 0.98. With τ_v = 2, s = 32 reaches about 0.99. With τ_v = 1, s = 16 reaches all of it."
- First form: τ_v = 2 and s = 32. This is cheaper than s = 64 and keeps a state that declines over more than one step.
- Add a sentence [F]: "The delay of a byte's effect at a cell n connections in is about n · τ_v steps."
- Correct sentence 174's figures accordingly.

### G2. Formula 8 opposes formula 14 at output-end cells (sentences 55, 95, 96, 146; C15, C41)

For the output end to be right, the cell of byte b must have a high arrival at the reading step in the contexts that come before b, and a low arrival elsewhere. In such a context its arrival rises above its prediction p, which is a running average over 16 steps. The traces of the context inputs are also high there. Formula 8 then weakens exactly those inputs. A state in which formula 8 makes no further change on average is one in which the cell's mismatch is uncorrelated with the traces of its inputs. In that state the cell cannot respond to the context in the way formula 14 builds.

**The sizes**, recomputed from the example (cell O, connection from u):

| Change over one window | Size |
|---|---|
| Formula 8 | +0.0372 |
| Formula 14 on the "no" that was given | −0.00011 |
| Formula 14 if the feedback had been "yes" | +0.0069 |

So formula 8 is about 5 times larger than a "yes" and about 340 times larger than a "no". The definition names this risk only for inner cells (section 8, item 1). At the output end the risk is worse, because there the cancelled part is the very part the task needs. Sentence 96's conclusion ("so the output end can begin to learn") does not follow while formula 8 acts at the same rate. In the order of opening for step 2, C15 is reached only through "then the order for step 1".

**Proposed wording.**

- New sentence [C], in C15: "In the first form formula 8 does not apply to connections into output-end cells. Their strengths change by formula 14 alone, and their activity is controlled by formula 10. Applying formula 8 there is a further form."
- Alternatively: "applies at the rate η/16".
- Move C15 at the output end to first place in the order of opening for step 2.

### G3. A "no" cannot be told from the absence of feedback (sentences 92, 93, 95, 172; C37, C39)

The feedback line carries r = 1 for yes and r = 0 for no. At the other 15 steps of the window the line is absent, which is also 0. To know when to apply formula 14 and when to update r̄ "at each position", a cell must know that φ = 0. No cell has φ: nothing marks the window except the arrival of a byte, and only encoder cells see that. Sentence 150 ("the first form uses only quantities a living cell holds") therefore fails for formula 14.

The living counterpart cited, dopamine, signals a worse outcome as a dip below a tonic baseline. A worse outcome is not signalled by silence.

**Proposed wording for sentence 92.** "The feedback for a position is carried by two feedback lines: a 'yes' line, active (1) for one step when the emission equalled the arriving byte, and a 'no' line, active (1) for one step otherwise. Every position activates exactly one of them. A cell applies formula 14, with r = 1 or r = 0, at a step at which either line is active, and updates r̄ at that step."

The same coding is needed for the magnitude of sentence 99, where d = 0 means "exact". It is also needed for q_y = −1 in sentence 97: a line cannot carry a negative value, so a separate "this line was wrong" line is needed.

### G4. A yes or no given alike to every cell of an end, and what it can tell one cell (sentences 94, 95, 96, 172 to 174; C40, C41, C52)

**At the output end.**

- Formula 14 changes every output-end cell, in proportion to its own activity trace ε_i. Activity control (formula 10) holds every cell's mean activity near 1/8, so every cell has an ε of that order.
- On a "yes" the emitted cell is strengthened, which is the right direction. Every other active cell is strengthened in the same context too, which is the wrong direction for each of them.
- On a "no" the cell of the right byte is weakened if it was active but not the most active, which is again the wrong direction.
- Only the emitted cell receives a change whose sign is right for it. Its advantage over the others is the ratio of its ε to theirs. Because ε has τ = 16, that ratio is moderate, not large.
- The first form has no source of variation from one occasion to the next: the tissue is deterministic and r̄ is a mean over all contexts. Under these conditions rules with the factor r − r̄ are known to be biased, so the change need not be in the right direction on average. Whether the bias is small enough is not visible on paper.
- Sentence 96 says that a "yes" changes "the inputs of the cell that was most active". This is incomplete, because all active cells change.

Proposed wording for sentence 96, [F]:

> "A 'yes' strengthens the active inputs of every output-end cell in proportion to its activity trace. The emitted cell, being the most active at the reading step, gains most, and the others gain in the same direction. A 'no' weakens them in the same proportions, including the cell of the right byte if it was active. The sign is right for the emitted cell only. Whether this suffices without variation between occasions is read at step 2."

**At the input end.**

- The yes or no concerns an emission shaped by bytes two and three positions back (G1). The encoder cells' traces at the feedback step hold mainly the byte just before. Their credit therefore goes mainly to a byte that had almost no part in the emission.
- Under sentence 173, formula 14 also changes the strengths of line connections.
- The expected sign of that change follows how often the position after the line's byte is right. Lines of bytes that come before positions that are hard to predict, such as the digits of an operand, are weakened on balance. Once a strength falls below 1/128, the connection is given up (formula 12).
- On paper, then, the input end of the first form learns to read more weakly the bytes after which the tissue errs. This is not what step 2 needs.

Proposed changes: in C52's first form, apply formula 14 at encoder cells only to their connections from cells (see G5). The order of opening for step 2 should open C52 with the second tissue that the roadmap already names (feedback at the output end only) as the first comparison.

### G5. Sentence 70 conflicts with sentence 173 and with the worked example (C15, C52)

- Sentence 70: "Connections from input lines change only by being given up and made … Their strength stays at its starting value of 1/8."
- Sentence 173: formula 14 applies "on all its connections, including those from input lines".
- The worked example changes the strength of line "5" by −0.0000117.

**Proposed wording for sentence 70.** "Connections from input lines are not changed by formula 8. In the first form they are also not changed by formula 14 (sentence 173), so their strength stays at 1/8 and they change only by being given up and made. Changing them by formula 14 is a further form of C52." Sentence 173 and the example would be corrected to match. This also removes the drift described in G4.

### G6. Section 15 has a second reading that is at least as close to the words (sentences 171, 172, 175; C52, C35)

"输出端信息的反馈" means "the feedback of the output end's information". In section 11, Mr. Xu used "输出端的信息" ("the output end's information") for what the output end gives. "在NLP任务中这是next byte prediction的结果" means "in language tasks this is the result of the next-byte prediction", which most directly names the predicted byte.

On this reading the input end receives the emitted byte (C35's second form). An encoder cell would then have the byte that was predicted and the byte that arrived, both on its own lines, and the comparison would be made inside the tissue. Sentence 172 names both readings, but it takes the yes or no as the first form without a reason. Sentence 175 ("the input end is reached at the same step") and roadmap draft 3, step 2, rest on that choice.

**Proposed wording, added to sentence 172.** "The words of section 11 use '输出端的信息' for the emission, which favours the second reading. The first form takes the yes or no only so that both ends receive the same quantity. Mr. Xu's reading is to be asked before the brief of step 2 is written, and both forms are run there if he does not settle it."

### G7. The growth of connections at the start of life is not determined (sentences 60, 62, 118; C6, C18)

All quantities start at 0 (sentence 118), so at step 0 every cell has θ = 0 and ā = 0 < a*. Formula 13 also requires "at least P steps … since it last made one", which is undefined for a cell that has never made one.

- If the condition counts as met, all 2,048 cells make a ninth connection at step 0.
- They keep doing so every P steps while ā < a* and θ = 0. A running average that starts at 0 approaches a* from below, so this can last for several periods.

Sentence 62's arithmetic (from θ = 0.1, about 102 steps) is right: 0.1/(1/1024) = 102.4. But it describes a threshold the first form never starts from.

**Proposed wording for formula 13.** "...and at least P steps have passed since the cell last made a connection, or since the start of life."

### G8. The conversion compares cells over the whole output face (sentences 79, 80; C33)

Choosing the most active of 256 output-end cells is a comparison over cells that lie along the whole output face, far beyond any neighbourhood. Sentence 80 admits it as the fixed conversion, and nothing in Mr. Xu's words excludes a fixed conversion outside the cells. It is consistent with sentence 135, since its only effect on cells passes through the emission and the feedback.

The reader is unsure, however, whether it agrees with section 10, under which the output end "learns how to give out". The competition among actions is done for the output end and is not learned. The first form of revision 2 ("exactly one line above the level") also compared all lines, but it did not choose a winner.

Proposed: keep it as the first form, and add to C33 a further form in which competition is formed by inhibitory cells near the output face and the conversion reads any line above 1/2, with no byte if several are. Also add to sentence 80: "This is the one comparison over many cells in the first form."

### G9. Marks and stated facts that do not hold

- **Sentence 4 [S].** "No ordered arrangement of cells is laid down" is the reviewer's "What follows", not Mr. Xu's words. Mr. Xu asked a question. The placing of the lines on opposite faces (C53) does lay down an axis. Propose: split the sentence, and mark the second part [F] with that reason.
- **Sentence 6 [F].** "About 8(1 − x) from the output end" is not a lower bound. Output-end cells lie at x from 0.83 to 1, so the bound is about 8(x_out − x), with x_out the coordinate of the nearest output-end cell.
- **Sentences 112 and 114.** See G1.
- **Section 2.2 and sentence 175.** "The shortest path from the input lines to the output end is about ten to eleven connections": my draws give a shortest path of 8 to 9 and a typical length of 10.
- **Sentence 144 [F].** "About half to two thirds of its starting draws are lines" holds only for cells at the face. Over all cells with x < 1/8 the share is 0.38 to 0.42 in my draws.
- **Sentence 147 [F].** "The threshold … keeps the mean activity near a*" holds only for cells whose activity at θ = 0 exceeds a*, because θ cannot fall below 0. Add "for cells whose drive allows it; others grow connections (formula 13)".
- **Sentence 96 [F].** See G4.
- **Sentence 150 [F].** See G3.
- **Sentence 153 [F].** See G7 and section 5.
- **Sentences 106 and 108 [F].** In the first form the feedback changes only strengths, at φ = 0. A neighbouring cell sees the effect only in the output-end cells' activity during the next byte's context, mixed with that context. Sentence 108's "changes in their arrivals … from which to correct itself" therefore describes no correction in the first form. The arithmetic of sentence 106 (2d + e = 9; (15/16)^9 ≈ 0.56) is right, but it applies only to further forms of C42. Propose "In further forms of C42, ..." in both sentences.
- **Sentence 94.** It quotes roadmap draft 2's step 2 ("the signal acts on the cells at the output end only"). Draft 3 has replaced it, and section 8, item 5 of the definition is likewise out of date.

Everything else checked holds:

- the arithmetic of sentences 62, 101, 106, 149 and 174 ((15/16)^8 ≈ 0.597, (15/16)^49 ≈ 0.042, (15/16)^16 ≈ 0.356, (15/16)^9 ≈ 0.559, (3/4)^15 ≈ 0.013, (15/16)^11 ≈ 0.492, (15/16)^27 ≈ 0.175);
- the 64 lines within reach of a cell at the input face;
- the living counterparts named (the mormyrid electrosensory lobe, homeostatic plasticity, motor cells being excitatory, spine growth and elimination, Bao and colleagues 2001), within the limits the definition states.

## 3. Task (1): the third reading's findings

| Finding | State | Place |
|---|---|---|
| F1 Richer feedback has no place | Settled | Term "feedback"; sentences 21, 22, 24, 87, 92, 97 to 100, 132, 141; C37 to C41. A remaining point on coding is G3 |
| F2 Groups at a distance are empty | Settled | Sentences 4 to 6; section 2.2. Every distance from 1 to 10 is present in four draws of my own |
| F3 Sentence 96 | Settled | Sentence 103 [C]; (3/4)^15 ≈ 0.013 (sentence 149) is right |
| F4 Sentences 99 and 104 | Settled | Sentences 55, 104, 109, 110 |
| F5 Return connections | Settled | Term; sentences 34, 35 |
| F6 The output end cannot start | Partly settled | Formula 14 with r − r̄ and the conversion (sentences 79, 95). Formula 8 still opposes it at the output end (G2), "no" is coded as silence (G3), and the factor is shared alike (G4) |
| F7 Formulas and sign | Settled | Formulas 1 to 14; sentence 113. Open points: G3 and G7 |
| F8 Trace arithmetic | Settled | Sentences 101, 106. The forward delay (G1) is a new matter |
| F9 Silent encoder cells | Settled | Formulas 12 and 13; sentence 144 |
| F10 Number of connections | Settled | Sentences 60, 61; C6. Start of life: G7 |
| F11 Marks | Settled | As in revision 3's section 7. New mark faults: G9 |
| F12 Order of opening | Partly settled | Section 5 exists, but for step 2 C15 at the output end belongs first (G2) |

## 4. Tasks (3) and (4): the worked example, the counts, and steps 1 and 2 by hand

**Worked example.** I recomputed the whole window with formulas 1 to 14 in the order of sentence 113, taking the same starting values and the same assumed outputs of the outside sources. All sixteen rows agree with the table to four decimals for E, I and O, including step 0 and steps 1, 8 and 15. The changes stated at step 0 also agree: +0.00059, −0.00029, −0.0000117, +0.00014, +0.00058, −0.00029, +0.00137, −0.00011, −0.00068 and +0.00005; θ_E = 0.0193; r̄_O = 0.0155. The example itself shows the fault of G5 (a line changed by formula 14) and the sizes of G2.

**Counts.**

- Marks: 177 sentences, of which 29 [S], 66 [F] and 82 [C]. The numbering runs 1 to 177 without gaps. The table has 51 choices.
- Distances: four draws of my own (seeds 1 to 4, my own generator use). Cells at each distance from 1 to 10 from both ends: 122 to 247 from the input lines and 122 to 238 from the output end. At distance 11 there are 3 to 36 cells from the input lines and 1 to 7 from the output end. 0 to 3 cells have no path to the output end. Encoder cells number 236 to 262, mean neighbourhood size is 88.9 to 91.1, the smallest neighbourhood holds 26 to 33 cells, and output-end cells lie at x of 0.830 or more. These agree with section 2.2 apart from the shortest path (G9).

**Step 1.** It is not trivially passed, and on paper it is not impossible:

- the groups at one, two and four connections from the input lines exist;
- silence is prevented by formulas 10, 12 and 13;
- "Better than at the start of life" is a real comparison.

The risk is to the bytes four and eight positions back. These are 64 and 128 steps back with s = 16. The state keeps (3/4)^64, which is effectively nothing, and traces keep (15/16)^64 ≈ 0.016. So only activity that keeps itself going across bytes can hold them. That is the part of an arrival that its inputs' traces foretell, and formula 8 cancels it. The definition's own risk for inner cells is real, and it falls on the memory that step 1 measures.

**Step 2, by hand from a tissue that is nearly always wrong.**

1. Early in life, the share of right positions is about the share of positions with an emission times about 1/256, if the most active cell is spread over all lines. So r̄ is of the order of 1/256 or less.
2. A "no" then changes strengths by η · r̄ · e · ε, about 10⁻⁵ in the example. A "yes" changes the emitted cell by about η · e · ε, about 0.007.
3. Formula 8 changes the same strengths by about 0.04 per window, against the direction a "yes" set (G2).
4. The emission sees the last byte at under 1 per cent of its effect (G1).
5. "No" is not distinguishable from silence (G3).

Under the first form as written, therefore:

- an output-end cell's rule cannot keep what a rare "yes" gives it;
- the shared yes or no has the right sign only for the emitted cell (G4);
- the coming byte cannot be predicted from the last byte.

I judge step 2 to be impossible on paper in the first form as written. After G1 to G3 are changed (s = 32 with τ_v = 2, formula 8 removed at output-end cells, two feedback lines), the rule can start from rare right emissions. Whether a shared factor without variation between occasions suffices is then a question for the run. It is not visible on paper.

## 5. Task (5): what a programmer would still have to decide

1. How a cell knows that feedback is present when r = 0 (G3).
2. Whether formula 14 changes line strengths (G5).
3. The interval condition of formula 13 at the start of life (G7).
4. The feedback at the first position of life, when there is no earlier emission.
5. The feedback at a step at which a position's yes or no and an answer's magnitude (sentence 99) both arrive: added, applied one after the other, or one replacing the other.
6. How a feedback line carries q_y = −1 (sentence 97).
7. "The largest possible difference" of sentence 99 for arithmetic that is not modular.
8. Ties in the choice of the nearest cell for an output line (sentence 5). These are negligible, but they are undetermined.
9. The step φ at which the experimenter records outputs for step 1. This belongs to the brief.

Everything else in my list of 28 is settled as the definition's section 7 states, and I checked each entry against the sentence it names.

The list of numbers in section 6 is complete with one exception. P serves as 128 steps for the running averages and as 128 positions (2,048 steps) for r̄. These are two different quantities fixed at one value, and they should be listed as two.

## 6. Task (6): the richer feedback of section 14

Revision 3 gives the richer feedback a place without a rewrite.

- **The right answer as bytes** enters through the feedback lines for each output line (sentences 97 and 98). It is received by the output-end cell of each line, at φ = 0 after the emission for that position, and is used by Δw = η · σ · q · e. This is permitted by sentence 88, because it comes after the emission.
- **The absolute difference** enters as one quantity for each answer, after the answer's last byte, to every output-end cell alike (sentence 99). It is normalised to lie between 0 and 1, uses the circular difference modulo 89 divided by 44 (sentence 100), and is used as (r̄_d − d) in formula 14.
- **The input end** receives it as sentence 177 provides.
- **The trace** for an answer of three bytes keeps 0.04 of its weight, and sentence 101 opens C16 for it. With G1's change to s = 32 the reach becomes 3 · 32 + 1 = 97 steps, so C16 would need τ of at least 32 to 64. Sentence 101 should be recomputed with the s finally chosen.

Missing are only the coding points of G3 (a value of 0, and q = −1), the rule for two kinds of feedback at the same step (section 5, item 5), and the normalisation for arithmetic that is not modular (section 5, item 7). Each is one sentence.

## References

Bao, Chan and Merzenich (2001); Lillicrap and colleagues (2016); Schultz, Dayan and Montague (1997): as cited in the definition. The bias of reward-modulated rules without variation between occasions is a general result of the theory of such rules. The reader's confidence in its exact size here is moderate.
