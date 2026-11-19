# Reading of the tissue defined, revision 5 (19 November 2026, written from 01:13 UTC by the clock of this machine)

The file read is `THE_TISSUE_DEFINED_REVISION_5_20261119.md` (called "the definition" below), from its first sentence to its last, after sections 7 to 17 of `WHAT_A_NEURON_HAS_20261118.md` (called "his sections") and sections 1 to 6 and 9 of the roadmap, draft 3. The reader did not read the earlier revisions before writing the findings. Afterwards the readings of revisions 3 and 4 and the check of the brief of step 1 were searched for the subjects of the findings, and each finding says whether it was raised before. The only programs run were `python3 -I` with the standard library, in the reader's scratch directory, for about eight minutes of processor time; no file other than this one was written in the repository.

## Verdict

**The definition stands after the listed rewordings, which change no formula of the first form.** No finding requires a change to formulas 1 to 12, to sentence 53 or to the order of operations as run at step 1. One finding (finding 1, on the growth of connections) shows on paper and by a calculation that the first further form named for that failure (a* = 1/32) cannot remedy it. The order of opening should therefore be changed, which changes no formula. If the coordinator judges that a first form whose growth can be predicted on paper should not be spent on in stage B, the change is to formula 12 (finding 1, alternative); that would be a change of a formula, and the reader does not recommend it as a condition for continuing.

## Summary

1. Every sentence was checked for its mark. 29 of the 152 sentences fail or are doubtful: [S] 2 of 28, [F] 14 of 62, [C] 13 of 62. No failure changes a formula.
2. The worked example, part 1, was recomputed from the text alone for all four cells. It agrees with the table to four decimals at every listed step, and the final strengths agree. Sentence 118 and the powers in sentences 99, 106, 115 and 124 agree.
3. Part 2 agrees up to step 129. After that point it cannot be reproduced from the definition, because the definition does not name the generator or the order of candidates for the sources of new connections. Sentence 143 ("one generator") also conflicts with the brief of step 1, which uses a second generator. At step 160 the reader obtains 0.0170 to 0.0190 for cell 333 against the published 0.0322.
4. Growth (point 4(a)). At birth no cell can reach a* = 1/8: a line connection is fixed at 1/8, only one line is active at a time, and the recurrent excitation is about half of what would sustain activity. Every one of the 1,984 cells other than feedback cells therefore grows, not only the cells driven during one byte value. Recurrent excitation sets in at about 14 connections (step 768 to 896). Growth then continues at 70 to 93 per cent of cells every 128 steps, and the mean is 43 connections at step 6,144. Line drive cannot reach a* = 1/32 either, so the first further form does not stop the growth.
5. The fixed tissue (point 4(b)) carries only the present byte at every distance. Its cells at distance 4 are silent. The comparisons that show learning are lags 1 and 2 above the present byte alone, together with a control in which connections grow but strengths do not change.
6. Mr. Xu's words: sentence 128 contradicts section 7, point 2, and section 9. The forms of C57 settle the example that Mr. Xu left open in section 17. Sentence 135 conflicts with sentence 74. Sentence 87 departs from the literal words of section 16; this was raised before and settled by a decision of the reviewer. Line connections can never be given up, which narrows section 10.

## Findings, in order of seriousness

### Finding 1 (sentences 59 and 60; choices C18, C29 and C17; section 15). Changes no formula of the first form; the alternative at the end would change formula 12.

*Reason.* Sentence 60 says that a cell driven during one byte value cannot reach a*. The true statement is wider. At birth no cell can reach a*, for two reasons.

First, a connection from an input line has the fixed strength 1/8 (sentence 66), and only one input line is active at a time (sentence 63). The drive of any cell from lines is therefore at most 1/8 = a*, and its mean is far lower. For seed 73129, an encoder cell whose sources included every line of its neighbourhood would be driven by a line at 18.9 per cent of positions on average and 40.1 per cent at most. Its mean activity from lines would be at most 0.050, which is below a* = 1/8. For 168 of the 240 encoder cells it would also be below 1/32.

Second, the loop gain is below 1 at birth. The loop gain is defined here as the factor by which activity carried around the connections between cells is multiplied on each pass, estimated as the share of excitatory minus inhibitory members of a neighbourhood, 0.539, times the number of connections, times 1/8. At 8 connections it is about 0.54, so activity from cells dies away.

Hence θ stays 0 and ā stays below a* at every cell, and formula 12 adds a connection to all 1,984 cells at every multiple of 128 steps. The reader's run confirms this: 1,984 connections were made at each of steps 128 to 768. The loop gain reaches 1 at about 15 connections. In the reader's run of the whole first-form tissue (seed 73129, birth and stream of the brief, generator of the brief, 6,144 steps):

- activity carried by recurrent connections appears between steps 768 and 896, at 13 to 14 connections. At the reading step of step 896, 98 per cent of the cells at distance 8 have activity above 0.01, against none before;
- θ is then above 0 at between 3 and 72 per cent of cells, varying from one sample to the next;
- giving up begins at about step 1,600;
- growth does not stop. 70 to 93 per cent of cells still make a connection in each period of 128 steps, because the mean activity at reading steps (0.05 to 0.12 after step 1,792) stays below a*. The mean number of connections is 22.5 at step 2,048, 33.1 at step 4,096 and 43.4 at step 6,144, and it was still rising by about 0.6 every 128 steps;
- the neighbourhoods hold 96.9 members on average (32 to 154). If the trend held, they would be full by about step 17,000 (byte 550), which is about 0.2 per cent of the stream of stage B.

The run was not continued, so whether growth stops before the neighbourhoods are full is not settled. What is settled is the following. Growth is set by the target a* and by the radius, not by what the cell carries. Sentence 59 ("Only the starting number, 8, is fixed by us") is therefore not true in effect: the number also rests on a*, P_s, the level 1/128 and the radius 1/8. There is also a cost to the code. At a full neighbourhood an encoder cell reads, with equal strengths, the lines of 9.0 byte values in use on average (19 at most), so its response to the present byte is coarse.

The first further form named in sentence 60 and second in the order of opening for step 1 is a* = 1/32. It cannot stop this growth for cells driven by lines, by the bound above. The check of the brief said that a cell responding fully to one or two frequent byte values could reach 1/32. A response through a line is at most 1/8, and two values with shares of at most 6.2 per cent each give a mean of at most 0.016.

*Was it raised before?* The growth of cells under their target was raised as finding 10 of the check of the brief, but only for cells driven by one byte value, and with a* = 1/32 as the remedy. The bound on line drive, the growth of every cell, and the onset of activity carried by recurrent connections were not found in the earlier readings.

*Is it a fault of the first form?* It is a fault in the sense that the growth condition aims at a level that the input the first form gives a selective cell cannot reach. The number of connections is therefore not the cell's own in any useful sense. It is not a fault that must stop the experiment. The definition names growth as a risk (sentence 60 and section 18, item 1), stage A will show it, and the first form is still a legitimate first form to run.

*Proposed wording.*

- Sentence 59: "The cell changes its number of connections itself, by formulas 11 and 12. The rule by which it does so uses numbers fixed by us (a*, P_s, the level 1/128) and is bounded by the radius 1/8 (C30). The starting number, 8, is also fixed by us."
- Sentence 60: "At birth no cell other than a feedback cell can reach a*. A connection from a line has the fixed strength 1/8 and one line is active at a time, so the drive from lines is at most a*. The connections between cells carry about half the activity they would need to sustain it. Every cell other than a feedback cell therefore makes a connection at every multiple of P_s until recurrent excitation raises activity. In the reader's calculation this happens at about 14 connections, and growth continues beyond it. Whether growth stops before the neighbourhoods are full, and whether cells stay informative, is read at step 1. a* = 1/32 cannot be reached through lines either."
- Section 15, step 1: put C18 before C29 in the order of opening when the failure is loss of selectivity with growth.

*Smallest change a living cell can do (a further form of C18; it changes formula 12 when opened).* A cell adds a connection only while it has been nearly silent over a time longer than the gaps between the occurrences of its byte values. Let s̄_i be a running share of active steps: s̄_i(k + 1) = (1 − 1/4,096) · s̄_i(k) + (1/4,096) · [x_i(k) > 1/128]. The condition ā_i < a* of formula 12 is replaced by s̄_i < 1/128. A cell that is active during even the rarest byte value, which occurs at 1.4 per cent of positions, then stops growing. A cell that is silent goes on growing until it is active at 1/128 of steps. The two numbers already appear in the definition, 4,096 in sentence 53 and 1/128 in formula 11, so no new number is introduced. The living counterpart is the growth of spines and synapses by neurons deprived of activity, which is integrated through intracellular calcium over hours (homeostatic structural plasticity). Its time course is unsettled. The change opens C18, and C17 in so far as ā is no longer used by formula 12.

### Finding 2 (sentence 128 and choice C57, second form; sentences 36 and 37; Mr. Xu's section 17). Changes no formula of the first form; it changes the stated second form of C57, which is not run.

*Reason.* Sentence 128 [F] says: "A chemical signal received alike by a region cannot choose which connection changes." This contradicts Mr. Xu's section 7, point 2, and section 9: how each receiving cell reads the signal is its own matter, and one signal can act differently in different cells. It also contradicts textbook neuroscience. A neuromodulator released alike over a region changes only those synapses that carry a local trace of recent activity. Examples are the eligibility traces of three-factor rules and synaptic tagging and capture (Frey and Morris, 1997). So a uniform signal does select, through the state of each connection. The conclusion drawn from sentence 128, that the second form of C57 "makes opposed signals add rather than select", is therefore not necessary.

Mr. Xu's example in section 17 (C's feedback says the strength should rise, D's says it should fall, and A also has separate connections with D and E) is exactly the question he left for discussion. The first form of C57 answers it: cells on a path have no say on a connection that is not their own. The second form answers it differently: their signals add. Sentence 37 says the definition leaves the matter open, but each of its forms decides his example, and no form is kept in which the outcome is not settled by us. Two further points:

- Sentence 36 quotes section 17 with an ellipsis that removes the example. The example is the substance of what was left open.
- Sentence 37's description "the influence of one cell on another is the sum of its effects over all paths" is not true of the first form. Formula 3 clips activity at 0 and 1, so the effects along paths do not add except where every cell on them is between those bounds.

*Proposed wording.*

- Sentence 128: "A chemical signal received alike by a region can change connections selectively when it acts together with a trace held at each connection. How opposed signals combine is not settled (section 17)."
- Second form of C57: "A third cell bears on a connection that is not its own through a local chemical signal released in that connection's region. How the signals of several cells on a path combine (for example, adding, or acting only on connections that hold a recent trace) is not settled by Mr. Xu, and is not settled here."
- Sentence 36: quote section 17 in full.
- Sentence 37, first form: "Each direct connection is set by its own two ends. A cell on a path between two cells bears on their connection only through its own connections. In the first form, which is not linear, the effects along different paths do not in general add."

### Finding 3 (sentence 135 against sentence 74 and Mr. Xu's section 7, point 1). Changes no formula.

*Reason.* Sentence 74 [S] states the accepted form of teaching: "the right answer is shown as input after the learner's own attempt". In the task of predicting the next byte, the byte that arrives on the input lines at φ = 0, after the emission read at φ = 31, is exactly that right answer. Sentence 135 [C] says: "No cell is given the right answer in the first form." It contradicts sentence 74, since encoder cells are given the right answer as input at every position.

*Proposed wording for sentence 135.* "No cell is given the right answer before its emission. After the emission, the right answer arrives as the next byte on the input lines (sentence 74). The per-line and per-answer forms give it, after the emission, also through feedback cells."

### Finding 4 (sentence 66 with formula 11; Mr. Xu's sections 8 and 10). Changes no formula if the choice is declared; a further form would change one.

*Reason.* Sentence 66 says that connections from input lines "change only by being given up and made". Their strength is fixed at 1/8, and formula 11 gives up only connections below 1/128, so a line connection can never be given up. Formula 12, however, adds lines at random. The input end's learning of "how to take in" (section 10) is therefore limited to the random addition of lines, which can never be undone. Section 8's "不断调整自己连接那些细胞" (keeps adjusting the cells it connects to) is met at the input end in one direction only. Sentence 66 has no entry in the table of choices. Whether a living sensory afferent synapse can be fixed for life is unsettled: such synapses are plastic in development and less plastic in the adult.

*Proposed wording for sentence 66.* "Connections from input lines keep their starting strength 1/8. Since formula 11 gives up only connections below 1/128, a line connection, once made, is never given up; an encoder cell's set of lines can only grow. This is our choice (a new entry, C58, 'Line connections'; further forms: giving up by the sending side's running average, C17; strengths of line connections changed by the coincidence form)."

### Finding 5 (sentences 58 and 143; section 13, part 2; sentences 150 to 152; section 12). Changes no formula.

*Reason.* Two programmers would not write the same tissue, because the definition does not say which generator draws the source of a new connection, nor in what order the candidates are listed. Sentence 143 says "one generator with a recorded seed", which reads as the birth generator continuing into life. The brief of step 1 uses a second generator, `Random(seed + 500000)`, with the candidates in neighbourhood order.

The reader's implementation, written from the text, agrees with part 2 up to step 129: cell 497 at 0.0625, 0.0625, 0, 0, 0.0021, 0.0021 and 0.0021, and cell 333 at 0, 0.0193, 0, 0, 0.0003, 0.0003 and 0.0041. At step 160 cell 333 has activity 0.0172 with the brief's generator, 0.0170 with the birth generator continued, and 0.0190 with `Random(seed)`, against the published 0.0322. In the same 160 steps formula 7 changes 6,146 starting connections between cells, 10 of them by more than 0.0001, against the published 6,188 and 12. 13,862 connections are made by step 1,024, against 13,875. The smallest starting strengths at step 1,024 are 0.1191, 0.1002 and 0.0419, against 0.1178, 0.1032 and 0.0455. These are the same in kind but not in value. Sentence 152 is marked [F], but it is the count of one run, and the generator of that run is not stated.

*Was it raised before?* The readings of revisions 3 and 4 mention a generator. The failure to reproduce part 2 was not found.

*Proposed wording for sentence 143.* "The birth uses the generator `Random(seed)`. The sources of new connections use a second generator, `Random(seed + 500000)`, drawing `choice` from the candidates in the order of the neighbourhood: cells in increasing index, then lines in increasing value." Recompute part 2 and sentences 150 to 152 with this generator, or state the generator used.

### Finding 6 (sentence 87; Mr. Xu's section 16). Changes no formula; at step 1 it changes only which silent cells are drawn at birth.

*Reason.* "我们的输入端不是所有的细胞都收到yes/no，只有某些收到" says that at the input end not all cells receive the yes or no, only some do. Under sentence 87 none does. The next sentence ("可以认为有专门的细胞...", "it can be taken that there are dedicated cells") allows dedicated cells, but it does not say that these cells lie outside the input end. In the first form the feedback cells lie anywhere in the square (sentence 100). The reading of sentence 87 therefore departs from the literal words of the first clause.

*Was it raised before?* Yes, by the fifth reading and by finding 9 of the check of the brief. The reviewer settled it by not applying E5. The reader records that the literal words still favour a reading in which some cells of the input end receive the feedback.

*Proposed wording.* Either put the question to Mr. Xu, or meet both clauses in the first form by drawing the 64 feedback cells among the cells within 1/8 of the input face. They then count as cells of the input end that receive the yes or no and pass it on. The second option changes the birth (sentence 6 and C56) but no formula, and it has no effect at step 1, where the feedback lines are never active, other than which silent cells other cells draw as sources. It should be settled before step 2.

### Finding 7 (the order of operations, sentence 120; choice C41; terms). Changes no formula.

- Operation 6 lists formulas 4, 6, 9 and 10 but not the update of the sliding threshold ϑ of sentence 53. Add "and, in the coincidence form, ϑ (sentence 53)".
- ε_i appears in C41 and in the order of opening for step 2 ("m_i · ε_i") and is defined nowhere in the definition. Define it, or remove it.
- a* is used in sentence 53 before sentence 55 defines it.
- "Emission line" (sentences 97 and 120) and "item" (sentences 116, 132 and 140) are not among the terms.
- There is no section 11: the numbering goes from 10 to 12.
- The rest of the order is complete. The reader's two implementations of part 1 and of the whole tissue needed no further choice except the generator of finding 5.

### Finding 8 (sentence 53). Changes no formula.

- "The threshold's time constant, 4,096 steps (128 bytes), is a number of this form on the grid". By sentence 134 and C47 the grid is 2 to 128 and their reciprocals, and only numbers of cells go beyond 128. Either the sentence should say "a number outside the grid", or C47 should be extended to time constants.
- "The living counterpart is the sliding threshold of Bienenstock, Cooper and Munro (1982) and homeostatic inhibitory plasticity" is stronger than the science. The first is a theory, with experimental support from metaplasticity in visual cortex; whether a threshold slides as x²/a* is unsettled. The second is cited to a model (Vogels and colleagues, 2011). Proposed: "The form follows the theory of Bienenstock, Cooper and Munro (1982), for which metaplasticity in visual cortex gives evidence, and the model of inhibitory plasticity of Vogels and colleagues (2011); whether living cells do either in this form is unsettled."

### Finding 9 (sentence 43 at inner cells; roadmap risk of section 18, item 1). Changes no formula.

*Reason.* The living counterpart given in sentence 43, structures like the cerebellum "where unchanging afferents carry the signal", applies only to cells that have unchanging afferents. Under formula 7 those are only the encoder cells with lines. An inner cell has no unchanging input. The fixed point of formula 7 there is an arrival that does not vary, which carries nothing. This is the risk named in section 18, item 1, but the sentence that claims the living counterpart does not say that it fails at inner cells.

*Proposed addition to sentence 43.* "At inner cells, which have no unchanging input, the rule has no such counterpart and tends to make the arrival constant (section 18, item 1)."

### Finding 10 (distance from the input lines; terms, sentence 7, section 12). Changes no formula.

*Reason.* The definition counts distance from any input line. 210 of the 256 byte values never occur in the stream. For seed 73129, 130 of the 232 cells at "distance 1" have only lines of values that never occur. Counted from the lines of values in use, as the brief does, the counts at distances 1, 2 and 4 are 102, 215 and 220, against 232, 189 and 193.

*Proposed wording.* Term "distance from the input lines": "... counted from the lines of the byte values that occur in the stream." Give the counts of section 12 both ways.

### Finding 11 (marks). Changes no formula.

**[S] sentences that fail or are doubtful (2):**
- Sentence 14 adds a clause of ours ("A living counterpart of three gates is unsettled"). "Three gates" is not in Mr. Xu's words. Move the clause to sentence 15.
- Sentence 36 removes the example of section 17 (finding 2).

Sentence 8 is accepted. The reader notes, however, that "不断调整自己连接那些细胞" also admits the reading "keeps adjusting its connections to those cells". The definition meets both readings through formulas 7, 11 and 12.

**[F] sentences that fail or are doubtful (14):**
- Sentence 5 contains a choice (C53). Split it.
- Sentence 7: x_out ≥ 0.83 is a count from three draws, not a necessity. Write "in the draws of section 12".
- Sentences 22, 24 and 138 rest on choices 85 and 87, which come later. Write "in the first form" and cite them.
- Sentences 40 and 42 claim necessity for an explicit prediction and mismatch. Sentence 53, run at step 1, shows that inner cells can do without them. Write "In the first form ...".
- Sentence 52: "taken to reach" is an assumption, that is a choice (C41). Cite it.
- Sentence 59 is false in effect (finding 1).
- Sentence 60 gives a reason that is incomplete (finding 1).
- Sentence 106: the reason for "2d + 1" is not given, and the reader cannot reconstruct it. Give the derivation.
- Sentence 128 is false (finding 2).
- Sentence 146 is a statement about the document, not a consequence.
- Sentence 152 is the count of one run with an unstated generator (finding 5).

Sentence 148 is accepted with a wording change: "at every connection leaving it", since feedback cells make no connections.

**[C] sentences that fail or are doubtful (13):**
- Sentences 45 (formula 1), 54 (formula 8), 66, 88, 114 and 120 have no entry in the table of choices. Formula 1 and the order of operations are choices: for example, the receiving factor x(k − 1) and the use of traces held before the step.
- Sentences 144 and 145 are references, not choices. C49 lists Mr. Xu's exclusions as a choice of ours, with the further form "Mr. Xu may restate"; they are his, not ours.
- Sentence 37 (finding 2).
- Sentence 43 (finding 9).
- Sentence 53 (finding 8).
- Sentence 87 (finding 6).
- Sentence 135 (finding 3).

All other [C] sentences are open choices, listed in the table.

**What a living cell can do.** No sentence asks for something a living cell certainly cannot do. Unsettled points are a signed mismatch at the synapse (sentence 52, stated as unsettled), the sliding threshold (sentence 53, finding 8), and line synapses fixed for life (sentence 66, finding 4). The reader is unsure about two more. Synchronous update of all cells (sentence 117) is an idealisation, but it is not stated as one. In sentence 89, dopamine cells respond to outcomes better than expected rather than to good outcomes as such; the sentence already says the pairing is not a finding about those cells.

### Finding 12 (Mr. Xu's words not entered; numbers without a place). Changes no formula.

**Section 7, point 2** excludes "the use by one cell of another cell's weights". Section 10 of the definition does not list this exclusion. Add: "No cell uses another cell's strengths."

**Section 8** describes adjusting connections as what a nerve cell does. In the first form a feedback cell adjusts no connection (sentence 85). The further form "with ordinary sources" exists; sentence 85 should say that this departs from section 8.

**Section 8, passages.** Sentence 64 says an encoder cell's passages are formed by its traces, state and connections. With τ_v = 2 and traces of 32 steps, an encoder cell in the first form reads the present byte and about a third of the previous one, not passages with gaps. This is within Mr. Xu's "may read only", but sentence 64 should say so.

**Numbers without a place in section 16:**
- the 8 emission lines of each feedback cell of form (b) and their strength 1/8 (sentence 97);
- the number of feedback cells of the per-line form (sentence 111) and of the magnitude form (sentence 113), which is not given at all;
- the factor 1/16 of the further form "η/16" (C15);
- the other numbers of further forms (128 feedback cells, radius 1/4, s = 16 and 64, τ = 64 and 128, starting numbers 4 and 16, shares 1/8 and 1/2). Section 16 names only 4,096 and 1/32 under "numbers of further forms".

### Mr. Xu's sections one by one (task 2)

- **Section 7.** Point 1 is entered (sentences 12, 14 and 74). Sentence 135 contradicts the accepted form of teaching (finding 3). Point 2: the local reading of the chemical signal is contradicted by sentence 128 (finding 2), and the exclusion of the use of another cell's weights is missing (finding 12).
- **Section 8.** Entered (sentences 1, 2, 8 to 11, 23, 25, 38, 39 and 61). Narrowed at the input end, where line connections can only be added (finding 4). Feedback cells adjust nothing (finding 12). Prediction in the first form is a running average of the arrival (sentence 43), which is the weakest reading of "predicting the new information", and inner cells under sentence 53 do not use it. The definition says so, and this is not a contradiction.
- **Section 9.** Entered (sentences 18, 35, 125 and 126). No signal reaches every cell. Sentence 128 contradicts "how each receiving cell reads it is its own matter".
- **Section 9a.** The goal is not a property of the tissue. Nothing is missing.
- **Section 10.** Both ends are entered. The input end's learning is narrowed by finding 4. The output conversion is not learned (sentence 76), which the definition leaves open (section 18, item 5).
- **Section 11.** Entered (sentences 79 and 82, and section 7). The feedback is given at one step of every position. At step 1 there is none (sentence 88), by the roadmap's choice and not by Mr. Xu's words.
- **Section 12.** Entered (sentences 59 and 133). In effect the number of connections is set by numbers of ours and the radius (finding 1). The time constant 4,096 is off the grid (finding 8).
- **Section 13.** Entered (sentences 21, 80 and 110 to 115).
- **Section 14.** Entered (sentences 81, 112 and 113). The circular difference and the scaling are listed choices (C50).
- **Section 15.** Entered (sentences 4, 5, 83 and 96 to 102).
- **Section 16.** Joint strength is entered (sentences 44, 51 and 52). "Only some receive" departs from the literal words (finding 6). The learned passing on is entered (sentence 90).
- **Section 17.** Entered as open, but both forms settle Mr. Xu's own example. The quotation leaves the example out (finding 2).

## Point 4(a): for which cells growth happens, how fast, and what stops it

**Which cells.** All 1,984 cells other than feedback cells, not only cells driven during one byte value. The drive from lines is at most 1/8 = a*, because line strengths are fixed at 1/8 and one line is active at a time. Its mean is at most 0.050 even with every line of a neighbourhood. The loop gain at 8 connections is about 0.54.

**How fast.** Exactly one connection every 128 steps (4 bytes) for every cell until recurrent excitation sets in. In the reader's run this is at 13 to 14 connections, between steps 768 and 896. After that, 70 to 93 per cent of cells still grow in each period, because the mean activity stays at 0.05 to 0.12, below a*. Some connections are given up from about step 1,600. The mean was 22.5 connections at step 2,048, 33.1 at step 4,096 and 43.4 at step 6,144, rising by about 0.6 every 128 steps. The neighbourhoods average 96.9 members. If the trend held they would be full by about byte 550, which is early in a stream of 262,144 bytes.

**What stops it.** Only recurrent excitation strong enough to keep each cell's ā at or above a*, or its θ above 0, at the moments of the check. In 6,144 steps this had not happened. a* = 1/32 cannot be reached through lines either, so it does not stop the growth of cells driven by lines.

**Is it a fault of the first form?** Yes, in that the target of formula 12 is out of reach of a selective cell, so the number of connections is not the cell's own in a useful sense. It is not a reason to stop the run.

**Smallest change.** Grow only while the cell's running share of active steps over 4,096 steps is below 1/128. This opens C18 (and C17 for ā). It changes formula 12 when opened; the first form is unchanged. For a growth failure, open C18 before a* = 1/32.

## Point 4(b): the fixed tissue at distances 2 and 4, and the comparison that shows learning

**What the fixed tissue's cells do.** In the fixed tissue (operations 5 and 7 off, θ kept), every cell keeps 8 connections, and the loop gain stays at about 0.54, so no activity is sustained by recurrent connections. With τ_v = 2, a byte's effect at a cell falls to about 2⁻³² of its size by the next byte. Every recorded output is therefore a function of the present byte alone.

- **Distance 1** (from lines in use): a cell is active, at most 1/8, while one of its few in-use lines is present. 102 of the 240 encoder cells have such a line at birth.
- **Distance 2:** a cell has activity of about 1/64 for each active encoder source. This is just above the recording level of 0.01. In the reader's run of 4,096 steps, 1 to 4 per cent of these cells were above 0.01 at a reading step, with a mean of 0.0002 to 0.0012. Many are silent and the rest are weak.
- **Distance 4:** activity of order 10⁻⁴, never above 0.01 in the run. These cells are silent by section 7.2 of the brief, and their score is at chance.

In the plastic first form, cells at distance 4 became active at about step 900 because of growth and recurrent excitation. The changes of strength by formula 7 are small by then: the smallest starting strength at step 1,024 is 0.042. So "better than the fixed tissue" at distance 4 means "not silent", and at distance 2 it means "better than a weak copy of the present byte".

**The comparison that would show that learning adds something about the bytes.**

1. At lags 1 and 2, the score must exceed what the present byte alone gives (C0) at the same distance. The fixed tissue cannot exceed C0 at any lag, since its outputs are a function of the present byte.
2. A further control from the same seed in which operation 7 is kept and operation 5 is off. This tissue grows connections but never changes a strength. The plastic tissue's excess over this tissue at lags 1 and 2, above C0, is what the strength rule adds. The reverse control (operation 5 on, operation 7 off) shows the strength rule without growth.
3. Cells are grouped by their birth coordinate x, for example the bands of width 1/8, as well as by present distance, since growth shortens distances and "distance 4" then names different cells in different tissues.
4. At lag 0, no tissue can exceed C0. The rule for lag 0 can only be "equal to C0"; this concerns the roadmap and the brief rather than the definition.

None of this changes a formula; it concerns the brief of step 1.

## What the reader computed

All runs were done with `python3 -I`, on the birth and stream of `check_step1_brief.py`, seed 73129.

- Worked example, part 1: all four cells, all eight listed steps and all ten columns agree to four decimals. The final strengths agree: E h 0.2554, c 0.4973; I E 0.5003, q 0.2510, g 0.2495; O u 0.7504, f 0.2498, F 0.2502. The text's "x = 0.4063" for O at step 0 is 0.40625, which the table rounds to 0.4062.
- Part 2: agrees to step 129 and differs from step 160 (finding 5).
- Sentence 118 (0.999, 0.996, 0.990 and 0.975; 0.027) and sentences 99, 106, 115 and 124 (0.362, 0.751, 0.0445, 0.0172 and 0.0003) agree.
- Growth and the fixed tissue: as stated in points 4(a) and 4(b).

## References added by the reader

- Frey and Morris (1997), Nature 385: 533-536 (synaptic tagging).
- Kirkwood, Rioult and Bear (1996), Nature 381: 526-528 (experience-dependent modification of the synaptic threshold in visual cortex).
- Butz and van Ooyen (2013), PLoS Computational Biology 9: e1003259 (homeostatic structural plasticity).

Page numbers are from memory and should be checked before citation.
