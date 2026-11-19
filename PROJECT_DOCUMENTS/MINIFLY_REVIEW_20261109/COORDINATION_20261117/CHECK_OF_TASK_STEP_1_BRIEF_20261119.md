# Check of the brief of step 1, "the tissue reads the stream and stays informative" (19 November 2026, written from 00:27 UTC by the clock of this machine)

Checked file: `TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md` (revision 1, from 00:03 UTC), with `check_step1_brief.py`, against sections 7 to 16 of `WHAT_A_NEURON_HAS_20261118.md`, roadmap draft 3, revision 4 of the definition, the fifth reading (errata E1 to E9) and the design block of `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`. The checker did not write the brief. No tissue was run; the only programs run were `python3 -I` with the standard library (the brief's check script, an independent recomputation, arithmetic on the births, and simulations of single cells, in all about nine minutes of CPU).

## Verdict and summary

**Verdict: not to be released as written. A short trial of about two CPU-hours should be run first (section 6), and the brief should be released for the full runs only with the changes of findings 1 to 6.**

1. The brief reproduces the definition faithfully and its check script reproduces section 17 exactly; the stream, chance level, counts at birth and cost arithmetic agree with an independent recomputation.
2. On paper, the first form's rule for giving up a connection (formula 11, the test of the sending side's 128-step running average against 1/128) removes, at step 128 (byte 4), at least 15,617 to 15,646 of the 15,872 starting connections in every seed, in both plastic tissues. This is a bound, not an estimate.
3. Afterwards a connection survives only if its sending cell or line is active at a mean of about 1/80 over each four bytes. New input-line connections survive their first test with a chance of 0.06 to 0.31 and then last a median of 16 to 17 bytes; connections from a cell driven at 1/8 last a median of 8 to 9 bytes.
4. A single cell two connections from the input lines, simulated under the definition, keeps one or two connections and loses its dependence on the byte (6 to 8 per cent of variance explained) under both strength rules alike. The strength rule, which the brief's two arms compare, is therefore not what the runs as designed would test.
5. The rule of success can be met at lags 1, 2 and 4 by a tissue that carries only the present byte (the brief's own countermodel C0 lies 0.03 to 0.07 above chance there), which the contract's design block does not allow.
6. The baseline "start of life" (the first 8,192 bytes) is measured on cells whose distances are taken at birth, although nearly all their connections are gone after byte 4; the classifier ignores amplitude; and the population sizes compared change between windows.
7. The coincidence rule as the brief writes it has a threshold that follows each byte (time constant of one byte) and strengthens inhibition when the cell is below its target, contrary to the rule it cites; simulated, it drives every active input to the upper bound 1.
8. The reversal of the feedback cells' kinds (E3) in the coincidence tissue is wrong for its output-end cells, which keep formula 7, and so would mislead step 2.
9. Under formula 7 without giving up, the anti-Hebbian weakening (risk H1) is real in a single cell: it halves the excitatory strengths over 100,000 bytes at the definition's activity level and removes them within 3,000 bytes when the senders are fully active.
10. Recommendation: a trial of 16,384 bytes on one seed with five tissues (both rules, formula 11 as written and by strength only, and the fixed tissue), at about two CPU-hours, decides whether the definition needs erratum E10 before 40 to 60 CPU-hours are spent.

## 0. Terms

- *Starting connection*: a connection drawn at birth (section 5.3, item 9 of the brief).
- *Sending-side test*: the clause "or when its sending side's running average ū_ij is below 1/128" of formula 11 (sentence 54 of the definition).
- *Activity bound*: an upper bound on a cell's activity over steps 0 to 127, computed from the birth by the iteration X_i = (1/8)·[the cell has a line of a byte value read in the first four positions] + Σ_j W_ij·X_j over its excitatory cell sources, capped at 1, where W_ij is itself bounded by 1/8 plus the largest change formula 7 can make in 128 steps given the bounds (iterated to a fixed point). It ignores inhibition and is therefore an upper bound.
- *Share of variance explained by the byte* (η²): the variance between the mean outputs for each byte value divided by the total variance of a cell's output at the reading step.
- *Single-cell model*: the model of section 5.3 below. All other terms are those of the definition and of the brief (its section 1).

## 1. Findings in order of seriousness

### Finding 1. Formula 11's sending-side test removes nearly every connection at byte 4 and thereafter keeps only connections from frequently active senders; both arms are affected alike (brief sections 2, 5.4 operation 7, 5.8 items 1 and 9, 15; definition sentences 53, 54, 57; choice C17)

*Reason.* A starting connection is tested at step 128. Its sending-side running average ū_ij then equals (1/128)·Σ (127/128)^(127−k) z_j(k), which is at most 0.633 times the sender's largest output in steps 0 to 127. A line keeps its connection only if its byte occurred in positions 0 to 3; a cell keeps its outgoing connections only if its activity reached at least 0.0123. At birth all cell strengths are 1/8 and a cell has about six excitatory sources, so activity falls by about a factor of eight per connection from the input lines (an encoder cell is at most 1/8; a cell at distance 2 is at most about 1/64). The activity bound (section 0) gives, for the three seeds:

| Seed | First four bytes | Line connections kept (of starting) | Cells whose activity bound reaches 0.0123 | Cell connections certainly given up (of starting) | All connections certainly given up |
|---|---|---|---|---|---|
| 73129 | `w58<` | 11 of 754 | 40 | 14,887 of 15,118 | 15,630 of 15,872 |
| 93683 | `p11+` | 10 of 833 | 47 | 14,794 of 15,039 | 15,617 of 15,872 |
| 116089 | `v57>` | 10 of 791 | 43 | 14,865 of 15,081 | 15,646 of 15,872 |

The bound holds with the strength changes that formula 7 can make in 128 steps (largest strength bound 0.22 to 0.39 at the fixed point). The bound was computed for formula 7; the coincidence rule's change in one step, η · e · x · (x − ϑ), is no larger than formula 7's bound for a cell whose activity is below 1/8, which holds for all but the 40 to 47 cells named in the table. After step 128 every connection is tested at every step. A new connection starts with ū = 0 and is given up 128 steps after it is made unless its sender was active enough in those four bytes; a surviving one is given up as soon as its sender has been silent for about 13 bytes (a line) or about 5 bytes (a cell whose output while active is 1/8). Task 5 (i) below gives the numbers. Every cell is then below its target activity, so each makes one connection every 128 steps, and most of these are given up 128 steps later.

The brief registers this mechanism for the input lines (X1) and for the feedback cells (X2), but not for connections between cells, which are 95 per cent of the tissue. Its two arms differ only in the strength rule (operation 5); a connection that lives for four bytes changes by formula 7 or by the coincidence rule by about 10⁻⁴ (section 5.3 below), so both arms are expected to arrive at the same state. The single-cell model confirms this: with giving up and making on, the first-form and coincidence cells are identical to four decimals at 3,000 and at 100,000 bytes.

*Consistency with Mr. Xu's words and with a living cell.* Giving up a connection whose sending terminal is silent has a living counterpart, but over hours to days; in the first form it acts within four bytes, which is shorter than the mean interval between occurrences of every byte value in the stream (16 bytes for the end mark and "=", 58 to 70 bytes for letters, digits and task bytes). The rule thereby removes exactly the connections from cells that respond to particular bytes, which is the information step 1 is to read.

*Proposed change (erratum E10 to the definition; the reviewer decides).* C17, first form: "A connection at least P_s steps old is given up when its strength is below 1/128." The sending-side clause becomes a further form of C17. Either end still determines whether a connection ends, because its strength is set by both ends (sentence 42). The change opens choice C17 and no other. Consequences to be stated in the brief: connections from never-used lines and from silent feedback cells are kept (expectation X2 is reversed, which helps step 2); under-active cells keep adding connections until their activity reaches a*, so the cost depends on the number reached (measured by the trial). Until the reviewer decides, the trial of section 6 runs both forms.

### Finding 2. The rule of success can be met at lags 1, 2 and 4 by a tissue that carries only the present byte (brief sections 3 item 3, 7.5, 7.6; contract, design block item 3)

*Reason.* The brief's countermodel C0(L), a perfect code of the present byte alone, scores 0.074 to 0.087 at L = 1, 0.050 to 0.059 at L = 2, 0.076 to 0.080 at L = 4, and 0.034 to 0.043 at L = 8 (section 17, part E), against chance 0.0217. A tissue that is at chance in W0 and carries only the present byte in W3 therefore gains 0.03 to 0.07 at L = 1, 2 and 4, and meets condition 2 there with the margin 0.02. At L = 8 it gains 0.012 to 0.021, at the margin. The contract states: "If one could pass the primary gate, the result stays E1/E2 or the gate is redesigned." The brief places C0 only among the readings (section 7.7).

*Proposed wording for section 7.6, condition 2.* "For L = 0: S(d, 0, W3) − B(d, 0) ≥ 0.02, or S(d, 0, W3) ≥ 0.95 and S(d, 0, W3) ≥ B(d, 0) − 0.01. For L ≥ 1: S(d, L, W3) − B(d, L) ≥ 0.02 and S(d, L, W3) ≥ C0(L) + 0.02." Here B is the baseline of finding 3. Because this makes the lags of four and eight bytes much harder, finding 7 applies.

### Finding 3. "The start of life" is measured on the wrong cells and the wrong population (brief sections 1 "Recording window" and "Distance d", 7.1 item 1, 7.3, 7.6)

*Reason.* (a) W0 covers bytes 0 to 8,191, that is 262,144 steps, of which only the first 128 precede the first giving up. The cells "at distance d" of W0 are those of the birth; by finding 1, nearly all the connections that define those distances are gone from byte 4 on. (b) In the plastic tissues connections turn over every few bytes, so the distance taken at the first step of a window does not describe a cell during the 8,192 positions that follow. (c) The number of cells at a distance differs between windows and tissues, and the nearest-centroid score rises with the number of features, so a change of S between W0 and W3 can come from a change of population size. (d) The roadmap's "better than at the start of life" is met most directly by the tissue as born, read late: that is the brief's fixed tissue, which the rule does not use.

*Proposed changes.* (1) Baseline B(d, L) = max(S(d, L, W0), S_fixed(d, L, W3)): the plastic tissue must beat both its own early life and its birth wiring read in the same window. (2) The distance of a cell in a window is the most frequent of its distances taken at the first step of each of the 64 blocks of 128 positions of the window; the receipt records, for each cell, the share of blocks at that distance. (3) The score is computed on a random sample of 128 cells at each distance (all if fewer), drawn by `Random(seed + 900000)` from the cells at that distance in that window, and the counts are reported.

### Finding 4. The classifier ignores amplitude, and float16 storage decides what small signals count (brief sections 7.1 item 2, 7.2, 7.3, 7.4)

*Reason.* Each feature is standardised, so a cell whose output varies with the byte at 10⁻⁴ counts as much as one that varies at 0.5. The tissue has no noise, so such variation is exact; but the brief itself calls a cell with output above 0.01 at fewer than 1/128 of positions silent. In float16, values below 6·10⁻⁵ are stored with an absolute step of 6·10⁻⁸ and values below 3·10⁻⁸ as 0, so whether a weak cell counts depends on the storage format. At birth, cells at distance 4 are expected at about 1/8⁴ of a line's effect; their S can therefore change between windows through a change of amplitude alone. In the single-cell model, the first form lowers the cell's output (mean 0.0056 to 0.0026, standard deviation 0.0080 to 0.0038) while the share of variance explained by the byte stays 1.000: by the classifier the cell is fully informative, and by section 7.2 it is constant at both times.

*Proposed wording for section 7.3, Samples.* "The features of t are the recorded outputs at t of the sampled cells at distance d, with every value below 0.01 set to 0." The same applies to the fixed tissue and to check 9.

### Finding 5. The coincidence rule as written has a threshold that follows each byte, and its inhibitory part is not the rule it cites (brief section 5.5; definition C15, second form)

*Reason.* (a) ϑ has time constant 32 steps, one byte. For an activity held at c, ϑ tends to c²/a*, so x − ϑ = c(1 − c/a*) is positive for every c below a* = 1/8. In a cell below its target, the rule strengthens every active input until the cell reaches a* in that byte's window, or until the strengths reach 1. The rule of Bienenstock, Cooper and Munro selects among inputs because its threshold averages over many stimuli; here it averages over one. (b) For an inhibitory input the same sign strengthens inhibition whenever the cell is below its target. The rule of Vogels and colleagues (2011), which the brief cites, weakens inhibition below the target. (c) Simulated (section 5.4 below), the excitatory and inhibitory strengths of the cell rise from 1/8 to 0.999 in 100,000 bytes, and the activity rises sevenfold; the dependence on the byte is unchanged (η² 0.999), because the rule adds no selectivity.

*Consistency.* The rule is a function of the two ends (section 16) and has a living counterpart. It does not use the cell's prediction, so in the coincidence tissue the inner cells no longer do what sentence 36 ([S], section 8: "predicting the information newly arriving from other cells") describes; the definition admits it as a further form of C15, and a favourable result would have to be reconciled with section 8. The brief should say so.

*Proposed wording.* "ϑ_i(k + 1) = (1 − 1/4096) · ϑ_i(k) + (1/4096) · x_i(k)² / a*, with ϑ_i(0) = a* (a time constant of 128 bytes, a new number of this brief on the grid of powers of two). For an inhibitory sending cell: Δw_ij = η · e_ij(k) · (x_i(k − 1) − a*) (Vogels and colleagues, 2011)." Both are the brief writer's choices and are to be marked as such in section 5.7.

### Finding 6. The reversal of the feedback cells' kinds in the coincidence tissue is wrong for the cells that emit (brief sections 1 "Coincidence tissue", 5.2 E3, 5.3 item 7, 6)

*Reason.* Erratum E3 (H4 of the fifth reading) reverses the kinds because the pairing of sentence 87 presupposes the cancelling sign of formula 7 at the receiving cell, and the receiving cell that matters is the output-end cell that emitted. In the brief's coincidence tissue the output-end cells keep formula 7 (operation 5). The pairing of sentence 87 is therefore the corrective one at the output end, and the reversal makes a "no" strengthen the context of a wrong emission there. It has no effect at step 1, but the brief keeps it "so that the same tissue can go on to step 2" (section 5.2).

*Proposed wording for section 5.2, E3, column "Applied here".* "Not applied: in the coincidence tissue the output-end cells keep formula 7, so the kinds of sentence 87 are kept. Whether inner cells that receive from feedback cells need the reversed kinds is for the brief of step 2." Correspondingly in sections 1, 5.3 item 7, 6 and check 1(c).

### Finding 7. The lags of four and eight bytes require a capability that the roadmap assigns to step 3 (brief sections 2, 7.6; roadmap section 3, capability e, and section 6, steps 1 and 3)

*Reason.* Step 1's criterion names lags 1, 2, 4 and 8. Capability e ("a byte's information is held in cells' states across later bytes") is introduced at step 3. With τ_v = 2 and traces of 32 steps, a byte four or eight positions back leaves 0.017 or 0.0003 in a trace and nothing in the state (section 17, F.1, F.2); it can be held only by sustained recurrent activity. The roadmap is therefore in tension with itself, and the brief inherits the tension by requiring all 15 pairs on all three seeds. With finding 2's correction, the lags of four and eight become a test of capability e.

*Proposed change.* The reviewer decides one of: (a) keep the 15 pairs, and register that step 1 then includes capability e; or (b) require L = 0, 1 and 2 and report L = 4 and 8 with their readings. The brief should state which, citing the roadmap.

### Finding 8. The brief's fallback cannot address the cause expected on paper (brief sections 10.1 to 10.3)

*Reason.* If round 1 fails, the brief runs only opening 2 (synaptic scaling in place of the threshold control, up to 20 CPU-hours). Scaling multiplies existing strengths, and connections still disappear by formula 11 within four bytes. Opening 3 (C17) is not run, because the definition gives no number for its further form. The 60 CPU-hours of the budget therefore have no branch that tests finding 1.

*Proposed change.* Replace opening 2 under this brief by the C17 change of finding 1 if the reviewer has not adopted it before release; otherwise keep opening 2.

### Finding 9. The brief resolves the conflict of E4 and E5 against the literal wording of section 16, and the resolution binds step 2 (brief sections 5.2 E5, 5.8 item 5)

*Reason.* Mr. Xu (section 16): "我们的输入端不是所有的细胞都收到yes/no，只有某些收到" (at the input end, not all cells receive the yes or no; only some do). E5 was written to meet this. The brief keeps E4 (a feedback cell has its line only) and turns E5 into a change of term only, so no cell of the input end receives the yes or no, at step 1 or, since the steps are one life, at step 2. The reason given (E4 removes the byte from feedback cells) is sound, but the decision is not the brief writer's to make silently.

*Proposed change.* Section 5.8, item 5, is put to the reviewer as an open decision before release, with the two readings named; the brief records whichever is chosen as a decision on Mr. Xu's words, not as a programmer's decision.

### Finding 10. The threshold control and the target a* make every byte-selective cell keep growing (definition formulas 9 and 12; brief sections 5.4 operations 6 and 7; choice C29)

*Reason.* Task 5 (ii) below. A cell driven only during one byte value cannot reach a mean activity of 1/8, because the shares of byte values are 0.014 to 0.062. Its threshold therefore stays at or returns to 0, and it is eligible to make a connection at 55 to 100 per cent of steps. It stops growing only when its activity reaches 1/8 on average, which requires responding at one-eighth of all positions or more. The control does not silence such a cell; it pushes it, through formula 12, towards responding to many byte values. An encoder cell driven by its line alone (fixed at 1/8) can never exceed 1/8 while its byte is present, so its mean is at most 0.002 to 0.008.

*Status.* A question that a run answers (whether growth leaves cells informative), not a fault to change before any run. The smallest candidate change, if the trial shows cells at distances 2 and 4 losing selectivity as they grow, is a* = 1/32 (C29, a number on the grid), which a cell that responds fully to one or two frequent byte values can reach.

### Finding 11. Registered expectations (brief section 15)

- X14 (the fixed tissue meets the 18 conditions, chance 0.15): the stream is stationary and the fixed tissue's thresholds settle within about ten bytes (task 5 (ii)), so W0 and W3 of the fixed tissue are equal in distribution. Its cells at distance 4 have activity of order 1/8⁴ of a line's effect and are silent by section 7.2. The checker's estimate is 0.01 or less; the reason "W0 starts with θ = 0, before the threshold control has settled" covers a few hundred of the 262,144 steps of W0.
- A new expectation should be registered: "At step 129, fewer than 300 of the 15,872 starting connections remain in each plastic tissue (finding 1)", chance 0.95.
- X1 is right in mechanism but understates it: a new connection from the line of the end mark or "=" survives its first test with a chance of only 0.25 to 0.31 (section 5.1 below).
- X12 and X13 (0.05 and 0.10): the checker agrees that failure is likely, for the reason of finding 1, which makes the two arms nearly equal; X13 should not exceed X12 by much.

### Finding 12. Cost (brief sections 8 check 10, 9)

*Reason.* The arithmetic agrees (section 4 below). The estimate assumes that connections are removed and appended only at steps when one is given up or made. By finding 1, about 16 cells make a connection at every step (2,048 cells, one each per 128 steps) and a similar number are given up, so array compaction and the Python-level making (building each cell's candidate list from about 96 members, then `rl.choice`) occur at nearly every step. At 20 to 40 microseconds a making, this adds about 0.3 to 0.6 ms a step, that is 0.7 to 1.4 CPU-hours a plastic tissue, or 4 to 8 CPU-hours over round 1. Round 1 stays under its ceiling at 12 connections a cell. Check 10 measures bytes 256 to 2,303, after the first giving up, so it will capture this; the brief should say so.

### Finding 13. Smaller completeness points

1. Section 5.4, operation 7: when a cell has no candidate, the brief does not say whether its step of last making is updated. Proposed: "If there is no candidate, nothing is made and the step of last making is not changed."
2. Section 7.3: K = 46 is stated for every window of every seed but section 17, A.6, shows seed 73129 only. The checker verified it for all four windows of all three seeds (smallest class 88 positions, in W0 of seed 116089). The check script should print it.
3. Section 1, "Distance d": a cell whose only line connection is to a byte value that never occurs, made within the last 128 steps, counts as d = 1. Proposed: "a cell with at least one connection from the line of a byte value present in the window".
4. Check 5 leaves open which byte replaces position 5. Proposed: "the byte at position 5 is replaced by the next byte value in use in increasing order of value".
5. Section 2 says the step-1 tissues are saved so that step 2 continues them; the first opening of step 2 (128 feedback cells, E2) and form (b) both need tissues born differently, so only the first form of step 2 can continue these. The sentence should say so.
6. Section 2: "Step 1 also settles ... which of the two rules of strength keeps that material." By finding 1 it cannot, unless E10 is adopted.

## 2. Task (1): faithfulness to the definition with its errata

| Place in the brief | Difference from the definition and errata | Consistent with Mr. Xu's words | Possible for a living cell |
|---|---|---|---|
| 5.5, coincidence rule | The definition names the form and gives no formula; the brief's formula (BCM with ϑ = running average of x²/a*, time constant 32 steps, same sign for both kinds) is its own | Yes for section 16 (both ends); it drops prediction at inner cells (section 8), which the definition admits as a further form | Yes, but see finding 5: the threshold is as fast as the stimulus, and the inhibitory sign is not the homeostatic one |
| 5.8 item 1, feedback cells present with silent lines | A decision on the fifth reading's open item 1 (E9) | Yes: D5 (one life) and item 4.8 (no cells added by an agent that knows the task changed) | Yes |
| 5.6, fixed tissue with threshold control | A second tissue in which operations 5 and 7 are not done | Yes: roadmap condition 3 (a second tissue from the same start) | Approximately: a lifelong block of all synaptic change and turnover has no exact counterpart, but the control acts on no learner and is acceptable as a model control |
| 5.4 operation 6, θ of feedback cells fixed at 0; formulas 7, 9, 12 not applied to them | E4 | Yes | Yes |
| 5.2 E5 and 5.8 item 5 | E5 reduced to a change of term | Departs from the literal wording of section 16 (finding 9) | Yes |
| 5.2 E3 in the coincidence tissue | E3 applied although the emitting cells keep formula 7 | Not a matter of his words; wrong for step 2 (finding 6) | Yes |
| 5.3, birth | Fixes the order of draws, the selection of feedback cells before drivers (open item 4), ties by index, and "within 1/8" as squared distance at most 1/64 | Yes (sentence 6, C54) | Not applicable (experimenter's construction) |
| 5.4, order of operations | Sentence 118 exactly; making uses θ and ā after operation 6; giving up before making, in index order | Yes | Yes |
| 5.8 item 9, line connections and formula 11 | Formula 11 applied to lines, as sentence 63 implies | Yes; but see finding 1 | The rule itself, yes; its time scale is not living |
| 4, stream | End mark byte 10 in place of the symbol 256; no start symbol; no reset | Yes (sections 8 and 12; C45) | Yes |
| 5.4, storage | float64 in the tissue, float16 records | Not applicable | Not applicable; the float16 records affect the measure (finding 4) |
| 10.3 opening 2 | A formula for synaptic scaling (multiplicative, rate 1/128 a step) that the definition only names | Yes | Yes in kind; living scaling is much slower than one step |

No sentence of the brief asks a cell for something a living cell cannot do. The two departures that matter are the coincidence rule's threshold and inhibitory sign (finding 5) and the reversal of kinds (finding 6).

## 3. Task (2): decisions left open and sentences in conflict

Open decisions: finding 13, items 1, 3 and 4; the distance of a cell within a window under turnover (finding 3); the meaning of the sampled population (finding 3). All other operations can be programmed from the brief alone: the checker could write the birth, the stream and the step from the brief's text, and the birth and stream agree with the check script's digests and counts (section 4).

Conflicts: section 2 against finding 1 (the arms cannot be told apart if connections live four bytes); section 2 against E2 (finding 13, item 5); section 5.2 E3 against operation 5 (finding 6); section 5.8 item 5 against section 16's wording (finding 9); the roadmap's step-1 criterion against its capability table (finding 7).

## 4. Task (4): re-run of the check script and independent recomputation

`python3 -I check_step1_brief.py` ran at 00:10 UTC in 23.6 seconds. Its output is identical, character for character, to the block in section 17 of the brief.

The checker's own program implemented the stream, the birth and the distances from the brief's text (sections 4 and 5.3), without the check script's code:

- Stream, seed 73129: SHA-256 `7772341c…8e8775`, as section 17; 16,362 items begun, mean item length 16.02 bytes; first 48 bytes as printed. Seeds 93683 and 116089: digests as printed; 16,355 and 16,379 items. The mean length 10 + 12/2 = 16 bytes gives 262,144 / 16 = 16,384 items, in agreement.
- Chance level: 46 byte values in every one of the four windows of all three seeds, so balanced-accuracy chance is 1/46 = 0.02174. A classifier whose assignments do not depend on the label has expected balanced accuracy (1/K)·Σ_c P(predict c) = 1/K exactly, as the brief says.
- Counts at birth, seed 73129, distances along the starting connections: d = 1: 232, d = 2: 189, d = 4: 193 (614 recorded cells); 64 not reached (the feedback cells); 240 encoder cells, 397 inhibitory, 15,872 starting connections of which 754 from lines. All agree with B.1. In addition: all 232 cells at d = 1 are encoder cells; at d = 2, 8 are encoder cells and 181 inner cells; at d = 4, all 193 are inner cells. So the coincidence rule acts on nearly all recorded cells at d = 2 and 4, and on none at d = 1.
- Cost: 2,048 · (12 · 20 + 25) = 542,720 operations a step; 80 · 3 µs + 542,720 · 1.5 ns = 1.054 ms a step; times 8,388,608 steps = 2.46 CPU-hours a plastic tissue. Fixed tissue: 2,048 · (12 · 4 + 25) · 1.5 ns + 40 · 3 µs = 0.344 ms a step, 0.80 CPU-hours. Round 1: 6 · 2.46 + 3 · 0.80 = 17.1 CPU-hours at 12 connections; at 32, 6 · 5.32 + 3 · 1.37 = 36.0. All agree. Records: 634 · 8,192 · 2 bytes = 10.4 MB a window. Agrees.
- Check 9(b), the positive control of the measure, was computed by the checker for seed 73129: the classifier on a standardised one-hot code of lags 0 to 8 gives 1.0000 at every lag in W0 and in W3, so check 9(b) can pass.

## 5. Task (5): what is visible before the full runs

### 5.1 (i) The giving up of connections from input lines

Computed per byte value on the stream of seed 73129, for a connection made at the start of a byte's window at 400 random positions, with formula 10 and 11 exactly (testing at the end of each window is exact, because within a silent window ū falls and within an active one it rises):

| Byte value (share) | Survives its first test at age 128 steps | Median life of survivors (line, output 1) | Median life of survivors (sender cell with output 1/8) |
|---|---|---|---|
| `A` (0.0146), `Z` (0.0139) | 0.06 to 0.08 | 16 to 17 bytes (longest 32) | 8 to 9 bytes (longest 16) |
| `p` (0.0156), `+` (0.0156) | 0.04 to 0.10 | 16 to 17 bytes | 8 bytes |
| `9` (0.0171) | 0.07 to 0.08 | 17 bytes | 9 bytes |
| `=` (0.0624), end mark (0.0624) | 0.25 to 0.31 | 17 bytes (longest 66 and 75) | 8 to 9 bytes |

*How fast:* a connection is given up 128 steps after it is made unless its byte occurred in those four bytes, and after about 13.3 bytes of silence once it has (5 bytes for a sender cell at 1/8). *For which bytes:* for all 46 byte values in use, not only the rare ones; the end mark and "=" are given up in 62 per cent of the gaps between their occurrences (F.5). The 210 values never used lose every connection at step 128 and at every re-making. *What a cell is left with:* an encoder cell holds, at any time, about one recent connection plus line connections of byte values that occurred in the last few bytes; the identity of its lines changes every few bytes. A longer running average does not help: with time constants of 4,096 or 8,192 steps and ū starting at 0, no new connection reaches 1/128 within its first 128 steps, so all are given up at age 128.

*Status:* a fault of the first form to be changed before any run (finding 1; erratum E10 to C17).

### 5.2 (ii) The threshold control at a cell whose input is active at about 1/46 of positions

Simulated over bytes 10,000 to 40,000 of seed 73129 for a single cell with formulas 2, 3, 9 and 10, whose arrival is c during the windows of one byte value and 0 otherwise:

| Byte value, arrival c | Mean activity (target 0.125) | Activity at the reading step of its byte | θ at those steps (largest) | Share of steps eligible to make a connection |
|---|---|---|---|---|
| `A`, 1 | 0.013 | 0.80 (smallest 0.54) | 0.20 (0.46) | 0.90 |
| `A`, 1/8 (one line) | 0.002 | 0.125 | 0 (0) | 1.00 |
| `=`, 1 | 0.056 | 0.82 | 0.19 (0.19) | 0.56 |
| `A`, 8 | 0.016 | 1.00 | 0.25 (0.67) | 0.87 |

The threshold rises during the cell's own byte (lowering its response at the reading step by about a fifth) and returns to 0 within about seven bytes; it never brings the mean activity near 1/8. The cell therefore stays eligible under formula 12 and keeps adding connections. The threshold's slow return also gives the cell a trace of having seen its byte in the last few bytes, a property that bears on lags 1 to 4. *Status:* a question for a run (finding 10), with a* = 1/32 as the candidate change.

### 5.3 (iii) The first form at a cell two connections from the input lines

*Model.* Stream of seed 73129. 24 sending cells at distance 1, each with the lines of two byte values in use (drawn by `Random(11)`), each line at the definition's fixed strength 1/8 (variant: 1), and no other source; their thresholds follow formula 9 (with strength 1/8 they stay 0 exactly). Cells 0 to 17 are excitatory and 18 to 23 inhibitory. A receiving cell R starts with 8 of them (`Random(12)`) at strength 1/8 and follows formulas 1 to 10 in the order of sentence 118; giving up and making (formulas 11 and 12, with the 24 cells as its neighbourhood) are on or off. R's activity at the reading step is read over bytes 1,000 to 3,000 and 98,000 to 100,000.

| Variant | Bytes | Mean activity | Share above 0.01 | Standard deviation | η² by the present byte | η² by the byte one back | Connections; mean excitatory strength |
|---|---|---|---|---|---|---|---|
| Fixed | 1,000 to 3,000 | 0.0058 | 0.35 | 0.0082 | 1.000 | 0.058 | 8; 0.125 |
| First form, no giving up | 1,000 to 3,000 | 0.0056 | 0.35 | 0.0080 | 1.000 | 0.058 | 8; 0.121 |
| First form, no giving up | 98,000 to 100,000 | 0.0026 | 0.07 | 0.0038 | 1.000 | 0.052 | 8; 0.063 |
| First form, senders at 1, no giving up | 1,000 to 3,000 | 0.0043 | 0.17 | 0.0077 | 0.869 | 0.072 | 8; 0.011 |
| First form, as defined (giving up on) | 1,000 to 3,000 | 0.0006 | 0.04 | 0.0033 | 0.057 | 0.020 | 1; 0.125 |
| First form, as defined (giving up on) | 98,000 to 100,000 | 0.0007 | 0.04 | 0.0032 | 0.078 | 0.016 | 2; — |

*Reading.* Without giving up, formula 7 weakens the active excitatory inputs as risk H1 says, slowly at the definition's activity level (strengths halved in 100,000 bytes) and quickly when the senders are fully active (from 0.125 to 0.011 within 3,000 bytes; the rate grows roughly with the cube of the activity). In this model without context input the cell's output stays a function of the byte (η² 1.000) while its amplitude falls towards the levels the brief calls silent and constant; whether the byte is lost relative to context in a tissue is the network question of H1, which only a run answers. With the definition as written, the giving up of section 5.1 decides the outcome before formula 7 acts: R keeps one or two connections and its activity no longer depends on the byte.

*Status:* the giving up is a fault to change before any run (finding 1); H1 is a question for a run, and the trial of section 6 can answer it at small cost.

### 5.4 (iv) The coincidence form at the same cell

| Variant | Bytes | Mean activity | Share above 0.01 | Standard deviation | η² by the present byte | Connections; mean strength (all; excitatory) |
|---|---|---|---|---|---|---|
| Coincidence, no giving up | 1,000 to 3,000 | 0.0062 | 0.35 | 0.0088 | 0.999 | 8; 0.136; 0.138 |
| Coincidence, no giving up | 98,000 to 100,000 | 0.0431 | 0.34 | 0.0617 | 0.999 | 8; 0.999; 0.999 |
| Coincidence, senders at 1, no giving up | 1,000 to 3,000 | 0.0571 | 0.35 | 0.0809 | 0.956 | 8; 0.277; 0.204 |
| Coincidence, as defined (giving up on) | 98,000 to 100,000 | 0.0007 | 0.04 | 0.0032 | 0.078 | 2; — |

*Reading.* Every active input, inhibitory included, rises to the upper bound, because the cell stays below a* and the threshold follows each byte (finding 5). The cell's activity rises and stays a function of the wiring; the rule adds no selectivity. With giving up on, the result equals the first form's to four decimals. *Status:* the threshold time constant and the inhibitory sign are faults of the brief's own formula, to be changed before release (finding 5); the giving up is finding 1.

The simulations used about eight minutes of CPU. The programs were written in this session's scratch space and are not kept; the models are fully stated above and in section 0.

## 6. Task (6): recommendation

**Run first a short trial; release the full runs only after it, with the changes of findings 1 to 6.**

*The trial.*
- Length: 16,384 bytes (524,288 steps, 4,096 periods of P_s) per tissue, seed 73129 only.
- Tissues (five, all born from seed 73129): first form with formula 11 as written; first form with formula 11 by strength only (finding 1); coincidence tissue with formula 11 as written; coincidence tissue by strength only, with the coincidence rule of finding 5; the fixed tissue.
- Read: the number of connections by kind of sender and receiver at steps 127, 129, 1,024 and every 128 bytes; activity summaries (section 7.1 item 3 and section 7.2) for cells at distances 1, 2 and 4 every 1,024 bytes, with distances by the mode of finding 3; and S(d, L) with features below 0.01 set to 0, on bytes 2,048 to 6,143 and 12,288 to 16,383, beside C0. The program, checks 1 to 10 and the analysis are those of the brief, so nothing written for the trial is lost.
- Cost: about 9 to 20 CPU-minutes for each plastic tissue (34 to 73 ms a byte at 12 to 32 connections) and 3 for the fixed tissue, with checks 1 to 10, under 2 CPU-hours in all, and about one hour of wall time with three processes.
- Decisions:
  1. If at step 129 fewer than 300 starting connections remain under formula 11 as written, and at byte 16,384 fewer than half the cells at distance 2 or 4 are neither silent nor constant there, while the strength-only tissues keep at least half: the reviewer adopts erratum E10, the brief is revised with findings 1 to 6, and round 1 is released with E10 in both arms.
  2. If both forms of formula 11 leave cells at distances 2 and 4 silent or constant: round 1 is not released; the definition's starting strength, the fixed line strength and a* (finding 10) are put to the reviewer, since neither C15 nor C29 addresses that state.
  3. If formula 11 as written does not remove the starting connections as predicted: the checker's finding 1 is wrong, and the brief is released with findings 2 to 6 only.
  4. In every case, the trial's measured cost per byte replaces the estimate of section 9.

*Reason, in costs and what each option can tell.* Releasing as written costs 17 to 36 CPU-hours for round 1 and up to 20 more for opening 2. By finding 1 its most likely result is that both arms lose their connections in the same way, which tells nothing about the strength rule that the arms compare, and its fallback does not touch the cause. Releasing after the listed changes without a trial commits 40 CPU-hours to a definition change (E10) whose consequences (growth, cost, activity at distances 2 and 4) are not known on paper. The trial costs about one twentieth of round 1, tests the prediction of finding 1 directly, measures what E10 does, and fixes the cost before the large spend.

## 7. Task (7): the contract's design block

The brief states all five items. Missing or insufficient:

1. **Recitation countermodel (item 3).** The stream countermodel C0 can pass the primary gate at lags 1, 2 and 4 (finding 2). The contract requires the gate to be redesigned. C<L is named but neither computed nor checked; check 9 should compute it, as it computes C0.
2. **Positive control (item 4).** The brief's positive control (check 9b) shows that the classifier reads a perfect code. No tissue known to stay informative under the step's conditions exists; the brief should say so plainly, and the trial supplies the nearest substitute (a tissue form that does keep cells informative, if one does).
3. **Cheap preflight before the costly run (item 5).** Checks 1 to 10 test the program, not the tissue. The contract asks for a preflight before a costly run; the trial of section 6 is that preflight.
4. **Exposure audit (item 2), for later steps.** The step-1 stream shows the 15,840 taught pairs with their answers, without feedback, in the same life that steps 2 to 5 continue. The brief should state that any control tissue for step 5 (a branch without the prior teaching) must have a step-1 phase with matched exposure, or the comparison is not matched.
5. **Promotion rule (item 5).** The rule should state what is reported if the fixed tissue meets the conditions; it does (section 7.7). Under finding 3 the fixed tissue enters the rule itself, and that sentence becomes unnecessary.

## References

- Bienenstock, Cooper and Munro (1982), Journal of Neuroscience 2: 32-48.
- Vogels, Sprekeler, Zenke, Clopath and Gerstner (2011), Science 334: 1569-1573. Their inhibitory rule changes an inhibitory strength in proportion to presynaptic activity times the postsynaptic rate minus a target rate.

Page numbers are as cited in the brief and were not checked.
