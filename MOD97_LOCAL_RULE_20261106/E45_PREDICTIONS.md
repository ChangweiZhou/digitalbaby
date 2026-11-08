# E4 (retention) and E5 (composition): design and predictions, stated before launch

**7 November 2026, before any run.** Learner for both: the harness window network (10-byte window, hidden width 256, about 155,000 parameters), stateless. The window stands in for working state at this horizon so that the life rule can stay fully local (sign-symmetric feedback within the step). Development by backprop (AdamW lr .001, decay 0.3, 24,576 updates). Life by the sign rule (AdamW lr .001, decay as stated per arm). Three seeds. Every sealed panel is scored after every stage. Shortcut ladder and shuffled control in every table.

**Addendum, same day, before launch (from Mr. Zhou):** the learner may edit its own local rules during life (learned per-synapse learning rates, gated or metaplastic writes, learned consolidation schedules). This is inside property P5 provided the edit itself uses only local information and broadcast signals. E4 and E5 as designed below do not use it; it is the first thing to add if arm C's self-generated replay proves the weak point (a learned write gate on the store), and it is a candidate component of E6.

## E4, retention

Development family: addition and subtraction mod 83 and mod 89 (codes p, q), half the pairs. Life: four new operations learned **in sequence**, 8,192 updates each, 25% of pairs each, no access to earlier tasks' data: T1 addition mod 97 (code r), T2 subtraction mod 97 (code t), T3 addition mod 73 (code s), T4 subtraction mod 73 (code u). After every stage, score sealed held panels of all four life tasks and of the development tasks.

| Arm | Life learner |
|---|---|
| A | shared path alone, sign rule, decay 1.0 (the setting of Experiments 1–3) |
| A0 | shared path alone, sign rule, decay 0.1 (separates decay-driven forgetting from interference) |
| B | shared path + fast store. **Changed at qualification, before launch:** a random sparse expansion (4,096 units, top 5%) could not separate windows sharing nine of ten bytes (the same overlap problem as the project's KC code), so the store is an exact-key episodic memory on the 10-byte window at each answer position, the perfectly separated limit of such a code (the project's HASH8), with a +6 logit on the majority stored byte and nothing for unseen keys, capacity 65,536, first-in first-out. Logits summed with the shared path; no replay |
| C | B + self-generated replay: a buffer of up to 1,024 earlier taught prompts per task; from stage 2 on, half of every batch is replayed prompts answered by the model's own output (shared path + store), not by ground truth. With an exact store the replayed answers are mostly correct, which is the complementary-systems claim: the fast memory teaches the slow path |
| C′ | C with the replayed answers replaced by random valid answers (rehearsal of inputs without content) |
| D | backprop ceiling: joint training on all tasks seen so far with ground-truth data (no forgetting by construction) |

Predictions:
- P4.1 Arm A forgets: T1 sealed accuracy falls below 30% by the end of stage 4, from at least 70% at the end of stage 1.
- P4.2 Arm A0 forgets less than A (decay is part of the forgetting) but still falls below 50% on T1.
- P4.3 Arm B retains **taught** T1 items (the store answers them by construction) but its **sealed** T1 accuracy tracks A within 10 points: the store protects memories, not the rule.
- P4.4 Arm C keeps T1–T3 sealed accuracy at or above 70% after stage 4, within 15 points of its end-of-stage value, while learning T4 to at least 70%.
- P4.5 Arm C′ tracks A: replaying inputs without content does not retain.
- P4.6 Arm D keeps all tasks at or above 85%.
Decision: E6 uses arm C's design if P4.4 holds in all seeds; if C fails but D passes, the self-generated answers are the weak point and E6 waits for a store that answers better.

## E5, composition

Development: single-step addition and subtraction mod 97 (codes r, t), half the pairs. Two-step problems are written `c a o1 b o2 c =` with code byte `c` for "compose", answer `(a o1 b) o2 c mod 97`. Operation types: (+,+), (−,−), (+,−), (−,+). Taught types in life: (+,+) and (−,−), 8,000 triples each. Held: (i) 600 new triples of the taught types (within-type), (ii) 600 triples of each untaught type (cross-type). Two renderings:
- **Direct**: the target is the final answer bytes and EOS.
- **Scratchpad**: the target is the intermediate result, a separator, then the final answer and EOS (`08;01`). At test the model must emit the intermediate itself; scoring is exact on the final answer with own EOS, intermediate correctness reported separately.

| Arm | Development | Life (8,192 updates, sign rule, decay 1.0) |
|---|---|---|
| Z | single ops | nothing (zero-shot on compositions) |
| DIR | single ops | direct compositions of taught types |
| SCR | single ops | scratchpad compositions of taught types |
| SCR-sh | single ops | scratchpad with shuffled final answers (control) |
| DIR-bp, SCR-bp | single ops | the same teaching by backprop (ceilings) |

Predictions:
- P5.1 Z scores under 5% on every composition panel (format never seen).
- P5.2 DIR cross-type is below 30% in every seed; DIR within-type is at least 60%.
- P5.3 SCR cross-type is at least 70% in every seed, and SCR within-type at least 80%; intermediate correctness at least 85%.
- P5.4 SCR-sh is under 10% on finals.
- P5.5 Single-op panels do not fall below 80% in any life arm (no forgetting of the parts within one stage).
Decision: if P5.3 holds and P5.2 holds, scratchpad composition is the form E6 uses; if DIR cross-type also passes, direct composition is used and the scratchpad becomes the fallback; if SCR fails cross-type, composition is not achieved at this scale and E6 is postponed.

## E5-R, composition with replay of the single operations (added 7 November after E4 and E5, before its launch)

Same as arm SCR, with the episodic store holding the single-step items the learner experienced in development (its taught halves of add97 and sub97) and half of every life batch being replayed single-step prompts answered by the model itself (shared path plus store), exactly as in E4 arm C. Arm SCR-R uses the sign rule; SCR-R-bp the same teaching by backprop. Compositions are never written to the store; composition panels are scored by the shared path alone.

Predictions:
- P5R.1 Single operations retained at or above 80% after life in SCR-R (they fell to 0–7% in every arm without replay).
- P5R.2 SCR-R cross-type final-answer exact at or above 70% in every seed, with the intermediate right in at least 85%.
- P5R.3 SCR-R within-type at or above 85%.
- P5R.4 SCR-R-bp cross-type stays below SCR-R (backprop overfits the taught types even with the parts retained), or, if it does not, the earlier gap was retention alone.
Decision: if P5R.2 holds in all seeds, E6 runs as designed (sequence of operations with replay, then compositions, then the early operations scored again). If it fails with the single operations retained, scratchpad composition itself is the limit at this scale and E6 is postponed.

## E5-C, canonical scratchpad (added 7 November after E5-R, before its launch)

The learner rewrites each step as a taught single-operation prompt and solves it: `c24+81-07=` → `r24+81=08;t08-07=01` EOS. The final step's window is then `08;t08-07=`, a taught subtraction prompt with leading bytes, and contains no first operator. Development for this experiment prefixes each single-operation record with 0–3 random letters so the network learns to ignore leading bytes (a stated design change; bare-prompt development never showed leading bytes). Arms under this development: SCR (plain scratchpad, as E5), SCRc (canonical, sign rule), SCRc-sh (canonical with shuffled finals), SCRc-bp (canonical, backprop). No replay (E5-R showed it hurts composition learning at this budget). Same taught types (+,+) and (−,−), same held panels.

Predictions:
- P5C.1 SCRc cross-type final-answer exact ≥70% in every seed.
- P5C.2 SCRc rewritten step prompts correct ≥85% on cross-type (both steps).
- P5C.3 SCRc-sh under 10%.
- P5C.4 SCR under this development stays ≤60% cross-type (replicating E5).
Decision: if P5C.1 holds in all seeds, P4 is established in canonical-scratchpad form and E6 resumes with that form plus E4's replay. If the rewritten prompts are right (P5C.2) but the finals fail, the leading-byte invariance is the limit; if the rewriting itself fails cross-type, the copy rule does not generalize across operator positions.

**Amendment before the E5-C rerun (same day).** The first E5-C run failed in every arm including the backprop ceiling (0–2% within-type). Teacher-forced accuracy was near 100% at every target position except the third operand of the rewritten second step (8–13%): by the time the learner writes it, the original operand has left the ten-byte window, so the rewrite is impossible by construction, a design error on my part. The rerun uses an 18-byte window in every arm (development and life, plain and canonical), which holds the whole problem plus the step being written. The first run's receipts stay in the folder as `e5c_*` without a tag; the rerun's carry `_w18`. Predictions P5C.1–P5C.4 are unchanged.

## E6-route: learned-key routing on the canonical scratchpad (added 7 November after E5-C, before launch)

RouteNet: four attention heads over the 18-byte window; query from the window, keys = learned slot-position embeddings, value = the slot's byte embedding; output head sees the four attended bytes plus a 16-d window summary ("att") or the attended bytes only ("attv"). All projections go through the sign-feedback path in life. Development with left noise on single operations, as E5-C. Arms: att, attv (sign rule), attsh (att with shuffled finals), attbp, attvbp (backprop). Same canonical scratchpad, same panels.

Predictions:
- P6.1 attbp cross-type final ≥70% in every seed.
- P6.2 att cross-type within 20 points of attbp.
- P6.3 attv cross-type ≥ att cross-type (the values-only head is more invariant).
- P6.4 attsh <10%; within-type ≥85% in att and attv.
Decision: P6.1 and P6.2 both held → P4 established with routing; E6 resumes with routing + canonical scratchpad + E4 replay. attbp fails → routing of this form is insufficient and the composition line stops here.

## E7, de-confounded composition with four operators (added 7 November, before launch)

Operations mod 97 with codes and symbols: add `r` `+`, sub `t` `-`, max `x` `>`, min `n` `<`. Development: all four single-step operations, half the pairs each, with 0–3 random leading letters, 18-byte window, hidden 256, backprop. Life: two-step problems `c a o1 b o2 c =`; the 16 ordered operator pairs are split into 12 taught (2,000 triples each) and 4 held: (+,−), (−,+), (>,<), (<,>). Every operator appears in both positions among the taught pairs. Held panels: 600 new triples per held pair; a within-type panel of 600 new triples pooled over four taught pairs. Arms: SCR plain scratchpad and SCRc canonical scratchpad with the sign rule; SCRc-sh shuffled finals; SCR-bp and SCRc-bp backprop ceilings. 8,192 updates, three seeds, shortcut ladder as before.

Predictions:
- P7.1 SCRc held-pair final-answer exact ≥70% in every seed.
- P7.2 SCR held-pair ≥40% in every seed.
- P7.3 SCRc-sh <10%.
- P7.4 SCRc-bp held-pair ≥85%.
Decision: P7.1 holds → P4 achieved in the de-confounded regime, the four earlier failures are explained by the confound, and step 2 is skipped; the persistent-life demonstration (step 5) then includes compositions. P7.1 fails but P7.4 holds → the local rule is the limit; test it at width 512. Both fail → binding needs a mechanism; step 2.

## E7-b, the local rule at width 512 and at double dose (added 7 November after E7, before launch)

E7 result: backprop plain scratchpad 90–93% held-pair mean (P4 achieved under backprop; the confound explained the earlier failures); sign rule 37–38% plain and 7–67% canonical, with within-type only 39–42% plain, so the local rule is under-trained on the twelve-pair teaching set at 8,192 updates. By the rule on file the local rule is tested at width 512. Two arms, plain scratchpad with the sign rule: (i) width 512, 8,192 updates (new births at 512); (ii) width 256, 16,384 updates. Three seeds each. Backprop at 256 remains the reference.

Predictions:
- P7b.1 width 512, sign rule, plain: held-pair mean ≥60% in every seed.
- P7b.2 width 256 at 16,384 updates, sign rule, plain: held-pair mean ≥50% in every seed.
Decision: either holds → P4 is achieved under the local rule in the de-confounded regime, and step 5 (persistent life) includes compositions. Both fail → the local rule's ceiling on composition is below backprop's at this scale; report it and proceed to step 5 without compositions, leaving steps 2–4 as options.

**Clarification from Mr. Zhou, 7 November, during E7-b.** A local rule does not mean a fixed small neighbourhood of neurons; which neurons a synapse takes its signals from may be adjusted dynamically during life. Consequences for the ladder: (1) learned feedback pathways (feedback weights that adapt during life, as in Kolen–Pollack or weight-mirror rules) are inside P5 and are the natural next arm if the sign rule stays below backprop on composition; (2) credit through time carried by dynamically weighted traces across neurons, rather than each unit's own gate, is inside P5 and is the natural next arm for the working-state rung. Neither is run yet; E7-b (width and dose) completes first.

## E8, the persistent-life demonstration (step 5; added 7 November, before launch)

One learner, 18-byte window, hidden 512, developed by backprop on add/sub/max/min mod 89 with left noise. Life by the sign-symmetric rule: eight new operations in sequence (add, sub, max, min mod 97, then the same four mod 73; distinct code bytes), 6,144 updates each, 25% of pairs each; then a ninth stage teaching two-step compositions of the mod-97 operations (the twelve taught operator pairs of E7, plain scratchpad, 8,192 updates). Arms: C store + self-generated replay (the demonstration); A no replay; D backprop on joint ground truth (ceiling). Every operation's sealed panel (600 pairs) scored after every stage; held composition pairs and within-type after the last. Three seeds.

Predictions:
- P8.1 Arm C: after the ninth stage every one of the eight operations is at or above 70% on its sealed panel; arm A has the first four below 10%.
- P8.2 Arm C: held composition pairs at or above 40% mean, within-type at or above 55% (replay halves the composition batch; E7-b at width 512 without replay reached 52–68%).
- P8.3 Arm D: all eight operations at or above 85%, held pairs at or above 85%.
- P8.4 Arm C's development operations (mod 89) are not retained (no replay covers them); reported, not gated.
Decision: P8.1 holding in all seeds is the persistent-learning demonstration (P1, P3, P5 in one life, P2 via the window); P8.2 adds composition in the same life. Either failing is reported with its stage-by-stage curve.

## E10, the ring with innate operations (step 3; added 7 November, before launch)

Model: the prompt is parsed by position into operand slots (two digits each) and operator slots; this parsing is supplied and stated as such. Learned interfaces, shared across slots: digits → a bump (softmax over 97 positions); operator byte → a soft choice among four innate operations on bumps (add = circular convolution, sub = circular correlation, max and min from products of cumulative distributions); final bump → two digit softmaxes. Composition is wiring: the intermediate bump feeds the second operation. A flat ablation keeps the interfaces and replaces the innate operations by a learned MLP on the concatenated bumps. Development by backprop on single-step problems (four operations mod 97, half the pairs). Life by the sign rule or backprop on two-step problems. Panels: E7's four held pairs and within-type; and the E5-style confounded set, taught (+,+) and (−,−), held (+,−) and (−,+). Three seeds.

Predictions:
- P10.1 Ring, single-step development only, zero-shot on two-step panels: ≥90% on every panel.
- P10.2 Ring, life on the confounded two-pair set only: ≥90% on the two untaught cross pairs.
- P10.3 Flat ablation: <30% cross-type on the confounded set; ≥70% on E7's held pairs after life on the twelve taught pairs.
- P10.4 Ring under the sign rule (twelve taught pairs) within 5 points of backprop.
Decision: P10.1 and P10.2 holding establish that composition can be innate at this scale with learning confined to interfaces; the supplied parsing then becomes the next thing to learn. P10.1 failing with P10.2 holding localizes the failure to the interfaces' transfer across prompt formats.

**Amendment before the E10 rerun (same day).** The first E10 run's development did not converge in 8,192 updates: single-operation accuracy after development was 5–93% depending on seed and operator (the 97-way bump encoder trains slowly through the innate operations, add and sub slowest). The qualification had checked the innate operations on exact bumps but not the learnability of the interfaces at that dose; that was my omission. In the one seed where development reached 93% on max/min, the ring with backprop life scored 92–100% on E7's held pairs and 100% on the confounded cross pairs, against 2–9% for the flat ablation. The rerun uses 32,768 development updates in every arm (tag `_d32768`); the first run's receipts stay untagged. Predictions P10.1–P10.4 are unchanged.

## E11: a diagonal memory cell with a time-local trace inside the E8 learner (7 November 2026, written before launch)

**Question.** Property P2 (working state) under a local rule, inside the persistent learner. Exp 3 showed that a self-path eligibility trace through a full GRU fails (2–14%) because credit runs through other units, while credit through time by the sign rule works (68–86%). Codex's second sweep found that for a *diagonal* cell the trace equals backprop through time. E11 puts such a cell into the E8 learner and asks whether the local, time-local rule (a) re-uses holding developed at birth and (b) acquires holding that birth never required, in a sequential life with the E8 store and replay.

**Learner.** The E8 window net (hidden 512) with a 10-byte window plus a diagonal memory cell of 512 units: `p = tanh(Wp z1)` (64-d), `c = tanh(Wc p + bc)`, `r = sigmoid(Wr p + br)`, `h_t = r ⊙ h_{t-1} + (1 − r) ⊙ c`, readout `z2 = tanh(W2 z1 + Wm h)`. Each unit reads only its own past state; the gates do not read `h`. The per-synapse trace `e_t = r_t e_{t-1} + ∂h_t/∂w` makes `Σ_t L_t e_t` the exact gradient for `Wc, bc, Wr, br` with `L_t` the readout learning signal at step `t`. Qualification (`exp45/QUALIFY_E11.json`): trace versus backprop through time, max relative error 1e-6 on all four tensors; within-step rule, 42–47% off. 800,081 parameters.

**Task.** `<code> a a <op> D1..Dk b b = c c EOS`, k uppercase letters between operator and second operand. The first operand is inside the window at the answer only for k ≤ 3 (qualification), so k ≥ 4 needs the cell. Sealed panels per operation at k = 0, 8, 12 and 20 (20 never presented), 400 held pairs each with seeded letters; the four development operations at k = 12 (300).

**Births** (backprop through time, 24,576 updates, four operations mod 89, codes p q v w, half the pairs): **M** memory birth, k ∈ 0..12; **N** no-memory birth, k ∈ 0..3 (letters familiar, holding never required). **Life**: four operations mod 97 (codes r t x n), a quarter of the pairs each, 6,144 updates per stage, k ∈ 0..12, in sequence, with the exact-key store and self-generated replay of half of every batch (E8 arm C): **T** sign rule everywhere and the exact trace for the cell's synapses; **S** sign rule everywhere, cell synapses by the within-step rule (state detached); **D** backprop through time on the joint ground truth of everything seen (ceiling, no store). AdamW lr .001, decoupled decay 1.0 (local) / 0.3 (backprop), as E8. Audit at every evaluation: the k = 12 panels scored again with the cell's state forced to zero ("silenced").

**Predictions.**
- P11.1 Birth M reaches ≥ 85% exact at k = 12 on all four mod-89 operations in every seed, with k = 20 within 10 points of k = 12. Birth N reaches ≥ 90% at k = 0 and ≤ 15% at k = 12.
- P11.2 From birth M: MT ≥ 70% at k = 12 on each mod-97 operation after its own stage; MS at least 15 points below MT at k = 12 in every seed; MD ≥ 85%. If MS ≈ MT the memory birth does not discriminate the rule and P11.3 decides.
- P11.3 From birth N (acquisition of holding by the local rule): NT ≥ 40% at k = 12 after each stage in at least two seeds (low confidence); NS ≤ 15% in every seed; ND ≥ 70%.
- P11.4 Extrapolation: at k = 20, T and D arms are within 10 points of k = 12 wherever k = 12 ≥ 50%.
- P11.5 Retention: after the fourth stage, in T arms each of the first three mod-97 operations is within 15 points of its post-stage k = 12 score, and the mod-89 development operations at k = 12 are within 20 points of their birth score.
- P11.6 Silenced audit: with the cell's state zeroed, k = 12 falls to ≤ 10% wherever the live score is ≥ 50%.
- Cost (smoke, qualification): T 190 ms per update, S 34, D 38. The trace is one scalar per synapse per sequence in the batch (2 × 64 × 512 × 64 floats), carried forward in time; no backward pass through time.

**Commands** (from `MOD97_LOCAL_RULE_20261106/`): `python3 exp11_cell.py qualify`; `python3 exp11_cell.py dev {M|N} SEED`; `python3 exp11_cell.py {M|N}{T|S|D} SEED`; seeds 73129 93683 116089.

**Amendment (7 November 2026, after stage 2 of the first launch).** The first launch's replay buffer was filled with *fresh* renders of each stage's items (new distractor letters), which had never been presented and whose exact keys the store did not hold, so every replay answer was an unanchored self-generated guess. That is not the E8 mechanism (E8 replays presented items, whose keys the store holds). Observed under the flawed buffer: from the no-memory birth under the trace rule, add97 at k = 12 fell from 90% to 2–4% during the subtraction stage and subtraction itself learned poorly (52% at k = 0). The buffer now holds 1,024 of the stage's presented prompts, as E8. Stage 1 (no replay yet) is deterministic and unchanged; the partial receipts of the first launch are archived under `exp45/archive_e11_freshbuffer/`. No prediction was changed.

**Second amendment (7 November 2026, after stage 2 of the second launch).** With presented prompts in the buffer, replay still failed to retain (add97 from the memory birth under the within-step rule: 80–85% at k = 0 → 1–4% during the subtraction stage). Cause: with distractor letters every presentation is a new exact key; a stage writes about 393,000 distinct prompts (three keys each) into the 65,536-key store, which evicts almost all of them within the stage and all of the previous stage's by the end of the next, so over 95% of replay answers were unanchored self-generated guesses. E8's taught set was 2,350 fixed renders, so its store held every key and replay was, in effect, the presented answer. E11 now keeps that explicitly: the buffer holds 1,024 episodes per stage (prompt and answer as presented) and replay draws from them; the exact-key store is still written and read at evaluation, as before. This is E8's effective mechanism with the store's capacity dependence on random letters removed; it is not self-generated replay, and the report will say so. Partial receipts of the second launch are archived under `exp45/archive_e11_storeoverflow/`. No prediction was changed.

## E11-D: why the trace learner forgets (7 November 2026, written before launch)

**Question.** E11 refuted retention (P11.5): under the sign rule with half-batch episode replay, the E8 learner plus the diagonal cell keeps only the current operation, while the backprop ceiling keeps everything. Four candidate causes, one change each, on the memory-birth trace learner (arm MT, three seeds, everything else as E11). Codex's E13 (same day) found that at batch 64 decay 0.25 alone lifts E8's retention from 76–80% to 94% and held composition from 48–55% to 81–85%, so decay is the first arm.

| Arm | Change from E11 MT | Tests |
|---|---|---|
| MT_wd0.25 | decoupled decay 0.25 instead of 1.0 on the local arms | decay as the cause (E13-A1) |
| MT_fr | replay re-renders the stored pairs with fresh letters and k, answer as stored | rote learning of fixed-letter episodes |
| MT_int | the four operations presented interleaved in one stream of 4 × 6,144 updates, no replay, scored after each 6,144 | blocked presentation (E9: interleaving holds while stationary) |
| MT_wd0.25_fr | decay 0.25 and fresh-render replay | the two together |

Birth: `e11_dev_M_{seed}.pt` unchanged. Panels, seeds, doses, store, loss, optimizer otherwise unchanged. Receipts `exp45/e11_life_MT{tag}_{seed}.json`; E11's MT and MD receipts are the baseline and ceiling.

**Compute note, on record.** The trace gradient is computed by the backward scan of the learning signal (`trace_grads_scan`), which is the forward trace's sum re-ordered: Λ_s = L_s + r_{s+1} Λ_{s+1}, dLoss/dWc = Σ_s (Λ_s ⊙ gc_s)ᵀ p_s. Qualification: scan versus forward trace, maximum relative error 1e-6 on all four tensors. The updates are therefore identical to E11's; the scan is not local in time and is used only for speed. The locality claim rests on the forward form, which E11 ran.

**Mechanism audit (what is carried over and the condition each needs).** Episode buffer replay (E11): needs the stored episodes to be the presented ones; satisfied by construction, 1,024 per stage. Fresh-render replay (new): needs the pair and operation recoverable from the stored prompt; the prompt's bytes 1–2 and the two before '=' are the operands, the operation is the stage's; checked in the dry run. Interleaved stream (new): `stream()` already draws uniformly over all task-pair entries; the buffer is unused. The exact-key store (E8/E11): written at every update and read at evaluation; inert for held items; no condition. Births: same architecture file, loaded by path. Dry run of the full loop at 4 updates per stage passed for all four arms.

**Predictions.**
- P11D.1 Decay 0.25: add and sub at k = 12 after the fourth stage ≥ 50% in every seed (E11 MT: 0–15%); the development operations mod 89 at k = 12 after the first life stage ≥ 40% (E11: 2–41%). If this holds, decay is the dominant cause, as in E8.
- P11D.2 Fresh-render replay at decay 1.0: add and sub final at least 20 points above E11 MT in every seed, otherwise fixed letters are not the cause. Low confidence.
- P11D.3 Interleaved, no replay: all four operations ≥ 70% at k = 12 after the fourth block in every seed (E9 analog: 91–92% in the window net).
- P11D.4 Combined: add and sub final at least as high as the better single arm in every seed.
- Shortcut ceiling as E11: max and min count as memory only above about 54%.

**Commands** (from `MOD97_LOCAL_RULE_20261106/`, seeds 73129 93683 116089): `E11_TRACE=scan E11_WD=0.25 python3 exp11_cell.py MT SEED`; `E11_TRACE=scan E11_REPLAY=fresh python3 exp11_cell.py MT SEED`; `E11_TRACE=scan E11_SCHEDULE=interleaved python3 exp11_cell.py MT SEED`; `E11_TRACE=scan E11_WD=0.25 E11_REPLAY=fresh python3 exp11_cell.py MT SEED`.
Smoke timing before launch: scan trace 37 ms per update, within-step 32 ms (forward trace 205 ms).

## E15: the fixed settings applied (7 November 2026, written before launch; Codex's E14 running in parallel)

Four one-change parts at the settings E11-D fixed (decay 0.25 on local arms; fresh-render replay of stored pairs; trace by the scan form, identical to the forward trace at 1e-6). Three seeds each.

| Part | Arm | Change | Question |
|---|---|---|---|
| A | MT_wd0.25_int | E11 memory birth, interleaved four-operation stream, no replay, decay 0.25 | can the sign rule learn a mixed stream once decay is fixed (E11-D's interleaved arm at decay 1.0 failed in two seeds) |
| B | NT_wd0.25_fr | E11 no-memory birth, blocked, decay 0.25 + fresh-render replay | acquisition of holding and its retention in one life |
| C | M_g2 births + MT_g2_wd0.25_fr | split write gate w and read gate o on the cell (`h = r h + w c`, readout `o ⊙ h`), new births by backprop through time, then the fixed-settings life | Codex's shelf check: separate gates 87–98% against 57–82% tied; does it hold in this learner |
| D | ring_sign_wd0.25, ring_conf_sign_wd0.25 | E10 ring, sign-rule life at decay 0.25 (births d32768 unchanged) | does lower decay stop the interface erosion that refuted P10.4 |

**Mechanism audit.** Interleaved stream: `stream()` draws uniformly over all task-pair entries (E11-D, ran). Fresh-render replay: pairs recovered from prompt bytes, stage's operation (E11-D, ran). Split gates: the trace is re-derived (gc = w(1 − c²), gr = h_{t−1} r(1 − r), gw = c w(1 − w), each with the r-recurrence; the read gate is instantaneous and trained by autograd); qualification compares forward trace, scan and backprop through time on the six cell tensors. Births for C are new (architecture changed); the baseline for C is E11-D's MT_wd0.25_fr (69 / 65 / 56 add+sub mean), whose births differ only by the absent gates. Ring decay: a single optimizer argument; default reproduces E10. Dry runs of all four parts passed (4 updates per stage; 512-update dry birth for C).

**Predictions.**
- P15.1 (A) With decay 0.25 the interleaved learner reaches ≥ 70% at k = 12 on add and sub in every seed after the four blocks (E11-D at decay 1.0: 78 / 15 / 12 on add). If it still fails in a seed, the mixed-stream failure is not decay and becomes a first-order problem for the language stage.
- P15.2 (B) From the no-memory birth under the fixed settings, add after its stage ≥ 60% at k = 12 (as E11 NT: 90 / 78 / 63) and after the fourth stage add+sub mean ≥ 40% in every seed (E11 NT: 26 / 24 / 25 total mean, add+sub ≈ 1).
- P15.3 (C) Split gates: births reach ≥ 85% at k = 12 as the single-gate births did; the fixed-settings life ends with add+sub mean at least 5 points above the single-gate learner in every seed (69 / 65 / 56), and k = 20 within 10 points of k = 12. If within ±5, the gates do not matter at this scale.
- P15.4 (D) At decay 0.25 the ring's single-operation interfaces after the sign-rule life are within 3 points of their birth values in every seed (at decay 1.0 they eroded), and held-pair composition does not fall during life.

**Commands** (from `MOD97_LOCAL_RULE_20261106/`): A `E11_TRACE=scan E11_WD=0.25 E11_SCHEDULE=interleaved python3 exp11_cell.py MT SEED`; B `E11_TRACE=scan E11_WD=0.25 E11_REPLAY=fresh python3 exp11_cell.py NT SEED`; C `E11_GATES=split python3 exp11_cell.py dev M SEED` then `E11_GATES=split E11_TRACE=scan E11_WD=0.25 E11_REPLAY=fresh python3 exp11_cell.py MT SEED`; D `E10_DEV_DOSE=32768 E10_WD=0.25 python3 exp10_ring.py ring_sign SEED` and `ring_conf_sign SEED`. Seeds 73129 93683 116089.

## E16: the integrated toy learner, the Phase A exit gate (8 November 2026, written before launch; Codex's E14-S in parallel)

One E11 cell learner (10-byte window, hidden 512, diagonal cell, exact trace in scan form) from the E11 memory births, at the adopted settings (decay 0.25, fresh-render replay at half of every batch, batch 64), on one stream carrying every toy family in six blocks of 6,144 updates: F1 the four mod-97 operations with 0–12 distractor letters (blocks 1–2, interleaved), F2 the twelve taught operator pairs in E8 scratchpad form, no distractors (blocks 3–4), F3 the four mod-73 operations with distractors (blocks 5–6, interleaved). Scored after every block: F1 and F3 sealed held pairs at k = 0, 8, 12, 20 (400 each); the four held cross pairs (600 each) and 600 held triples of taught pairs; the development operations mod 89 at k = 12; the silenced-cell audit on every panel. Arm T is the learner; arm D is a backprop-through-time twin trained at decay 0.3 on batches drawn with equal weight from the families seen so far. Three seeds (73129, 93683, 116089). Harness `exp16_integrated.py`; mechanics qualification `exp45/QUALIFY_E16.json`; dry runs of the full six-block loop on both arms passed at 4 updates per block (one slip found and fixed in arm D's batch assembly before launch).

**Mechanism audit (what is carried over and the condition it needs here).** Memory births: E11's (`e11_dev_M_*`), developed on mod-89 operations with k ≤ 12; the mod-97 and mod-73 codes and the composition code byte `c` are new to them. Interleaved stream: uniform over a family's entries (E15-A, ran). Fresh-render replay: entries are (family, operands); composition entries re-render identically since they carry no letters (the E8 way); operation entries get fresh letters and k (E11-D, ran); the round trip from presented bytes back to an entry is asserted in qualification. The buffer takes 1,024 entries per block, so after six blocks it holds 6,144 and the replay half of a batch is 32 draws from it; F1 replay therefore thins as later families arrive, which is the retention test. The E8 exact-key store runs beside the net as in E11/E15; with letters it overflows and is close to inert, and for compositions it holds only taught prompts, so it cannot score a held pair. Window: the composition prompt is 10 bytes, so the code byte is visible only for the first output byte and the first operand leaves the window during the intermediate; both operators stay visible through the final answer. The twin sees the same prompts with the same window, so the gate is relative to it. Shortcut ceilings on the held pairs (final answer from a visible-only rule: ignore the first operator, ignore the second, copy c): 1–2% for +- and -+, 34–69% for >< and <> (as in E8, the comparison pairs are weakly informative; the carry pairs are the test). Response cap raised to 6 bytes for the scratchpad answers.

**Predictions.**
- P16.1 (coexistence, the gate) After block 6, arm T is at or above 80% of arm D on each family: F1 mean at k = 12, F3 mean at k = 12, held-pair final-answer mean. The twin itself is expected at ≥ 90 / ≥ 90 / ≥ 80.
- P16.2 (retention after the stream moves on) Arm T's F1 mean at k = 12 after block 6 is within 10 points of its value after block 2 in every seed; add and sub separately within 20 points. (E11-D at these settings, blocked: 52–72% on add+sub against a ceiling of 88–98 after four operation stages.)
- P16.3 (composition by the local rule inside the integrated learner) Arm T's held-pair final-answer mean ≥ 70% after block 4 in every seed, and the carry pairs +- and -+ each ≥ 60%; E14 saw one pair in one seed of six fail, so one such failure here would be consistent with E14, two would say the integrated setting makes it worse.
- P16.4 (the cell is still what holds) After block 6 the silenced-cell audit on F1 at k = 12 costs arm T at least 30 points, and k = 20 is within 10 points of k = 12 (extrapolation survives the later families).
- P16.5 (the twin) Arm D holds everything: F1 ≥ 90% at k = 12, held composition ≥ 80%, F3 ≥ 90% at k = 12 after block 6. If the twin itself fails composition, the 10-byte window is the limit for that family and E17 inherits the problem.

**Forecast.** Unloaded 36.5 ms per update for arm T on a mixed batch and 31 ms for arm D; six processes on this machine; forecast taken from the first block's measured pace once it prints, not from the smoke test (E15 ran at about 940 s of CPU per 6,144-update block with twelve processes).

**Commands** (from `MOD97_LOCAL_RULE_20261106/`): `python3 exp16_integrated.py T SEED` and `python3 exp16_integrated.py D SEED`, seeds 73129 93683 116089.

**E16 verdicts (8 November 2026, after the run; details in `E45_REPORT.md`, E16).** P16.1 refuted (F3 0.88–0.97 of the twin; F1 0.87 / 0.85 / 0.69; composition 0.61 / 0.58 / 0.49). P16.2 refuted (F1 −10.6 / −12.0 / −20.8 from block 2 to block 6). P16.3 holds (held pairs 87.0 / 84.3 / 87.0 after block 4; carry pairs 75–84%). P16.4 holds (silenced cell 3–7%; k = 20 within 2–9 points of k = 12). P16.5 holds (twin 95 / 93 / 94 on F1, composition, F3). Measured pace: 13–15 minutes per block with six processes; 80 minutes for the six lives.

## E16-D: why the integrated learner forgets (8 November 2026, written before launch)

E16 acquired every family within 1–9 points of the twin and then lost the earlier ones when the stream changed form (operations −11 to −21 over four blocks, compositions −30 to −41 over two), with the earlier family in the replay half of every batch. Two differences from the twin can carry that: how earlier families are sampled (the learner replays a 1,024-entry-per-block buffer uniformly, so an earlier family's share and operand diversity both shrink; the twin draws fresh items of every seen family with equal weight) and the learning rule itself. Three arm-T controls on the E16 stream, one change each, same births, seeds and settings (`exp16_integrated.py`, `E16_REPLAY`):

| Control | Change from E16 arm T | Question |
|---|---|---|
| oracle | the whole batch is the twin's joint stream (fresh items of every seen family, equal weight), learned by the sign rule with the trace | with perfect replay, does the local rule retain like backprop? |
| balanced | the same 1,024-per-block buffer, kept per family; each replay draw from a family chosen uniformly | is it the share? |
| all_balanced | every distinct presented entry kept per family; draws balanced per family | is it the share plus operand diversity? |

**Mechanism audit.** The default path (`uniform`) is untouched and reproduces E16 (the new branches are entered only when a per-family buffer exists or the mode is oracle). Oracle shares the twin's stream constructor and seeds, so its data are the twin's data exactly; it differs from arm D only in the rule. In balanced and all_balanced the current family is also in the buffer from its second block, as the twin's current family is in its joint stream. all_balanced buffers hold at most the family's entry count (9,408 / 24,000 / 5,328); sampling is over distinct entries, then fresh-rendered. Dry runs of all three modes passed on the full six-block loop (4 updates per block).

**Predictions.**
- P16D.1 (oracle) With the twin's data the local rule keeps F1 within 5 points of its block-2 value and the held pairs within 10 points of their block-4 value in every seed. If so, the E16 forgetting is replay sampling, not the rule. If the oracle still loses more than 10 points on the compositions, the sign rule interferes on its own under a balanced stream and no replay design fixes it.
- P16D.2 (balanced) Balancing the share alone halves the composition loss (≤ 20 points instead of 30–41) but stays more than 10 points below the oracle on the carry pairs: the 2,048 replayed triples are memorized rather than kept as a rule (the E11-D fixed-episode effect on operands).
- P16D.3 (all_balanced) Within 5 points of the oracle on every family in every seed: share plus diversity is enough, and the 1,024-entry buffer was the limit.
- P16D.4 Acquisition of each family while current is unchanged by any control (within 3 points of E16 arm T), since the new half of the batch is the same.

**Forecast.** Nine processes on twelve cores (six performance); E16 ran 13–15 minutes per block with six processes, so about 20 minutes per block, two hours for the nine lives; corrected from the first block's print.

**Commands:** `E16_REPLAY={oracle,balanced,all_balanced} python3 exp16_integrated.py T SEED`, seeds 73129 93683 116089.

**E16-D amendment, 8 November 2026 12:15 EST, written at block 3 of 6, before any result of blocks 4–6 is seen.** Codex's review of the E16-W brief applies here. With 1,024 entries per block, the `balanced` arm changes expected family shares only where a family has more blocks in the buffer than another: during F2.2 it lowers F1's replay share from 2/3 to 1/2, and during F3.1, where E16 lost most of the compositions, the buffer holds two F1 blocks and two F2 blocks, so balanced and uniform have the same expected shares and differ only in repetition and batch correlation. P16D.2 therefore has almost no intervention behind it in the block that matters and is withdrawn as a test of share; the balanced arm is kept as the control that isolates diversity when read beside `all_balanced` (same shares, every distinct entry instead of 1,024 per block). The `oracle` arm differs from `all_balanced` in share (F2 at one third of the batch instead of one quarter during F3, and the current family at one third instead of one half) as well as in drawing from the full family; it bounds what the local rule can retain with the twin's data, and share is not isolated by any pair of arms. Reading rule for the results: oracle ≈ twin → rule not the limit; all_balanced ≈ oracle → diversity is the limit; all_balanced ≈ balanced ≈ E16 → neither diversity nor repetition, and the remaining differences are share and the rule, to be separated by a follow-up with an explicit share arm and a backprop arm on the learner's own batches.

## E16-D2: the two arms E16-D lacks (8 November 2026 12:40 EST, written before E16-D's results and before launch)

Prepared while E16-D runs so that the follow-up Codex's review called for can start as soon as E16-D is read. Two arms on the E16 stream, same births, seeds and settings (`exp16_integrated.py`): **share** (`E16_REPLAY=share`): per-family buffers of 1,024 per block, the replay half split equally among the earlier families and the current family never replayed, so F1 is 50% of every batch during F2 (33% in E16) and F1 and F2 are 25% each during F3 (20% each in E16); exposures are implied by the schedule and recorded as buffer sizes. **M**: backprop through time at decay 0.25 on exactly arm T's uniform-replay batches (same buffer, same draws, no store), so the rule is the only difference from T. Dry runs of both passed on the full six-block loop. Whether both launch depends on E16-D: if `all_balanced` already matches the oracle, diversity is the answer and only M is needed to close the rule question; if `all_balanced` does not, both run.

**Predictions.** P16D2.1 (share) Raising the earlier families' exposure cuts the F1 loss during F2 and the F2 loss during F3 by at least a third relative to E16 arm T in every seed; if the losses are unchanged, share is not the cause. P16D2.2 (M) Backprop on T's own batches keeps the held pairs within 10 points of their block-4 value and F1 within 5 points of its block-2 value; if M loses as much as T, the replay data are the limit under either rule and the rule is cleared; if M retains, the sign rule with the trace is implicated given these data. P16D2.3 Acquisition of each family while current is within 3 points of E16 arm T in both arms.

**Commands:** `E16_REPLAY=share python3 exp16_integrated.py T SEED`; `python3 exp16_integrated.py M SEED`.

**E16-D verdicts (8 November 2026, 13:00 EST; details in `E45_REPORT.md`, E16-D).** P16D.1 holds: oracle F1 −0.6 / −0.1 / −1.2 from block 2, compositions −0.3 / −1.0 / +0.4 from block 4. P16D.2 withdrawn pre-result; balanced ≈ uniform observed (compositions 65 / 56 / 49 against 57 / 53 / 46). P16D.3 holds on compositions (84.5 / 79.8 / 83.3 against oracle 82.9 / 84.0 / 87.3) and F3, misses on F1 in seed 116089 (77.7 against 89.6). P16D.4 holds within 3–4 points. E16-D2 launched 13:00 EST with both arms (the rule on file: all_balanced did not match the oracle on every family).

**E16-D2 verdicts (8 November 2026, 14:25 EST; details in `E45_REPORT.md`, E16-D2).** P16D2.1 refuted (share: operations' loss during F2 15 / 24 / 44 against 15 / 20 / 39; compositions' loss during F3 24 / 30 / 37 against 30 / 31 / 41). P16D2.2 holds on operations (M within 0.8 / 2.9 / 1.5 of block 2), misses on compositions (M 93.6→74.2, 93.2→65.4, 93.2→76.7). P16D2.3 holds (share compositions after block 4 88.5 / 84.5 / 86.0 against 87.0 / 84.3 / 87.0). Codex's E16-W reached the same conclusions on the window net.

## E16-R: bounded replay, the second attempt at the Phase A gate (8 November 2026 14:50 EST, written before launch)

E16-D showed that replaying every presented entry (unbounded) restores retention; E16-D2 and E16-W showed that share is not the lever and the rule is secondary. The scaling requirements forbid a buffer that grows with life, so E16-R asks at what fixed per-family capacity operand coverage suffices. Replay mode `reservoir` (`exp16_integrated.py`, `E16_REPLAY=reservoir E16_CAP=C`): each family keeps a list of at most C entries filled by Algorithm R over its presentations (one counter and one fixed-size list per family; bounded memory, constant work per event; duplicates allowed, distinct count recorded), draws balanced per family, fresh-render as before. Capacities **2,048 / 4,096 / 8,192**, three standing seeds, arm T; the E16 twin (D), the E16-D `balanced` arm (2,048 entries for F1 and F2 by construction) and the `all_balanced` arm (9,408 / 24,000 / 5,328 distinct) are the comparators and are not rerun. Nine lives.

**Mechanism audit.** The family label is still supplied by the harness (the learner-side stratification is the follow-up noted in 9.30). Reservoir over presentations, not over distinct entries, so that memory stays bounded; at C = 2,048 the F1 reservoir holds about 1,850 distinct entries of 9,408 (close to the balanced arm's 2,048 fixed entries, but uniform over the family's history rather than two 1,024 samples); at 8,192 about 5,500 of 9,408 for F1 and about 7,400 of 24,000 for F2. Dry run passed on the full loop (cap 100, replacement verified: 99 distinct of 100). A name collision between the response-length cap and the reservoir cap was found in the first dry run and fixed before launch (the reservoir would have held six entries).

**Predictions.**
- P16R.1 (monotone in capacity) After block 6, held-composition mean and F1 mean at k = 12 rise with capacity in every seed; C = 2,048 is within 5 points of the balanced arm on both, C = 8,192 within 5 points of all_balanced on both.
- P16R.2 (the gate) At C = 8,192 the learner is at or above 0.80 of the twin on compositions and F3 in every seed and on F1 in at least two seeds (all_balanced: compositions 0.91 / 0.87 / 0.89, F3 0.99 / 0.97 / 0.95, F1 0.94 / 0.95 / 0.82). If so, the Phase A gate is met by a bounded mechanism, with the family-label caveat.
- P16R.3 (the knee) C = 4,096 recovers at least two thirds of the gap between the balanced arm and all_balanced on held compositions in every seed (balanced 64.8 / 55.6 / 49.1; all_balanced 84.5 / 79.8 / 83.3).
- P16R.4 Acquisition while current within 3 points of E16 arm T.

**Forecast.** Nine processes on a free machine; E16-D ran at 18–19 minutes per block with nine; about two hours.

**Commands:** `E16_REPLAY=reservoir E16_CAP={2048,4096,8192} python3 exp16_integrated.py T SEED`, seeds 73129 93683 116089.

**E16-R verdicts (8 November 2026, 16:15 EST; details in `E45_REPORT.md`, E16-R).** P16R.1 monotone holds; 2,048 within 5 of balanced on compositions (1.3 / 5.0 / 4.9) and on F1 in two seeds (0.9 / 1.0 / 6.9); 8,192 within 5 of every-entry on F1 (1.5 / 1.1 / 2.1) but 8.2 / 7.1 / 7.0 short on compositions: partial. P16R.2 ratios at 8,192: compositions 0.82 / 0.79 / 0.81, F3 0.97 / 0.97 / 0.96, F1 0.93 / 0.94 / 0.85: misses by 0.01 in one seed. P16R.3 recovered fractions 46 / 69 / 56%: misses in two seeds. P16R.4 literal misses in three of eighteen comparisons (+3.5, +3.8, +3.8), all better acquisition. Measured pace 17 minutes per block with nine processes.

## E16-R2: one reservoir, one total budget, no family labels (8 November 2026 16:30 EST, written before launch; launch waits for Codex's E16-WR to free the machine)

E16-R's reservoir was per family, which needs a label the learner is not given and is bounded only while the number of families is fixed (Codex, 9.33). Replay mode `total` (`E16_REPLAY=total E16_CAP=C`): a single reservoir of C entries filled by Algorithm R over every presentation of the life, one counter, no family key anywhere in the replay path; the replay half is drawn uniformly from it. The receipt records, for reading only, how the label-free reservoir happens to be composed by family. Capacities **24,576** (equal to E16-R's three reservoirs of 8,192) and **65,536**; three standing seeds; six lives. Comparators: E16-R reservoir 8,192, E16-D every-entry, the twin.

**Mechanism audit.** In this stream every family occupies two of six blocks, so by the last block a presentation-uniform reservoir holds about equal numbers of each family's entries, and the 24,576 arm should match the per-family 8,192 arm closely; the comparison therefore tests whether the label does anything here, not whether label-free replay survives unequal family durations (that needs a stream with unequal blocks, noted for later). During blocks 3–4 the label-free reservoir is 1/2 to 2/3 compositions (presentation share), against the per-family arm's fixed thirds, so F1's replay share is lower in this arm during the composition blocks; if F1 falls more there than in E16-R, that is the label's one real effect in this stream. Dry run passed (cap 300: reservoir composition shifted from all-F1 to 128 / 82 / 89 by family across the life, as expected).

**Predictions.**
- P16R2.1 (the label is unnecessary) After block 6 the 24,576 arm is within 5 points of the per-family 8,192 arm on every family in every seed (8,192: F1 88.3 / 88.1 / 79.8; compositions 76.3 / 72.7 / 76.3; F3 90.8 / 90.2 / 90.1).
- P16R2.2 (more total budget closes the gap) At 65,536 the held compositions after block 6 are within 3 points of the every-entry arm (84.5 / 79.8 / 83.3) in every seed and the ratio to the twin is at or above 0.85 on compositions; F1 within 3 points of every-entry.
- P16R2.3 The label's one effect: during the composition blocks the 24,576 arm's F1 at block 4 is lower than the 8,192 arm's by at most 5 points (8,192: 86.7 / 83.7 / 64.6).
- P16R2.4 Acquisition within 4 points of E16 arm T on both families (the tolerance is widened from 3 after E16-R, before launch).

**Forecast.** Six processes on a free machine: about 15 minutes per block, 95 minutes. **Commands:** `E16_REPLAY=total E16_CAP={24576,65536} python3 exp16_integrated.py T SEED`.

**E16-R2 verdicts (8 November 2026, 17:20 EST; details in `E45_REPORT.md`, E16-R2).** P16R2.1 holds on 8 of 9 (compositions seed 116089: 70.3 against 76.3). P16R2.2 partial (compositions −2.0 / −5.0 / −5.4 from every-entry; ratios 0.88 / 0.81 / 0.83; F1 +3.0 / +1.6 / +5.4). P16R2.3 misses in direction (label-free F1 at block 4 higher by 3.7 / 4.7 / 10.0); the audit's reasoning about the reservoir's composition during the composition blocks was wrong (updated at block ends). P16R2.4 holds. Gate at 65,536: F1 0.97 / 0.96 / 0.88, compositions 0.88 / 0.81 / 0.83, F3 0.99 / 0.98 / 0.95, all ≥ 0.80.
