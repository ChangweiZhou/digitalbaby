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

## E16-R3: the parser removed; verbatim raw-byte replay at the same budget (8 November 2026 17:35 EST, written before launch; launch after Codex's E16-WR frees the machine)

E16-R2's reservoir still stores parsed entries (family, operands) and re-renders them with fresh letters: a parser inside the replay path that no learner is given. Replay mode `raw` (`E16_REPLAY=raw E16_CAP=65536`): the same single label-free Algorithm-R reservoir, but each slot holds the presented bytes, prompt and answer exactly as they appeared, and replay returns them verbatim; no task recognition, no operand extraction, no recomputed or re-rendered answer anywhere. The receipt records the distinct count, the composition by first byte (for reading only) and the reservoir's size in bytes. One capacity, 65,536, three standing seeds, three lives; comparator E16-R2's `total 65536` (same budget, parsed and re-rendered) and the twin. Dry run passed (cap 300; about 16 bytes per slot).

**Mechanism audit.** The only difference from E16-R2's arm is what a slot holds and that replay skips `render`; the reservoir update, the draw, the batch assembly and the learner are shared code. E11-D found that replaying 1,024 fixed episodes per stage lets the window path memorize those prompts (fixed letters) instead of keeping the rule; at 65,536 slots over the life the stored prompts cover most operand pairs and many letter strings, so the question is whether verbatim coverage at this budget suffices where fixed-episode replay at 1,024 did not. Answers replayed are the observed correct answers (as in every cell-harness arm), not the learner's own, so E16-V's wrong-target problem does not enter.

**Predictions.**
- P16R3.1 Verbatim replay at 65,536 is within 5 points of E16-R2's parsed arm on every family in every seed after block 6 (parsed: F1 92.8 / 90.8 / 83.1; compositions 82.5 / 74.8 / 77.9; F3 92.8 / 90.4 / 88.6). If it is more than 10 points lower on the operations at k = 12, fixed letters are being memorized and the parser-free version needs more budget or a different storage form, to be found by a capacity point at 131,072 before anything else changes.
- P16R3.2 The gate holds without the parser: ratio to the twin at or above 0.80 on every family in every seed.
- P16R3.3 Extrapolation survives: k = 20 within 10 points of k = 12 on F1 after block 6, and the silenced-cell audit on F1 stays at or below 10% (family mean).
- P16R3.4 The reservoir holds at most 65,536 × 25 bytes, about 1.6 MB, recorded exactly in the receipt.

**Forecast.** Three processes on a free machine, about 13 minutes per block, 80 minutes. **Command:** `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T SEED`.

**E16-R3 verdicts (8 November 2026, 18:55 EST; details in `E45_REPORT.md`, E16-R3).** P16R3.1 holds (raw against parsed: F1 +0.4 / −0.9 / −0.9; compositions −2.3 / 0.0 / +2.2; F3 −0.6 / +0.3 / −3.0). P16R3.2 holds (ratios F1 0.98 / 0.96 / 0.87, compositions 0.86 / 0.81 / 0.85, F3 0.98 / 0.98 / 0.92). P16R3.3 holds (k = 20 within 1.2 / 2.4 / 4.9 of k = 12; silenced 2.6 / 5.5 / 6.8%). P16R3.4 holds (1,048,860–1,049,796 bytes). Measured pace 6 minutes per block with three processes.

## E16-S and E16-L: the environment's half of the scaffold, one change each (8 November 2026 19:05 EST, written before launch; launch when Codex's E16-WR2 has had the machine)

After E16-R3 the learner's replay path is label-free, parser-free and bounded. Two scaffolds remain on the environment's side: the stream arrives in six blocks of one family each, and the loss is masked to the answer bytes. Each is removed as one change from E16-R3's arm (raw bytes, 65,536 slots, observed answers), with the twin re-run where the stream changes. Harness knobs `E16_STREAM`, `E16_ADMIT`, `E16_LOSS` in `exp16_integrated.py`; dry runs of all four arms passed on the full loop.

| Arm | Change from E16-R3 | Question |
|---|---|---|
| S0: `raw 65536, online` | the reservoir admits each presented item as it arrives (Algorithm R per item) instead of at checkpoint ends | a control: a stream without stages needs online admission, so its effect is measured alone first |
| S1: `raw 65536, drift, online` | no stages: each new item's family is drawn from weights that move smoothly through the life (raised-cosine bumps, half-width half the life: F1 alone at the start, F1 and F2 equal at one third, F2 alone at the middle, F2 and F3 equal at two thirds, F3 alone at the end); scored at the same six checkpoints | does retention survive when the stream never announces a boundary |
| S-D: `D, drift` | the twin on the drifting stream (joint ground truth of every family whose weight has been positive, equal weight) | the comparator for S1 |
| L: `raw 65536, loss all` | next-byte loss on every byte of prompt and answer (padding excluded), the answer mask removed; scoring unchanged | does the learner still acquire and keep the families when the answer is not pointed to |

**Mechanism audit.** Online admission changes only when a raw slot is filled; the count of admitted items is recorded per checkpoint (verified: 64 + 32 per update after the first). The drift stream shares the family entry streams and seeds across arms; the twin's "seen" set is evaluated at checkpoint starts (F2 enters at one third, F3 at two thirds). Under drift, F1's exposure falls to zero by mid-life and is never replayed by schedule; what the learner keeps of it is what the reservoir and the rule keep. The every-byte loss changes the training signal only; replay, reservoir, scoring and shortcut ceilings are untouched; the store's write mask follows the loss mask. The L arm's comparator is the answer-loss twin (conservative: an every-byte twin would likely be lower).

**Predictions.**
- P16S.0 (control) Online admission is within 3 points of E16-R3 on every family in every seed.
- P16S.1 (no stages) At the last checkpoint S1 is at or above 0.80 of S-D on every family in every seed, and within 5 points of S0 on F1 and on compositions (the drift stream gives F1 slightly less total exposure and F3 slightly more, so F1 may sit a few points lower).
- P16S.2 (twin) S-D holds every family at or above 90 / 85 / 90.
- P16L.1 (acquisition under every-byte loss) F1 after checkpoint 2 within 10 points of E16-R3 (93.6 / 92.7 / 90.4) and compositions after checkpoint 4 within 10 points (87.6 / 86.6 / 89.2): the answer bytes are a quarter to a third of the loss, so acquisition is slower but not blocked.
- P16L.2 (retention under every-byte loss) At the last checkpoint within 5 points of E16-R3 on every family, ratio to the twin at or above 0.80 on every family in at least two seeds. Predicting the prompt bytes is rehearsal of the operands' format and should not hurt retention; if L is more than 10 points below on compositions, the answer mask was doing work that a language stream will not do for us, and the loss weighting becomes a design question.

**Forecast.** Twelve lives; raw arms cost about 2,150 CPU-seconds each alone; with twelve processes about 15 minutes per checkpoint, 90 minutes. **Commands:** `E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online python3 exp16_integrated.py T SEED`; `... E16_STREAM=drift E16_ADMIT=online ... T SEED`; `E16_STREAM=drift python3 exp16_integrated.py D SEED`; `E16_REPLAY=raw E16_CAP=65536 E16_LOSS=all python3 exp16_integrated.py T SEED`.

## E16-A: the architecture audit, E16-R3 with and without the exact-key store (8 November 2026 19:20 EST, written before launch; queued behind E16-S/E16-L)

Since E4 the local arms have carried E8's exact-key episodic store beside the network (65,536 ten-byte keys, gain 6 on the output logits at scoring, written on every training batch). It is the functional descendant of the mushroom-body associative memory that left the computational path on 6 November, and it has never been ablated in the cell harness (E16-V, check 5). E16-A runs E16-R3's arm (raw bytes, 65,536 slots) with the store absent from the local arm (`E16_STORE=off`: no writes, no logit bias), three standing seeds, three lives; comparator E16-R3. Default path unchanged; dry run passed; the receipt's `settings.store` records the state.

**Mechanism audit.** With distractor letters nearly every single-operation prompt is a new key, so the store overflows within a block and its bias at scoring is near zero; composition prompts carry no letters, so the store memorizes taught triples, but every scored composition panel (held pairs, held triples of taught pairs) is unseen and gets no bias. The store can therefore help only the k = 0 single-operation panels on prompts that happen to have been presented, and those are held pairs by construction. The expected effect is nil.

**Predictions.** P16A.1 Without the store, every family at every checkpoint is within 2 points of E16-R3 in every seed (F1 93.2 / 89.9 / 82.2; compositions 80.2 / 74.8 / 80.1; F3 92.2 / 90.7 / 85.6 at the end). P16A.2 Acquisition at k = 0 after block 2 within 2 points as well (the one place a lookup could show). If both hold, the store is removed from the harness defaults and the record states that the original PN–KC–MBON architecture has left the main model rather than survived as a module; if the store accounts for more than 5 points anywhere, the record says where and the component stays until its role is understood.

**Command:** `E16_REPLAY=raw E16_CAP=65536 E16_STORE=off python3 exp16_integrated.py T SEED`, three seeds, about 40 minutes alone on the machine.

**E16-S and E16-L verdicts (8 November 2026, 21:20 EST; details in `E45_REPORT.md`, E16-S and E16-L).** P16S.0 holds within 3.4 (F1 −3.4 / −0.2 / −1.7; compositions +0.9 / +0.9 / +2.1; F3 −0.6 / +0.7 / +3.8; one comparison 0.4 over the tolerance). P16S.1 holds (ratios to the drift twin F1 0.97 / 0.93 / 0.84, compositions 0.86 / 0.87 / 0.86, F3 0.99 / 0.96 / 0.97; against S0 within 5 except compositions in 93683, +5.7). P16S.2 holds (94–95 / 93–94 / 93–94). P16L.1 refuted (F1 after block 2: 32 / 41 / 33 against 93 / 93 / 90). P16L.2 refuted (every family far below; ratios 0.32–0.59). A twin under the every-byte loss (`E16_LOSS=all python3 exp16_integrated.py D SEED`) is queued to separate objective from rule; prediction written now: backprop under the every-byte loss acquires the first operations to at least 80% at k = 12 after block 2 in every seed; if it does not, the objective is the limit under either rule.

**E16-A verdicts (8 November 2026, 22:10 EST).** P16A.1 holds: no-store against store differs by 0.00 on every operation panel at every checkpoint in every seed and by at most 0.04 points on compositions (one item in 600, seeds 73129 and 116089). P16A.2 holds (k = 0 after block 2 identical). The store is inert; retired for new harnesses, knob kept in the E16 harness for receipt parity.

## E17: the window removed (8 November 2026 22:40 EST, written before launch; births overnight, lives after)

The 10-byte window is the last structural crutch in the learner and the one a language stream cannot have. E17 replaces it with recurrent state only: every byte arrives alone through an 8-d embedding, and all working memory is the diagonal state. Two architectures (`arch17.py`, selected in `exp16_integrated.py` by `E16_ARCH`):

| Arm | Architecture | Change from the E16 learner |
|---|---|---|
| cell1 | the E11 cell as it is: input-driven gate r, `h = r h + (1 − r) c`, 512 units | the window's 80-d input becomes the current byte's 8-d embedding; nothing else |
| bank | a bank of diagonal units with fixed retention rates in five groups (0.5, 0.8, 0.95, 0.99, 0.999), an input-driven write gate, `h = R h + w c`; no learned recurrence | as above, and the learned retention gate replaced by fixed timescales: a trainable temporal basis |

Everything else is E16-R3's arm: sign rule with the exact trace (scan form), decay 0.25, the raw-byte reservoir of 65,536 with observed answers, the answer-masked loss, blocks, three standing seeds; the twin is backprop through time on the same architecture; the store is absent (E16-A). New births are needed (the old ones have a window): `python3 exp16_integrated.py dev SEED` under `E16_ARCH`, exp11's development on the windowless net (mod-89 operations, 0–12 letters, 24,576 updates by backprop through time).

**Staging.** Phase 0, the births, is a feasibility test of the architectures under backprop: if a windowless twin cannot learn the development operations, the local-rule question is moot for that architecture. Phase 1, the lives, runs only for architectures that pass.

**Mechanism audit.** The trace stays exact because no gate reads the state: for both architectures the backward scan reproduces backprop through time on the state's synapses to 1e-7 (`exp45/QUALIFY_E17_cell1.json`, `QUALIFY_E17_bank.json`), and the within-step (state-detached) gradient differs from it by 46–51% (cell1) and 93–97% (bank), so the trace is load-bearing, more so for the bank. The readout keeps a direct path from the current byte (z1 from the 8-d embedding), so the net can still emit the successor of the byte it is reading; everything about operands, operators and position must now be carried in h. The response loop recomputes the prefix at each generated byte, as before. Pace under load: 100–125 ms per update (the state loop is the cost; the window net was 36 ms), so a life is about 4,100 s of CPU and a birth about 2,700 s. Dry runs of births (512 updates) and lives (4 updates per block) passed for both architectures and both arms; the store is off by default for these architectures.

**Predictions.**
- P17.0 (feasibility, the births) Both windowless twins reach at least 85% at k = 12 on all four development operations in every seed (E11's windowed births: 91–99%). If the bank fails and cell1 passes, fixed timescales cannot carry the parse and a learned retention gate is necessary; if both fail, one diagonal layer is insufficient without a window and a second layer is the next step.
- P17.1 (the window was not the learner) On the E16 stream, cell1 under the local rule ends within 10 points of E16-R3 on every family in every seed (E16-R3: F1 93.2 / 89.9 / 82.2; compositions 80.2 / 74.8 / 80.1; F3 92.2 / 90.7 / 85.6) and at or above 0.80 of its own twin.
- P17.2 (fixed timescales suffice) The bank is within 10 points of cell1 on every family in every seed. If the bank is more than 10 points below on compositions only, the learned gate is doing the parse of the two-operator prompt.
- P17.3 (extrapolation improves without the window) At k = 40, where the window never helped, both windowless learners are within 10 points of their k = 12 score on F1 after block 2 (a k = 40 panel is scored in addition to k = 0, 8, 12, 20 for this experiment).
- P17.4 (the window was not the retainer) Composition loss from block 4 to block 6 under the local rule is no larger than E16-R3's (7.4 / 11.8 / 9.1 points) plus 5 in every seed.
- P17.5 Composition acquisition is the risk: if either windowless learner is below 60% on held compositions after block 4 while its twin is at or above 85%, the windowless learner has a credit problem the windowed cell did not.

**Commands.** Births: `E16_ARCH={cell1,bank} python3 exp16_integrated.py dev SEED`. Lives: `E16_ARCH={cell1,bank} E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py {T,D} SEED`. Receipts `e17_dev_{arch}_{seed}`, `e17_life_T_{arch}_raw65536_{seed}`, `e17_life_D_{arch}_{seed}`. Forecast: six births on a free machine about 50 minutes; twelve lives about 2.5 hours; the lock `/tmp/MINIFLY_MACHINE_LOCK` is held throughout.

**E17 amendment (8 November 2026, 23:10 EST, before the lives launch; births not yet started).** After the static audit: (i) the held and within composition panels are re-keyed by split and pair and drawn operand-disjoint from every taught triple, so E17's composition panels are fresh 600-triple panels rather than E16-R3's; P17.1's 10-point tolerance on compositions therefore also absorbs panel sampling (about ±2 points); (ii) every life saves its final weights and records the hash of the birth it loaded; (iii) the pipeline admits a birth only from a complete receipt with its checkpoint hash recorded, and holds the host lock under its own name. No prediction changes.

**E16-L twin verdict (8 November 2026, 23:15 EST).** Backprop under the every-byte loss: mod-97 operations 92 / 91 / 94 at k = 12 after block 2 (prediction: ≥ 80 in every seed): holds. The objective is not the limit under backprop; the local rule on the recurrent learner is (ratios to this twin 0.31–0.35 on operations, 0.53–0.61 on compositions). Next arm: position-weighted learning signal into the state's synapses (λ = 0 and 0.1 on prompt positions), one change from E16-L.

**E17 P17.0 verdict (8 November 2026, 23:35 EST).** Refuted for both architectures: cell1 reaches 23–31% on add89 and 13–17% on sub89 at k = 12 after 24,576 updates; bank 22–30% and 9–13%; comparisons 53–73%; k = 0 no better. Lives not launched. Phase 0b (births only) follows with three one-change variants; predictions below.

## E17, Phase 0b: three one-change variants against the birth gate (8 November 2026 23:55 EST, written before launch; births only, overnight)

P17.0 was refuted for both windowless architectures under backprop. The reading in the report is representational: writes into the state depend on the current byte only, so two digits can interact only in the readout's single hidden layer. Three variants, births only (backprop through time, 24,576 updates unless stated), same gate (at least 85% at k = 12 on all four development operations in every seed), three standing seeds each, nine births (`arch17.py`, `E16_ARCH`):

| Variant | Change from Phase 0 | What it tests |
|---|---|---|
| cell1d2 | a second hidden layer in the readout between (z1, h) and the logits | whether the readout's depth was the limit while the state's content was sufficient |
| bank2 | a second fixed-rate bank whose write is computed from the first bank's state | whether writes that depend on accumulated context are what is missing |
| bank at 49,152 updates | double dose, nothing else | the slope: whether the single bank is merely slow |

**Mechanism audit.** Trace qualification (`exp45/QUALIFY_E17_cell1d2.json`, `QUALIFY_E17_bank2.json`): cell1d2's trace reproduces backprop through time on the state's synapses to 1e-7; bank2's second bank likewise, while its first bank's trace deviates by 64–88% from backprop through time because credit that reaches the first bank through the second bank's memory is not carried (the e-prop truncation). This matters for a future local-rule life, not for these backprop births. Births are scored at k = 0 / 4 / 8 / 12 / 20 as before; the gate script admits a birth only from a complete receipt with its checkpoint hash recorded; no life is launched by this phase.

**Predictions.**
- P17.0b.1 (readout depth) cell1d2 improves add89 and sub89 at k = 12 by at least 15 points over cell1 in every seed but does not reach the gate: the state's content, not the readout, is the limit.
- P17.0b.2 (context-dependent writes) bank2 reaches the gate in at least two seeds. If it does, context-dependent writes are the missing operation and the local-rule life on bank2 is the next run, with its first bank's truncated trace named as the known approximation.
- P17.0b.3 (dose) the single bank at double dose gains at most 15 points on add89 and stays below 60%: dose is not the answer.
- P17.0b.4 If none of the three reaches the gate in any seed, a byte-at-a-time learner needs state-dependent writes in a non-diagonal recurrence, for which no exact local trace exists; that becomes a design decision for Mr. Zhou (e-prop approximation, or a short window of two or three bytes as an explicit compromise), not a run.

**Commands.** `E16_ARCH={cell1d2,bank2} python3 exp16_integrated.py dev SEED`; `E16_ARCH=bank E11_DEV_UPDATES=49152 python3 exp16_integrated.py dev SEED` (receipt `e17_dev_bank_dry49152_SEED`, the dose tag). Nine births, about 45 minutes for the six standard ones and 75 for the double-dose three, under the host lock.

**E17 Phase 0b verdicts and Phase 1 launch (9 November 2026, 08:45 EST).** P17.0b.1 holds (cell1d2 add89 54–67 and sub89 45–59 at k = 12: +27 to +36 and +32 to +45 over cell1, below the gate). P17.0b.2 holds in every seed (bank2 add89 93–95, sub89 92, max/min 95–99: the windowed births' level). P17.0b.3 holds (bank at 49,152 updates: add89 28–29, sub89 12–14; no gain). P17.0b.4 not reached. Context-dependent writes are the missing operation. **Phase 1 launches now on bank2 only:** P17.1–P17.5 apply with bank2 as the windowless learner and `e17_life_D_bank2_*` as its twin. One clause added before launch: the first bank's trace is the e-prop truncation (64–88% from backprop through time in qualification), so if the local life falls more than 15 points below its twin on acquisition of F1 while the births matched, the truncated credit to the first bank is the suspect, and the separating arm is a life with the first bank frozen after birth (only the second bank and the readout learn), to be run before any other change. Commands: `E16_ARCH=bank2 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T SEED`; `E16_ARCH=bank2 python3 exp16_integrated.py D SEED`.

## E17, Phase 1b: the first bank frozen after birth (9 November 2026, 09:20 EST, written before launch; runs after Phase 1 under the lock)

Phase 1's first block already shows the pre-registered pattern: the bank2 twin learns the first family to about 90% at k = 12 in one block while the local learner stands at 1–3%, where the windowed cell learner stood at 85–93% at the same point. The clause written before Phase 1 names the suspect, the first bank's truncated trace (64–88% from backprop through time in qualification), and the separating arm: a life in which the first bank's write path (`l1`, `lp`, `lc`, `lw`) is frozen at its birth values and only the second bank (`lp2`, `lc2`, `lw2`, whose traces are exact given the first bank's state) and the readout learn. Knob `E16_FREEZE=bank1` (`exp16_integrated.py`; dry run on both arms with the frozen tensors verified unchanged and the rest changed). Arms: T frozen (`E16_ARCH=bank2 E16_FREEZE=bank1 E16_REPLAY=raw E16_CAP=65536`) and D frozen (`E16_ARCH=bank2 E16_FREEZE=bank1`), three standing seeds, six lives.

**Predictions.**
- P17.1b.1 (the truncated trace is the cause) With the first bank frozen, the local learner acquires F1 to at least 70% at k = 12 after block 2 in every seed (Phase 1 unfrozen: expected under 20% from its first block). If so, the sign rule with exact traces works on this architecture and the failure of Phase 1 is the truncated credit to the first bank, which is then the design question (an approximate trace for stacked diagonal layers, or a frozen developmental first bank, as the fly's sensory periphery is).
- P17.1b.2 (what freezing costs the twin) The frozen twin stays within 5 points of the unfrozen twin on F1 after block 2: the first bank's birth features are sufficient for the operations.
- P17.1b.3 If the frozen local learner still fails (under 40% on F1 after block 2) while its frozen twin learns, the truncated trace is not the cause; the sign rule's credit through two diagonal layers is, and the next arm compares sign feedback against exact feedback on the readout path alone (a diagnostic, not a mechanism).
- P17.1b.4 Compositions and retention are reported as in P17.1–P17.5 but are secondary until P17.1b.1 is decided.

**E17 Phase 1 verdicts (9 November 2026, 12:50 EST; six lives complete).** P17.1 refuted in every seed: the bank2 learner under the local rule ends at 0–2% on every family, below its own birth (95 on the development operations at birth, 1 after block 1). P17.5's premise fails: the bank2 twin reaches 16–21 on held compositions after block 4, not 85; two stacked fixed-rate banks under backprop learn the operations (F1 92–96, F3 90–93) and not the two-operator compositions (held 16–21, within 53–60) where the windowed twin reached 93. P17.2 and P17.3 not evaluable; P17.4 void. The probe of the final weights refutes the state-runaway account: under the sign rule the state is silenced (first bank |h| at a few hundredths, second-bank gates shut in two seeds and jammed open in the third, readout drive from the state near zero) while the twin keeps the birth's scale. Report section "E17, Phase 1 and 1b".

**E17 Phase 1b verdicts at block 2 (9 November 2026, 12:50 EST; lives continue to block 6).** P17.1b.1 refuted (F1 at 21.3 / 15.2 / 31.8 after block 2). P17.1b.2 holds (frozen twin 92.9 / 94.1 / 94.4, unfrozen 93.6 / 94.3 / 93.4). P17.1b.3's condition met in every seed: the truncated trace is not the cause. Unforeseen by the clauses: the frozen learner acquires F1 in block 1 (70.7 / 71.4 / 80.6) and loses it in block 2, the first block with replay.

## E17, Phase 1c: feedback or data (9 November 2026, 12:50 EST, written before launch; runs under the lock after Phase 1b)

The frozen learner and its frozen twin differ in the feedback path (sign-symmetric against exact), the update through time (the exact trace against backprop, which the frozen first bank makes equivalent for the learned synapses), the decay (0.25 against 0.3) and the data pipeline (half-batch replay from the raw reservoir from block 2 against the twin's joint stream). Two arms, three standing seeds each, six lives: `E16_ARCH=bank2 E16_FREEZE=bank1 E16_FEEDBACK=exact E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T SEED` (receipt `e17_life_T_bank2_frz1_xfb_raw65536_SEED`; exact feedback through every layer, trace and pipeline unchanged; dry-run with the frozen tensors verified unchanged) and `E16_ARCH=bank2 E16_FREEZE=bank1 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M SEED` (receipt `e17_life_M_bank2_frz1_raw65536_SEED`; the twin's update on the learner's pipeline, online admission now included for M as for T).

- P17.1c.1 (the feedback is the cause) The exact-feedback learner holds F1 at or above 70% at k = 12 after block 2 in every seed and at or above 60% after block 6 (sign feedback: 15–32 after block 2). If so, the sign-symmetric error signal through two diagonal layers is what the stacked architecture cannot tolerate, and the local-rule question for windowless learners is the feedback path, not the trace.
- P17.1c.2 (the data pipeline is not the cause) Arm M holds F1 at or above 85% after block 2 and after block 6 in every seed, within 5 points of the frozen twin: the onset of replay does not hurt the exact update.
- P17.1c.3 If the exact-feedback learner also loses F1 in block 2 (under 40 after block 2) while M holds, neither feedback nor trace is the cause; the separating arm is then the frozen bank2 under the local rule on the twin's stream (`E16_REPLAY=oracle`, no reservoir), which removes the replay while keeping the rule.
- P17.1c.4 If M loses F1 too (under 60 after block 2), the learner's data pipeline breaks the frozen bank2 under any rule, and the arm to run is the frozen bank2 on F1 alone without replay.
- P17.1c.5 Compositions are reported and not predicted: the twin itself does not acquire them on this architecture (Phase 1).

**E17 Phase 1c verdicts (9 November 2026, 18:00 EST; six lives complete).** P17.1c.2 holds in every seed: arm M (backprop on the learner's pipeline, first bank frozen) holds F1 at 92.4 / 95.5 / 95.2 after block 2 and 93.0 / 92.5 / 92.1 after block 6, within 1 point of the frozen twin; the onset of replay does not hurt the exact update. P17.1c.1 refuted: with exact feedback the frozen learner stands at 45.8 / 49.9 / 66.9 after block 2 (sign feedback: 21.3 / 15.2 / 31.8) and at about 10 after block 3, 15–17 after block 6. P17.1c.3's condition (under 40 after block 2) is not met in any seed at block 2 and is met in every seed one block later. Before taking P17.1c.3's named arm, the learner's gradient was compared with backprop through time's directly (`arch17.qualify bank2 up`, `exp45/QUALIFY_E17_bank2_up.json`): with the first bank frozen and exact feedback, the two agree on every learned synapse except the projection `lp2` that feeds the second bank's write, where the within-step gradient the learner uses has relative error 0.87 to the exact credit through time; the scan extended one layer upstream reproduces that credit to 1.6e-7. The "exact trace" was exact for the write synapses (`lc2`, `lw2`) and not for the projection feeding them. That is the one remaining difference from arm M apart from decay (0.25 against 0.3), so the named oracle arm is set aside for the measured gap.

## E17, Phase 1d: the trace extended to the write projection (9 November 2026, 18:15 EST, written before launch; runs under the lock now)

`E16_TRACEUP=proj` (arch17.TRACEUP): given the scan's Lambda, the credit through time to the projection p2 = tanh(Wp2 h1) is dL/dp2_s = (Lambda_s * gc_s) Wc2 + (Lambda_s * gg_s) Ww2, delivered through sign(W) under sign feedback or W under exact feedback, and dL/dWp2 = sum_s (dL/dp2_s * (1 - p2_s^2))^T h1_s; it replaces the within-step gradient. No new approximation given Lambda (qualification: 1.6e-7 against backprop through time with the first bank frozen). Two arms on the frozen bank2, three standing seeds, six lives: `E16_ARCH=bank2 E16_FREEZE=bank1 E16_TRACEUP=proj E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T SEED` (receipt `e17_life_T_bank2_frz1_tup_raw65536_SEED`) and the same with `E16_FEEDBACK=exact` (`e17_life_T_bank2_frz1_xfb_tup_raw65536_SEED`). Dry runs archived (frozen tensors unchanged, `lp2` changed).

- P17.1d.1 (the trace gap was the cause) With exact feedback and the extended trace, the frozen learner holds F1 at or above 80 after block 2 and at or above 75 after block 6 in every seed (arm M: 92–95; the only remaining difference is decay 0.25 against 0.3). If so, the collapse of Phases 1b and 1c was the truncated credit to the write projection: a trace gap, not the feedback and not the pipeline, and the E11 cell escaped it only because its input window made within-step credit sufficient.
- P17.1d.2 (what sign feedback costs on this architecture) With sign feedback and the extended trace, F1 at or above 60 after block 2 and at or above 50 after block 6 in every seed (without the extension: 15–32 and 17–20). The gap between P17.1d.1 and P17.1d.2 is the cost of sign-symmetric feedback here.
- P17.1d.3 If the exact-feedback, extended-trace learner still falls under 40 after block 2 while M holds, identical gradients collapse under one decay and not the other, and the arm is the same life at `E11_WD=0.3`.
- P17.1d.4 Compositions are reported and not predicted; the twin does not acquire them on this architecture.

**E17 Phase 1d stopped (9 November 2026, 18:12 EST).** The six lives were stopped seven minutes after launch when the 9 November audit's embedding finding was confirmed on the Phase 1–1c weights (review 9.60): the local arms' prompt-symbol embeddings receive no gradient and decay to nothing, which the Phase 1d arms inherit. No Phase 1d result exists; P17.1d.1–4 are not evaluated.

## E17, Phase 2: the corrected rerun, embeddings without decay (9 November 2026, 19:00 EST, written before launch)

`E16_BOOKDECAY=off` for births and lives (tag `_nbd`): the embedding table in a no-decay group; the frozen arm (`E16_FREEZE=bank1`) freezes `book`, `l1`, `lp`, `lc`, `lw`; every receipt records which parameters and which byte rows present in the first batch received no gradient. Births `e17_dev_bank2_nbd_<seed>` (three standing seeds, running now), then the twin `D` (running after the births). The three sign-rule arms below are launched only on Mr. Xu's go, pending Mr. Zhou's ruling on what may propagate (review 9.60).

- P17.2.1 (births) each `bank2_nbd` birth reaches at least 85% at k = 12 on the four development operations (Phase 0b with decay: 92–99).
- P17.2.2 (the twin) `D_bank2_nbd` ends within 3 points of the Phase 1 twin on the F1 and F3 family means in every seed, and its held compositions stay under 30: the architecture's composition failure is not an embedding artefact.
- P17.2.3 (unfrozen, sign rule) `T_bank2_nbd_raw65536` reaches F1 at or above 60 after block 1 in every seed (Phase 1 with decay: 1–3). If under 20, the block-1 silence has a second cause in the first bank's learning (the truncated credit through the second bank, or the feedback).
- P17.2.4 (frozen, sign rule) `T_bank2_nbd_frz1_raw65536` holds F1 at or above 70 after block 2 and at or above 60 after block 6 in every seed (with decay: 71–81 → 15–32 → 17–20). If it holds, the Phase 1b collapse was the decayed embeddings alone.
- P17.2.5 (frozen, exact feedback, trace to the projection) `T_bank2_nbd_frz1_xfb_tup_raw65536` holds F1 at or above 85 after blocks 2 and 6, within 5 points of the frozen twin: with the embeddings kept and the trace complete, the only remaining difference from backprop is decay 0.25 against 0.3. If P17.2.4 misses and P17.2.5 holds, the write-projection credit matters beyond the embeddings; if both miss while the twin holds, a further cause remains.
- P17.2.6 Compositions are reported, not predicted.

Commands: `E16_ARCH=bank2 E16_BOOKDECAY=off python3 exp16_integrated.py dev SEED`; `... D SEED`; `E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T SEED`, with `E16_FREEZE=bank1` and with `E16_FREEZE=bank1 E16_FEEDBACK=exact E16_TRACEUP=proj`. Dry runs on a 512-update dry birth archived in `exp45/archive_e17p2_dry/` (the frozen tensors unchanged, the rest changed, seven stages each).

**Phase 2 go (9 November 2026, 19:05 EST).** Mr. Xu: "Launch after twins." The three sign-rule arms (P17.2.3–P17.2.5) are queued behind the twins by `e17p2_lives.sh`; predictions unchanged.

**Phase 2, births and twin verdicts (9 November 2026, 19:40 EST).** P17.2.1 holds in every seed: `bank2_nbd` births at k = 12 reach add89 93.0 / 94.7 / 95.7, sub89 89.3 / 94.3 / 85.0, max89 97.7 / 94.7 / 97.0, min89 94.3 / 95.7 / 94.0 (seed 116089's subtraction sits on the bar). P17.2.2 holds in 93683 and 116089 and misses in 73129 on the F1 clause: `D_bank2_nbd` ends at F1 92.3 / 93.7 / 93.3 against the Phase 1 twin's 95.9 / 92.3 / 94.4 (−3.6 / +1.4 / −1.1; band 3), F3 92.6 / 91.6 / 91.9 against 93.3 / 90.3 / 93.2, held compositions 15.5 / 21.0 / 17.8 (under 30 in every seed): the architecture's composition failure is not an embedding artefact. The nine sign-rule lives launched at 19:39 EST under the lock. Added at launch, so that P17.2.5's comparator shares the regime: the frozen twin without embedding decay, `D_bank2_nbd_frz1`, three seeds, launched beside them (twelve lives); P17.2.5's "frozen twin" reads as this one.

**Phase 2, sign-rule verdicts (9 November 2026, 22:10 EST; lives 19:39–22:08 EST).** P17.2.3 **misses in every seed**: F1 after block 1 is 2.8 / 1.6 / 1.5, under the 20 named in the prediction's second clause, and the development operations fall from 93–95 at birth to 1–2; the silence has a second cause in the rule's own update to the digit embedding rows (row norm 4.1 → 0.3 with sign flips in block 1). P17.2.4 **holds in 93683 and 116089, misses in 73129 by 0.6**: F1 82.1 / 86.9 / 83.9 after block 2, 59.4 / 62.4 / 61.2 after block 6 (against 70 and 60). P17.2.5 **holds in every seed**: 93.7 / 93.6 / 95.0 after block 2 and 87.5 / 90.1 / 91.9 after block 6, within 5 of `D_bank2_nbd_frz1` (94.1 / 95.0 / 93.1 and 90.2 / 90.0 / 91.5) at both checkpoints. P17.2.6 reported: held compositions 13.4 / 14.3 / 15.0 in the exact-feedback arm, 5.8 / 9.2 / 8.5 in the frozen sign rule, 14.9 / 17.8 / 18.1 in the frozen twin; hard pairs at 1–4 in every local arm. By the pre-written reading: P17.2.4 holds by the letter in two seeds while the arm keeps a third of its gap to the twin (0.66–0.69 on mod-97), and P17.2.5 holds, so the write-projection credit or the exact feedback matters beyond the embeddings; which of the two is not separated (Phase 3).

## E17, Phase 3: the two changes one at a time, and the embeddings alone frozen (9 November 2026, 22:40 EST, written before launch; not launched, awaiting Mr. Xu's go)

Nine lives on the Mac, seeds 73129 / 93683 / 116089, `bank2` with embeddings out of decay, raw reservoir 65,536 at block admission as in Phases 1–2, comparators from Phase 2 (`T_bank2_nbd_frz1_raw65536`, `T_bank2_nbd_frz1_xfb_tup_raw65536`, `D_bank2_nbd_frz1`). New knob `E16_FREEZE=book` (tag `_frzb`): the embeddings alone frozen after birth, both banks learning under the sign rule.

- P17.3.1 (exact feedback alone, `T_bank2_nbd_frz1_xfb_raw65536`) F1 at or above 80 after block 6 in every seed, within 8 of the frozen twin: the feedback carries most of arm 3's effect (Phase 1c, with decay, had it delaying the collapse by a block).
- P17.3.2 (extended trace alone, `T_bank2_nbd_frz1_tup_raw65536`) F1 after block 6 at least 5 points above the frozen sign rule's 59.4 / 62.4 / 61.2 in every seed, and below P17.3.1's arm: the trace helps and is not the larger part.
- P17.3.3 (embeddings alone frozen, `T_bank2_nbd_frzb_raw65536`) the birth survives block 1: development operations at or above 60 and F1 at or above 40 after block 1 in every seed. If the development operations fall under 20 with the embeddings frozen, the sign rule's update to the first bank's recurrent weights destroys the birth as well, and the unfrozen windowless learner is not viable under this rule.
- P17.3.4 Compositions are reported, not predicted; no arm is expected above 25 on the held pairs.

Reading, written before the result: P17.3.1 and P17.3.2 holding together place the remaining gap in the feedback, which the propagation answer excludes; P17.3.2 holding with P17.3.1 missing places it in the trace, which is local in the sense of 9.60's case G and stands or falls with the sign rule itself. P17.3.3 decides whether Phase 3's unfrozen arm is worth a Phase 4. Cost about 6,700–7,500 CPU-s per life; nine processes on the Mac, about 2 h 15 min of wall time from the Phase 2 pace. Dry runs of the three command shapes (four updates per block, seven stages each, 130–141 CPU-s) in `exp45/archive_e17p3_dry/`; launcher `e17p3_lives.sh`.

**Phase 3 go (9 November 2026, 23:21 EST).** Mr. Xu: "Launch Phase 3." The nine lives launched at 23:21:40 EST under the machine lock by `e17p3_lives.sh` (block admission, as Phase 2); predictions P17.3.1–P17.3.4 unchanged. Expected finish about 01:40 EST, 10 November, from the Phase 2 pace.

**Phase 3 verdicts (10 November 2026, 01:05 EST; lives 23:21–00:57 EST).** P17.3.1 **misses in every seed**: exact feedback alone ends at F1 67.9 / 72.9 / 72.8, 17–22 under the frozen twin. P17.3.2 **first clause holds, second misses**: the trace alone ends at 80.1 / 84.9 / 84.9, 21–24 above the frozen sign rule's 59.4 / 62.4 / 61.2, and 12 points *above* the exact-feedback arm, not below it; the trace, not the feedback, is the larger part of Phase 2's closure. P17.3.3 **misses in every seed**: with the embeddings frozen the development operations fall to 1.3 / 1.5 / 1.2 after block 1 and F1 to 1.8 / 1.8 / 1.1; by the prediction's own second sentence, the sign rule's update to the first bank destroys the birth as well, and the unfrozen windowless learner is not viable under this rule. P17.3.4 reported: held pairs 8.8 / 8.3 / 8.8 (feedback), 12.2 / 13.2 / 12.0 (trace), 1.0 / 2.0 / 1.7 (embeddings frozen); none above 25. By the pre-written reading, P17.3.2 holding on its first clause with P17.3.1 missing places the remaining gap in the trace, which is local in the sense of 9.60's case G and stands or falls with the sign rule itself; P17.3.3 decides against a Phase 4 on the unfrozen arm. No further Mac run is queued.

**E16-F7 and E16-M3 verdicts (10 November 2026, 10:30 EST; cloud sessions, receipts recomputed here).** The two jobs' own predictions files (`E16F7_PREDICTIONS.md` at d96129c7, `E16M3_PREDICTIONS.md` at bec6d07f) were committed before their launches and are unchanged. E16-F7: P16F7.1, .2, .3 and .5 **hold** (births ≥ 86.3; twins ≥ 93.7 / 91.2 / 92.2; 6 of 6 on the family gate; 1 of 6 on the hard pairs); P16F7.4 **misses** on its correlation clause (Spearman 0.109 over eighteen seeds against 0.35), the threshold clauses holding without a new seed below 65. E16-M3: P16M3.1 **holds** in all three seeds; P16M3.2 **misses in both failing seeds, in the reverse direction** (the rule's hard pairs at share 0.75 are 27 and 15 points above share 0.5, not below); P16M3.3 **misses at share 0.25** in both seeds and holds at 0.75; P16M3.4 **misses** (300007 at 0.796 on compositions at share 0.25; hard pairs 59.2 at 0.25 against 75.7 at 0.75). The premise of P16M3.2, that the reservoir over-doses the rule with verbatim replay, was mine and was wrong. Review 9.62 and 9.64.

## E17 Phase 4 predictions (10 November 2026, 10:30 EST; written before any Phase 4 life; launch on Mr. Xu's go)

Mr. Xu's answers of 10 November (review 9.67): a frozen first bank does not meet the aim; feedback relayed layer by layer, or by a sub-network or long-range connection, is allowed. Measured at birth before these predictions (`exp45/E17_ALIGN_AT_BIRTH.md`): under the sign rule as run, the update to the first bank's projection has cosine 0.09 / −0.06 / 0.05 with the true gradient, to the embeddings 0.13–0.32 and to the first layer 0.16–0.32, and exact feedback does not change that; with credit carried through time at every layer and the feedback still by weight signs, every layer is at 0.45–0.81.

Nine lives on the Mac, seeds 73129 / 93683 / 116089, `bank2`, embeddings out of decay, **nothing frozen**, raw reservoir 65,536, block admission, replay share 0.5, as Phases 2–3:
- `T_bank2_nbd_tall_raw65536` (`E16_TRACEUP=all`, new): sign-of-weight feedback, credit through time at every layer.
- `T_bank2_nbd_tup_raw65536` (`E16_TRACEUP=proj`): the Phase 3 trace, to the write projections only, with nothing frozen.
- `M_bank2_nbd_raw65536`: backprop through time on the learner's own pipeline.

Comparators on file: the unfrozen twin `D_bank2_nbd` (mod-97 92.3 / 93.7 / 93.3, mod-73 92.6 / 91.6 / 91.9, held pairs 15.5 / 21.0 / 17.8 after block 6); the unfrozen sign rule of Phase 2 (`T_bank2_nbd_raw65536`, 1–3 on everything from block 1); the frozen learner with the Phase 3 trace (80.1 / 84.9 / 84.9 on mod-97 after block 6).

- **P17.4.1 (the birth survives)** Under the new arm, the mod-97 mean after block 1 is at or above 80 in at least two of the three seeds, and above 50 in all three. (The unfrozen sign rule had 2.8 / 1.6 / 1.5; the unfrozen twin 92.4 / 92.2 / 93.6.)
- **P17.4.2 (closure at the end)** After block 6 the new arm is at or above 0.80 of the unfrozen twin on mod-97 and on mod-73 in at least two of the three seeds.
- **P17.4.3 (one layer is not enough)** With the trace extended to the write projections only and nothing frozen, the mod-97 mean after block 1 is below 50 in at least two of the three seeds: the embeddings and the first layer, still credited within the step only, are enough to lose the birth. Held with less confidence than the others (about 60%); if it misses, the cheaper trace suffices and the new arm is not needed.
- **P17.4.4 (the pipeline's ceiling)** Backprop on the learner's pipeline ends at or above 0.90 of the unfrozen twin on mod-97 and on mod-73 in all three seeds.
- **P17.4.5 (compositions)** Reported, not predicted to compose: no arm above 25 on the held-pair mean, the hard pairs at or under 15 in every arm, as the twin (1.5 / 12.1 / 5.4).

Reading, written before the result. P17.4.1 and P17.4.2 holding withdraw 9.66's "the first bank must be frozen": what the first bank needs is credit that reaches it through time, and the windowless learner is then plastic in every layer under weight-sign feedback, at the price stated in 9.67 item 4 (credit relayed back along the item, a per-item backward pass or its forward equivalent). P17.4.1 missing while P17.4.4 holds says weight signs compounded through two banks and time do not train the first bank, and the next candidate is a learned feedback pathway of the kind Mr. Xu's answer names (a sub-network carrying the error), not a longer trace. P17.4.4 missing says the blocks and the reservoir, not the rule, lose the unfrozen learner, and Phase 4's rule arms are then uninformative. Nothing here tests composition; the two-bank architecture does not compose under backprop either (Phase 1), and that stays the architecture's open problem. Evidence level E2; no reuse on never-reinforced inputs is claimed.

Cost and forecast. Measured per update on the Mac at batch 64: new arm 48.7 ms, plain rule 54.7, backprop 49.4 (`exp45/E17P4_CHECK.json`). Expected CPU-s per life: new arm 6,000–6,800; one-layer trace 7,500–7,900; backprop on the pipeline 5,500–6,500. Nine processes: about 2 h 20 min of wall (range 2 h to 2 h 40), from Phase 2 (twelve processes, 7,515 CPU-s in 2 h 29) and Phase 3 (nine processes, 5,602 CPU-s in 1 h 35). Dry runs of the three commands at four updates per block pass seven stages (`exp45/archive_e17p4_dry/`, 201–204 CPU-s each, run three at once). Mechanism audit of what the new arm inherits: the reservoir, block admission and replay share are Phase 2's unchanged; the optimizer groups are Phase 2's (embeddings without decay); the scan trace is switched off in the new arm because the backward pass carries the same credit (checked: identical to 2e-7 on the layers that had it); evaluation runs the plain forward pass and is untouched. Launcher `e17p4_lives.sh` takes and releases the machine lock.

**Phase 4 go (10 November 2026, 10:31 EST).** Mr. Xu: "Launch Phase 4". The go was given after 9.67 item 4 was put to him in plain words (the new arm relays credit back along the item in time with weight signs, which is backprop through time over one item with weight signs in place of weights) and is read as accepting that for a diagnostic, not as a ruling on it. The nine lives launched at 10:31:24 EST under the machine lock by `e17p4_lives.sh`, from the harness at a3c0b675; each process's environment was checked while running (`E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536`, plus `E16_TRACEUP=all`, `E16_TRACEUP=proj` or nothing for arm M; no freeze). Predictions P17.4.1–P17.4.5 unchanged. Expected finish about 12:50 EST (range 12:30–13:10).

**Phase 4 verdicts (10 November 2026, 13:00 EST; lives 10:31–12:35 EST).** P17.4.1 **holds in every seed**: the new arm (weight-sign feedback, credit through time at every layer, nothing frozen) is at 92.9 / 92.1 / 90.7 on mod-97 after block 1. P17.4.2 **holds in every seed**: after block 6 it is at 0.95 / 0.92 / 0.93 of the unfrozen twin on mod-97 and 0.98 / 0.95 / 0.97 on mod-73. P17.4.3 **misses as written**: the one-layer trace with nothing frozen is at 65.0 / 75.3 / 74.9 after block 1, not below 50; neither pre-written reading of it obtained, since it then ends at 44.2 / 42.1 / 42.6, about half the twin: one layer keeps the learner alive and does not suffice. P17.4.4 **holds in every seed**: backprop on the learner's pipeline ends at 0.98–1.01 of the twin on mod-97 and 1.01–1.02 on mod-73. P17.4.5 as expected: no arm above 25 on the held pairs (new arm 12.8 / 14.7 / 14.4), hard pairs at or under 8.6. Not predicted and reported: the new arm ends at 10.7 / 22.2 / 43.2 on the unrehearsed development operations against 48.6 / 41.4 / 53.6 for backprop on the same pipeline. Review 9.70.

## Repeats on the corrected panels, after the 10 November source audit (written 10 November 2026, 15:00 EST, before any repeat; launch on Mr. Xu's go)

Twelve window-learner lives on the Mac, receipts tagged `_cp`, originals untouched (`audit_rerun_lives.sh`). Drift, seeds 73129 / 93683 / 116089: learner (raw reservoir 65,536, online) and reference. Staged, seeds 400009 / 600013: learner (raw reservoir 65,536, block admission), backprop on the learner's pipeline, and reference, on one machine and one panel. Each receipt also carries strict-panel and no-store scores.

- **P-CP.1 (family gate on corrected panels)** All three drift seeds pass the family gate again.
- **P-CP.2 (hard pairs on corrected panels)** At most one of the three reaches 0.80 of the reference on `+-`/`-+` alone (two did on the original panels; the corrected-panel rate elsewhere is 2 of 15).
- **P-CP.3 (E16-M2's reading survives a common panel)** In both staged seeds, backprop on the learner's pipeline ends at or above 0.90 of the reference on held compositions, and the learner at least 0.10 below it in ratio.
- **P-CP.4 (the store on compositions)** The learner's held-pair mean with and without the score-time store differs by less than 1 point in every one of its five lives.

Reading, written before the result. P-CP.1 and P-CP.2 replace the three original-panel rows of the eighteen-seed table; the eighteen then form one population. P-CP.3 missing would withdraw E16-M2's conclusion that the reservoir costs the rule what it does not cost backprop. P-CP.4 missing would mean the learner's composition scores owe something to the store and every rule-against-backprop composition ratio needs the no-store column. Mac runs are not bit-identical to the cloud runs of the same seeds; these are fresh matched lives, not re-scores. Cost: original lives took 4,500–6,300 CPU-s; the added scoring is about a tenth more; twelve processes, about 2 h 45 min of wall (range 2 h 20 to 3 h 10) from Phase 2's twelve-process pace.

**Repeats on the corrected panels: go (10 November 2026, 14:27 EST).** Mr. Xu: "Launch repeats". The twelve lives launched at 14:27:14 EST by `audit_rerun_lives.sh` from the harness at c45747a5, under the exclusive lock (token 20261110T142715-24918-1136). Each process's environment was read while running: three drift learners (`E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift`), three drift references (`E16_STREAM=drift`), and for 400009 and 600013 a staged learner and backprop on its pipeline (`E16_REPLAY=raw E16_CAP=65536`) and a staged reference; all with `E16_OUTTAG=_cp`. No receipt tagged `_cp` existed before launch. Predictions P-CP.1–4 unchanged. Expected finish about 17:10 EST (range 16:50–17:40).

**E18 and E19 verdicts (10 November 2026, 17:30 EST; cloud sessions; recomputed here from the receipts).** E18 (predictions at 0ea40628): P18.1, P18.2, P18.3, P18.4 and P18.6 **hold in every seed**; P18.5 **misses** (B2 is 5.8 / 12.5 / 10.8 below B1). P18.4 and P18.6 hold as written and are package comparisons: B0 and B1 differ in four things, and B3 cannot learn its input code. E19 (predictions on the worker's branch): P19.1 holds (gradient equality to rounding, also against an independently written forward pass); P19.2 holds (69.0 / 70.5 / 69.3 against a gate of 60); P19.3 **misses in every seed** (0.63 / 0.65 / 0.79 against 0.85), with decay differing between the two arms; P19.4 holds (deep traces +8.0 / +14.5 / +23.8). Review 9.71 and 9.79.

**Repeats on the corrected panels: verdicts (10 November 2026, 17:15 EST; lives 14:27–16:45 EST; all twelve exit codes zero, completion records and weight hashes verified).** P-CP.1 **holds**: the three drift seeds pass the family gate again. P-CP.2 **misses**: all three reach 0.80 of the reference on the plus-minus pairs alone (0.81 / 0.82 / 0.84), where I predicted at most one; these are the seeds the reservoir size was chosen on, and the fifteen later seeds stand at 2 of 15. P-CP.3 **holds in both staged seeds**: backprop on the learner's pipeline 0.98 and 0.94 of the reference on held pairs, the learner 0.83 and 0.64. P-CP.4 **holds**: the learner's held-pair mean is identical with and without the store in all five of its lives. Review 9.81.

## Two runs from the audit session's next-steps note (written 10 November 2026, 18:45 EST, before launch)

`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/NEXT_STEPS_NOTE_20261110.md`, sections 3.3 step 1 and 3.4, run on the owner's machine on Mr. Xu's instruction to implement the note. Launcher `nextsteps_lives.sh`; dry runs of the four command shapes in `exp45/archive_nextsteps_dry/` (seven stages, completion records, births admitted by the manifest, every symbol role reached); with the current harness an answer-loss dry life reproduces the Phase 4 dry receipt score for score and loss for loss.

**Zero replay under both updates.** Six window-learner lives on the drifting stream, seeds 400009 / 600013 / 300007, raw reservoir 65,536 admitted online, `E16_RSHARE=0`, receipts tagged `_mac`: the sign rule (`T`) and backprop on the same pipeline (`M`). They differ in the update alone (same stream, decay 0.25, panels, machine). The record so far has the rule at zero replay (cloud, review 9.64) and no backprop comparator. Ratios are to the drift references on file (cloud receipts; levels, not digits).
- **P-Z.1 (the rule's result repeats on this machine)** `T` ends below 0.50 of the reference on mod-97 and at or under 5 on the plus-minus pairs in all three seeds.
- **P-Z.2 (backprop forgets too)** `M` ends below 0.60 of the reference on mod-97 and at or under 10 on the plus-minus pairs in all three seeds. Held at about 75%.
- **P-Z.3 (but more slowly)** `M`'s mod-97 mean is at least 5 points above `T`'s in at least two seeds. Held at about 60%.

Reading, written before the result. P-Z.2 holding says that losing everything without replay belongs to this stream under either update, and "the rule retains nothing without replay" (9.64 item 4) is not a statement about the rule. P-Z.2 missing, with backprop retaining well, says the forgetting is the rule's. Either way the answer-mask and in-view caveats of 9.76 apply to what is retained.

**Every next byte as the target, in a life.** Six lives of the windowless two-bank learner, seeds 73129 / 93683 / 116089, embeddings out of decay, raw reservoir 65,536, block admission, share 0.5, `E16_LOSS=all`: the Phase 4 rule (`T`, `E16_TRACEUP=all`) and backprop on the same pipeline (`M`). Each has a partner on file that differs in the loss mask alone (Phase 4, review 9.70: rule 87.9 / 86.1 / 86.9 on mod-97 and 90.8 / 86.9 / 89.4 on mod-73; backprop on the pipeline 90.9 / 92.3 / 94.2 and 93.5 / 93.2 / 93.2), and the two differ from each other in the update alone. Scoring is unchanged. With this mask 4,080 bytes are supervised per 256 rows of first-block items against 768 under the answer mask.
- **P-E.1 (backprop tolerates every-byte targets)** `M` ends at or above 0.90 of its answer-loss partner on mod-97 and on mod-73 in at least two seeds. Held at about 65%.
- **P-E.2 (so does the corrected rule)** `T` ends within 10 points of its answer-loss partner on mod-97 in at least two seeds. Held at about 55%: E16-L lost addition and subtraction under this loss, but under the rule as it then was.
- **P-E.3 (the update alone)** `T` ends at or above 0.85 of `M` on mod-97 in at least two seeds.
- **P-E.4** Reported: the unrehearsed development operations, plus-minus pairs, and scores at 20 and 40 distractors, for both arms beside their partners.

Reading, written before the result. P-E.1 and P-E.2 holding say the answer mask was not needed: the stream's own next byte is enough of a teacher for single operations under this rule, which is Mr. Zhou's delayed-observation idea in its plainest form. P-E.2 missing with P-E.1 holding says the rule, not the objective, loses the skill, and the next single change is predictable distractors. P-E.1 missing says every-byte targets cost backprop too, and the unpredictable prompt bytes are the first suspect.

Cost and forecast. Per life on this machine: window learners about 5,700–6,300 CPU-s, two-bank lives about 6,600–6,800, plus scoring. Twelve processes: about 2 h 30 min of wall (range 2 h 10 to 3 h), from the twelve-process paces of Phase 2 (2 h 29) and the corrected-panel repeats (2 h 17).

**E16-M4: not judged (10 November 2026, 20:00 EST).** A container restart stopped all twelve lives inside their last stretch; receipts end at checkpoint 5 (`_killed`). P16M4.1–P16M4.5 are written for the last checkpoint and are not judged. Not rerun; the replay-share line closes here. What the five checkpoints show is described in review 9.68 and is not a verdict.

## E22 version 2: one rule at birth and in life (written 10 November 2026, 23:40 EST, before launch; Mr. Xu's go: "Yes, run it")

Brief: `OPUS_TASK_E22_RULE_BORN_LIFE_20261110.md`, version 2, rewritten after Codex's audit of version 1 (review 9.87). Six lives on the owner's machine, one host, seeds 73129 / 93683 / 116089: the windowless two-bank learner under the working life rule (a sign-feedback temporal sweep, `E16_TRACEUP=all`), started from E18's rule-born birth and, as its partner, from the backprop birth; receipts tagged `_e22`. Launcher `e22_lives.sh`; validator and scorer `e22_validate.py`, written before any life ran. Dry runs of both commands: seven stages, completion records, births admitted by the manifest with their training configuration checked, equal panel hashes, and no setting differing between partners except the birth fields.

Definitions: checkpoint 6 unless a block is named; twelve distractors; whole answers; four operations weighted equally; hard mean = (add97 + sub97) / 2.
- **P22.1 (acquisition)** After block 1 the rule-born learner's mod-97 mean is at least 80 and its hard mean at least 70, in at least two seeds.
- **P22.2 (final competence, absolute)** At checkpoint 6 the rule-born learner's mod-97 mean is at least 75, its hard mean at least 65 and its mod-73 mean at least 75, all in the same seed, in at least two seeds.
- **P22.3 (parity, one-sided)** In at least two seeds, the same two for both families, the rule-born learner is no more than 5 points below its partner on the mod-97 mean and on the mod-73 mean.
- **P22.4** Reported with no mark: the development operations at every checkpoint, scores at 20 and 40 distractors, compositions.

If both partners end under P22.2's marks, the result is common loss whatever P22.3 says. A cell that fails the validator is not scored. Reading as in the brief. Cost: six processes, about 6,700 CPU-s each; about 1 h 30 min of wall (range 1 h 15 to 1 h 50), from the six- to nine-process paces of Phase 3 and Phase 4.

**Next-steps runs: verdicts (10 November 2026, 23:59 EST; lives 17:12–19:51 EST; twelve exit codes zero, receipts verified).** P-Z.1, P-Z.2 and P-Z.3 **hold**: without replay the rule ends at 0.33 / 0.33 / 0.12 of the reference on mod-97 and backprop on the same pipeline at 0.45 / 0.37 / 0.30, both at about 1 on the plus-minus pairs; backprop is 11.1 / 3.8 / 16.6 points higher on mod-97. P-E.1 **holds in every seed**: backprop under every-byte targets is at 0.95–0.98 of its answer-loss partner. P-E.2 and P-E.3 **miss in every seed**: the sign-feedback sweep under every-byte targets ends at 10.3 / 5.4 / 9.7 on mod-97, against 87.9 / 86.1 / 86.9 for its partner and 89.3 / 90.6 / 89.2 for backprop. Review 9.88.

**E22 version 2: launched (10 November 2026, 19:52 EST).** On Mr. Xu's go ("Yes, run it") the six lives started at 19:52:08 EST under the exclusive lock from the harness at c6732624; source and manifest hashes are in `exp45/logs/e22_source_hashes.txt`; each process's environment was read while running. Predictions P22.1–4 unchanged.

**E22 version 2: verdicts (11 November 2026, 01:15 EST; lives 19:52–21:14 EST on 10 November; six exit codes zero; all three pairs pass the validator written beforehand).** P22.1 **holds in every seed** (rule-born after block 1: mod-97 90.9 / 91.9 / 88.7, hard mean 85.0 / 87.2 / 82.0). P22.2 **holds in every seed** (end: mod-97 88.1 / 84.8 / 84.3, hard mean 84.5 / 77.0 / 76.4, mod-73 90.7 / 87.5 / 87.1). P22.3 **holds in every seed** (partner minus rule-born: mod-97 −0.2 / +1.2 / +2.6, mod-73 +0.1 / −0.6 / +2.2). No seed has both partners under the marks. Review 9.85.


## Addendum, 10 November 2026, 22:17 EST (clock): E20-P, probes on saved E20-A checkpoints

Predictions P-PR.1 to P-PR.7 and their readings are in `E20P_PREDICTIONS.md`, written and pushed before any trained checkpoint is probed. Script `e20_probe.py` (fixtures pass for the three seeds, on untrained nets and stub answers only), launcher `e20p_run.sh`. Not run: it waits for Mr. Xu's go.


## Verdicts, E20-P (10 November 2026, 22:51 EST, clock; review 9.93)

Seeds 73129 / 93683 / 116089, from `exp45/E20P_VERDICTS.json` (scorer `e20p_score.py`).

- P-PR.1 holds: 91.3 / 89.5 / 77.5% of failed twins have a wrong intermediate.
- P-PR.2 holds: 92.5 / 92.0 / 94.8%.
- P-PR.3 misses: 0.3 / 2.1 / 7.0 points, against 10.
- P-PR.4 holds: 88.8 / 88.1 / 89.8%.
- P-PR.5 holds: 100 / 100 / 100%; shuffled labels 0.7 to 0.9%, untrained net 2.2 to 3.7%.
- P-PR.6 cannot be judged: the operand reads out on every item, so there is no second group. The expectation behind it was wrong.
- P-PR.7 holds: 0.47 / 0.42 / 0.13.


## Addendum, 10 November 2026, 23:27 EST (clock): E20-P2, follow-up probes

Predictions P-P2.1 to P-P2.4 and their readings are in `E20P2_PREDICTIONS.md`, written and pushed before `e20_probe2.py` touches a trained checkpoint. Exploratory: designed after E20-P's results. Not run: it waits for Mr. Xu's go.
