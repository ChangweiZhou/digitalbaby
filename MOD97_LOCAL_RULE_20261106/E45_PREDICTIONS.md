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
