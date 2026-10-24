# V146: number map plus identity code, with a downscaling phase. Can the rule overtake memorisation?

24 October 2026. **Design memo only. Nothing in it has been implemented or run**, except the background analyses it cites. It builds on:
* V143 (LEARNED-BANK V1);
* V144 and V145, the Codex exercises on the consolidated branch;
* NUMBER-CODE V1 (`minifly_number_code_design_20261024/`);
* the grokking-versus-symbol-set check in `analysis/`.

It keeps the owner's endpoint: complete, self-generated answers with their own EOS on never-taught addition relations. It is to be refined in three cycles (refine, audit, test run, exercise) before any lock.

**Ruled out on every path, including reference and ceiling learners:**
* back-propagation of any kind (exact gradient, BPTT);
* attention and Transformers;
* ELM solvers;
* pretrained models;
* supplied arithmetic structure: operand parsing, number or carry features, answer length, digit masks, "apply +1 b times" routines.

All learning uses local rules: Hebbian, competitive or delta rules, or three-factor rules with fixed random feedback.

## 1. What the evidence says

| Evidence | Result | Reading |
|---|---|---|
| V143 parity (4 worlds) | Learned bank 0.92 against frozen 0.47 and shuffled 0.56 (balanced held); system answer 0.70 | Local feature learning can make a native bank's addressing match a task, when the task's interaction is rank 1 |
| Failure analysis (`../minifly_number_code_design_20261024/analysis/`) | Digits reach the learner as arbitrary symbols (code distance against numeric distance: Spearman 0.09). Addition is a Latin square, so same-answer prompts share no operand. A row-plus-column smoother scores 0.00 on held sums. A one-hidden-layer evaluator network fits every taught pair and transfers 0.02 (units digit) and 0.00 (sum) | Symbol codes make memorising cheap and the rule invisible |
| Grokking versus symbol-set size (`analysis/grokking_vs_p.log`) | Same one-hot MLP with strong weight decay, 20,000 steps: p = 10 never groks (held ≤ 0.28); p = 23 at 75% training and p = 47 do (held reaches 1.00 long after training is fitted) | Single-digit addition is below the critical data size: the rule never becomes cheaper than the lookup |
| V145 actual-interface check (Codex) | Symbol codes: taught 100%, held 13.3% (the copy baseline). Number-line oracle: taught 38.8%, held 26.7%. Circle oracle: 82.5% / 16.7%. Learned codes (64 passes): held 0–10% | Held ÷ taught is 0.13 for symbols against 0.69 for the line. The line makes the rule cheap but destroys separability: digits are collinear, '0' is a zero pattern, and every byte was replaced. Naming and output tracking then fail |

The three earlier evidence rows came from exact-gradient evaluator learners. Those numbers explain the past; no such learner appears in V146.

**The diagnosis in one sentence.** Grokking needs both solutions available, with the rule the cheaper one and a long pressure towards small weights after fitting. Our runs never had that:
* symbol codes: the rule is never cheaper at p = 10, and weight decay is about 3e-5 per update;
* line codes: memorising is impossible, and naming the answer is hard, so nothing fits.

**What the human-learning catalog contributes** (grades from `human_learning_mechanisms_20261023`). Humans never infer held-out single-digit facts from a table. Facts are generated over a magnitude system and then stored. Three catalog-supported changes follow:

| Change | Catalog support |
|---|---|
| A code that is both separable and ordered: units tuned to numbers, on a line | Single-neuron number codes (B); mental number line (B) |
| Magnitude grounded in quantity and counting experience | Neuronal recycling of the number sense (B); counting-to-retrieval shift (B); number-line board games (B, mostly one group) |
| Pressure towards simple solutions after fitting | Synaptic homeostasis (C); synaptic scaling (C); sleep-dependent rule discovery (B, one central study); starting small (D) |

The triple-code model (separate symbolic and magnitude codes) is **not** in the catalog; dual coding (B) is the nearest row.

## 2. The question

> When MiniFly's input carries both its unchanged byte-identity code and a number map learned locally from raw-byte quantity and counting experience, and a downscaling phase follows fitting, does held-out complete addition rise above every matched control? Does it rise during the downscaling phase (a grokking-like signature) rather than with fitting alone?

V146 is an evaluator-side **local-rule** implementation (Stage A). Its naming store is a delta-rule readout whose speed is calibrated to the native Full151 compartments, as in V143 cycle 2. Swapping in the native compartments is the next round (section 11).

## 3. Architecture (changed)

```
byte event x ──► identity receptor R(x) (unchanged FE-PERM 88-d pattern, all 256 bytes)
            └──► number map m(x)        (M = 32 tuned units on a 1-D sheet, learned)
[R(x) ; g·m(x)] ──► FE-PERM permute-then-add trace over the record (88+M channels, inherited timing)
trace ──► combination layer h = relu(W p + b)        (H units; three-factor rule, fixed random feedback)
h     ──► fixed sparse expansion: KC = top-k(B_PN→KC · V h)   (V fixed random; native PN→KC wiring and top-k)
KC    ──► naming store: 9 opponent pairs (8 byte bits + EOS)  (delta rule; Stage B: native compartments)
own output byte ──► fed back as the next byte event (identity + map), until own EOS or 8 decisions
```

**3.1 Identity code: kept for all 256 bytes.** It keeps memorising, output tracking and EOS easy. This fixes V145's main confound, which replaced every byte's code with a d ≤ 8 code.

**3.2 Number map: new; the same mechanism for every byte, with no digit gate.**
* **Units.** M = 32 units on a 1-D sheet with a neighbourhood function. This is a declared generic prior: cortical maps are topographic (cortical map plasticity, B). Which bytes acquire graded positions is decided by the data, not by the design.
* **Organisation (competitive Hebbian, self-organising-map style; local).**
  * The map's input is the FE-PERM context trace at each event.
  * The winning unit and its neighbours move their weights towards the input; the neighbourhood width shrinks over the experience phases.
  * A run of k identical bytes produces traces that change smoothly with k, so the map orders runs by length.
  * In a record such as `*****5` the digit event's trace still contains the run, so each digit is pulled towards the position of its quantity.
* **Recall without context (delta rule).** An association from the identity code R(x) to the map activation is learned during experience, so that later a byte alone recalls its bump m(x).
* **No zero vectors.** Bumps have unit mass. g is set once, from the RMS of the identity patterns at birth.
* **The map is frozen before addition**, so its effect is cleanly attributable. Continued plasticity during addition is a declared later variant.

**3.3 Combination layer.** V143's three-factor rule with fixed random feedback G, on the augmented trace:
* the third factor is the store's own residual;
* the warmup ramps the rate over the first 1,024 records.

In the number map, a+b becomes a simple direction, so a local rule *can* find it. In symbol codes no such direction exists.

**3.4 Expansion and naming.** A fixed random sparse expansion with top-k (the native wiring) turns a smooth sum-like coordinate into separable codes, and the store names them by lookup.
* This splits the work as "combine, then expand, then name".
* It answers V145's line-code failure: naming a high-frequency function of magnitude is a lookup, not a smooth readout.

**3.5 Downscaling phase (simplicity pressure).**
* Addition training continues past taught saturation, as grokking requires.
* Every R passes, a rest event multiplies the combination-layer and readout weights by (1 − ρ). This is the synaptic-homeostasis and synaptic-scaling analogue of weight decay.
* There is no replay buffer and no stored raw history: training passes simply continue.
* ρ and R are dev-tuned; the proposal is ρ ∈ {0.01, 0.03} per rest and R = 5 passes.

**3.6 Starting small.** A declared dev-only variant, not in the main roster: the expansion's top-k starts lower and grows.

## 4. Experience (all charged as exposure)

| Phase | Records | Feedback | Purpose |
|---|---|---|---|
| Q: quantity | A run of k `*` bytes, then the digit k, then EOS (k = 0..9, so the k = 0 record is just `0`); equal frequency per digit, random order | Next-byte observation only (no answers, no arithmetic) | Grounds the map in magnitude from raw bytes |
| C: counting | Full records `0123456789` + EOS; equal digit frequency (V145's fix) | Next-byte observation only | Orders neighbours by temporal adjacency |
| A: addition | `a+b=` with observed answer bytes and EOS, over 40 taught orbits (both orientations); gradual warmup 1/16 → 1 | Supervised warmup, as inherited | The task |

* Held orbits (15, both orientations) never appear in any phase.
* Phases Q and C contain no `+` and no sums.
* The digit bytes in Q and C are the same bytes used in A. This is the point of the experiment, and it is declared.

## 5. Arms (per world)

| Arm | Change from FULL | Question |
|---|---|---|
| FULL | Q + C, then A with downscaling | Candidate |
| NO-DOWNSCALE | ρ = 0, same passes | Is simplicity pressure needed (the grokking signature)? |
| SYMBOL | No number map (identity only); same layer, expansion, store and downscaling | Is the map needed? |
| MAP-PERM | Q and C with a fixed, locked digit permutation π (digit d paired with π(d) stars; counting in π order) | Does *numeric* grounding matter, or would any structured map do? |
| MAP-NOEXP | Map initialised but given no Q or C experience; identical A | Does map learning matter? |
| FROZEN-LAYER | Combination layer frozen at birth | Is local feature learning needed downstream of the map? |
| SHUF-ADD | FULL, with addition answers permuted among taught orbits at about chance agreement (orbit-consistent) | Leakage and structure control |
| ORACLE-MAP | Evaluator-supplied bumps at the true digit positions; no Q or C | Plumbing ceiling only, never a candidate |
| BIRTH-MORE-ADD | No Q or C; extra addition passes matched to FULL's total exposed tokens and measured CPU (V145's rule) | Total-efficiency control |

**Shortcut ladder,** scored on the same held orbits by evaluator-only rules:
* COPY_A and COPY_B;
* exact and unordered lookup;
* answer frequency;
* nearest taught answer under the *actual* frozen augmented code;
* FIXED_k truncations of the candidate's own stream.

## 6. Cycle 0: before any map-learning code is written

Run ORACLE-MAP and SYMBOL through the complete local pipeline at the real interface (own bytes, own EOS), on 2 dev worlds, with downscaling on and off.

* **Continue only if** ORACLE-MAP, with downscaling, beats the strongest shortcut by at least 3 held orbits in both worlds, with taught ≥ 0.90.
* **Otherwise the verdict is CANNOT_TEST_AS_SPECIFIED.** The combination or naming stage is then the bottleneck. Redesign it, for example the expansion width or the opponent readout, before learning any map. Do not run Q and C.
* **What cycle 0 can already show:** whether downscaling produces a delayed held rise *with the ideal map*. That is the grokking question in its cleanest form.

## 7. Endpoint, measurements and decision rule

**Primary.** Final complete held accuracy: own payload plus own EOS within 8 decisions, averaged over orientations within each orbit and then over the 15 orbits.

**Per main world, FULL must:**
* beat SYMBOL, MAP-PERM, MAP-NOEXP, SHUF-ADD and the strongest shortcut by at least 3 held orbits (0.20);
* reach taught ≥ 0.90;
* answer at least one two-byte held orbit correctly.

**Gate.** ORACLE-MAP passes the cycle-0 criterion in that world.

**Verdict over 4 main worlds.**
* **Promising** if all 4 pass, **mixed** if 3 pass, **negative** otherwise.
* This is a directional screen, not confirmation: 4 of 4 positive signs is a one-sided sign-test probability of 1/16.

**Secondary (pre-registered): the grokking signature.**
* Held accuracy in FULL rises by at least 2 orbits between the first checkpoint with taught ≥ 0.95 and the final checkpoint.
* NO-DOWNSCALE does not rise that way.
* Reported per world, never used to pick checkpoints.

**Map meters (descriptive, not gates):**
* order: Spearman correlation of map position against digit, and against run length;
* tuning width;
* separability (minimum distance between digit bumps);
* recall accuracy from identity alone.

**Also reported:**
* taught accuracy;
* one-byte and two-byte strata;
* EOS at the correct length;
* invalid codes and truncation;
* learning curves against exposed tokens and against CPU;
* BIRTH-MORE-ADD comparisons for any efficiency statement.

**How outcomes will be read:**

| Outcome | Reading | Next |
|---|---|---|
| FULL passes, with the grokking signature | Grounded map plus simplicity pressure lets the rule beat memorisation at p = 10 | Native port (V147) |
| FULL passes, without the signature | The map alone makes the rule cheap enough; downscaling is not needed | Native port; drop downscaling or keep it for retention |
| ORACLE passes, FULL fails, the map meters are poor | The map learner fails, not the pipeline | Fix map learning (Q/C design, neighbourhood schedule) |
| ORACLE passes, FULL fails, the map meters are good | The learned map's geometry differs from the oracle in a way that matters | Compare geometries; no new mechanism yet |
| MAP-PERM is as good as FULL | Structure, not numeric grounding, drives transfer | The theory is wrong; stop and rethink |
| FROZEN-LAYER is as good as FULL | The map makes addition kernel-easy; local feature learning is not needed | Simplify |
| ORACLE fails cycle 0 | Combine, expand and name cannot carry the rule at this interface | Redesign that stage; no map work |

## 8. Integrity and accounting (lessons of V143–V145 adopted)

* **Fail-closed reducer.** It requires the exact locked roster, worlds and arms. Every gate must be evaluated; a missing arm or world means INVALID. Source, closure or hash failures invalidate the verdict. The V143 post-hoc checker's fail-open flags are fixed here.
* **Closure and lock.** The runtime closure is computed from `sys.modules` and lazy imports, plus assets, and each phase re-checks the lock.
* **Recoverable runs.** Whole-learner checkpoints are tested by pickling the combined object (V145's failure). Resume replays one completed unit and must be bit-identical.
* **Ledger and tables.** An append-only attempt ledger is written before training. Every failed, diagnostic and surrogate attempt is charged. Map and code tables are persisted before any downstream fit.
* **Wording.** No "confirmation" wording on dev worlds.

## 9. Worlds, cycles and budget

**Worlds.**
* Dev: `V146-D1`–`D3`, for cycle 0, tuning and cycles.
* Rehearsal: `V146-D4`, fresh, never used for tuning.
* Main: `V146-M1`–`M4`, generated only after the lock.
* The 55-orbit relation split is as in V143–V145. Four worlds are four splits and initialisations of the same finite table, not independent domains.

**Cycles.**
1. Cycle 0 plus implementation: local rules written as explicit equations; unit checks (no zero vectors; identity path bit-equal to FE-PERM for every byte; probes leave state unchanged; map and identity recall); tiny test run.
2. Dev runs: tune M, k, H, ρ, R and the neighbourhood schedule on D1–D3 only; measure cost.
3. Fixes, dress rehearsal on D4, lock.

**Budget.**
* Stage A is numpy, local rules only. My estimate is minutes per history, but it is not yet measured, as V145 cautioned.
* The main roster is 9 arms × 4 worlds = 36 histories.
* **Proposed cap: 6 CPU-hours** for the whole round, including all dev, failed and diagnostic work. The cycles measure real cost before the lock, and an INCOMPLETE status is reported if the cap is hit.

## 10. What changed from NUMBER-CODE V1 and V145, and why

| Earlier design | V146 | Reason |
|---|---|---|
| Replace digit codes (V1) or all byte codes (V145) with a low-dimensional learned code | Keep the identity code for all bytes; *add* a number map through the same mechanism for every byte | Separability for memorising, naming and EOS (V145: taught fell in every replaced-code arm) |
| Line oracle `a/9` | Tuned bumps of unit mass | No zero vectors; separable and ordered |
| Shared affine "+1" transition | Self-organising map grounded in quantity runs, plus counting | A fitted transition does not guarantee a number line (V145 review); quantity runs supply magnitude geometry from raw bytes |
| Smooth shallow readout from the code | Combine, then fixed sparse expansion, then lookup store | Naming is a high-frequency function of magnitude |
| No simplicity pressure | Downscaling phase after fitting, with a NO-DOWNSCALE control | The grokking condition; human analogue: synaptic homeostasis and scaling |
| Evaluator exact-gradient references | None. Local rules only, everywhere | Owner's no-backprop rule. ORACLE-MAP through the local pipeline is the ceiling |
| Classes as answers (V1) | Own bytes and own EOS | Owner's endpoint (V145 review) |
| Random-start counting | Full-record counting and equal-frequency quantity records | Frequency confound (V145 review) |

## 11. Later rounds (not part of V146)

1. **V147, native port.** Native Full151 compartments become the naming store, with V143-style banks and the combined output. Retention is tested after genuine later learning.
2. **Multi-digit sums and carries,** through a generic learned recurrent state. No supplied column or carry procedure.

## 12. Owner decisions needed before implementation

1. **Curricula.** Quantity and counting records as added, charged experience (section 4).
2. **The map prior.** The 1-D topographic sheet as a declared generic prior (section 3.2).
3. **Budget.** A cap for the round (proposal: 6 CPU-hours).
4. **Staging.** Stage A (local rules, native-calibrated store) before the native port.
