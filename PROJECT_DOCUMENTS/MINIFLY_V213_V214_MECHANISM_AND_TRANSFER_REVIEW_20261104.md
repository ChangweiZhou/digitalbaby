# MiniFly research note: V213/V214 learning dynamics and forward-transfer audit

**Date:** 2026-11-04  
**Status:** Research interpretation and proposed next diagnostics; **not** an approved architecture, validated scientific result, or authorization to run experiments  
**Scope:** V212 -> V213 observed-transition learning; V214A/B prequential recall/transfer; relationship to the experience-driven continual-learning strategy

## Executive assessment

The latest results sharpen, but do not resolve, the persistent-learning bottleneck:

1. **V213:** A compact shared transition learner failed to acquire a learned event-effect rule in its engineering rehearsal. A pooled count-table reference could solve most of the same diagnostic twins. The primary scientific problem is therefore **learning dynamics**, not merely whether the data contain any information about the rule. The official V213 roster did **not** run.
2. **V214B:** A recovered Round 20 byte predictor reproduces an advantage from causal episodic recall and reports positive prequential code-length transfer from prior worlds to **one** new target world. This does not yet establish transferable object dynamics, autonomous answer generation, or retained plasticity.
3. **V214A:** The prospective multi-family, matched-base/sham/fork protocol is designed and driver-checked but has **not** run on the actual learner; successful test-double checks are infrastructure evidence only.

**Main priority:** Diagnose why V213's two-branch/gate transition learner does not separate event effects, rather than adding another hand-coded object rule or scaling the same failed method.

**Secondary priority:** Complete a properly paired V214 recall-vs-disabled-vs-sham forward-transfer comparison across independent target worlds, with **answer-vs-nonanswer bits** broken out.

## Verified observations versus hypotheses

### V213: development failure, not an official scientific outcome

Source: [V213 directory](../minifly_v213_observed_transition_design_20261104/), especially `DESIGN.md`, `learner.py`, `world.py`, `CYCLE3.md`, and `rehearsal/CYCLE3_TEST.json`.

- The simulator exposes opaque events, ordered participant IDs, movement and later position observations, with the hidden attach/release mapping kept outside the learner.
- The learner retains last-observed positions and a per-pair scalar `g`. Two shared softmax emission tables predict persistence versus copying the mover's observed position; an opcode-specific two-parameter state transition controls their mixture.
- In two engineering worlds the aligned, shuffled-opcode, frozen-gate and no-prior-write arms each solved **0/32 held strict twins**. The known-rule oracle solved **32/32**; the pooled count-table reference solved **29/32 held** and **27/32 taught**.
- The cumulative record budget was exceeded by **1,416**: 116,416 charged records vs the locked 115,000 cap. No official world opened. Do not convert the engineering outcome into an official result.
- `test_learning.py` checks the analytic local update against finite differences. This validates the tested derivative algebra; it does **not** establish optimization, adequate gradient magnitude, or learning of transferable dynamics.

**Candidate explanation (unverified): coupled branch/gate bootstrap.** For opcode `o`,

```
g_new = (1-g_old)*sigmoid(a_o) + g_old*sigmoid(b_o)
p(y) = (1-g_new)*p_self(y) + g_new*p_other(y)
d log p(y)/d g_new = [p_other(y)-p_self(y)]/p(y)
```

When both emission branches start near uniform (`p_self ~= p_other ~= 1/4`), the gate learning signal is close to zero. If the branches then receive similar mixture-weighted updates, branch specialization and opcode learning can obstruct each other. This is a plausible **symmetry/credit bootstrap hypothesis**, **not** the established reason V213 failed.

The next *small* diagnostic must record, over the same short training trace:
- `||p_self - p_other||`, sign and magnitude of gate credit, opcode parameter changes, and branch-row learning curves.
- Which branch predicts the observation better **when both are scored independently**, rather than only through their mixture.
- Whether the model learns a single event-effect mapping when the emission branches are separately calibrated or temporarily supplied with diagnostic oracle emissions.
- Whether independently taught branch distributions remain distinguishable when the opcode gate is trained without oracle event labels.

Use explicit `oracle-assisted diagnostic` labels for any experiment that supplies true branch meaning. Such experiments locate failure; they are **not** evidence of autonomous transition-rule acquisition.

Before any larger roster, compare ALIGNED with frozen, shuffled and no-write variants on identical observables. Inspect sample efficiency against the already successful pooled-count reference. Do not add a new task-specific attachment/detachment rule inside the learner as a "fix."

### V214B: a real but limited prequential transfer signal

Source: [V214B directory](../minifly_v214b_round20_fresh_seed_20261104/), especially `REPORT.md`, `cmr.py`, `run.py`, `results/SUMMARY.json`, `results/LOCK.json`, and `results/AUDIT.json`.

The recovered GLN + H1/H2 word-association predictor scores each bit before observing it and updates online as subsequent bits arrive. This measures **causal prequential compression**, not a freely generated answer or frozen target loss.

**Mixed 25-world stream (fresh seed):**

| Recall mode | Bits/byte |
|---|---:|
| Recall disabled | 0.6600 |
| Cue-shuffled recall | 0.6578 |
| True-cue recall | **0.5776** |

This confirms the ordering on one fresh mixed stream. The recall-disabled and shuffle arms are not perfectly equal-cost, equal-coverage controls; the original sham changes retrieved candidates.

**One held target (9,085 bytes; 80 questions):**

| Prior worlds | Target bits/byte | Bits saved versus no prior worlds |
|---:|---:|---:|
| 0 | 0.6089 | 0 |
| 2 | 0.5390 | +635 |
| 8 | **0.5321** | **+697** |
| 24 | 0.5443 | +587 |

The histories share prefixes, the target byte stream is identical, and the predictor continues updating on that target. These comparisons support **warm-history target compression**, not necessarily transfer of relational/world-dynamics rules. There is only one target world; the different history lengths are correlated. The 24-world result's decline relative to eight worlds does **not**, by itself, establish loss of plasticity.

**Important mechanistic bound:** Reported mean bits per answer payload for the 8-world target are about 3.386 (person) and 3.174 (object), versus 3.651 and 3.531 at birth. Thus the largest mean saving per answer is about 0.357 bits. Across 80 questions this can account for at most ~28.6 of the ~697 overall bits saved (under the reported span means): at least ~96% of aggregate savings is on **nonanswer bytes**. This is an arithmetic upper bound based on the published mean spans, not a new scoring run. It warns against interpreting the total code-length benefit as direct evidence of object-state learning.

The target uses newly generated named entities and held-out movement templates, while sharing generic vocabulary, question formatting, and pickup/drop templates with prior worlds. The existing audit also reports no identical target entity tokens or exact movement templates in the sampled history. These properties support a bounded lexical/template holdout, **not** an independent new causal ecology.

### V214A versus V214B

Source: [V214A protocol](../minifly_v214_prequential_recall_transfer_design_20261104/).

- **V214A** validates the scorer, checkpoint-fork accounting, prequential order, and control-flow checks with a **test double**. Its real parent was unqualified; official streams were never generated and the actual original learner was not run in that protocol.
- **V214B** recovers the Round 20 compressor and performs a smaller fresh-seed reproduction, but does **not** automatically complete V214A's 16-family × 3-arm × 4-checkpoint matched experiment.
- The restored source is an opportunity to qualify V214A's real parent. Do not call the prospective protocol completed until the actual causal recall, sham and disabled variants are run on identical target families and their source/metric assumptions are checked.

## Research decisions

### Priority 1 — One small V213 identifiability/optimization diagnosis

Do not rerun the full eight-world official roster or enlarge the state network yet.

A minimal controlled sequence should ask:

1. Can the emission tables learn **persistence and mover-following separately**, with the same observed-position feedback?
2. With those distributions distinguishable, can the opcode gate learn which predictive branch is appropriate?
3. When both are learned jointly from birth, where exactly does the signal vanish or cease to align with the correct outcome?
4. Can the resulting shared parameters predict *previously unpaired* entity combinations without additional training, under frozen weights?

For diagnostic branches, use the same examples, observed outcomes, entity IDs, initialization accounting and update budgets. A correctly labeled oracle-assisted branch is informative but may not count as a learner success. Maintain the strong pooled-count comparator; if it is superior at much lower cost, admit that.

**Advancement condition:** learned, label-blind event-effect parameters must outperform shuffled, frozen and no-write controls on a matched counterfactual twin panel. A differentiable update matching finite differences is insufficient.

### Priority 2 — Properly isolate V214 forward transfer

Complete multi-family V214 with:
- RECALL, BASE and cue-shuffled arms on identical chronological input streams and target bytes.
- Matched birth versus 2/8/24-world checkpoints; *the same target learning opportunities* for all forks.
- Causal answer-payload versus nonanswer code lengths, with first-quarter target loss separately reported.
- Independent world/family confidence intervals; no pseudoreplication over correlated checkpoints or bytes.
- Recall hit/availability, extra lookup cost, H1/H2 contribution, and the exact shuffled-cue information/coverage difference.
- A separate **frozen-start** or read-only probe panel when testing retained knowledge rather than online compressor adaptation.

Measure `G(k)=C(0)-C(k)` as the same-arm target code savings; for a recall-specific claim also require a contrast such as `G(RECALL,k)-G(BASE,k)` and a properly controlled sham. A higher generic compression score is not evidence of a learned object transition.

If 24-world transfer is lower than eight-world transfer in independent families, distinguish **stale-memory interference**, **mixer/calibration drift**, **changed initial predictive headroom**, and **plasticity loss** by evaluating both immediate frozen-start quality and improvement after a matched amount of new target evidence. Do not infer loss of plasticity solely from a lower cumulative transfer score.

### Priority 3 — Integrate only once the underlying capability is real

The long-term goal remains one continuous-life learner with sparse memory, causal retained context, local learning and event-driven predictions. But a continuous-life stream is a **training and evaluation protocol**, not proof that the learner has learned its environment's transition structure.

Do not introduce another QA-specific role parser, privileged object tracking, future-value leak, or benchmark-specific tie-order convention to bridge the gap. Keep observed event metadata and simulator-hidden rules separate. Reintroduce language only after learning from consequences works in a language-free, observable-event setting.

## Bottom line

- **Supported:** Bounded associative recall, consequence-based action updates in other experiments, context-retained binding, a recovered prequential memory-assisted compressor, and positive warm-history compression on one held target.
- **Not established:** Autonomous object dynamics, robust learned state transitions, general semantic inference, or durable forward transfer of causal rules over independently changed worlds.
- **Current bottleneck:** Identifying a local, stable learning mechanism that both preserves relevant state and learns reusable event effects from future observed consequences.
- **Next action:** Diagnose V213's joint-learning failure at the gradient/branch level; then validate V214's transfer signal across multiple held worlds with matched controls. Neither step requires a new benchmark-specific answer architecture.

**This note authorizes no training, workflow execution or repository code change.**
