# After LEARNED-BANK V1: what the mechanism catalog suggests, and the next steps (24 Oct 2026)

A discussion memo. Nothing here has been run. It reads the LEARNED-BANK V1 result (`minifly_learned_bank_v1_20261024/REPORT.md`) against the human learning mechanism catalog (`human_learning_mechanisms_20261023/`, 150 entries with evidence grades A to E). It then proposes an order of experiments, one step at a time.

## 1. Where V1 leaves us
V1 gave partial support for condition 1 (3 of 4 fresh worlds) by its locked rule. A feature layer trained by a local three-factor rule made the learned bank's address kernel task-aligned on held-out parity.
* Learned-bank balanced held accuracy: 0.73 to 1.00, against 0.36 to 0.61 for frozen random features.
* M1 missed the pre-registered margin over the shuffled-label control.

Four problems remain:
1. **The fly's own answer is worse than its learned bank.**
   * System held accuracy is 0.48 to 0.81, against 0.73 to 1.00 for the learned bank alone.
   * The fixed shared bank answers held prompts below chance, and the two banks' margins are simply added.
2. **Data hunger.**
   * About 11,000 records were needed to learn from 74 examples (150 passes; learning levels off by about pass 100 to 125).
   * The sparse KC store memorises the taught set within about 20 passes, after which the error that trains the feature layer is small.
3. **Code collapse.**
   * Mean overlap of learned-bank codes rose from 0.22–0.34 to 0.70–0.83.
   * The cause: answer bits that never vary give every prompt the same error sign. The collapse slows the storage of arbitrary labels, and will matter more with 10 answer classes.
4. **The testbed is tiny.**
   * One answer bit and 15 held orbits per world.
   * At chance, a control's balanced accuracy has a sampling SD of roughly 0.13, and that noise decided M1.

### The main reading
Problems 2 and 1 are largely artifacts of a tiny dataset repeated many times.
* Once every taught item is memorised, training error carries no information about generalisation. The feature layer loses its error signal.
* No rule for weighting the two banks can learn which one generalises: on taught items both look perfect.
* In a stream of mostly fresh problems, the error on first exposure *is* the generalisation error. Both the feature layer and any arbitration then have a signal for as long as new problems keep arriving.
* Children learn arithmetic from thousands of distinct problems, not from 74 problems repeated 150 times.

## 2. What the catalog offers, by problem
| Problem | Catalog mechanisms (grade) | Status in MiniFly |
|---|---|---|
| 1. System answer | Complementary learning systems (B); model-based vs model-free arbitration by reliability (B); hippocampus–VTA novelty loop (B) | Not present: banks are summed. A reliability or novelty gate needs errors on novel items to learn from, i.e. a fresh-problem stream (step 2 below). |
| 2. Data hunger | CLS: slow cortical learner, fast hippocampal store (B); hippocampal replay (B); targeted memory reactivation in sleep (B); schema-dependent rapid consolidation (B) | Partly present. V1's "the memoriser starves the feature learner" is the CLS argument for why cortex must learn slowly while the hippocampus memorises. Replay is not used yet. |
| 3. Code collapse | Rescorla–Wagner, error relative to expectation (A); intrinsic excitability homeostasis (C); synaptic scaling (C); heterosynaptic competition (C); pattern separation (B) | Partly present. The shared bank has the native one-sided KC homeostasis and the learned bank does not. A baseline-subtracted third factor was tested on the surrogate in cycle 2: it halved the collapse without improving held accuracy, so it was not adopted. |
| 4. Tiny testbed | (none: a task-design issue) | — |
| Beyond condition 1 (carries, multi-step answers) | Activity-silent working memory (B); competitive queuing for serial order (B); prefrontal–basal ganglia gating (C) | Not present. The native fast stores are themselves a synaptic short-term memory, close to the activity-silent account. |

Radical options to keep in view (exploratory, mostly grade C):
* **Dendritic conjunction units (C).** A single unit with nonlinear dendrites can compute an XOR-like interaction, which is the operation parity needed. Such units could stand in for, or sit beside, the learned hidden layer.
* **Behavioural-timescale plasticity, BTSP (C).** Plateau-gated one-shot plasticity: a candidate for allocating a new feature in a single event, instead of over hundreds of passes.
* **Latent-cause inference (B).** A large prediction error creates a new inferred context instead of overwriting the old one. It is a candidate for learning several tasks without interference.

A caution: most of the useful entries are grade B or C. They motivate designs; they do not validate them. Each step still needs its own controls (BASE, FROZEN, SHUF, ORACLE), a lock, and fresh main worlds.

## 3. Proposed order of steps
Each step asks one question and changes one thing.

### Step 1. Screen a larger item space (evaluator side only; about half a day; no native runs)
* **Prompts:** two-digit operands (`37+45=`). The answer depends only on the units digits.
* **Tasks:** units-digit parity first, then the units digit of the sum (the column rule of addition).
* **Held out:** units-digit pairs (orbits) that are never shown with any tens digit.
* **Training:** a stream of mostly fresh problems instead of a small set repeated many times.
* **Screen:** exact gradient, the local three-factor rule, and frozen random features, as `diag_mlp.py` did in V1. Record held accuracy as a function of the number of problems seen.
* **Questions:**
  * Is the column rule learnable from this input code with N fresh problems?
  * Do frozen random features fail it?
  * Does the FE-PERM front end keep units digits distinguishable in two-digit prompts? If not, the front end becomes the first problem to fix.
* **If it passes,** the testbed has 10 answer classes, many held items, much less chance noise in the controls, and a real novelty signal.

### Step 2. LEARNED-BANK V2 on the native stores: only the data change
* V1's code, lock pattern, controls and decision-rule style are reused on the step-1 testbed.
* **Question:** does local feature learning extend to a 10-class column rule that ignores the tens digit, and does a fresh-problem stream remove the data hunger?
* **Measured:** held accuracy against the number of real records seen.

### Step 3. Collapse fix (move it before step 4 if V2 shows collapse limiting 10-class storage)
Apply the native KC homeostasis to the learned bank, reusing an existing mechanism, or the Rescorla–Wagner baseline on the third factor. Compare against V2 with everything else fixed.

### Step 4. Bank arbitration
Weight the two banks by a reliability learned from first-exposure errors, in the spirit of model-based versus model-free arbitration and of novelty gating. This is learned from data, not routed by hand, and it is possible only once the stream contains novel problems.
* **Success:** the system answer is at least as good as the better bank, on both taught and held problems.

### Step 5. CLS replay during rest
* Replay stored episodes from the fast store during rest to train the slow learner.
* **Question:** does it cut the number of real records needed for a given held accuracy?
* **Note:** the slow learner must stay slow enough that its own error does not vanish. That is the core CLS prediction, and V1 already saw the failure mode.

### Later: beyond condition 1
Carries and multi-digit answers need state carried across steps. Candidates: activity-silent working memory, which the native fast stores may already provide; competitive queuing for order; and gating. These come only after steps 1 to 4 give a fly whose own answer generalises a single column.

## 4. What this memo does not claim
* Nothing in it has been tested; the order is a proposal.
* The catalog's mechanisms are human or animal findings of varying strength. A good fit in a table is not evidence that a mechanism will work in MiniFly.
* V1's support is partial and confined to one binary interaction in a toy setting.
