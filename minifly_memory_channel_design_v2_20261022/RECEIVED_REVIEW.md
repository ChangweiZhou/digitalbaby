**The newest work has resolved the previous byte-recognition failure, but it has exposed a more consequential problem: the proposed native memory interface stores recognizable information without reliably reconstructing complete values.** That is now a more concrete obstacle than “MiniFly needs more reasoning machinery.”

My recommendation is **one tightly scoped memory-channel experiment before either another arithmetic redesign or substantial Transformer training**. It should determine how much of the failure comes from cross-address interference, how much from the write rule, and how much from the conditions under which memories are read.

I checked both active branches. The latest implementation-side commit is **`667ba0e0`**, published **October 22, 2026, at 8:12 p.m. Eastern**, following the external-memory diagnostics. Claude’s branch remains at `8e90d2e6`, the v5 proposal reviewed previously. I reviewed the new reports, saved metrics, diagnostic implementation and proposed host design; I did not rerun training.  

## 1. What has changed since the last review

There are now **two distinct research tracks**, not one continuous model-upgrade sequence.

| Track | New progress | Remaining limitation |
|---|---|---|
| **Native arithmetic: v5.1-R2** | Learned byte identities no longer alias; the repaired representation was exercised with native value learning | Completed histories still produced zero exact taught and untaught answers; the paired experiment stopped at its resource limit |
| **Native external memory: V1-R1** | Random-value tests establish item-specific storage, clearly exceeding no-write and shuffled-write controls | Payload errors and unreliable stopping leave complete recall at zero |
| **Transformer host** | A specific architecture, training protocol and causal controls are designed | No host or learned key-to-memory bridge has been trained or qualified |

The external-memory proposal explicitly permits attention and backpropagation **in the separate host and its offline-trained interface**, while preserving local native learning inside MiniFly. It does not retroactively change the original arithmetic track’s constraints. This is a declared new application of MiniFly, not evidence that the original arithmetic objective has been achieved.   

### The old sensory problem is genuinely no longer the immediate blocker

V5.1-R2 replaces the unsuccessful random prototype initialization with initialization from actually observed sensory vectors. Both tested births had **zero aliases among observed byte types**. Context grouping remained variable—two occupied classes in one birth, one in the other—but the experiment proceeded to test the complete path rather than endlessly optimizing a component score. 

That was the right decision. A single context class is a warning about the intended segmentation mechanism, but it does not by itself make learning impossible when raw bytes and bounded positional buffers remain available.

The integrated result was nevertheless poor. Six complete histories had **zero exact taught and held-out answers**. Representative taught outputs were `1125`, `1127` and `1107` for quite different questions. Correct and shuffled teaching also produced very similar teacher-prefix metrics. This makes common output-pattern learning a plausible interpretation of that run, rather than adequate acquisition of individual arithmetic facts. 

The run stopped after 4,303 main record applications at its CPU limit. Missing endpoints remain missing, so its official conclusion is appropriately **`RESOURCE_LIMIT_NO_EFFICACY_VERDICT`**, not a completed comparative rejection of the architecture. 

**My update:** do not spend another iteration repairing byte identities. That repair has done its job. The next question is whether distinct inputs produce sufficiently distinct, usable value-writing and retrieval effects.

## 2. The external-memory test gives a cleaner diagnosis

The external-memory diagnostic removes arithmetic and the unqualified learned encoder from the immediate problem.

It teaches arbitrary four-byte values using target-independent random receptor inputs passed through the canonical native PN/KC circuitry. The inputs are a **surrogate addressing test**, not a learned byte encoder and not the future deployment interface. The test uses the canonical 5,177-cell anatomy and nine output compartments: eight payload bits and EOS. 

This makes the central question simple:

> When the native system receives a particular address and an actual value, does subsequent activity at that address contain information about that value?

The answer is now clearly **yes, but imperfectly**.

### Final OLD recall in the third diagnostic cycle

| History | Correct payload bits | Correct four-byte payload, **with length artificially supplied** | Complete value with own EOS |
|---|---:|---:|---:|
| Correct writes | **91.4%** | **31.25%** | **0%** |
| No writes | 49.4% | 0% | 0% |
| Shuffled OLD values | 48.8% | 0% | 0% |

The forced-length column is diagnostic only. It is not a permissible replacement for autonomous output. The test contained 16 OLD values, so 31.25% means **five complete payloads** were recoverable by the current bit thresholding when the stopping problem was removed. 

This is stronger evidence of content storage than a high bit score on decimal answers. Here the values are random bytes, and the matched controls are near 50%. The gain cannot be explained simply by learning that answers tend to contain ASCII digits. 

But it also establishes that **EOS is not the only problem**. A perfect stopping decision would not repair the eleven incorrect payloads in that endpoint.

### Additional repetition helps content, but has not solved complete recall

In the separate acquisition curve, payload-bit accuracy rose from **87.3% after one pass to 95.5% after 64 passes**. At 64 passes, five of sixteen payloads were completely correct under forced length, and the EOS-at-position-four rate was 25%. Complete autonomous recall remained zero. 

Two conclusions follow:

**It is wrong to say the native memory learns nothing.**

**It is equally wrong to describe it as a working arbitrary-byte memory service.**

The useful present description is **a noisy, address-dependent associative memory signal**. That is enough to justify further investigation, but not enough to claim a general persistent core.

## 3. What the new evidence says about the mechanism

### A. Passive forgetting is not the leading explanation in this particular test

In the third cycle, OLD payload-bit accuracy was approximately **91.6% immediately after acquisition and 91.4% after the two rests and intervening NEW writes**. The target-aligned response margin declined, but most bit signs remained correct. 

This is different from the earlier adaptation-dependent collapse. It argues against making longer memory time constants the default repair here.

There is an important limitation: OLD received sixteen passes, whereas NEW received only one. Furthermore, the diagnostic code scores OLD after the NEW phase but does **not** report a corresponding NEW-acquisition curve. Therefore, stable OLD bits do not yet establish a good retention–plasticity trade-off; NEW may simply have made a relatively weak intervention. 

**The next test must measure both what remains usable and what has newly been acquired.**

### B. Different addresses still affect one another

The first diagnostic reports about **9.8% shared active support between different keys at the same output position**, with 184 active cells per address. A single write changes other keys’ readouts as well as its own. The reported mean other-key change norm is roughly one-eighth of the self-change norm. These are measurements from one technical configuration, not a general capacity estimate. 

The important point is not that any overlap is unacceptable. Overlap can be useful and does not automatically imply incorrect recall. What matters is **effective interference**:

> Does teaching one value move another value’s readout in a direction that damages its answer?

The latest memo correctly distinguishes raw support overlap from this signed functional effect. Mean-centering receptor drives does not equalize cell recruitment or make all cells equally likely to win. Repeatedly recruited cells may carry a disproportionate share of the interaction. 

I would therefore prioritize the proposed **one-write/cross-read influence measurement**, not simply optimize an overlap percentage.

For addresses indexed by key and output position, measure

\[
I_{ji}
=
m_j(\text{after a write at }i)
-
m_j(\text{after a matched passive event at }i).
\]

Do this with both target polarities. The diagonal shows whether the intended address is controllable; off-diagonal signs show which other answers are affected. This is much closer to the actual memory problem than counting shared cells.

### C. “Error-corrective teaching” must demonstrate that it corrects

The implemented coefficient is

\[
s=a(n)\frac{\tanh(m/2)-y}{2}.
\]

For small margins,

\[
s\approx-\frac{a(n)y}{2}+\frac{a(n)m}{4}.
\]

Thus early writes may be dominated by the target term and behave approximately like accumulation. That does not mean the prediction-dependent term remains negligible later, or that the native nonlinear update is equivalent to simple summation. The saved margins grow substantially over the acquisition curve.  

The proposed matched control—remove only the prediction-dependent term—is therefore useful. It asks whether the “correction” component actually improves exact retrieval at the same addresses, schedule and clock.

This distinction also appears in fast-weight memory research, where replacing purely additive storage with a delta-like update can improve correction of existing associations. That literature motivates a comparison; it does not validate MiniFly’s particular native implementation. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v139/schlag21a.html?utm_source=chatgpt.com)

### D. There is another interface mismatch worth testing

The diagnostic source reveals a potentially important difference between writing and reading:

- During teaching, the native state advances after each payload/EOS event.
- During recall, the code resets a private snapshot and reads every position from that same snapshot **without inter-position ticks**. 

Consequently, later positions can be written and read under different adaptation conditions, even when their sparse address vectors are identical.

This is **a declared API choice, not an undisclosed coding bug**, and the current evidence does not establish that it causes the errors. However, earlier project results already made state-dependent access important enough that I would not leave this difference untested.

A cheap diagnostic can compare, on the same saved memory:

**the current simultaneous-snapshot read**, versus **a private passive sequence of reads that advances the relevant state between positions**.

Neither condition needs the correct answer or its length. The fixed surrogate addresses already specify the preceding positions. A difference in performance would localize a read/write compatibility issue before changing the stored values or learning rule.

## 4. My view of the Transformer direction

**The separate Transformer-host experiment is reasonable. It changes the research question in a legitimate and potentially useful way.**

Instead of asking native MiniFly to discover arithmetic, interpret sequences and generate answers by itself, it asks:

> Can a frozen general-purpose host use newly written information whose only persistent source is native MiniFly plasticity?

The design makes this division explicit. It proposes a roughly 0.92-million-parameter byte Transformer, a frozen key-producing interface, native local writes, and a decoder trained to interpret native responses. Test associations are generated after the host is frozen, and old write records are removed from its query context. 

The general separation of a language model and nondifferentiable external memory has precedent, including *Memorizing Transformers*. That precedent supports testing the architecture; it does not establish an advantage for the native store over simpler alternatives. [arXiv](https://arxiv.org/abs/2203.08913?utm_source=chatgpt.com)

### What I would not do

I would not train a large host and hope it conceals an unexplained memory failure.

But I also would not require **perfect raw sign-thresholded native recall** before allowing any learned readout experiment. A decoder may exploit magnitudes, correlated response patterns and calibration information that direct thresholding discards. The current revised design acknowledges this possibility. 

The correct boundary is:

**First establish stable, target-specific information and characterize its errors. Then test whether a bounded decoder can recover it on fresh associations.**

For independent random values, the host cannot infer missing content from the key. It can only exploit information actually present in the memory responses or accidentally leaked through the experiment. That makes fresh random associations an especially useful causal test.

### The hybrid must earn its own contribution

A successful demonstration should freeze the entire host before new associations arrive and show that its outputs follow the native writes. A particularly clean test is to use the **same key with different independently drawn values in two separate memory lives**. The frozen host should follow the value stored in the corresponding life, not a learned default.

No-write, shuffled-value, zeroed-read and foreign-memory controls remain necessary. The design already includes this general family of controls, along with exact-table and simpler memory references. 

The exact-table comparison is not an insult to MiniFly. On arbitrary key–value recall, a table is an appropriate benchmark. Native memory eventually needs to justify itself through some measured property—bounded-state behavior, useful forgetting, robustness, transfer across related cues, or another explicitly tested advantage—not merely the ability to approximate a table.

## 5. What I would run next

I largely agree with the latest diagnosis memo, but I would make its proposed work more sequential. **Do not simultaneously change normalization, teaching, stopping, the learned encoder and the host.** That would recreate the attribution problems of earlier versions.

### First: finish the functional diagnosis on the existing memory

Use saved states and small controlled perturbations to measure:

| Measurement | Question it resolves |
|---|---|
| Signed one-write/cross-read influence | Are failures caused by broad destructive interactions between addresses? |
| Current versus sequential passive read | Does read/write state mismatch contribute to errors? |
| Payload and EOS errors by position and item | Are mistakes dispersed, concentrated in particular keys, or shared across positions? |
| Actual prediction and target terms at write time | Does the residual meaningfully correct existing errors? |

These should produce a decision, not another open-ended audit. The current arithmetic postcycle plan and latest external-memory memo already prioritize influence and write-time instrumentation; the read-protocol comparison is the main addition I would make.  

### Second: test one justified storage intervention

If excessive cross-address influence is the dominant finding, compare the unchanged representation with **one generic activity-normalization or recruitment-control intervention**, calibrated without target values and frozen before evaluation.

Keep the native population, values, write schedule and evaluation unchanged. Avoid allocating private cells per key or selecting favorable addresses after seeing answers.

Alongside it, the target-only accumulation control can establish whether the current corrective term is useful. Preserve the gradual warmup rather than silently increasing strength.

Crucially, comparisons across memory loads need care. With the current schedule, increasing the number of facts also changes the number of completed records and therefore the teaching strength reached at a given number of passes. A load comparison can otherwise confuse interference with training dose. The latest memo correctly flags this. 

### Third: separate stopping from content, using variable-length values

The fixed four-byte test has one positive EOS target and four negative EOS targets per record. That makes a common stopping bias plausible, but not proven.

I would use variable-length values within the declared interface limit and compare the existing EOS readout with **one training-only calibrated readout**. If the response distributions themselves remain poorly separated, balanced EOS teaching becomes a different, explicitly identified intervention—not another parameter quietly folded into calibration. 

Both payload and stopping must remain visible. A successful EOS adjustment cannot be described as solving memory when most payloads are still wrong.

Also test never-written keys. A populated associative memory may produce confident responses at unfamiliar addresses; correct abstention is a separate service requirement, not guaranteed by recall on known keys.

### Fourth: only then qualify the actual learned interface

Random surrogate addresses are useful for isolating the backend. They do not establish that a learned key encoder will create similarly usable addresses.

The learned interface should face the same influence, occupancy, read/write-compatibility and recall measurements. Only after that should the full host experiment proceed.

A small decoder experiment can occur once a usable signal is established; it need not wait for perfect raw thresholding. But the decoder must be trained on separate episodes and frozen before scoring fresh values. Otherwise it can become another place where the test associations are stored.

## 6. What should happen to the original arithmetic project?

**Keep it separate, and stop interpreting these generic-memory results as arithmetic progress.**

The new native-memory work is useful because it tests whether a broader interface can actually preserve and deliver content. It may repair a prerequisite that the current arithmetic wrapper lacks. It does not discover addition.

Conversely, the earlier successful narrow memory demonstrations are not erased by these failures. The output representation, address geometry, task, dose and read protocol have changed. What has not been established is reliable inheritance of the old capability by these newer interfaces. The external-memory design explicitly recognizes that boundary. 

For the arithmetic track, I would retain the repaired sensory path and the earlier working native-memory reference. I would **not** introduce another segmentation mechanism, larger cell population or consolidation scheme before resolving whether the actual encoded frames permit reliable taught acquisition.

When taught acquisition is adequate under the new interface, the next arithmetic question returns to the one v3 exposed:

> What mechanism makes the same learned relationship usable on an untaught combination?

That remains a different problem from reducing storage noise.

## Overall judgment

The newest progress is **diagnostic rather than a demonstrated capability advance**, but the diagnosis is considerably sharper:

**Byte identity:** repaired on the tested alphabet.

**Integrated addition:** still unsuccessful in completed histories; the latest paired run was resource-limited.

**Native arbitrary-value storage:** demonstrably content-specific, but noisy.

**Complete autonomous recall:** still zero in the newest surrogate tests.

**Transformer integration:** designed, not trained.  

The project should now avoid two opposite errors: dismissing the native mechanism because complete recall is zero, and treating 91–95% bit accuracy as a practically working memory.

**The next useful functionality is reliable access to stored content under overlapping addresses and realistic read conditions—not another hidden layer.** Resolve that with one bounded experiment and explicit decision criteria. Then test a frozen host’s ability to use new native memories, while preserving arithmetic-rule learning as a separate, still-unresolved objective.