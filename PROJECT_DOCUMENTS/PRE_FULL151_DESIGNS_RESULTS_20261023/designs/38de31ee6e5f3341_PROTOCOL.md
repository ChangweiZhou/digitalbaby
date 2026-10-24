# V36 — Generalizing Ownership
## Refined design after V35 and the second fly re-audit

## Status

**Design / preregistration only. No V36 scientific outcomes have been observed.**

V36 follows the completed V35 result.

The central V35 finding was not merely that exact local memory retains. The decisive causal contrast was:

\[
\text{LOCAL-R}: s=s_B+r_c
\]

versus

\[
\text{LOCAL-V}: s=v_c.
\]

Both arms used:

- the same V32 temporal base;
- the same exact raw 12-byte address;
- the same eligibility schedule;
- the same number of local vectors;
- the same 0.6 local update scale.

Yet LOCAL-R retained roughly 30% of untouched A while LOCAL-V retained roughly 99%.

Therefore the large LOCAL-V − LOCAL-R gap is **not an addressing or capacity effect**. It isolates whether owned state is interpreted relative to a drifting shared base or owns the absolute prediction.

At the same time, LOCAL-V transferred essentially no revision to the other rendering of the same logical key.

V36 therefore asks:

\[
\boxed{
\textbf{Can absolute local ownership be made to generalize across causally similar contexts
without giving up its resistance to unrelated overwrite?}
}
\]

This is now more urgent than bounded capacity.

Bounded capacity is deferred until we know whether the owned state can represent a *logical object* rather than only an exact byte string.

---

# 1. Empirical constraints inherited from V35

The following V35 conclusions are treated as fixed evidence.

## 1.1 Exact local absolute ownership works

LOCAL-V achieved approximately:

- 99% untouched-A retention after D;
- 100% D-current acquisition;
- immediate exact revision after a contradiction;
- no collateral damage to unrelated exact local memories;
- whole-stream prediction better than the shared base.

This establishes exact-context ownership as a real anti-overwrite primitive.

It does **not** establish a bounded lifelong architecture.

## 1.2 Additive local residuals remain vulnerable to base drift

LOCAL-R and LOCAL-V had identical exact addressing and capacity.

The sole architectural difference was:

\[
s_B+r_c
\quad\text{versus}\quad
v_c.
\]

The V35 base-drift diagnostics showed that the shared base score on old contexts moved substantially after later waves.

Therefore V36 includes a frozen-base LOCAL-R control to causally verify that this drift explains the LOCAL-R failure.

That control is diagnostic only and is not a deployable candidate.

## 1.3 Exact ownership does not create logical-object identity

For true revision of rendering 0:

- LOCAL-V revised rendering 0 rapidly;
- the already learned alternate rendering remained essentially at the obsolete value;
- the held-out rendering did not inherit the new value.

This failure is mechanical under exact addressing.

The result separates:

\[
\boxed{\text{storage ownership}}
\]

from:

\[
\boxed{\text{addressing / identity formation}}.
\]

## 1.4 The alias endpoint is task-valid

The V35 current-fact minimum-norm oracle can transfer revision to the never-trained alias substantially above the live learners.

Thus failure of held-out alias transfer is not an intrinsic impossibility of the workload.

V36 may use alias transfer as a genuine endpoint.

---

# 2. The missing 2×2

V35 tested only the left column:

| value semantics | exact address | similarity / logical address |
|---|---:|---:|
| additive residual on shared base | LOCAL-R | **untested** |
| absolute locally owned value | LOCAL-V | **untested** |

V36 fills the right column.

The principal live causal comparison is therefore:

\[
\boxed{
\text{exact vs similarity-linked addressing}
\times
\text{residual vs absolute value semantics}
}
\]

with the same shared base.

---

# 3. What the fly re-audit contributes — and what it does not

The additional connectome analysis motivates a **continuum of overlap** rather than exact identity versus global sharing.

The useful structural idea is:

> related inputs should reactivate substantially overlapping local substrate, while unrelated inputs should overlap much less.

This could naturally supply:

- retention through low unrelated overlap;
- generalization through partial related overlap;
- revision through reactivation of the same local substrate.

However, V36 does **not** assume that simple width plus sparsity automatically creates this regime.

## 3.1 V34 already tested a nearby sparse-expansion hypothesis

V34 used a fixed 768-unit expansion with global Top48 sparsity.

Its fixed code produced approximately:

\[
J_{\rm same}\approx0.349
\]

for the two trained renderings and

\[
J_{\rm different}\approx0.079,
\]

a separation ratio of only about:

\[
4.4\times.
\]

The global sparse ordinary arm learned well but retained only about 6% of untouched A after D.

Thus:

\[
\boxed{
\text{sparse expansion at a few-fold separation can delay interference without producing persistence}
}
\]

is already established.

V36 will not repeat V34 under a larger width unless a read-only representation gate first demonstrates a qualitatively stronger separation regime.

## 3.2 The current byte workload contains large common syntax

A learned record context is:

\[
\texttt{~|~|} + \text{header} + \text{6-byte key} + \text{separator}.
\]

Two renderings of the same key differ only in header and separator and share 10 of 12 raw causal bytes.

But different logical keys also share substantial fixed syntax.

Therefore a random sparse expansion cannot be assumed to behave like odor patterns whose unrelated sensory-channel overlap is near zero.

A useful fly-like code for byte data may ultimately need recurrent/statistical preprocessing that discounts ubiquitous syntax.

That is not added to V36.

---

# 4. Non-negotiable causal interface

Every deployable V36 candidate receives only:

\[
\text{past bytes}
\rightarrow
\text{prediction}
\rightarrow
\text{actual next byte}
\rightarrow
\text{online update}.
\]

No learner receives:

- wave ID;
- task ID;
- logical-key ID;
- alias ID;
- target-position ID;
- revision flag;
- corruption flag;
- semantic memory ID;
- future information.

Logical identity and alias metadata remain evaluator-only.

There is no replay of untouched A.

There is no reset at wave or record boundaries.

---

# 5. Fixed workload

Reuse the exact V35 / V34 multi-wave generator:

\[
A\rightarrow B\rightarrow C\rightarrow D.
\]

Per full trajectory:

- 64 logical keys;
- 16 keys per wave;
- 32 initial target presentations per key;
- two alternating learned renderings;
- one never-trained alias;
- four one-off corrupt A keys;
- four disjoint truly revised A keys;
- eight untouched A keys;
- 2,084 target events;
- 29,176 total next-byte prediction events.

Unique examples and total presentations must be reported separately.

---

# 6. Fresh scientific seeds

Use:

\[
\boxed{36101,\ldots,36108}
\]

for the main ownership/addressing block.

Use separate fixed engineering/preflight seeds:

\[
\boxed{963601,\ldots,963604}.
\]

Before science, audit both blocks against all accessible prior manifests and embedded source seed lists.

If any collision is found before outcomes exist, replace the entire affected block with the next unused consecutive block and lock it.

---

# 7. Shared base — held fixed

All main V36 arms use the exact source-locked V35 Stage-II base:

> the V32 current temporal full-covariance causal byte learner.

The implementation is copied from the executing V35 source, including:

- current temporal representation;
- fixed eight-bank router;
- full covariance writer;
- historical 0.6 update scale;
- historical direction cap;
- softmax temperature 0.1.

As in V35, auxiliary output must **never** affect base learning.

For every main arm and seed:

- base prediction before auxiliary combination is identical;
- base update is identical;
- base state hashes must be identical at every checkpoint.

A separate BASE trajectory is unnecessary.

The runner records counterfactual base-only loss and accuracy from every arm.

Any base-state mismatch invalidates the experiment.

---

# 8. Private address representation

Use the exact V35 lag96 address code:

\[
k(c)\in\mathbb R^{96}.
\]

No encoder learning.

No retuning.

Before science report on the V36 streams:

- same-key cosine distributions for rendering 0/1 and 0/2;
- different-key distributions;
- full tails, not merely means;
- nearest-neighbor ranks;
- false-neighbor rates at the prespecified similarity threshold.

---

# 9. Main arm roster

Six live arms are run on the eight fresh scientific seeds:

| Arm | Addressing | Value semantics | Role |
|---|---|---|---|
| **EXACT-R** | exact 12-byte context | additive residual \(s_B+r_c\) | V35 LOCAL-R replication |
| **EXACT-V** | exact 12-byte context | absolute local \(v_c\) | V35 LOCAL-V replication |
| **SIM-R** | similarity-linked slot | additive residual \(s_B+r_j\) | missing 2×2 cell |
| **SIM-V** | similarity-linked slot | absolute local \(v_j\) | primary missing cell |
| **PRED-V** | similarity + predictive confirmation | absolute local \(v_j\) | self-supervised identity candidate |
| **FROZEN-R** | exact context | additive residual | nondeployable base-drift causal control |

Thus:

\[
6\times8=\boxed{48\text{ main scientific trajectories}}.
\]

No capacity bound or eviction is imposed in the main block.

---

# 10. Common V35 eligibility rule

To preserve comparability, retain V35's parameter-free local-state eligibility concept.

The base-only prediction is computed before any auxiliary update.

After observing byte \(y\):

- if the base top-1 prediction was wrong, the exact current context becomes eligible for auxiliary state;
- once eligible it remains eligible.

For SIM/PRED arms, similarity retrieval can make a tentative prediction before the exact context has permanent membership; permanent binding is specified below.

The base-error rule is **not** interpreted as a revision detector or special biological mechanism.

Report target-eligibility coverage as in V35.

---

# 11. EXACT-R

Identical in meaning to V35 LOCAL-R.

Each eligible exact context \(c\) owns:

\[
r_c\in\mathbb R^{256}.
\]

Prediction:

\[
s=s_B+r_c.
\]

Update:

\[
r_c^+
=
r_c
+
0.6(y-s).
\]

Only \(r_c\) changes.

All other local states are bit-for-bit unchanged.

---

# 12. EXACT-V

Identical in meaning to V35 LOCAL-V.

Each eligible exact context owns:

\[
v_c\in\mathbb R^{256}.
\]

Prediction:

\[
s=v_c.
\]

Update:

\[
v_c^+
=
0.4v_c+0.6y.
\]

The shared base continues learning independently but does not affect the prediction of an owned exact context.

---

# 13. Similarity slot representation

SIM-R, SIM-V, and PRED-V store local **slots**.

A slot \(j\) contains:

- a set of exact member contexts \(M_j\);
- the lag96 vector of each member;
- a normalized prototype \(p_j\), equal to the normalized mean of member lag vectors;
- one value state:
  - residual \(r_j\) for SIM-R;
  - absolute \(v_j\) for SIM-V/PRED-V.

No logical-key ID is stored.

No alias ID is stored.

No evaluator metadata enters the slot.

## 13.1 Fixed similarity threshold

Use:

\[
\boxed{\tau=0.85}.
\]

This threshold is selected **before V36** from V35's already observed lag96 geometry.

It is not tuned on V36 outcomes.

The V36 report must display the entire V36 same/different similarity distributions so the threshold's operating characteristics are transparent.

---

# 14. Similarity retrieval

For current exact context \(c\):

### Exact member hit

If \(c\in M_j\) for some slot, retrieve that slot directly.

### Unbound context

Otherwise compute:

\[
s_j=p_j^\top k(c)
\]

for every slot.

Let:

\[
j^*=\arg\max_j s_j.
\]

If:

\[
s_{j^*}\ge\tau,
\]

the slot is a **tentative similarity candidate** for the current prediction.

Otherwise there is no auxiliary retrieval and prediction is base-only.

This tentative lookup is causal and requires no identity label.

---

# 15. SIM-R permanent binding

SIM-R tests similarity address with residual value semantics, without predictive confirmation.

When a previously unbound exact context becomes eligible:

- if a tentative candidate with similarity \(\ge\tau\) exists, permanently bind the context to that slot;
- otherwise allocate a new slot.

The context's lag vector is added to the slot and the prototype is recomputed.

Prediction for a retrieved slot:

\[
s=s_B+r_j.
\]

Only the retrieved slot's residual updates.

This arm intentionally exposes the failure mode of false similarity merges.

---

# 16. SIM-V permanent binding

Same address rule as SIM-R.

The only difference is absolute ownership.

Prediction:

\[
s=v_j.
\]

Update:

\[
v_j^+
=
0.4v_j+0.6y.
\]

Thus SIM-V fills the previously missing cell:

\[
\boxed{\text{similarity-linked address + absolute owned value}.}
\]

This is the primary V36 arm.

---

# 17. PRED-V — predictive-confirmed similarity ownership

PRED-V tests a stricter autonomous identity rule:

\[
\boxed{
\text{similar context}
+
\text{compatible predictive consequence}
}
\]

rather than similarity alone.

For an unbound context:

1. find tentative candidate \(j^*\) using the same \(\tau=0.85\);
2. use the candidate's current absolute value for the tentative prediction;
3. observe the true next byte \(y\);
4. if the candidate's **pre-update top-1 byte equals \(y\)**, permanently bind the new exact context to that slot;
5. otherwise allocate a new slot when the context satisfies the common eligibility rule.

Once contexts are bound to a slot, later contradictory outcomes update that one shared slot and therefore propagate revision to all linked members.

No global revision classifier is used.

The rule asks only whether a proposed identity link predicted the observed consequence correctly.

## 17.1 Held-out alias probe

A never-trained held-out alias is never permanently bound by evaluator probing.

At a read-only probe it may tentatively retrieve the most similar slot if \(s\ge\tau\).

This permits genuine held-out transfer measurement without modifying learner state.

---

# 18. FROZEN-R — base-drift causal control

FROZEN-R is intentionally nondeployable.

It is identical to EXACT-R except:

- after the end of wave A, the shared base state is frozen permanently;
- exact residual memories continue learning causally.

The A boundary is evaluator metadata and is therefore used only in this diagnostic control.

FROZEN-R cannot qualify as an architecture.

Its sole purpose is to test:

\[
\boxed{
\text{Does removing shared-base drift restore additive local residual persistence?}
}
\]

If FROZEN-R rises close to EXACT-V while EXACT-R remains poor, the V35 base-drift interpretation is causally confirmed.

---

# 19. Main primary endpoints

Always report absolute levels.

## Persistence

\[
\boxed{\text{untouched A after D}}
\]

averaged over the two trained renderings.

Also report after A/B/C.

## Current learning

\[
\boxed{D_{\rm current}}.
\]

## Exact revision

For revised rendering 0:

- prequential accuracy after 1/2/4/8 changed occurrences;
- probe accuracy;
- obsolete-target probability.

## Known-alias revision

Rendering 1 of the same logical key.

This becomes a **primary V36 addressing endpoint**.

## Held-out alias transfer

Never-trained rendering 2.

This is also primary, because V35 oracles established that the task can express this property.

## Prediction

Report:

- whole-stream bpb;
- target bpb;
- non-target bpb;
- base-only counterfactual bpb.

## Sample efficiency

Report:

- first target occurrence;
- second logical-key occurrence;
- first exact recurrence;
- first recurrence after slot creation;
- number of observations before two aliases share one slot.

---

# 20. Addressing diagnostics

For SIM/PRED arms report:

- slots allocated by wave;
- exact contexts per slot;
- target versus non-target slots;
- number of logical keys per slot, evaluator-only;
- number of slots per logical key;
- same-key merge rate;
- different-key false-merge rate;
- false merge rate weighted by presentations;
- held-out nearest-slot accuracy;
- nearest similarity;
- top1–top2 similarity margin;
- slot prototype drift;
- revision propagation across linked members.

Logical-key metadata is diagnostic only.

---

# 21. Prespecified main contrasts

Use paired two-sided 95% Student-\(t\) intervals across eight seeds.

## V1 — fresh replication of absolute anchoring

\[
\boxed{\text{EXACT-V}-\text{EXACT-R}}
\]

on untouched A after D.

This replicates V35's large R3 effect.

Strong support requires:

- both arms validly acquire A and D;
- paired lower bound \(>0\);
- mean gain \(\ge40\) pp.

## V2 — frozen-base mechanism check

\[
\boxed{\text{FROZEN-R}-\text{EXACT-R}}.
\]

Primary diagnostic interpretation:

- if FROZEN-R is within 10 pp of EXACT-V on A-after-D, base drift explains most of the EXACT-R failure;
- if it remains far below EXACT-V, the additive-residual explanation is incomplete.

This is not an architectural qualification gate.

## V3 — does similarity addressing buy alias transfer under absolute ownership?

\[
\boxed{\text{SIM-V}-\text{EXACT-V}}
\]

on known-alias revised accuracy after four changed outcomes.

Require:

- mean gain \(\ge30\) pp;
- paired lower bound \(>0\);
- SIM-V untouched A after D \(\ge85\%\).

This prevents calling alias transfer a success if it destroys ownership.

## V4 — does absolute ownership still matter under similarity addressing?

\[
\boxed{\text{SIM-V}-\text{SIM-R}}
\]

on A-after-D.

This completes the 2×2.

## V5 — does predictive confirmation repair false similarity merges?

\[
\boxed{\text{PRED-V}-\text{SIM-V}}
\]

Primary mechanism outcomes:

- lower different-key false-merge rate;
- no more than 5 pp loss in known-alias merge rate;
- no more than 5 pp loss in A-after-D;
- whole-stream bpb noninferior within 0.10 bits/byte.

If SIM-V already has negligible false merges, PRED-V may be unnecessary; report that rather than forcing a positive claim.

---

# 22. Candidate qualification

A V36 **generalizing ownership candidate** must satisfy all:

### Q1 — acquisition

- A-after-A mean \(\ge90\%\);
- minimum seed A-after-A \(\ge75\%\);
- D-current mean \(\ge90\%\).

### Q2 — strict persistence

\[
\boxed{A_{\rm afterD}\ge90\%}.
\]

### Q3 — exact revision

After four changed outcomes:

\[
\text{exact revised accuracy}\ge90\%.
\]

After eight:

\[
P(\text{obsolete target})\le0.10.
\]

### Q4 — known-alias revision

After four changed outcomes:

\[
\boxed{\text{known-alias revised accuracy}\ge70\%}.
\]

### Q5 — held-out alias transfer

After four changed outcomes:

\[
\boxed{\text{held-out alias revised accuracy}\ge50\%}.
\]

### Q6 — whole-stream utility

Mean whole-stream bpb may be no worse than the base-only counterfactual by more than:

\[
0.10\text{ bits/byte}.
\]

### Q7 — no major false identity collapse

Evaluator-known different-key permanent merges must remain below:

\[
\boxed{5\%}
\]

of target-context bindings.

Report non-target false merges separately.

Passing Q1–Q7 establishes a promising **unbounded generalizing ownership mechanism**, not a bounded lifelong architecture.

---

# 23. Graded verdicts

Use the V35 vocabulary:

- `qualified`
- `supported_below_qualification`
- `favorable_unresolved_precision`
- `no_mean_advantage`
- `adverse`
- `invalid_acquisition`

Add:

- `addressing_invalid_false_merges`
- `alias_task_invalid` only if the new V36 oracle/task check unexpectedly fails
- `fly_separation_gate_failed`

Do not compress a 20–50 pp supported effect into a binary failure.

---

# 24. Fly separation preflight

The additional fly hypothesis is tested first as **representation geometry**, not immediately as a learning claim.

Use the exact same lag96 input.

Construct a fixed KC-like code:

\[
96\rightarrow4096.
\]

For each KC:

- choose exactly 5 distinct lag96 coordinates;
- fixed seeded Gaussian contact weights;
- normalize the five weights once;
- compute tanh drive;
- retain the top 5% by absolute activity:

\[
K=\boxed{205}
\]

active units.

No learning in the code.

The numerical choices are engineering abstractions inspired by the connectome scale, not a claim of biological identity.

## 24.1 Fly-code representation gate

On the four fixed preflight seeds report, separately for 0/1 and 0/2 aliases:

\[
J_{\rm same},
\qquad
J_{\rm different},
\qquad
R_J=\frac{J_{\rm same}}{J_{\rm different}}.
\]

The fly-separation hypothesis is considered instantiated only if both alias pairs satisfy:

\[
\boxed{J_{\rm same}\ge0.30}
\]

\[
\boxed{J_{\rm different}\le0.04}
\]

and

\[
\boxed{R_J\ge10}.
\]

Also report full distributions and code utilization.

If any condition fails:

\[
\boxed{\texttt{fly_separation_gate_failed}}
\]

and **no fly learning trajectories run**.

This is a legitimate scientific outcome.

It means simple widening/sparsifying the present lag code does not reproduce the proposed fly overlap regime on this byte workload.

---

# 25. Conditional fly learning block

Only if the representation gate passes, run two additional arms on the same eight scientific seeds:

## FLY-O

- 4096-dimensional 5%-active code above;
- one global ordinary score matrix;
- ordinary normalized 0.6 delta update;
- no covariance;
- no RLS;
- no local slots;
- no semantic labels.

## HASH-O

Matched control:

- same 4096-dimensional width;
- exactly 205 active features;
- same 4096×256 learned score matrix;
- same 0.6 ordinary update;
- active set deterministically generated from the exact raw 12-byte context hash;
- independent exact contexts therefore have near-random sparse overlap;
- no logical or alias identity enters the hash.

HASH-O asks whether low overlap alone gives retention while destroying alias transfer.

The comparison isolates whether the **structured overlap** of the fly-like code adds useful generalization beyond sparse capacity.

Conditional work:

\[
2\times8=16\text{ trajectories}.
\]

Thus maximum V36 workload is:

\[
48+16=\boxed{64\text{ trajectories}}.
\]

---

# 26. Conditional fly hypothesis

The strong version of the proposed fly explanation predicts:

### F1 — retention

If \(R_J\ge10\), FLY-O should retain substantially more untouched A than V34's lower-separation sparse ordinary lineage.

For the fresh run require at least:

\[
A_{\rm afterD}\ge50\%
\]

before calling it practically interesting.

### F2 — structured overlap versus hash

Compared with HASH-O, FLY-O should show substantially greater:

- known-alias revision;
- held-out alias transfer;

while retaining current D.

### F3 — ordinary writer sufficiency

If FLY-O reaches:

- A-after-D \(\ge75\%\);
- D-current \(\ge90\%\);
- exact revision \(\ge90\%\);
- known-alias revision \(\ge70\%\);

with no covariance/local ownership,

then low-overlap distributed coding becomes a serious alternative mainline.

If the representation gate passes but FLY-O still catastrophically forgets:

\[
\boxed{
\text{the hypothesis that separation alone makes an ordinary writer persistent is falsified on this workload}
}
\]

and the project should retain local ownership as the mainline.

---

# 27. Important interpretation limit on sparsity

A nonzero different-key overlap is not equivalent to permanent ownership.

With a fixed shared score matrix, even a small collision probability accumulates across an indefinitely long sequence.

Therefore a successful finite four-wave FLY-O result would establish:

\[
\boxed{\text{strong interference suppression over this horizon}}
\]

not asymptotic lifelong persistence.

Longer streams and capacity saturation remain required.

---

# 28. Why V36 does not bound the local store yet

V35 LOCAL-V stored roughly 1,800 exact contexts.

That is clearly too large for a lifelong system.

However, bounding exact LOCAL-V before solving alias identity would optimize a memory system that treats two renderings of one logical fact as unrelated items.

V36 therefore sequences the problems:

1. establish ownership;
2. establish autonomous cross-context identity;
3. then bound / evict / consolidate.

If SIM-V or PRED-V succeeds, V37 should become the bounded-capacity round.

If both fail, the next problem is addressing/representation, not eviction.

---

# 29. Noise versus revision remains separate

After one contradictory outcome, no causal learner can know with certainty whether:

- the world has permanently changed;
- or one noisy observation occurred.

V36 therefore does not add a global noise/revision classifier.

Report:

- direct damage to the contradicted slot;
- unrelated-slot damage;
- whether linked aliases inherit the corruption;
- recovery after later correct observations if available.

A linked-memory mechanism may spread a one-off corruption to aliases. That cost must be visible.

If this becomes the dominant failure after V36, a future round should test **slot-local provisional contradiction state**, not a global familiarity gate.

---

# 30. Mandatory invariants

For main arms:

- shared base hashes identical across all deployable main arms for a seed;
- exact lag encoder fixed;
- probes read-only;
- unrelated exact local states unchanged unless they intentionally share the same similarity slot;
- slot membership changes only by the prespecified causal binding rule.

For PRED-V:

- link confirmation uses only the candidate's pre-update prediction and the subsequently observed byte;
- evaluator identity metadata must never enter the rule.

For FROZEN-R:

- explicitly label wave-boundary freezing as noncausal diagnostic intervention.

For conditional fly arms:

- identical stream;
- matched width, active count, output coefficient count and writer between FLY-O and HASH-O.

---

# 31. State and sample accounting

Report for every arm:

- shared-base bytes;
- local value bytes;
- slot/address bytes;
- number of exact member contexts;
- number of slots;
- mean members per slot;
- maximum members per slot;
- auxiliary state growth by wave;
- fixed representation bytes;
- output coefficient bytes;
- operation-count proxy.

Do not call SIM/PRED memory bounded.

Do not count aliases or repeated presentations as new unique facts.

---

# 32. Main interpretation tree

## A. FROZEN-R ≈ EXACT-V

Base drift is causally confirmed as the reason additive exact residual ownership failed.

The full-vector anchored residual is then merely a reparameterization of absolute ownership and should not become another mainline mechanism.

## B. SIM-V preserves but does not transfer

Similarity clustering failed to create useful shared identity.

Inspect threshold/tails/slot purity; do not immediately change the writer.

## C. SIM-V transfers but loses retention through false merges

The missing primitive is selective identity binding.

PRED-V becomes the main result.

## D. PRED-V retains and transfers

The architecture becomes:

\[
\boxed{
\text{shared statistical base}
+
\text{absolute owned slot}
+
\text{similarity/predictive-equivalence addressing}
}
\]

V37 should then bound capacity.

## E. PRED-V preserves known aliases but not held-out aliases

Storage and learned alias binding are functioning, but one-shot unseen generalization remains insufficient.

The next problem is representation-level invariance.

## F. Fly separation gate fails

Do not run a large sparse-code learning experiment.

Interpretation:

> simple random expansion of the current byte representation does not reproduce the fly-like overlap geometry; the fly's sensory preprocessing cannot be skipped.

## G. Fly gate passes and FLY-O succeeds

Distributed low-overlap coding becomes a viable alternative to explicit slot ownership.

A longer-horizon comparison against PRED-V becomes necessary.

## H. Fly gate passes and FLY-O fails

Separation alone is not sufficient; retain item-local ownership as the mainline.

---

# 33. Out of scope

V36 does not test:

- bounded slot counts;
- eviction policies;
- utility-based retention;
- memory compression;
- forgetting-factor RLS;
- MESU;
- covariance variants;
- KC compartment counts;
- APL homeostasis;
- learned representation plasticity;
- replay;
- semantic IDs;
- Transformers;
- natural text scaling.

Those remain downstream.

---

# 34. Execution contract

One self-contained Python file and one command.

Suggested name:

```text
elm_v36_generalizing_ownership.py
```

Suggested invocation:

```bash
python3 elm_v36_generalizing_ownership.py --install-deps
```

Requirements:

- NumPy/SciPy only unless strictly necessary;
- one BLAS thread;
- automatic macOS `caffeinate`;
- process lock;
- deterministic resume;
- atomic per-job writes;
- fixed source/protocol/plan hashes;
- engineering and scientific seed separation;
- compact outputs;
- no per-byte model dumps.

The conditional fly block runs automatically only if the locked representation gate passes.

---

# 35. Required artifact package

- executing `.py`;
- locked V36 protocol;
- `RUN_MANIFEST.json`;
- `PLAN.json`;
- `REPORT.json`;
- `SUMMARY.md`;
- `SUMMARY.csv`;
- per-seed compact records;
- retention-by-wave;
- acquisition curves;
- exact/known/held-out revision curves;
- obsolete-target curves;
- base-drift control results;
- lag similarity distributions;
- slot-link diagnostics;
- false-merge diagnostics;
- state/resource accounting;
- noise diagnostics;
- fly representation gate;
- FLY/HASH results if gate passes;
- observer-effect audit;
- artifact checksums.

---

# 36. Central V36 falsifiable hypotheses

### H1 — anchoring

\[
\boxed{
\text{the exact LOCAL-R deficit is caused predominantly by shared-base drift}
}
\]

tested by FROZEN-R.

### H2 — missing 2×2 cell

\[
\boxed{
\text{similarity-linked absolute ownership combines LOCAL-V persistence with cross-rendering transfer}
}
\]

tested by SIM-V.

### H3 — autonomous identity

\[
\boxed{
\text{predictive confirmation reduces false similarity merges while preserving useful alias binding}
}
\]

tested by PRED-V.

### H4 — fly separation

\[
\boxed{
\text{if a fixed sparse expanded code actually reaches a ≥10× same/different overlap separation,
an ordinary shared delta writer may become substantially more persistent}
}
\]

tested only if the representation gate is met.

V36 is designed to decide these questions without yet mixing in bounded capacity or another family of write rules.
