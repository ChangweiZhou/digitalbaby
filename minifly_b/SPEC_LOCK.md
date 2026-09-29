# Package B — SPEC_LOCK (T1, T3, S1, S2, S3, S4)

**Status:** written before any science world was run. Every constant below was fixed from source
constants, numerical/graph feasibility on the technical world `190000`, or literature form, and **never**
from a behavioural score. The code that implements it is `src/tfam_graph.py` (T) and `src/topo_model.py`
(S), run by `src/runner.py`, audited by `src/audit_topo.py`, analysed by `src/analyze_b.py`, all locked in
`SOURCE_LOCK.json` together with `GRAPH_MANIFEST.json`, `RESOURCE_BUDGET.json` and `ARM_ROSTER.json`.

## 0. Common platform and birth (all arms)

* Task, branches, probes and timing: the shared, unmodified
  `REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928/{fixture.py,common_platform.py}` —
  `run_fourstore_life(world, actor)`; FE0→native Full151, four output stores, five causal branches
  `W, N_old_fact, N_old_rel, N_new_fact, N_new_rel`, output before teacher, read-only probe clones.
  The historical `run_science.py` is never executed.
* Birth: every store of every arm is created by `portable_birth.canonical_fresh_native()`
  (V2 addendum). The runner records per store the raw host B digest
  (`872094396ed4…` on this host), the canonical digest `32a3726c…`, and the installed graph digest, and
  refuses to start unless all four stores agree and carry the canonical birth. T graphs and S supports
  are derived only from this canonical B.
* Store identity: FourStore store `j∈{0,1,2,3}`; relation records write only stores 0–1 (platform rule).
* Receipts: gzip JSON, one per world-arm, written write-once (`os.link` on the final name; never
  overwritten). Science receipts keep full S partner-change lists for branch `W` and counts plus graph
  digest after every structural event for the other branches; technical receipts keep all branches in full.

## T. Birth-time static graphs (T1, T3) and matched controls (T0_1, T0_3)

Machinery reused unmodified from the frozen T2 generator (`t2_graph.py`): exact PN row degrees
91/92 (90 rows of 92 by SHA256 rank), effective input-type degrees `D_a`, canonical per-KC degree `c_k`,
deterministic Havel–Hakimi realisation on the 88 byte-driven types, common mixing by `8E` valid
degree-preserving type-level switches to `G*`, PN-row assignment and per-KC native weight-multiset
assignment. Streams are SHA256 counter streams keyed `T2-v2|<F>-v1:<world>|<domain>` with
`F∈{T1,T3}` so the three T families never share a stream. No fixture, label, byte history or score is read.

**Label-free module assignment (both families).** Loads are fixed source-side degrees only.
Types (load `D_a`) are placed in `n` parts by LPT: descending load, tie SHA256 rank, each into the currently
lightest part, cap `ceil(88/n)` per part. KCs (load `c_k`) are then placed by descending load into the
open part with largest remaining deficit relative to that part's type load (cap `ceil(5177/n)`).
Result on every world: type-load and KC-load per part are exactly equal (imbalance bound 0).

**T1 — hierarchical modular.** `n=8` leaf modules; top module = `leaf // 2` (4 top modules of 2 leaves).
Edge cost `h(a,k)=0` same leaf, `1` same top module but different leaf, `2` different top module.
From `G*`, exactly `16E=441,152` proposals; accept a valid switch iff
`Σh` strictly decreases **and** the cross-level floors hold: `#h=2 ≥ 0.10E (2,757)` and
`#h≥1 ≥ 0.25E (6,892)` ("bounded cross-level edges"). Zero accepted switches ⇒ `T1_GRAPH_NOT_INSTANTIATED`.

**T3 — clustered with exact sparse bridges.** `n=8` flat clusters (different stream from T1).
Bridge = edge whose type cluster ≠ KC cluster. From `G*`, proposals until the bridge count equals
exactly `T3_BRIDGES = int(0.20E) = 5,514`, at most `80E` proposals; accept a valid switch iff bridges do not
increase and never fall below 5,514 (neutral moves let the search cross plateaus).
Not reaching the exact count ⇒ `T3_GRAPH_NOT_INSTANTIATED`.
*Origin of 0.20:* graph-only feasibility on technical world 190000 — strict descent stalled at 4,806
bridges under the one-PN-per-type-per-KC and degree constraints, and 4,279 after balanced partitions;
a 5% target (the first draft) was infeasible. 20% (random-graph expectation ≈ 87.5%) is the smallest
round budget with margin; its reachability on all 64 science worlds is verified in `GRAPH_MANIFEST.json`
before science.

**Matched controls.** `T0_x` = the same world's `G*` plus exactly `A_w` (the candidate's accepted switch
count) uniformly random valid switches from the `null` stream, then the identical PN/weight assignment.
T and its control match: PN/KC counts, every PN row degree, every `D_a`, every KC degree `c_k`, edge count,
each KC's exact weight multiset (hence global histogram and each KC's weight sum), top-k readout sparsity,
all non-B birth state and the label-blind generation budget (same mixing, same accepted-switch count).

**Lifetime.** Installed read-only in all four stores before the first byte; clones share it; the graph
digest is part of every state digest. Audit: graph digest = pre-science manifest; every branch and store
ends with the birth graph; zero structural events.

## S. Genuine lifetime rewiring (S1–S4), static control S0, yoked random controls Srand_1–4

**S.1 Representation.** Per KC `k`, a birth-fixed *support* = its canonical partners ∪ a pool of
`POOL=16` further PNs drawn once per world (SHA256 stream `SFAM-v1:<world>|pool`, identical for every S
arm and store; zero-degree KCs get no pool). A slot is *on* (a live edge carrying one canonical weight)
or *off* (candidate). A partner change moves an edge's weight from an on slot to an off slot **of the same
KC**: edge count, every KC degree and every KC weight multiset are invariant; PN row degrees may drift and
are reported. Only CSR `indices`/row membership change; there is no weight learning (that is family P).
The graph is rebuilt as a new read-only CSR (copy-on-write).

**S.2 Event clock and budgets (locked constants).**
* Evidence is updated only inside `teach(..., write=True)` of that store — the native write events
  the branch permits. Probes, byte feeds, value reads and clones never touch structure. A no-write
  branch therefore also withholds structural learning from the withheld records (causal-branch rule).
* Structural event `n` of store `j` fires after its `24·n`-th write-enabled teach (`K_EVENT=24`), after the
  native write for that record. Events per life (W branch): 25 for stores 0–1, 16 for stores 2–3.
* At most `R_MAX=64` partner replacements per event per store (S4: ≤64 new tentatives plus resolution of
  earlier ones). Lifetime turnover ≤ 25·64 = 1,600 moves per store (5.8% of edges).
* Evidence decays with model time, `τ_S = 86,400 s` (the fixture's day scale), using `t` of the write.
  Inputs available at a write: the pre-outcome KC code `x = pending_x` (0/1) and the pre-outcome FE0
  type activity `p` that produced it (captured in `byte` with the parent's own FE0 advance). PN activity
  `a_p = p[pn_type_index[p]]`. The reinforcement bit `r`, answer, labels, fixture and scores are never read
  by any rule (unit test: r=0 vs r=1 from one state give identical S state).
* State: `C` co-activity per support slot, `U` KC usage, `A` PN usage, decayed event count `ν`:
  `C←C·d; C[s]+=a_{pn(s)}` for all slots of active KCs; `U←U·d+x`; `A←A·d+a`; `ν←ν·d+1`, `d=exp(-Δt/τ_S)`.
  All S arms (including S0) carry and update the same state, so the state budget is identical.
* Ties: birth-fixed per-slot / per-KC SHA256 ranks (`SFAM-v1|<world>|tie`). Per-event random draws:
  counter stream `SFAM-v1:<world>:<arm>:<store>|event<n>` — branch-independent, so branch divergence can
  arise only from the rule acting on different histories.

**S1 — SET-like.** Among live edges on KCs with ≥1 free candidate, prune the `R_MAX` smallest weights
(tie rank); regrow each at the same KC to a uniformly random free candidate; the weight moves with the
edge. (With no weight learning, the weakest weights stay weakest: S1 is random re-partnering of the
weakest-weight edges; SET's magnitude criterion without training.)

**S2 — co-activity guided.** For each eligible KC: weakest partner `w=argmin_on C` (tie rank), best
candidate `b=argmax_off C` (tie lower PN). Gain `g=C_b−C_w`; take up to `R_MAX` KCs with `g>0`, largest
gain first (tie KC rank); move `w→b`. Receipt stores `C_w, C_b` for every change (pre-change evidence,
accumulated from pre-feedback codes only).

**S3 — homeostatic.** KC rate `ρ_k=U_k/ν`, target `ρ*=0.05` (Full151 `active_fraction`, a source constant),
deviation `δ_k=ln((ρ_k+10⁻³)/ρ*)`. Eligible KCs with `|δ_k|>ln 2`, largest `|δ|` first, up to `R_MAX`.
Under-used (`δ<0`): replace the partner with least PN usage `A` by the candidate with most `A`, only if
larger. Over-used: replace the most-used partner by the least-used candidate, only if smaller.

**S4 — sample then stabilise.** At event `n`: first resolve every tentative started at `n−2`
(`S4_TRIAL=2` events): keep (“accept”) iff the new partner's windowed co-activity `C_new` since the trial
began is strictly greater than the displaced partner's counterfactual `C_old` (both accumulated at the
KC's active writes); otherwise roll back exactly. Then choose up to `R_MAX` eligible KCs without an open
trial by partial Fisher–Yates on the event stream; in each, replace the weakest-`C` partner by a uniformly
random candidate as a live tentative edge. At most one open trial per KC; tentative edges are real edges
(they count in the fixed edge budget) and their state (≤128 trials × 6 fields) is in the checkpoint/digest.

**S0 — static primary control (shared by S1–S4).** Canonical birth graph fixed for life, with exactly the
S state (`C,U,A,ν`, clock, event log) carried and updated identically but never used. Because that state
is write-only, S0_1…S0_4 would be the same trajectory; one S0 arm serves all four S contrasts (this sharing
is declared before science and does not change any contrast).

**Srand_x — yoked random diagnostic (non-autonomous).** Same support, clock and budgets; at each
(branch, store, event) it performs exactly the candidate's realised number of partner changes (for S4:
tentatives + rollbacks) on uniformly random eligible live edges to uniformly random candidates. It reads only
that count from the committed same-world candidate receipt — never partner identities or evidence — and is
reported as a diagnostic outside the 26-contrast family.

**Clone/restart.** `clone_round` shares immutable graph/state arrays (all flagged read-only) and copies
mutable native state; any write installs new arrays. `snapshot/restore` carries B bytes, slots, evidence,
tentatives and pending PN activity, verifies the B digest and that the slots rebuild it.

## Endpoints, roster and statistics (unchanged shared contract)

Science worlds exactly `190001–190064`; technical world `190000`; no calibration worlds are used by
Package B. Per world, final checkpoint:
`E1 = acc(W, old_fact canonical) − acc(N_old_fact, …)`, `E3 = acc(W, old_relation_heldout) − acc(N_old_rel, …)`.
Primary Package-B contrasts (12 of the shared 26): `T1−T0_1`, `T3−T0_3`, `S1−S0`, `S2−S0`, `S3−S0`,
`S4−S0`, each on ΔE1 and ΔE3; two-sided simultaneous 95% Student-t with Bonferroni `m=26`
(`t_{63, 1−0.05/52}`); Hoeffding bound on `[−2,2]` if a paired variance is exactly zero. A claim of
improvement requires a lower bound > 0. Reported alongside: absolute W/N, E2 forms, new-cohort learning,
taught guards, `Sx−Srand_x` diagnostics, graph metrics and resources. No tuning, replacement, omission or
added world after any score is seen.

## Failure handling

A technical construction failure is `*_NOT_INSTANTIATED` for that variant (not a scientific null). During
science a scientific integrity failure (runner assertion, audit failure) stops the run; the actual error and
the committed count are reported; committed receipts are never rerun or replaced. An interrupted process may
resume only uncommitted world-arms under the same lock.
