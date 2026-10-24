# A11 one-hour modular redesign — proposal, not a runnable protocol

Date: 2026-09-24. Status: **DRAFT / NO SCIENCE AUTHORIZATION**.

## Decision that must be made first

There are two different meanings of “split into one-hour experiments”:

1. **One-hour execution packets:** keep the original 576-life, nine-cell A11-A
   estimand, but execute it in resumable jobs lasting at most one hour. This
   reduces failure/restart risk, **not** total scientific cost. No packet by
   itself certifies the A11 competence envelope.
2. **One-hour scientific modules:** replace the omnibus certification with
   smaller, separately locked questions. This can reduce total work, but the
   resulting program must not claim `A11_COMPLETE_*` or the unmeasured
   nine-cell envelope. It requires a new protocol and analysis plan before
   any science result is inspected.

**Recommendation:** use option 2 for scientific scope, with option 1 as its
execution mechanism. Archive A11 v3.8 intact. Do not silently relabel a
partial A11 run as completed A11.

## Why one hour cannot yet be guaranteed

The existing V88 runner seals a receipt after a whole learner life; it has no
qualified mid-life save/restore boundary. A single K8/L16 life has 455,690
event rows. If that life takes more than one hour, scheduling whole lives in
hourly batches cannot enforce a hard one-hour cap. Event rows have unequal
CPU costs (teaching, rest, probe), so dividing rows by historical throughput
is not a valid runtime estimate. Existing A11 technical ledgers contain no
learner runtime measurement.

Before freezing any hour-sized scientific module, perform **qualification-only**
timing on the exact two anchors and representative L1/L4/L16 technical seeds.
Record wall and CPU time, peak aggregate RSS, disk, worker count and per-event
stage cost, including warm-up. This must use a distinct technical namespace;
none of its outcomes enter science. The old FB1 run duration is not an A11
runtime benchmark.

Set a 45-minute target and a 60-minute hard execution ceiling per packet,
leaving time for receipt sealing and checks. Admit a packet only if its
predeclared workload has a defensible upper runtime bound below the ceiling.
If even one smallest restartable unit exceeds the ceiling, **do not launch**
under a one-hour promise. First implement and qualify a true mid-life
checkpoint containing learner state, HASH8 birth key/route map, event cursor,
truth and relevance sets, elapsed learner time, endpoint accumulators, source
hashes and RNG state. Prove uninterrupted and resumed runs agree on final
state/write and scored output digests for both anchors and L>1 boundaries.
An operating-system kill without such a checkpoint is not a valid split.

## Proposed narrow scientific modules

M1–M3 use one predeclared base-seed bank that is fresh relative to all prior
science; the same base-seed identities recur across these cells so their load
and lifetime comparisons remain paired. Both frozen anchors see each cell's
identical exogenous worlds. Each module has a separately frozen
question/analysis family.
The table gives the original n=32 reference cost, **not** a one-hour runtime
claim. Actual n and precision must be fixed before science from the technical
timing and precision plan. If a scientifically useful n cannot fit in an hour,
the module is either exploratory, spans multiple hourly packets, or is
deferred; never disguise n=8 screening as n=32 confirmation.

| Module | Narrow question | Cell added | Rows at n=32, two anchors | Decision scope |
|---|---|---|---:|---|
| M1 | What is current/return performance at moderate load? | K8/L1 | 1,931,840 | One-cell estimate; no scaling claim |
| M2 | Does high load change performance relative to M1? | K16/L1 | 1,931,840 | Paired K8→K16 load contrast only |
| M3 | Does continued fresh learning change performance relative to M1? | K8/L4 | 7,378,304 | Paired L1→L4 lifetime contrast only |
| M4, optional | Does the longer horizon materially change the M3 picture? | K8/L16 | 29,164,160 | Separate, expensive long-horizon study |

M1+M2+M3 are three of the original nine cells: 11,241,984 structural rows at
n=32 and two anchors, **18.4%** of the original A11-A 60,958,272 rows. This
removes 81.6% of planned rows only by relinquishing the other six cells and
their claims. It does not prove an 81.6% CPU reduction. M4 alone would add
29,164,160 rows and should not be included in the first campaign by inertia.

Treat M1–M3 as a predeclared sequence of separate studies, not a post-hoc
search for favorable cells. If later modules are commissioned based on earlier
scientific outcomes, label the later estimates as conditional/exploratory
unless their sequential decision and error-control rules were frozen before
M1. Preserve pairing within each comparison and analyze **complete base-seed
vectors**, not individual lives or event rows as independent observations.

## Separate A11-C

Structured transfer/exception C is a different scientific question. The
original 64-world, two-anchor, three-branch/two-suffix C plan has 640
condition evaluations and its own C0/C1 value, evidence and resource gates.
Do not add it to M1–M3's hour budget. Keep C0 read-only; propose C as a
separate modular protocol only if its sourced numeric locks and independent
review exist. A small C pilot can characterize feasibility but cannot inherit
the original C confirmation/revision labels or be pooled with them without a
predeclared analysis.

## Packet contract and hourly report

An execution packet is a manifest of exact (module, base seed, cell, anchor,
event range) identities, hashes and expected checkpoints. It is never a new
inferential unit. Packets may finish out of order, but hourly reports show:

- committed complete paired base-seed vectors per module;
- in-flight segments separately from completed lives;
- elapsed wall/CPU time, peak RSS, peak/persistent disk and failure reason;
- source/runner hashes and checkpoint/resume identity;
- descriptive partial estimates clearly marked **NOT FINAL INFERENCE**.

No output-dependent world replacement, cell addition, threshold change or
sample increase is permitted. Technical interruptions resume the identical
frozen input. If a one-hour packet fails or overruns, report that fact and
the committed counts; do not count partial state as a completed observation.
Formal interim inference, if desired, requires its own pre-science group-
sequential or anytime-valid plan at completed paired-seed counts, not hourly
clock ticks.

## Resource and acceptance gates

The original A11 §40 caps (four workers, 12 GiB aggregate RSS, 72 CPU hours,
24 wall hours, 64 MiB persistent experiment files, 256 MiB peak incremental
experiment disk and <10 MB final bundle) are **not** automatically inherited
or satisfied merely because packets last one hour. A replacement module
protocol must explicitly allocate both per-packet and cumulative caps and
freeze a streaming/receipt policy. The seven technical CSVs already occupy
293 MiB; all-science per-event CSV persistence is not a viable default.

Before launching any module, require: exact source/anchor/namespace locks;
portable dependency and hash manifest; qualified exact checkpoint/restart if
needed; technical runtime evidence that the proposed packet fits one hour;
module-specific estimand, n, precision and multiplicity plan; resource caps;
independent path-specific audit; and a machine-readable authorization flag.
If any fails, retain `NO_SCIENCE_AUTHORIZATION` and revise the draft. No A11
scientific code or frozen source is changed by this proposal.
