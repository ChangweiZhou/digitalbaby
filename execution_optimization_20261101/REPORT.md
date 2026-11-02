# V176 execution optimization — engineering results

**Selected:** exact CPU float64 native backend, fused compartment updates and common-prefix evaluation. **No scientific feature fit, native B life, arithmetic evaluation or core adoption occurred.** This is E0 execution evidence. Original model equations, feedback, warmup, event times, fixed12-life roster, training dose and endpoints remain intact.

## Result

| H-arm operation,35-byte engineering prompt | Literal | Selected | Speed ratio |
|---|---:|---:|---:|
| One complete training record |0.2070 CPU s|0.0549 CPU s|3.77×|
| One complete own/teacher-forced evaluation pair |0.3350 CPU s|0.0288 CPU s|11.63×|

At71-byte prompts, training fell0.3756→0.0797 CPU seconds and evaluation0.6979→0.0524. These are bounded engineering measurements, not a sustained full-run benchmark.

The unchanged full-roster forecast falls **9,036→2,165 CPU seconds**, approximately **36 minutes including1.5× margin**, all prior technical costs and a300-second supplemental engineering allowance. The conservative variant without shared-prefix evaluation forecasts42 minutes. Both are below7,200 CPU seconds. Short timing samples, below35-byte extrapolation, actual trained-H behavior, I/O and hardware contention introduce uncertainty; this is not a guaranteed elapsed duration. RSS peaked below0.5GiB in native qualification. See `RESOURCE_REFORECAST.json`.

## Changes and why they preserve the experiment

1. **Skip irrelevant work during no-write active events.** Original `advance73` computes firing/readout/feedback and zero write arrays even when `plastic=False`. The specialized compiled kernel performs the identical scalar float64 adaptation and fast/slow decay expressions, without fastmath, and retains event/presentation/time counters. Plastic updates still use the original implementation. Full151's nonzero allocation genes are retained, not replaced by neutral genes.
2. **Fuse sixteen passive compartments.** Mutable arrays use a contiguous buffer with nonoverlapping compartment views. The same scalar update runs in the same order; compartments retain independent values and adaptation. No pooled learned store or altered update is introduced.
3. **Copy mutable state only.** Clones copy fast/slow/adaptation, counters, front end, thresholds and other mutable bookkeeping; fixed connectivity and frozen H are shared. The old signed-teacher counterfactual formula and its two hypothetical branches remain. Cheap clones reduce their overhead without deriving a different signed update.
4. **Exact TopK membership without a full sort.** A partition identifies the cutoff. Strictly higher scores enter first; exact ties use ascending original indices, retaining positive-drive eligibility and per-side counts. Native projection and floating-point precision are unchanged.
5. **Bounded, label-free encoding cache.** At most64 entries, keyed by exact receptor-vector and threshold bytes, with frozen H and bridge. It stores derived computation, not learned cue/answer associations. Actual generated prefixes must match their own exact keys; teacher prefixes cannot substitute. Cache arrays are immutable and clone ownership is qualified.
6. **Share the evaluation prompt once.** Own and teacher-forced branches start at the identical post-prompt native state, then fork independently. Own generation finishes before expected target bytes are consumed by the teacher-forced branch. Physical training histories, target counters, native times and reported endpoints remain identical; redundant evaluator execution is removed.
7. **Restore packed views explicitly.** Deserialization rebinds compartment views to the packed buffers and reapplies immutable H/bridge/cache flags. Self-contained checkpoints avoid the older process-global asset-ID registry. Continuing after restore was tested, not merely comparing the saved snapshot.

## Qualification evidence

The initial optimized version passed8,668 bitwise comparisons. The final fused version was exercised on the same engineering cases; saved RAW checks were retained when continuing to the unstarted H tests. Final continuation added4,834 comparisons with **maximum absolute state difference0**. Failed attempts also retain their completed comparisons; they are not independent efficacy evidence or counted as completed scientific runs.

Coverage includes all256 input bytes across two128-byte records, empty/one-byte/35/71/128-byte prompts, repeated cues, enabled/disabled writing, the1023→1024 warmup boundary fixture, generated off-path bytes,128-step rest, output/EOS/CAP/teacher-forced scores, checkpoint restore and further training. State comparisons include every compartment's fast/slow/adaptation, clocks, event/presentation counters, teacher counts, wrapper counters/front end and diagnostic stats. Independent TopK tests cover zero drive, all ties and randomized integer/continuous scores.

`OWNERSHIP_QUALIFICATION.json` verifies that mutation of a clone cannot affect its parent, compartment views do not overlap, immutable assets are shared and restored views point to the correct buffers. `SHARED_PANEL_QUALIFICATION.json` verifies exact complete-output and teacher-forced endpoint equality for both RAW/H on three prompt lengths, without changing the source learner state. Finite fixtures are not proof for every possible state; the specialization rejects unsupported rep/feedback configurations.

## GPU and CPU parallelism

The local M3 Pro GPU works outside the sandbox. MPS rejects native float64 tensors. Changing native state to float32 would be a separately qualified numerical change, so no GPU native backend was selected.

Synthetic resident-operand H projection tests found MPS about4.2× slower at batch1,3.3× slower at batch16, and1.71× faster at batch256. Small-batch float32 outputs differed by up to~2e-6. These exclude transfer cost and are not a full training benchmark. Source/probe work is a small share of the plan, so a GPU port is deferred. No attention, Transformer, arithmetic feature or different learning rule was added.

Independent-life CPU parallelism could reduce wall time, but would change the frozen single-worker execution policy and would not itself reduce total CPU work. It is unnecessary to clear the current budget after the exact serial optimizations and was not used.

## Failures, accounting and limits

Engineering failures are preserved under `history/attempt001–003`: an initially overrestrictive neutral-gene guard, packed-state restoration, and a continuation-test mismatch caused by comparing reset versus accumulated diagnostic stats. No failed scientific trajectory was restarted. The numerical continuation test now resets stats on both cloned branches. Failure states were saved in ignored `local/` storage.

All supplemental work is engineering, outside the earlier stopped scientific session, and inherited in the forecast. Counts reconcile to at most133 engineering teaching records/266 target applications and161 own/TF pairs. A manually counted engineering query ceiling128 was exceeded because a failed stage had exercised both arms; this is disclosed rather than discarded. The prospective supplemental ceiling is192. The scientific roster/dose is unchanged; `EXECUTION_AMENDMENT.json` records the additional engineering allowance and future total query accounting. The engineering CPU reserve300 seconds covers~94 measured stage CPU seconds plus imports/setup/unmeasured checks.

The first profile appended three engineering records to the old ignored telemetry path before output isolation was corrected. They are counted here and are not science. Original frozen source files/design and the unrelated shared-workspace changes were not edited. Raw checkpoints, caches and response logs are excluded from compact delivery.

## Ready state

`optimized_native.py:Model` and `pair_shared` are the selected qualified native execution components. **The complete source/probe/native main driver still needs prospective integration qualification and a complete source lock. Its main-fitting CLI remains blocked.** This package removes the measured native execution bottleneck; it does not claim the previously incomplete scientific driver is complete or silently launch the experiment.

The next integration must keep end-to-end resource counts persistent across processes, preserve failure snapshots, use the fixed source/bridge admission gates and avoid changing original scientific outcomes. Arithmetic and learning-mechanism efficacy remain untested by this optimization work.
