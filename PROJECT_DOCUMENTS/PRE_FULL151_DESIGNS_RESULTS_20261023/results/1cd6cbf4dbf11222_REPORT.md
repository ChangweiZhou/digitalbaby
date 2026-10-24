# Byte encoder → FULL_151 bridge: completed E0 report

**Result:** the V9.2 byte-context predictor now causally drives FULL_151's native 88-PN → 5,177-KC value path in this isolated fork. The preset E0 gates passed. This is an interface result, **not** evidence of better memory, unseen-task transfer, or autonomous output.

## What was connected

The prior V9.2 checkpoint already contained a learned 1–4-byte predictor and the inherited FULL_151 memory in one object, but the predictor was a parallel head. In this fork, its frozen next-byte distribution is projected through the fixed 256-byte × 88-receptor table, mixed at gain 0.5 with FE0's receptor state, then passed to the existing FULL_151 PN→KC encoder. The same bridged FE read supplies the native `pending_x` captured for `teach()` and the native association readout. This is a deliberate sensory-structure change; no FULL_151 connection, plasticity, or reward rule was changed. The parent source files and text checkpoint were not edited.

The bridge adds no trainable parameters beyond the already present V9.2 predictor (720,896 prediction synapses; 2,884,608 mutable predictor bytes in its parent resource record). It adds one 256×88 projection and a gain/mode normalization per native PN read. There is no measured large-run runtime or learning-benefit claim.

## Prespecified assays and results

| Assay | Primary result | Causal controls | Gate |
|---|---:|---|---|
| Equal-FE0-state clamp, engineering pilot | 16/16 new swapped-order pairs produced different native KC codes | FE0-only and context-predictor lesion: 0/16 | Pass |
| Equal-FE0-state clamp, confirmation | **128/128** new pairs produced different native KC codes; median Hamming distance 48 of 5,177 KCs (range 2–112) | FE0-only and predictor lesion: **0/128** each | Pass (threshold ≥90%) |
| Raw-byte stream, engineering pilot | 16/16 context presentations changed native pre-outcome `pending_x` when predictor input was lesioned | FE0 p, fast/slow/adapt state and clocks matched exactly | Pass |
| Raw-byte stream, confirmation | **128/128** new context presentations changed `pending_x`; median Hamming distance 84 of 5,177 KCs (range 34–198) | FE0 p, fast/slow/adapt state and clocks matched exactly in all 128 | Pass (threshold ≥90%) |

The clamp keeps FE0 and the fly's prior state identical while swapping the four-byte history. The intact predictor supplies different PN and KC codes; replacing its context-dependent contribution with bias-only removes the difference. Confirmation predictor total-variation distance between paired contexts had median 0.365, and the resulting PN distance median was 0.702. This is evidence that information present in the learned predictor reaches the inherited KC population. It does not establish that this particular projection preserves semantic relations or helps a task.

The stream check is separate and uses disjoint contexts. Each context was fed byte by byte with predictor plasticity disabled, then a common `!` outcome byte was presented without teaching. FULL_151 captured the KC code immediately before `!` entered the receptor state. Intact and lesioned arms had identical raw bytes, timing, FE0 state and native memory state, so their `pending_x` difference is attributable to the predictor's feedforward contribution. The 64 confirmation pairs yield 128 context presentations; these are correlated within pairs and no independent-trial confidence interval is claimed.

Six engineering tests pass, including bitwise FE0 parity at gain 0, frozen predictor weights, causal pre-byte timing, clone/read purity, checkpoint restore, and native write/read use. A one-reward **technical sentinel** showed that the bridge changes `pending_x`, subsequent FULL_151 fast/slow state updates, and association readout relative to FE0 alone. That sentinel is not a scored E1 learning experiment.

## Audit and caveat

The receipt-only audits passed: source and checkpoint hashes match; pair counts, swaps, disjoint phases, controls and gates recompute exactly. The parent checkpoint SHA256 is `ac79d7f4dcc28a131049b93d6b8390a2423bbe4208e43df1cb2d4052857b11c9`. All experiment arms used its same frozen predictor parameters. `caffeinate` was attached to both one-shot phase runners and exited when each runner finished.

One secondary field in the completed stream receipts is invalid: `predictor_total_variation` was calculated with the intact distribution in both arms and is consequently zero. The actual lesion is applied inside `PredictionBridgeFE0.read()`, so the native KC primary endpoint and state controls are unaffected. The field is excluded; see [STREAM_DIAGNOSTIC_ERRATUM.md](STREAM_DIAGNOSTIC_ERRATUM.md). No completed phase was rerun to repair it.

The fixed permutation control also distinguished all clamp pairs. That matters: *discriminability alone* does not establish that the chosen byte-to-PN mapping is the best alignment. This experiment closes the narrow causal access question. A separate, prospectively designed behavioral comparison is needed to learn whether the bridge improves FULL_151 acquisition, retention, or use on previously unreinforced inputs. It should keep exposure-independent shortcuts and exact-key recitation as explicit countermodels and report E1/E2/E3 at the actual level earned.

## Files

- Architecture and implementation: [DESIGN.md](DESIGN.md), [STREAM_ADDENDUM.md](STREAM_ADDENDUM.md), [bridge.py](bridge.py)
- Test and runners: [test_bridge.py](test_bridge.py), [run_experiment.py](run_experiment.py), [stream_experiment.py](stream_experiment.py)
- Receipts: [pilot.json](results/pilot.json), [confirm.json](results/confirm.json), [stream_pilot.json](results/stream_pilot.json), [stream_confirm.json](results/stream_confirm.json)
- Independent audit: [AUDIT.json](results/AUDIT.json), [AUDIT_STREAM.json](results/AUDIT_STREAM.json)
