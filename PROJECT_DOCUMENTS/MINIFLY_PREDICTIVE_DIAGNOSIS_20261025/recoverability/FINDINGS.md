# Recoverability: bounded signed reanalysis

25 October 2026. This reanalysis reads immutable compact artifacts from commit `9803c3d4b8e5428abc54e263c32daa2cb12d8846`. It imports no learner and performs no native inference, fitting, teaching or trajectory replay. Historical files and the arithmetic pointer are unchanged. [RESULTS.json](RESULTS.json) contains every selected phase, world-level signed contrast and input SHA/blob identifier; [analyze.py](analyze.py) reproduces the reduction using `git show`.

The historical experiment supports **E1 taught-association acquisition and interference**, not E3 unseen-addition generalization. Evaluation values were taught at their scored keys. A key lookup countermodel solves them. The arithmetic goal and gradual warmup remain unchanged.

## What the paired continuation identifies

Four assessment worlds each contain two correlated value lives. Each cohort/life has 18 facts, including two empty values: complete nonempty and forced-length payload-exact both use 16 facts, whereas bit accuracy uses 576 actual payload bits. We average head seeds within a life, lives within a world, and four worlds equally. Head seeds reuse the same native responses and do not create additional histories.

`S_NEW` starts from the same W state after OLD teaching and the first one-hour rest. It receives the same NEW inputs and simulated time, with NEW writes disabled, then the same final rest. The signed comparison is **W final minus S_NEW final**. This is a matched total effect of allowing the NEW writes under this protocol; it includes their downstream nonlinear state consequences. It is not a single-event response derivative.

| Direct native OLD readout | OLD end | Final with NEW writes | Final with NEW writes off | Write effect, percentage points |
|---|---:|---:|---:|---:|
| RAW complete own-byte/EOS | 0.78% | 0.00% | 0.78% | −0.78 |
| RAW payload exact, evaluator length | 19.53% | 0.78% | 20.31% | −19.53 |
| RAW payload bits | 90.76% | 70.12% | 91.08% | −20.96 |
| STD complete own-byte/EOS | 0.00% | 0.00% | 0.00% | 0.00 |
| STD payload exact, evaluator length | 54.69% | 0.00% | 53.91% | −53.91 |
| STD payload bits | 97.92% | 75.89% | 97.81% | −21.92 |

Thus zero complete accuracy does not identify a total failure to store payloads. NEW writes cause substantial net damage to OLD payload readout; pure time and passive NEW exposure do not explain the observed size in this protocol. Conversely, forced-length payload recovery is an evaluator diagnostic, not a valid generated-answer result. This does not establish irreversible erasure from every native state variable.

There is **no S_OLD branch** in these artifacts. N disables both OLD and NEW writes; SH changes OLD target assignments. Neither substitutes for a counterfactual that removes only the relevant prior episode while matching later teaching. The historical write contrast remains useful but cannot prove E3 reuse of prior learning.

## The same responses can support better stopping and worse payloads

The frozen MEAN/SEP joint heads and DIRECT decoder operate on the same saved native response record. The following STD MEAN minus DIRECT differences are net changes, averaged by the same hierarchy:

| Checkpoint/cohort | Complete byte/EOS change | Payload-exact change | Payload-bit change |
|---|---:|---:|---:|
| OLD immediately after teaching | +2.60 pp | −40.10 pp | −10.07 pp |
| OLD final, NEW writes off | +2.08 pp | −44.01 pp | −15.57 pp |
| NEW final | +3.91 pp | −41.15 pp | −6.67 pp |

A single headline complete score can therefore hide opposite readout effects. These are signed aggregate differences, **not counts of individual bits damaged versus rescued**. We did not open query-level tensors or reconstruct logits. All 448 corresponding payload/bit rows agree with the separate `PAYLOAD_EOS_DIAGNOSIS` reduction, and all 16 phase summaries agree with `DIAGNOSTIC_PHASE_ROWS`.

The learned joint head used offline backpropagation historically; this reanalysis does not recommend it as a local-only MiniFly mechanism or introduce it into the arithmetic track.

## What early information is available

Compact W evaluation rows preserve OLD after passes 1, 4 and 14, OLD after NEW passes 1, 4 and 14, both rests, final OLD/NEW and S_NEW. For example, STD DIRECT OLD payload exact changes 46.88% → 50.78% → 54.69% during OLD teaching; after NEW passes 1/4/14 it is 56.25% → 22.66% → 0%. RAW's first NEW pass also slightly improves OLD payload before later damage. A useful theory must allow beneficial cross-effects as well as harmful interference.

No per-query native margins, teaching residuals, event-level pre/post states, or address-overlap values occur in the compact inputs read here. Historically tracked raw tensors/checkpoints were intentionally not restored. Training-episode early probes were omitted historically and remain missing. The proposed signed teaching-influence operator and its additivity error cannot be reconstructed from these metric rows.

We do **not fit a predictor** to this material. There are only four grouped worlds, all with identical delay and write-dose schedules, so time/write-count predictors have no cross-world dose variation. Direct final OLD complete accuracy is constant zero in all four RAW and all four STD worlds; STD direct final OLD payload exact is also constant zero. A prediction exercise on those outcomes would reward the floor, not validate an explanatory operator. Bit accuracy varies and can be described, but model selection on four already-inspected worlds would not provide a defensible prospective test.

The evidence supports keeping write interference and readout corruption as distinct hypotheses. It does not select a new architecture, establish a universal retention law, prove the original operator predicts future failures, or change the unseen-addition goal. Any independent predictive test must freeze its quantities before future authorized outcomes are observed.

## Accounting and reproduction

This reduction: **0 new training records, 0 native events, 0 head fits, 0 model inference calls, 0 trajectory replays**. No V147 held scores were opened. Reduction process and git-read child CPU are recorded in `RESULTS.json`; source inspection/preparation time is not fully metered. Run:

```sh
python3 PROJECT_DOCUMENTS/MINIFLY_PREDICTIVE_DIAGNOSIS_20261025/recoverability/analyze.py
```

Only the explicit compact JSON/reports and reduction/source definitions in the script are read. The script verifies Git blob object IDs, content SHA-256, row identity uniqueness, fixed denominator/roster counts, cross-summary agreement and paired aggregation. It does not independently regenerate the historical emissions from raw tensors; the original report's artifact-reconstruction audit remains separate evidence.
