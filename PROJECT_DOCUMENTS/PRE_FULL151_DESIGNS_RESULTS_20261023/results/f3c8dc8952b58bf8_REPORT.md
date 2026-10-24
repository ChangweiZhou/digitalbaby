# V40 results — complementary two-layer memory

**Source SHA256** `10fa234bc0a2e84668c4a33a833b5186d35f8cebe1b53762c158995e1663524d`  
**Stage** `full` · **seeds** 16 · **verdict** **`qualified`**

Label-free: the only teaching signal is the next-byte residual. Workload and slot layer are V39's, imported; the load axis is added here and the stream is byte-identical to V39 at V39's own settings.

`wall` = min(retention, delayed held-out transfer, E1 one-shot transfer), all on the untouched wave-A cohort. A single component is maximised by surrendering another, so the wall is the endpoint.

## The wall by arm and load

| arm | 64 | 128 | 256 |
|---|---:|---:|---:|
| `SUPER` | 0.875 | 0.387 | 0.104 |
| `SLOTS` | 0.070 | 0.094 | 0.066 |
| `HYBRID` | 0.906 | 0.596 | 0.443 |
| `HYBRID-SHUFFLE` *(control)* | 0.016 | 0.004 | 0.008 |
| `HYBRID-FROZEN` | 1.000 | 1.000 | 1.000 |
| `HYBRID-RANDOM` *(control)* | 0.875 | 0.420 | 0.140 |

## Components at each load

| cell | A initial | retention | delayed held-out | E1 | **wall** | store | ops/byte |
|---|---|---|---|---|---|---:|---:|
| `SUPER@64` | +1.000 [+1.000, +1.000] | +0.934 [+0.864, +1.003] | +0.883 [+0.797, +0.969] | +1.000 [+1.000, +1.000] | **+0.875 [+0.790, +0.960]** | 8.8 MB | 523,679 |
| `SLOTS@64` | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.070 [-0.030, +0.171] | **+0.070 [-0.030, +0.171]** | 2.0 MB | 91,818 |
| `HYBRID@64` | +1.000 [+1.000, +1.000] | +0.949 [+0.895, +1.004] | +0.914 [+0.856, +0.972] | +1.000 [+1.000, +1.000] | **+0.906 [+0.848, +0.965]** | 10.4 MB | 617,680 |
| `HYBRID-SHUFFLE@64` | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.016 [-0.007, +0.038] | **+0.016 [-0.007, +0.038]** | 10.8 MB | 614,375 |
| `HYBRID-FROZEN@64` | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | **+1.000 [+1.000, +1.000]** | 10.8 MB | 229,538 |
| `HYBRID-RANDOM@64` | +1.000 [+1.000, +1.000] | +0.938 [+0.875, +1.000] | +0.883 [+0.797, +0.969] | +1.000 [+1.000, +1.000] | **+0.875 [+0.790, +0.960]** | 8.8 MB | 523,679 |
| `SUPER@128` | +1.000 [+1.000, +1.000] | +0.598 [+0.532, +0.663] | +0.387 [+0.344, +0.429] | +1.000 [+1.000, +1.000] | **+0.387 [+0.344, +0.429]** | 8.8 MB | 524,865 |
| `SLOTS@128` | +1.000 [+1.000, +1.000] | +0.992 [+0.981, +1.004] | +0.992 [+0.981, +1.004] | +0.094 [-0.004, +0.192] | **+0.094 [-0.004, +0.192]** | 2.1 MB | 105,679 |
| `HYBRID@128` | +1.000 [+1.000, +1.000] | +0.721 [+0.665, +0.776] | +0.598 [+0.551, +0.645] | +1.000 [+1.000, +1.000] | **+0.596 [+0.549, +0.643]** | 10.8 MB | 635,771 |
| `HYBRID-SHUFFLE@128` | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.004 [-0.004, +0.012] | **+0.004 [-0.004, +0.012]** | 10.8 MB | 627,642 |
| `HYBRID-FROZEN@128` | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | **+1.000 [+1.000, +1.000]** | 10.8 MB | 242,111 |
| `HYBRID-RANDOM@128` | +1.000 [+1.000, +1.000] | +0.629 [+0.567, +0.691] | +0.422 [+0.377, +0.467] | +1.000 [+1.000, +1.000] | **+0.420 [+0.375, +0.465]** | 8.8 MB | 524,865 |
| `SUPER@256` | +1.000 [+1.000, +1.000] | +0.197 [+0.169, +0.225] | +0.107 [+0.076, +0.138] | +1.000 [+1.000, +1.000] | **+0.104 [+0.075, +0.134]** | 8.8 MB | 528,662 |
| `SLOTS@256` | +1.000 [+1.000, +1.000] | +0.987 [+0.978, +0.997] | +0.986 [+0.976, +0.997] | +0.066 [-0.008, +0.141] | **+0.066 [-0.008, +0.141]** | 2.1 MB | 119,359 |
| `HYBRID@256` | +1.000 [+1.000, +1.000] | +0.500 [+0.451, +0.549] | +0.449 [+0.406, +0.492] | +1.000 [+1.000, +1.000] | **+0.443 [+0.400, +0.486]** | 10.8 MB | 648,291 |
| `HYBRID-SHUFFLE@256` | +0.906 [+0.893, +0.919] | +0.859 [+0.828, +0.890] | +0.861 [+0.830, +0.893] | +0.008 [+0.000, +0.015] | **+0.008 [+0.000, +0.015]** | 10.8 MB | 653,192 |
| `HYBRID-FROZEN@256` | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | **+1.000 [+1.000, +1.000]** | 10.8 MB | 254,893 |
| `HYBRID-RANDOM@256` | +1.000 [+1.000, +1.000] | +0.246 [+0.216, +0.276] | +0.148 [+0.101, +0.196] | +1.000 [+1.000, +1.000] | **+0.140 [+0.097, +0.182]** | 8.8 MB | 528,662 |

## Gates

### G0 — acquisition validity (≥ 0.90, every load) · **pass**

| cell | A initial | valid |
|---|---|:-:|
| `HYBRID-FROZEN@128` | +1.000 [+1.000, +1.000] | True |
| `HYBRID-FROZEN@256` | +1.000 [+1.000, +1.000] | True |
| `HYBRID-FROZEN@64` | +1.000 [+1.000, +1.000] | True |
| `HYBRID-RANDOM@128` | +1.000 [+1.000, +1.000] | True |
| `HYBRID-RANDOM@256` | +1.000 [+1.000, +1.000] | True |
| `HYBRID-RANDOM@64` | +1.000 [+1.000, +1.000] | True |
| `HYBRID-SHUFFLE@128` | +1.000 [+1.000, +1.000] | True |
| `HYBRID-SHUFFLE@256` | +0.906 [+0.893, +0.919] | True |
| `HYBRID-SHUFFLE@64` | +1.000 [+1.000, +1.000] | True |
| `HYBRID@128` | +1.000 [+1.000, +1.000] | True |
| `HYBRID@256` | +1.000 [+1.000, +1.000] | True |
| `HYBRID@64` | +1.000 [+1.000, +1.000] | True |
| `SLOTS@128` | +1.000 [+1.000, +1.000] | True |
| `SLOTS@256` | +1.000 [+1.000, +1.000] | True |
| `SLOTS@64` | +1.000 [+1.000, +1.000] | True |
| `SUPER@128` | +1.000 [+1.000, +1.000] | True |
| `SUPER@256` | +1.000 [+1.000, +1.000] | True |
| `SUPER@64` | +1.000 [+1.000, +1.000] | True |

### G1 — the complementary gap exists · **pass**

Both single layers must fail in the predicted complementary way. If either already holds the wall there is no gap and the design is void.

- `SLOTS` E1 at the highest load: +0.066 [-0.008, +0.141] (must be ≤ 0.10) — **True**
- `SUPER` wall decay across the load axis: +0.771 [+0.672, +0.869] (must be ≥ 0.30) — **True**

### G2 — primary: the hybrid beats both single layers · **pass**

At load **256**, on `wall`, margin ≥ 0.10 with the paired lower bound above zero.

| contrast | wall difference |
|---|---|
| `HYBRID` − `SLOTS` | +0.3770 [+0.2766, +0.4773] |
| `HYBRID` − `SUPER` | +0.3389 [+0.3007, +0.3770] |

| cell | retention | delayed | E1 (wave A) | E1 (all waves) | wall |
|---|---|---|---|---|---|
| `HYBRID@256` | +0.500 [+0.451, +0.549] | +0.449 [+0.406, +0.492] | +1.000 [+1.000, +1.000] | +0.984 [+0.978, +0.990] | +0.443 [+0.400, +0.486] |
| `SUPER@256` | +0.197 [+0.169, +0.225] | +0.107 [+0.076, +0.138] | +1.000 [+1.000, +1.000] | +0.984 [+0.978, +0.990] | +0.104 [+0.075, +0.134] |
| `SLOTS@256` | +0.987 [+0.978, +0.997] | +0.986 [+0.976, +0.997] | +0.066 [-0.008, +0.141] | +0.023 [+0.000, +0.045] | +0.066 [-0.008, +0.141] |

### G3 — specificity · **pass**

A shuffled layer or a count-matched random arbitration that matches the hybrid means this is ensembling, not complementary memory.

| contrast | wall difference |
|---|---|
| `HYBRID` − `HYBRID-RANDOM` | +0.3037 [+0.2669, +0.3406] |
| `HYBRID` − `HYBRID-SHUFFLE` | +0.4355 [+0.3912, +0.4799] |

### Prediction non-inferiority · **pass**

`HYBRID` − `SLOTS` whole bpb: -0.4749 [-0.5131, -0.4367] (must be ≤ +0.10; lower is better).

## Reference points

| quantity | value |
|---|---:|
| V39 superposed wall @32 / @256 | 0.936 / 0.092 |
| V39 superposed E1 / slot E1 | 0.998 / 0.027 |
| V38 E1 held-out (all policies) | 0.008 |
| V39 TRACE-CORR / LEGACY-U peak store | 2.00 MB / 22.56 MB |

## Interpretation limits

- G0 and G1 are hard gates in that order. G1 exists because a hybrid is only
  worth testing if both single layers fail in the predicted complementary way;
  if either already holds the wall, the premise dissolves.
- `HYBRID-SHUFFLE` and `HYBRID-RANDOM` are nondeployable controls and can never
  qualify. `HYBRID-RANDOM` is an offline count-matched read policy over the
  hybrid's own state, as V37's RAND-MATCH was; it never learns or writes.
- The clutter cohort is fixed across loads, so it is proportionally milder at
  high load. Load and clutter are not varied together on purpose.
- `SUPER` carries no shared base learner, matching V39, so its whole-stream bpb
  is not comparable to the arms that do. The wall is the endpoint for that reason.
- The read composition below memory authority is `written`, fixed by a pre-run
  calibration (written): the additive rule predicts
  1.168 bpb against
  1.356 but reaches only
  0.188 one-shot transfer against 1.000, so it
  cannot express the mechanism. The table is in MANIFEST.json.
- The superposed layer uses the stationary lag front end. V39 measured the
  phase-conditioned alternative at 0.62 same-context self-overlap across a run,
  which is why the higher-separation code is not used here.
- E1/E2 fire for every key at its first exposure, so the stream-wide figure
  spans all four cohorts while the wall's other components are the untouched
  wave-A cohort. The wall uses the wave-A restriction so all three components
  share a cohort; both figures are reported.
- Revision endpoints are recency-protected; the wall deliberately uses the
  wave-D delay for both of its retention components.
- Synthetic stream with controlled renderings. No scaling or language claim.
