# MiniFly connectome navigation engineering cycles

**Three bounded engineering cycles completed. Functional navigation admission is held at the anatomy and interface gate.** The extraction, numerical rollout, evaluator and local write plumbing ran. The functional heading and displacement tests require compartment mapping and sensory and steering adapters.

## Cycle 1 Anatomical extraction and boundary audit

The fixed-gain engineering graph was extracted from the archived FlyWire v630 node and edge tables. The roster uses named central-complex families, the Iteration 5 PN, KC, MBON and DAN IDs, and matching APL labels. It retains **6,704 neurons**, including 5,177 KCs and 1,213 selected central-complex neurons, with **427,258** internal nonzero edge records. The conservative signed current mask admits **417,204**. It uses agreeing acetylcholine and GABA source labels, documents the inherited KC cholinergic override, and separates DAN records from additive currents. The extract uses 106 occupied ordered role-pair gain groups and zero fitted values. Those groups demonstrate coarse sharing for this fixture. A future fitted parameter budget must still count adapters and any finer sharing.

The extract drops **40.8%** of incoming absolute edge mass from the parent graph. It includes **zero direct MBON-to-selected-CX records** from the six selected MBONs. A read-only parent scan found **61 omitted one-relay candidates** connecting those MBONs to selected CX cells in two directed nonzero hops. All six MBONs have at least one such path. These are paths to adjudicate by identity, compartment and sign. They are not validated goal signals. The historical aggregate edge table does not locate synapses in heading columns. The FB plasticity mask remains unregistered.

[Extraction receipt](CYCLE1.json) · [Relay audit](BOUNDARY_AUDIT.json) · [Engineering extract](EXTRACT.npz)

## Cycle 2 Matched graph and arena fixtures

The null swaps changed **70,746 edge destinations** while preserving directed degree by sign and the source and destination role classes. Native and rewired graphs each completed 120 fixed-gain stress steps with finite bounded activations under synthetic drives. This checks numerical plumbing. It gives no heading phase, displacement, steering or topological performance result.

The independent analytic navigator solved **32/32** paired goal probes. The longest took 31 of 200 steps. The fixed-zone policy solved **0/32** paired probes. The engineering start points were at least 0.34 arena units from the demonstration visit line, and the opposite and permuted reward histories each had 16 positive events among the same 32 opportunities. An analytic turn-and-translate fixture confirmed that reversing event order can change final displacement. The analytic controller uses the evaluator's goal and is a solvability control.

[Graph and arena receipt](CYCLE2.json)

## Cycle 3 Local writes and restart

The odor fixture used actual PN-to-KC and KC-to-MBON edges, engineered PN patterns and a scalar outcome. Direct DAN-to-KC contact restricted eligibility to **5,583** KC-to-MBON edges. The fixture uses a pooled engineering modulator with the selected anatomical mask. It does not implement the full compartment-specific DAN, APL or MBON feedback dynamics. The paired and shifted reward arms changed 446 eligible weights each, and the no-write arm changed zero. The run exercised assignment reversal, sign preservation and read-only response scoring. A fresh process restored the local state and reproduced the next-update hash exactly. The receipt records cue responses. A behavioral odor reversal endpoint and the learned arena policy remain unopened.

[Local update and restart receipt](CYCLE3.json)

## Independent scalar reward calibration

On the mod-97 smoke test, the cold arm obtained **0/128** rewards and made zero writes. With the archived shared-path initialization, the aligned arm earned 119/128 rewards and exercised the update. All warm arms scored **28/32** before and after on the small monitoring probe. The warm check demonstrates usable reward delivery and update instrumentation at this dose. It does not establish a gain over no writes or yoked reward.

[Scalar probe and checkpoint checksum](SCALAR_PROBE.json)

## Next qualification gate

Resolve anatomical sector and boundary mappings, audit the candidate MBON relay paths, and qualify sensory and steering adapters before opening anatomical heading and displacement scoring. Keep the derived relay and each engineered adapter explicit in the model manifest. The engineering checks used fixed gains. Main navigation, construction fitting, odor behavior gates and held-start learning were outside this rehearsal.

[Proposed design](../minifly-whole-repo-audit/MINIFLY_CONNECTOME_NAVIGATION_AND_LOCAL_LEARNING_DESIGN_20261106.md)

## Reproduction

The extraction and relay audit require NumPy and PyArrow. From this project root, use a Python 3.11 environment with compatible NumPy and PyArrow, then use the existing PyTorch environment for the scalar probe.

```sh
dir=output/minifly-connectome-navigation
PYTHONPATH="$dir" python -B -c 'import json; from anatomy import extract, ROOT; (ROOT/"CYCLE1.json").write_text(json.dumps(extract(), indent=2)+"\n")'
python -B "$dir/boundary_audit.py"
scratch=$(mktemp -d)
python -B "$dir/qualify.py" 2
python -B "$dir/qualify.py" 3 "$scratch"
python -B "$dir/qualify.py" resume "$scratch"
python -B "$dir/scalar_probe.py"
```
