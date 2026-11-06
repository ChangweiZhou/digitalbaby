# MiniFly v630 navigation interface follow-up

**The compass map is ready for engineering. The fixed-gain model still fails the turn-in-darkness and goal-steering checks.** The follow-up recovered same-release synapse coordinates, measured the compass geometry, and implemented boundary probes. The strongest results were stationary heading retention after separating gains, and movement direction from a separately clamped compass. Displacement accumulation, learned-place navigation and fan-shaped-body writes remain gated.

| Question | What the follow-up established |
| --- | --- |
| Where is heading represented? | Mapped **47/47 EPG cells** using 42,970 FlyWire v630 input contacts inside the atlas ellipsoid body. Mapped EB, PB and FB anatomy from 393 v630 skeletons. On 28 unchanged root IDs held out from phase alignment, the median discrepancy versus a separate v783-derived geometry was **31.5°**. Use the continuous atlas coordinates as engineering readouts and retain that uncertainty. |
| Do the PEN1 paths have the expected offset? | The v630 coordinate export contained 430 directed EPG–PEN1 pairs. Of the pairs in the original signed extract, 29 EPG→PEN1 pairs had localized PB contacts and 236 PEN1→EPG pairs had localized EB contacts. Weighting by the observed contact fractions exposed median offsets of **+47.8° and −47.9°**. The aggregate edge weights gave **+5.5° and −4.2°**. |
| Can the circuit hold or turn the heading? | The direct landmark adapter anchored the native compass with **2.1°** median error while the landmark was present. The original fixed gains held the stationary heading in **0/16** probes. A separated-gain setting reached **14/16**, versus **0/16** after rewiring, with **88.7°** median error on turns. Using the observed PB/EB fractions on their own still left median turn error well above 20°. ER4d stimulation gave a weak landmark-dependent effect relative to its constant-drive control. |
| Does the movement pathway carry information? | The observed EPG→PFNd/v→hΔB→PFR edges produced **22.8°** median direction error versus **59.6°** after rewiring, across 128 clamped-heading cases. This was a feedforward velocity probe with known heading supplied and no displacement state. |
| Can the circuit steer toward a supplied goal? | The parent graph has **24 signed PFL3→DNa02 contacts**, 12 onto each side. Both unilateral motor canaries passed. With the direct compass cue and an externally supplied FC2 goal, the native circuit issued the correct turn sign on **29/64**, versus **32/64** after rewiring and **32/64** with the goal absent. The Δ7 sign sensitivity scored **35/64** on both graphs. |
| Can the selected mushroom-body outputs teach this circuit? | Found **61** one-relay candidates, including **27** FB tangential cells with 22 glutamate, four dopamine and one serotonin prediction. Twelve conservative signed two-hop paths reached FC2 or PFL3 at the lowest weight threshold, and zero did when both hops had absolute weight ≥3. The dopamine cell-contact rule selected all **421/421** eligible FC2→PFL3 edges. The coordinate export localized 188 of those FC2→PFL3 pairs in FB and contained zero of the 55 candidate dopamine→PFL3 pairs. The available evidence leaves the plasticity mask unqualified. |

The coordinate-based arm keeps the historical signed graph and scales the two compass routes by the fractions of public contacts located in the intended compartment. It excludes pairs without coordinate support in that arm and rewires the retained graph independently. The Δ7 glutamate-as-inhibitory arm is a separately reported physiological assumption for inputs to EPG and PFL3. The fixed-gain sweeps used development angles and no gradient fitting. Landmark stimulation, supplied goal, body-motion axes and phase readouts are declared boundary adapters. Atlas membership and contact positions give continuous coordinates, with uncertainty at compartment borders and in assigning exact glomeruli.

The yaw audit also recovered **40 GLNO→PEN1** edges from four omitted GLNO cells, including **30** signed GABA edges. Left GLNO cells feed right PEN1 cells and right GLNO cells feed left PEN1 cells. Using those 30 inhibitory weights under an assumed opponent GLNO response yielded **82.1°** median error at the best of three tested input strengths, versus **88.4°** for the matched rewired graph at that strength. The target is **20°**. Physiological calibration still has to establish the drive and timing of this observed input path. [Angular-velocity physiology](https://elifesciences.org/articles/23496) · [Central-complex anatomy](https://elifesciences.org/articles/66039)

The engineering result narrows the next repair to dynamic yaw transfer and goal-relative steering. The next useful comparison is a physiological GLNO/PEN1 input and rate calibration against the same native, rewired and recurrence-off circuits, followed by the supplied-goal steering test. Displacement and local reward belong after those checks pass.

## Reproduction

The original [three-cycle qualification](CYCLES.md) and its [hash manifest](BUNDLE_HASHES.json) are preserved. The current bundle contains the [skeleton map](COMPARTMENT_MAP.json), [synapse localization](SYNAPSE_COMPARTMENTS.json), [relay and steering audit](RESOLVED_BOUNDARY.json) and [functional probe receipt](INTERFACE_PROBE.json). The raw public downloads can stay in a scratch directory. The original extract and parent tables supply the historical signed weights.

```sh
dir=output/minifly-connectome-navigation
scratch=$(mktemp -d)
curl -fLsS 'https://raw.githubusercontent.com/navis-org/fafbseg-py/master/fafbseg/data/JFRC2NP.surf.fw.zip' -o "$scratch/roi.zip"
curl -fLsS 'https://storage.googleapis.com/flywire-data/codex/data/fafb/630/synapse_coordinates.csv.gz' -o "$scratch/synapse_coordinates.csv.gz"
curl -fLsS 'https://www.immortalfly.app/docs/circuit/' -o "$scratch/reference.html"
python -B "$dir/map_compartments.py" --ids | while read -r root; do
  curl -fLsS "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_630/$root" -o "$scratch/$root.skel" || exit 1
done
OPENBLAS_NUM_THREADS=1 python -B "$dir/map_compartments.py" "$scratch" --reference-html "$scratch/reference.html"
python -B "$dir/resolve_boundary.py"
OPENBLAS_NUM_THREADS=1 python -B "$dir/map_synapses.py" "$scratch" --reference-html "$scratch/reference.html"
OPENBLAS_NUM_THREADS=1 python -B "$dir/qualify_interfaces.py"
```

The boundary scan uses PyArrow with a compatible NumPy build. The public source URLs and SHA-256 values are embedded in the derived files. The independent v783 comparison is optional and can be omitted by dropping both `--reference-html` arguments. The runner uses v630 geometry throughout. [FlyWire v630 skeleton access](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.get_skeletons.html) · [FlyWire Codex data](https://codex.flywire.ai/faq) · [Velocity physiology](https://www.nature.com/articles/s41586-021-04067-0) · [PFL3 physiology](https://www.nature.com/articles/s41586-023-07006-3) · [DNa02 physiology](https://elifesciences.org/articles/102230)
