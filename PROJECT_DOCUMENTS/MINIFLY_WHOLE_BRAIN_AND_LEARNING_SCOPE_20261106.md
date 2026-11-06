# MiniFly whole brain sampling and learning scope

**Research note · November 6, 2026**

Early MiniFly ran a whole-brain FlyWire reference and sampled from that graph. The learning lineage later narrowed to an extracted mushroom-body pathway. This changed the scope of the research. Results from the extracted learner cannot answer whether a broader fly circuit would support working state, navigation, or compositional computation.

## What the early project ran

The original fixed-weight spiking reference used FlyWire materialization 630, with **127,400 neurons and 14,687,178 directed weighted connections**. The report records **132 initial full-reference trials**. It also records uniform and cell-type-stratified sampling at six fractions, alongside activity-selected and connectivity-guarded reductions. The tests covered taste and antennal mechanosensory responses and measured how well reduced graphs preserved those responses. [Original MiniFly report](https://github.com/ChangweiZhou/digitalbaby/blob/511cbb30011683df8d1b2768b7e290032b7aa2ef/PROJECT_DOCUMENTS/PRE_FULL151_DESIGNS_RESULTS_20261023/results/30032ad45886007e_REPORT.md) · [Published reference model](https://www.nature.com/articles/s41586-024-07763-9)

I counted `cell_class == CX` in the saved v630 model metadata. These counts use the historical annotation for central-complex cells.

| Model | Total neurons | Central-complex neurons | Kenyon cells |
| --- | ---: | ---: | ---: |
| Full reference | 127,400 | 2,858 | 5,177 |
| Activity core | 1,722 | 11 | 0 |
| Guarded core | 4,083 | 38 | 0 |
| Ten-partner guarded core | 5,832 | 75 | 2 |
| Twenty-partner guarded core | 8,783 | 150 | 36 |
| Mushroom-body-preserving scaffold | 9,619 | 38 | 5,177 |

The 11 central-complex cells in the activity core fall outside the forced sensory and readout interface. The recorded selection rule therefore retained them because they spiked in at least one construction trial. This is evidence of their inclusion and activity under the tested stimuli. The trials establish sensory and motor response preservation within their declared scope. Heading integration and goal-directed navigation were separate capabilities. [Original MiniFly report](https://github.com/ChangweiZhou/digitalbaby/blob/511cbb30011683df8d1b2768b7e290032b7aa2ef/PROJECT_DOCUMENTS/PRE_FULL151_DESIGNS_RESULTS_20261023/results/30032ad45886007e_REPORT.md) · [Central-complex heading integration study](https://www.nature.com/articles/nature22343)

## Where the scope narrowed

Iteration 5 extracted **302 olfactory projection neurons, 5,177 Kenyon cells, and selected dopamine teaching and mushroom-body output neurons**. I joined the three saved interface rosters against the historical annotations by FlyWire root ID. The rosters contain **zero central-complex cells**. V181 later inherited the projection-neuron-to-Kenyon-cell pathway as a fixed byte encoder. Iteration 7 did test a separate comparator derived from lateral-horn connectivity. The more recent 32-unit attractor used a synthetic ring and a rewired control. The implementations examined here contain no validated anatomical central-complex learning or navigation subsystem. [Iteration 5 learning report](https://github.com/ChangweiZhou/digitalbaby/blob/511cbb30011683df8d1b2768b7e290032b7aa2ef/PROJECT_DOCUMENTS/PRE_FULL151_DESIGNS_RESULTS_20261023/results/41187b7afe625c50_REPORT.md) · [V181 encoder](https://github.com/ChangweiZhou/digitalbaby/blob/064135ae2a4a95ca4fa7d473f180f57b024e5efc/minifly_v181_mini_local_plasticity_dev_20261102/sensor.py) · [Lateral-horn comparator](https://github.com/ChangweiZhou/digitalbaby/blob/511cbb30011683df8d1b2768b7e290032b7aa2ef/PROJECT_DOCUMENTS/PRE_FULL151_DESIGNS_RESULTS_20261023/results/135886e065763edc_REPORT.md) · [Synthetic ring construction](https://github.com/ChangweiZhou/digitalbaby/blob/501ca0c007da59e6aae581b02c6277f2f9caef94/minifly_three_remaining_experiments_20261106/output/minifly-remaining-experiments/models.py#L133)

The review is right to flag the narrowing of the learning architecture. Its statement that the central complex was never in the model needs to be corrected. The whole-brain simulation and sampled reductions preceded the extracted learning core. The negative results from that core identify limits of tested learners, interfaces, and tasks. They do not establish that omitting the central complex caused those limits. A future circuit comparison can test that hypothesis only by giving the candidate a defined function and measuring it against a matched control.

## Recount inputs

The local archive paths below are relative to the `second_try_copy` root.

- Full reference membership and annotations are in `minifly/data/flywire630/metadata.parquet`. The historical annotation export is `minifly/sources/flywire630_annotations/supplemental_files/Supplemental_file1_annotations.tsv`.
- Reduced membership is in `minifly/models/*/metadata.parquet`. The sampling and activity-selection code is `minifly/src/reduce.py`. The forced interface lists are in the model-specific `protocol.json` files.
- Iteration 5 membership is in `minifly/iteration5/data/input_neurons.csv`, `minifly/iteration5/data/coding_units.csv`, and `minifly/iteration5/results/interface_neurons.csv`.
