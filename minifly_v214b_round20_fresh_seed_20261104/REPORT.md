# MiniFly V214B — Round 20 fresh-seed reproduction

V214B recovered Claude’s Round 20 compressor and answer scorer from the supplied session export and executed the unchanged compressor on fresh MiniWorld streams. The launcher reconstructs the generator prefix and original rendering and scoring protocol. Recovery records the source event IDs and SHA-256 hashes. All seven scoring runs passed the local audit, covering 635,177 byte applications with repeated prefixes charged each time.

## Mixed-world recall

One continuous stream contained 25 worlds and 1,500 questions. Lower prequential bits per byte (bpb) is better.

| Mode | Claude’s archived run | V214B |
| --- | ---: | ---: |
| Recall disabled | 0.6551 | 0.6600 |
| Shuffled cue | 0.6515 | 0.6578 |
| True cue | 0.5707 | **0.5776** |

True cues reduced stream loss by 12.5% versus recall disabled and 12.2% versus shuffled cues. They also lowered answer-payload loss against both controls for person and object questions in both halves. This reproduces the ordering on a fresh stream. The shuffle changes retrieved candidates and can also change retrieval coverage and cost. The recall-disabled arm removes the extra mixer inputs.

## Transfer to one fresh target world

Each run used true recall and scored the same 9,085-byte target with 80 questions. The target used fresh generated entity tokens and held-out movement templates. Generic vocabulary, question formats, and pickup/drop templates were shared. The compressor continued learning online during the target segment.

| Prior worlds | Target bpb | V214B bits saved vs cold | Claude’s archived bits saved | Prior-stream code bits |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6089 | 0 | 0 | 0 |
| 2 | 0.5390 | +635 | +775 | 7,821 |
| 8 | **0.5321** | **+697** | +731 | 26,687 |
| 24 | 0.5443 | +587 | +563 | 74,844 |

All three histories lowered target loss in both runs. Eight prior worlds produced the lowest fresh target loss, while two had the highest observed target savings per prior-stream code bit. Each checkpoint reprocessed its full history from an empty state. Shared history prefixes and their losses matched exactly. The checkpoints are correlated and cover one target world. The transfer roster did not cross history length with recall-disabled or shuffled modes, so it cannot attribute the warm-history gain specifically to correct cues or establish an optimal history length.

## Protocol and reproduction

V214B scores each bit before observing it and updates the compressor before predicting the next bit. Within a byte, the next prediction can use previously observed bits of that byte. Answer-payload scoring excludes the trailing delimiter. These are conditional compression scores under Claude’s protocol. Exact generated answers, native MiniFly deployment, and prose transfer were outside this run. V214A used a separate test-double protocol.

Fresh seeds were 17 for the mixed stream, 23 for transfer history, 197 for the target world and 198 for target events. The launcher pins Python hash seed 7 because hash addressing and set iteration affect the run. Claude’s archived commands did not specify a hash seed. The historical sham seed and recall decay were kept unchanged.

Run `PYTHONHASHSEED=7 python -B run.py` in a fresh copy with NumPy and no existing `results/` directory. Run `python -B audit.py` afterward to check the regenerated streams, scores and receipts. The compact bundle carries source and aggregate evidence. Generated streams, per-byte arrays and the session export are stored separately from the bundle.
