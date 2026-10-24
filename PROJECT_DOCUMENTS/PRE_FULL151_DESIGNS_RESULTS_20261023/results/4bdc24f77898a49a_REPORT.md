# Huang 2024 Zenodo data — intake inspection, not a new experiment

13 September 2026. Source archives remain unchanged in the project root. No MiniFly model, configuration, fit, threshold or previous result was changed. No learning simulation, physiological outcome comparison or spike redetection was run.

**These files materially improve access to the physiology underlying MiniFly's effective learning model.** They provide individual voltage-imaging and behavioral records for the same Huang et al. Nature 2024 study already used through its source workbooks and model repository. They add temporal and subject-level detail, not an independent biological replication or another connectome.

## Identity and integrity

The macOS download-origin metadata on all five archives points to `https://zenodo.org/records/10998457/files/FigureN.zip`. This record matches the DOI cited in the locally deposited paper: `10.5281/zenodo.10998457`, associated with *Dopamine-mediated interactions between short- and long-term memory dynamics*, DOI `10.1038/s41586-024-07819-w`.

All 5,110 non-directory ZIP members were read with CRC verification. All 5,074 substantive MATLAB records decoded, contained finite numeric data and passed the declared shape/time/index checks. The remaining 36 members are five readmes and 31 macOS metadata entries. Archive SHA-256 hashes, original paths and download origins are recorded in [intake_manifest.json](intake_manifest.json). Local integrity passed; official server checksums and the server's complete file inventory were not requested or compared. Thus this audit verifies these five downloaded archives, not completeness of the entire Zenodo deposit.

| Archive | Compressed bytes | Voltage records | Behavior records | Main content identified from folders and paper captions |
|---|---:|---:|---:|---|
| Figure1.zip | 121,597,393 | 1,520 | 0 | Five-second recordings across 11 DAN/MBON label groups; spike-detection metadata |
| Figure2.zip | 190,003,282 | 770 | 60 | Shock, sucrose and odor responses; innate odor-related walking/turning |
| Figure3.zip | 260,934,825 | 1,056 | 72 | CS+/CS− before/mid-training and at 5 min/1 h; neural and behavioral records |
| Figure4.zip | 171,235,412 | 576 | 288 | Attractive/repulsive conditioning through 24 h; odor/shock integration; feedback blocking and optogenetic behavior controls |
| Figure5.zip | 176,746,566 | 732 | 0 | 60/360/900-s inter-stimulus intervals; early/late unpaired re-exposure; 5-min, 35-min, 3-h and 24-h tests |
| **Total** | **920,517,478** | **4,654** | **420** | **5,074 MATLAB files across 105 leaf folders** |

Counts refer to files, not independent flies or independent experimental units. Several files contain repeated trials, and some voltage records are duplicated across figure folders.

The total expanded payload would be 989,868,411 bytes. The inspection processed one ZIP member at a time, without expanding the archives to disk. The largest member was 348,112 bytes; the audit process peaked at 120,766,464 bytes RSS and completed in about 8.8 seconds. Memory is not a practical obstacle for this source collection. Only metadata tables, five tiny original readmes and this inspection code/report were saved.

## Actual fields and interpretation

Every voltage file contains six fields:

- `PhotobleachCorrectedTrace`: fluorescence trace after photobleaching correction.
- `HighpassedTrace`: high-pass-filtered fluorescence signal.
- `DetectedSpikes`: integer sample locations of the authors' detected spikes; these are not second-valued timestamps.
- `SpikeDectionLimit`: the original field spelling, containing the cell-type-dependent detection threshold.
- `imagingRate`: 1,000 samples/s in all inspected voltage files.
- `t`: seconds, starting at zero, with 0.001-s spacing.

Voltage windows are 5, 10, 15 or 20 seconds. A file labelled `24hr` is a short measurement taken at that retention time, not a continuous 24-hour recording. All spike-location arrays contain strictly increasing integer values within 1..N, consistent with MATLAB indexing. Conversion to physical spike times must respect that indexing and the explicit `t` vector. The author's readmes and the paper Methods describe preprocessing and detection; this audit did not reproduce either.

These are **analyzed fluorescence/voltage-imaging traces, not raw image movies, intracellular voltage in millivolts, or synaptic weight measurements**. The local paper explicitly distinguishes deposited analyzed traces from raw voltage-imaging data available from its authors. Upstream motion correction, denoising and ROI selection therefore cannot all be re-audited from these archives alone.

Behavior files use another schema. Figure 2 includes forward speed, rotational speed, downsampled variants and `SamplingRate="1000hz"`. Figures 3 and 4 include rotational speed and `OdorDuration="5s–10s"`, with sampling-rate metadata absent. Their `t` vectors are integer sample indices, 1..15000, not the second-valued voltage vectors. The readmes give forward speed in mm/s and rotational speed in degrees/s. A shared reader that treats every `t` as seconds would be wrong.

The directory spelling `Panel aΓÇôd` is a ZIP filename-encoding artifact; the metadata inventory retains the exact archive path and a separately decoded display label. Likewise, the authors’ ASCII labels (a, a with a trailing grave accent, and v) should be mapped carefully to α, α′ and γ. α2, α′2 and α3 must remain distinct. Source paths were not renamed.

See [member inventory](member_inventory.csv) for every record's keys, dimensions, time range, local fly label, condition label and hashes; [group inventory](group_inventory.csv) summarizes the 105 folders.

## Duplicate and identity audit: resolve before statistical reuse

The MATLAB files are not byte-identical, but hashing their actual voltage signal arrays, spike locations, time vectors and acquisition rate identifies **113 pairs of identical voltage payloads**. There are **4,541 distinct voltage payloads among the 4,654 voltage files**. Distinct payloads still do not imply distinct animals.

| Duplicate location | Identical-payload pairs |
|---|---:|
| Figure 2 ↔ Figure 3 | 78 |
| Figure 3 ↔ Figure 4 | 32 |
| Within Figure 3 | 2 |
| Within Figure 4 | 1 |

Most cross-figure duplicates are compatible with reuse of recordings in different presentations of the study. Some identical recordings also have different local `FlyNN` labels across figures. The labels therefore cannot be used as global subject identities. Nor should an absence of exact duplicates be treated as proof that two records came from different flies.

**Two within-Figure-3 pairs need specific clarification.** For `PPL1-a3_Fly03_MB502B,pAce`, the CS+ and CS− records labelled `Pre` respectively match those labelled `1hr` exactly in all six stored data fields. This was independently rechecked by direct array comparison, recorded in [cross-time duplicate checks](cross_time_duplicate_checks.json). It is not merely agreement in the mean response. The available files do not establish which label is correct or why the same payload occurs at two times. No file was corrected or removed, and this observation is not an accusation about the cause.

The remaining within-Figure-4 pair is the same Fly11 shock-only payload under the ACV and OCT integration folders. It may be a shared control, but should not be counted as two independent control measurements.

All locations are preserved in [duplicate_records.csv](duplicate_records.csv). Before any refit or independent validation, identify recording reuse and cohort relationships, resolve or explicitly isolate the two cross-time ambiguities, and define splits at the appropriate animal/cohort level. Merely splitting by ZIP, filename or time point would permit leakage. Do not discard duplicates silently: a single recording may legitimately support multiple published panels.

## What this can improve for MiniFly

**The strongest immediate value is measurement reconciliation.** Previous physiology work used aggregate source-workbook measurements, including the 86 extracted conditioning observations and the 60 used for the iteration-4 reduced fit. These traces allow a future, separately specified audit of time windows, baseline subtraction, subject pairing and uncertainty behind relevant aggregates. They can show whether a discrepancy arises in measurement extraction or model prediction. They do not automatically resolve the existing mismatch between released model-prediction medians and source-code replay; that remains a software/parameter-version question as well.

**They supply temporal constraints on teaching and expression.** The Figure 2 shock/sucrose/odor traces and Figure 4 odor-plus-shock records can constrain absolute baseline firing, positive and negative response changes, and integration timing. Figure 3 and Figure 4 blocking conditions constrain the history-dependent feedback that the current effective model compresses. This is more direct evidence for those interfaces than assigning physiological gains from synapse counts.

**Figure 5 is particularly relevant to competing persistence rules.** It contains the actual animal-level recordings for spacing and time-dependent unpaired re-exposure effects already considered as model challenges. A later comparison could require a proposed rule to preserve both effects, instead of accepting improved artificial association-load performance alone. These conditions have already informed earlier project evaluations, so they cannot now be called a wholly untouched independent test set simply because their detailed traces have arrived.

**Behavioral records provide an external check beyond neural response differences.** Turning and walking data can support a separate comparison between neural expression and behavior in the studied conditions. They are not automatically paired neural/behavioral measurements from the same identified animal, and they do not calibrate the current RBF observer by themselves.

## What remains missing

- No PN/KC population recordings or odor→88-PN-channel mapping were identified. These records do not fill the physiological input-encoding gap.
- No same-specimen neuron/synapse IDs connecting the imaged animals to FlyWire, BANC or hemibrain were identified. Existing cross-specimen restrictions remain.
- No explicit contingency-reversal condition was found in these five archives. Unpaired re-exposure/extinction is not equivalent to switching the punished cue. They cannot directly answer the preceding old-memory/new-memory reversal question.
- Retention labels in this download extend to 24 h. No 48-h or multi-day record was found, although the paper discusses a 48-h extended-data experiment. This audit has not checked whether an additional deposit file provides it. Long slow lifetimes remain poorly identified by the present horizon.
- No individual synaptic weights or hard synaptic bounds are measured here. These files cannot establish that the current learner suffers Jiang-style weight-bound exhaustion.
- The files belong to the same study underlying the current physiological reference. More granular access strengthens auditability; it does not create a second biological replication or validate the artificial high-load KC task.

## Recommended disposition

Register these as a new source-data layer and keep iteration10 and the accepted parent frozen. Before the next mechanism experiment, reconcile condition/animal identities, exact duplicates, time units, and a small prespecified set of published measurements. The two cross-time duplicates deserve targeted source clarification; an author-provided correction/cohort map would help more than guessing.

If a separate 48-h/extended-data archive is present on the download page, it would be useful to retain in the same project folder. The present work does not require another download and has not contacted authors or retried Zenodo.

Reproduce the metadata inspection from the project root with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 _omnibus10_env/bin/python minifly/data_intake/huang_zenodo_20260913/inspect_archives.py
```

The original five ZIPs are referenced in place, not copied into this intake directory or a new result bundle. This is a data inspection, not MiniFly iteration 11.
