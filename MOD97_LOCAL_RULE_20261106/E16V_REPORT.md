# E16-V — independent source and receipt verification

Status: COMPLETE_WITH_DISCREPANCIES. Audit only: zero life reruns, zero network executions. No original scientific receipt, harness, prediction or report was edited.

Pinned branch: `codex/minifly-consolidated-20261023`; source commit `0eaeaf3df573740b290bb1e83f3eac1ef360625e`. This audit covers the E16/E16-D/E16-D2 sections as they existed at that commit, all 21 target lives, the complete SUMMARY_E16 tables, and the 12 JOB-selected E16W receipts used in cross-study numerical claims. The added check7 from brief amendment `4e87196` is also delivered, with its amended brief preserved as E16V_AMENDED_SOURCE_BRIEF.md. Later E16-R results are outside this frozen audit.

The acquisition and retention results are real as recorded. The audit supports a benefit from richer replay, but does not establish operand coverage as the unique cause, clear the learning rule, or isolate a cell-specific cause. Several literal predictions were reported as holding after their thresholds were widened or their scope narrowed.

|Check|Verdict|Finding|
|---|---|---|
|1 Numbers and literal predictions|DISCREPANCY|All 3124 numeric table components match within displayed rounding. Prose bounds and the literal acquisition gates have discrepancies; see claims and gates CSVs.|
|2 Independent shortcut ceilings|CONFIRMED|12 panels ×4 existing predictors =48 exact fraction comparisons. Extra intermediate-copy is identical to ignore_o2; modal answer is a panel-specific constant upper bound.|
|3 Exposure and buffers|DISCREPANCY|36 source-derived whole-batch family-share rows;60 logged buffer snapshots match source counts. Realized draw frequencies are NOT_MEASURED. Share did not double exposure or alter first loss-bearing blocks.|
|4 Stream construction|CONFIRMED_WITH_LIMITS|Oracle/D and T/M have identical source-level input/target construction under the pinned helpers. No recorded batch digests attest the actual historical streams. Store and decay differences remain.|
|5 Cell ablation|DISCREPANCY|18/168 individual operation scores exceed10%; maximum32.25%. Low family means do not establish≤10% everywhere. Store was not ablated.|
|6 Settings and provenance|DISCREPANCY|Standing births absent from pinned tree;9 dry receipts present. All D receipts label wd=.25 despite source optimizer=.3. Source-at-run, birth and batch hashes not recorded.|
|7 Added replay coverage/target-error audit|DISCREPANCY / MEASURED LIMIT|Both arms replay only1,024 distinct compositions cumulatively. Uniform has13.6–21.4% wrong full-scratchpad replay targets per late stage. Share targets were not saved; its correctness is NOT_RECORDED.|

## 1. Report numbers and pre-result gates

Every numeric component of all five SUMMARY_E16 tables and all four E45 E16/D/D2 tables was independently recomputed: 3124 comparisons, zero beyond the displayed half-unit rounding interval. That agreement does not validate every prose inference. `E16V_TABLE_CROSSWALK.csv` gives source line, column, published token, exact recomputation and tolerance.

|Arm/seed|F1 b2→b4→b6|Held b4→b6|F3 b6|F1/held/F3 vs twin|
|---|---|---|---|---|
|T_73129|93.2500→78.0625→82.6250|87.0417→56.9167|90.8750|0.8663/0.6101/0.9693|
|T_93683|92.1875→71.7500→80.1875|84.2917→53.2083|90.1875|0.8519/0.5784/0.9743|
|T_116089|86.1250→47.0000→65.3125|87.0417→45.9583|82.3125|0.6916/0.4898/0.8815|
|D_73129|93.8750→91.5625→95.3750|93.7917→93.2917|93.7500|1.0000/1.0000/1.0000|
|D_93683|93.8125→91.6875→94.1250|91.2083→92.0000|92.5625|1.0000/1.0000/1.0000|
|D_116089|94.1250→94.6875→94.4375|93.7083→93.8333|93.3750|1.0000/1.0000/1.0000|
|T_oracle_73129|94.2500→91.8125→93.5625|83.2500→82.8750|90.9375|0.9810/0.8883/0.9700|
|T_oracle_93683|92.8750→92.4375→92.8125|85.0000→84.0000|89.2500|0.9861/0.9130/0.9642|
|T_oracle_116089|90.8125→84.8750→89.5625|86.8750→87.2917|87.8750|0.9484/0.9303/0.9411|
|T_balanced_73129|92.6250→69.9375→82.0625|89.0417→64.7500|91.0625|0.8604/0.6941/0.9713|
|T_balanced_93683|91.5000→68.1875→76.9375|85.7500→55.6250|89.7500|0.8174/0.6046/0.9696|
|T_balanced_116089|87.5625→44.4375→69.0000|88.5833→49.0833|85.7500|0.7306/0.5231/0.9183|
|T_all_balanced_73129|93.6875→87.7500→89.7500|90.1667→84.4583|92.5000|0.9410/0.9053/0.9867|
|T_all_balanced_93683|92.5000→84.9375→89.1875|87.4167→79.8333|89.5625|0.9475/0.8678/0.9676|
|T_all_balanced_116089|90.1875→60.6875→77.6875|91.3750→83.3333|88.3125|0.8226/0.8881/0.9458|
|T_share_73129|93.7500→78.8750→84.0000|88.5000→64.5417|90.5000|0.8807/0.6918/0.9653|
|T_share_93683|92.8750→69.3750→78.3125|84.5000→54.5417|88.4375|0.8320/0.5928/0.9554|
|T_share_116089|91.0625→46.9375→66.2500|85.9583→49.5417|83.1875|0.7015/0.5280/0.8909|
|M_73129|92.3125→88.0000→91.5000|93.6250→74.1667|92.8750|0.9594/0.7950/0.9907|
|M_93683|92.5625→80.7500→89.6875|93.2500→65.4167|91.8125|0.9529/0.7111/0.9919|
|M_116089|91.9375→91.0625→90.3750|93.2083→76.7083|92.5625|0.9570/0.8175/0.9913|

**Literal gate corrections.** P16D.4 required every control's acquisition on each current family to be within 3 pp of T. It does not hold as written: improvements beyond 3 pp also violate this two-sided prediction. P16D2.3 likewise required both new arms to be within 3 pp on every family, and does not hold. These are prediction misses, not evidence that improved acquisition is bad. P16.4 explicitly concerned F1: its ≥30pp ablation cost and ≤10pp family-mean extrapolation gap both pass. Do not broaden that gate retrospectively; the report's broader k20 wording does not hold for F3 seed116089 (10.375pp drop).

|Prediction|Arm|Seed|Measurement|Exact value|Criterion|Pass|
|---|---|---|---|---|---|---|
|P16.1|T|73129|F2_ratio|0.610094|>=0.8|False|
|P16.2|T|73129|F1_loss_pp|10.625000|<=10|False|
|P16.2|T|73129|sub_loss_pp|22.250000|<=20|False|
|P16.4|T|73129|F1_silencing_cost_pp|79.937500|>=30|True|
|P16.4|T|73129|F1_k12_minus_k20_pp|2.062500|abs<=10|True|
|P16D.4|T_oracle|73129|F2_acquisition_delta_pp|-3.791667|abs<=3|False|
|P16D.4|T_all_balanced|73129|F2_acquisition_delta_pp|3.125000|abs<=3|False|
|P16D2.1|T_share|73129|F1_relative_loss_reduction|0.020576|>=1/3|False|
|P16D2.1|T_share|73129|F2_relative_loss_reduction|0.204703|>=1/3|False|
|P16D2.2|M|73129|F2_loss_pp|19.458333|abs<=10|False|
|P16D2.3|M|73129|F2_acquisition_delta_pp|6.583333|abs<=3|False|
|P16.1|T|93683|F2_ratio|0.578351|>=0.8|False|
|P16.2|T|93683|F1_loss_pp|12.000000|<=10|False|
|P16.2|T|93683|sub_loss_pp|27.750000|<=20|False|
|P16.4|T|93683|F1_silencing_cost_pp|75.062500|>=30|True|
|P16.4|T|93683|F1_k12_minus_k20_pp|4.187500|abs<=10|True|
|P16D.4|T_all_balanced|93683|F2_acquisition_delta_pp|3.125000|abs<=3|False|
|P16D2.1|T_share|93683|F1_relative_loss_reduction|-0.149847|>=1/3|False|
|P16D2.1|T_share|93683|F2_relative_loss_reduction|0.036193|>=1/3|False|
|P16D2.2|M|93683|F2_loss_pp|27.833333|abs<=10|False|
|P16D2.3|M|93683|F2_acquisition_delta_pp|8.958333|abs<=3|False|
|P16.1|T|116089|F1_ratio|0.691595|>=0.8|False|
|P16.1|T|116089|F2_ratio|0.489787|>=0.8|False|
|P16.2|T|116089|F1_loss_pp|20.812500|<=10|False|
|P16.2|T|116089|add_loss_pp|39.000000|<=20|False|
|P16.2|T|116089|sub_loss_pp|36.750000|<=20|False|
|P16.4|T|116089|F1_silencing_cost_pp|61.875000|>=30|True|
|P16.4|T|116089|F1_k12_minus_k20_pp|9.375000|abs<=10|True|
|P16D.3|T_all_balanced|116089|F1_minus_oracle_pp|-11.875000|abs<=5|False|
|P16D.4|T_oracle|116089|F1_acquisition_delta_pp|4.687500|abs<=3|False|
|P16D.4|T_oracle|116089|F3_acquisition_delta_pp|5.562500|abs<=3|False|
|P16D.4|T_balanced|116089|F3_acquisition_delta_pp|3.437500|abs<=3|False|
|P16D.4|T_all_balanced|116089|F1_acquisition_delta_pp|4.062500|abs<=3|False|
|P16D.4|T_all_balanced|116089|F2_acquisition_delta_pp|4.333333|abs<=3|False|
|P16D.4|T_all_balanced|116089|F3_acquisition_delta_pp|6.000000|abs<=3|False|
|P16D2.1|T_share|116089|F1_relative_loss_reduction|-0.127796|>=1/3|False|
|P16D2.1|T_share|116089|F2_relative_loss_reduction|0.113590|>=1/3|False|
|P16D2.2|M|116089|F2_loss_pp|16.500000|abs<=10|False|
|P16D2.3|T_share|116089|F1_acquisition_delta_pp|4.937500|abs<=3|False|
|P16D2.3|M|116089|F1_acquisition_delta_pp|5.812500|abs<=3|False|
|P16D2.3|M|116089|F2_acquisition_delta_pp|6.166667|abs<=3|False|
|P16D2.3|M|116089|F3_acquisition_delta_pp|10.250000|abs<=3|False|

P16D.2 was withdrawn before final retention endpoints and is not scored as a successful share intervention. All other literal predictions and every seed are in `E16V_GATES.csv`. The first E16 coexistence gate still fails; richer replay is a bounded development positive, not autonomous learning.

|Section|Quoted claim|Recomputed|Verdict|Explanation|
|---|---|---|---|---|
|E16|6 blocks, 6144 updates, three standing seeds|6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000/6.0000|CONFIRMED|21 receipts have six blocks; all record stage_updates=6144. Historical wall 80min is not independently derivable from CPU counters.|
|E16|current held composition84–87; carry75–84; shortcut1–2; within4–9 of twin|87.0417/84.2917/87.0417/82.3333/83.8333/75.3333/75.0000/77.3333/79.8333|CONFIRMED|Rounded envelopes; table crosswalk supplies all exact values. Carry shortcuts independently rebuilt, 0.5–1.67% across seeds.|
|E16|each family within1–9 points of BP while current|2.8750/2.3750/11.0625|DISCREPANCY|F3 seed116089 gap11.0625pp; overall current-family gaps span0.875–11.0625pp, not1–9.|
|E16|F1 block4 values78.1/71.8/47.0 and partial recovery in F3|78.0625/71.7500/47.0000/82.6250/80.1875/65.3125|CONFIRMED|All three improve between blocks4 and6; F1 losses begin during F2.|
|E16|F2 carry18–38, within56–64, comparison72–85 after6|33.5000/37.6667/74.0000/82.5000/22.3333/27.6667/77.3333/85.5000/18.3333/17.6667/72.0000/75.8333|CONFIRMED|Rounded envelopes, within55.5/61.1667/64.1667; comparisons72–85.5. These comparisons alone do not establish composition.|
|E16|F1 one third during F2, F2 one fifth during F3|F1 F2 blocks=1/2,1/3; F2 F3 blocks=1/4,1/5|DISCREPANCY|Quoted shares apply to each second block, not both blocks; realized counters unavailable.|
|E16|silenced operations3–7 and compositions4–7|2.6875/2.8125/5.1250/6.3125/3.4375/5.4375/3.7500/6.7917/7.2917|QUALIFIED|Rounded family averages support approximate wording; individual operations reach13.25%, individual held pairs need separate table. Store remains enabled.|
|E16|k20 within2–9 points; P16.4 within10 holds|2.0625/0.5000/4.1875/1.1875/9.3750/10.3750|QUALIFIED|P16.4 explicitly names F1: its mean drops2.0625/4.1875/9.375pp, all≤10, and cell-silencing costs≥30pp. A broader all-family reading fails: F3 seed116089 drops10.375pp (82.3125→71.9375). Individual add73 falls14.75pp. Do not retrospectively broaden the original F1 gate.|
|E16|development operations9–52 in both arms|9.1667/22.7500/51.8333/38.0000/7.6667/55.7500|DISCREPANCY|T means9.1667–51.8333; D means7.6667–55.75. Mod89 is not replayed.|
|E16|P16.1 coexistence fails; F1passes2,F3passes3,compfails3|[[0.8663171690694627, 0.851925630810093, 0.6915949702183984], [0.6100937918713711, 0.5783514492753623, 0.4897868561278863], [0.9693333333333334, 0.974341661039838, 0.8815261044176707]]|CONFIRMED|Threshold0.8: T has2/3 F1,0/3 F2,3/3 F3 passes.|
|E16|P16.2 losses10.6/12.0/20.8, add/sub16–39|10.6250/12.0000/20.8125|CONFIRMED|All F1 family losses exceed10pp; per-operation losses16–39. P16.3 ≥70 held and≥60 carry after4 passes3/3.|
|E16|P16.5 twin holds everything|0.5000/-0.7917/-0.1250|QUALIFIED|Supports life-family means and all-family gate, not development operations or every individual cell. D data, decay and Store differ from T.|
|E16-D|9 lives,18–19min per block,115min|9|NOT_INDEPENDENTLY_VERIFIED|Receipts establish9 six-block lives; elapsed wall cannot be recovered from cumulative process CPU alone.|
|E16-D|oracle loses≤1.2 F1 and ≤1.0comp,3–9 belowtwin|0.6875/0.0625/1.2500/0.3750/1.0000/-0.4167|QUALIFIED|F1 seed116089 loses1.25pp, not≤1.2 literally (half-decimal boundary). F2maxloss1pp. Absolute current/final F3 gap reaches5.5; full comparisons in tables. Decay and Store confound rule-only interpretation.|
|E16-D|all-entry compositions80–85,within1.6–4.2 oforacle|1.5833/-4.1667/-3.9583|CONFIRMED|Absolute gaps1.5833/4.1667/3.9583pp; first is above oracle; operations89.75/89.1875/77.6875.|
|E16-D|balanced indistinguishable fromuniform; observednull|7.8333/2.4167/3.1250|QUALIFIED|Final gains7.8333/2.4167/3.125pp, not exact null; no statistical equivalence test. Expected shares equal during block3 but differ during blocks4 and6; per-item repetition also differs.|
|E16-D|weak seed61duringcomposition; oracle75–85;shareaddressesexactlythatblock|60.8125/60.6875/74.8750/84.8750|DISCREPANCY|All-entry60.8125/60.6875 versusoracle74.875/84.875. Share does NOT change exposure in first compositionblock3, where large loss already occurs; only changes block4 relativebalanced.|
|E16-D|P16D.3 missesF1by11.9;P16D.4 acquisitionwithin3–4|11.8750/3.1250/3.1250/4.3333|DISCREPANCY|F1difference11.875; all-entry composition acquisition+3.125/+3.125/+4.3333pp. Original P16D.4 says within 3 pp, not3–4; fails literal gate. Other family/arm deviations in E16V_GATES.csv.|
|E16-D2|6lives,23min/block,2h25min|6|NOT_INDEPENDENTLY_VERIFIED|6 complete receipts; process CPU is not elapsed wall.|
|E16-D2|shareF1loss15/24/44 versus15/20/39|14.8750/23.5000/44.1250/15.1875/20.4375/39.1250|CONFIRMED|Rounded correctly; first loss-bearing F2block has unchanged expected share; cannot exclude share as a mechanism generally.|
|E16-D2|compositionloss24/30/37 versus30/31/41,reduction4–20%|23.9583/29.9583/36.4167/30.1250/31.0833/41.0833/20.4703/3.6193/11.3590|DISCREPANCY|Share losses23.9583/29.9583/36.4167pp: last rounds36,not37; subtracting rounded endpoints can create this discrepancy. Reduction3.6193–20.4703% supports rounded4–20. Literal doubling exposure is wrong:25→25 and20→25% inF3.|
|E16-D2|Mopswithin3ofblock2,comp loses17–28,ends65–77|0.8125/2.8750/1.5625/19.4583/27.8333/16.5000|CONFIRMED|M F1 losses0.8125/2.875/1.5625; comp losses19.4583/27.8333/16.5; final65.4167–76.7083.|
|E16-D2|localforgetsextraabout20points|17.2500/12.2083/30.7500|QUALIFIED|Final advantages17.25/12.2083/30.75pp; mean20.0694, not20in everyseed. Loss differences and endpointdifferences are distinct because acquisition differs.|
|E16-D2|shareacquisitionwithin2comp,+0.6to+5ops|1.4583/0.2083/-1.0833/0.5000/0.6875/4.9375|DISCREPANCY|Compositiondeltas+1.4583/+0.2083/−1.0833;operations+0.5/+0.6875/+4.9375. Original P16D2.3 says everyfamily within 3 pp in botharms: fails; M composition acquisition is also >3pp higher in everyseed. Predictions cannot be loosened after results.|
|E16-D2|rulecleared;shareisnotthelever;coverageisthecause;cellnotspecific|Positive all-entry intervention; partial share effect; both rules forget under sparse replay|OVERCLAIM|Support for replay coverage as important, not uniquely isolated cause. No cell-only causal inference; no fixed-buffer composition-priority sweep rules out allshare interventions. Rich-data comparison also changes decay/Store; actual batches were not logged.|
|E16-D2/E16W|windowloss46–48;sharegain6–11andreduction14–23%|45.5833/46.4167/48.0833/6.2500/10.8750/8.1667|CONFIRMED|Uses pinned JOB-selected receipts, not glob-selected partial runs. Percentage reductions13.7/23.4/17.0 round14/23/17.|
|E16-D2/E16W|windowBPends43–50local31–37ceiling91|43.4167/45.2917/49.8750/35.3333/31.2500/37.2917/90.5000/90.7500/91.8750|CONFIRMED|Ceiling90.5/90.75/91.875; differentdata,notrule-only. Local record-stream matching verified by originalE16W study, not reexecuted here.|
|E16-D2/E16W|oldops93–95→76–85duringcompose|93.3333/76.0000/94.9167/78.8750/95.0417/85.3333|CONFIRMED|Scalar family means; does not prove same cause as the cell learner.|

Verdict: DISCREPANCY in prose/interpretation and literal gate reporting; numerical tables confirmed. Historical wall forecasts80min/115min/2h25 are not independently verified from process-CPU fields.

## 2. Independently rebuilt shortcut ceilings

Rebuilt the arithmetic panel construction using only Python Random and arithmetic, without importing the harness or model. Each of 12 held panels has600 distinct triples. All 48 existing predictor fractions match exactly. The earlier WORKING status incorrectly counted36 checks; corrected here. Copy-intermediate is exactly the already-recorded ignore-second-operation predictor, not an independent control. Modal answer is the most common correct output in the sealed panel; its reported score is an optimistic panel-derived constant upper bound, not an independently trained predictor.

|Seed|Pair|Ignore1/a|Ignore1/b|Ignore2|Copy c|Copy intermediate|Modal constant|
|---|---|---|---|---|---|---|---|
|73129|+-|1.5000|1.0000|1.3333|1.1667|1.3333|2.0000|
|73129|-+|1.5000|1.1667|1.6667|1.1667|1.6667|2.5000|
|73129|><|67.5000|66.5000|34.1667|67.0000|34.1667|3.3333|
|73129|<>|66.6667|68.5000|34.8333|66.1667|34.8333|2.5000|
|93683|+-|1.1667|1.3333|0.8333|0.8333|0.8333|2.3333|
|93683|-+|1.3333|0.6667|1.6667|0.5000|1.6667|2.1667|
|93683|><|66.8333|65.1667|35.8333|65.6667|35.8333|2.3333|
|93683|<>|68.3333|65.3333|31.8333|68.8333|31.8333|2.8333|
|116089|+-|0.5000|1.0000|1.3333|1.6667|1.3333|2.5000|
|116089|-+|0.6667|1.1667|0.8333|1.6667|0.8333|2.0000|
|116089|><|68.6667|67.0000|34.3333|66.3333|34.3333|2.6667|
|116089|<>|67.1667|67.6667|33.0000|67.8333|33.0000|2.1667|

Verdict: CONFIRMED. Carry-pair shortcuts are low; comparison-pair visible-only scores reach≈65–69%, so comparison accuracy alone cannot establish composition.

## 3. Expected exposure, recorded buffer counts, missing realized counters

F1=mod97 operations, F2=composition, F3=mod73 operations. Fractions are EXPECTED WHOLE-BATCH exposure; they are not realized historical frequencies. T and M use uniform rows; D and oracle use their own equal-seen-family stream. Each replay half has32 items except branches that draw the whole batch from the current or joint stream.

|Arm|Block|F1|F2|F3|
|---|---|---|---|---|
|uniform|1|1|0|0|
|uniform|2|1|0|0|
|uniform|3|1/2|1/2|0|
|uniform|4|1/3|2/3|0|
|uniform|5|1/4|1/4|1/2|
|uniform|6|1/5|1/5|3/5|
|balanced|1|1|0|0|
|balanced|2|1|0|0|
|balanced|3|1/2|1/2|0|
|balanced|4|1/4|3/4|0|
|balanced|5|1/4|1/4|1/2|
|balanced|6|1/6|1/6|2/3|
|all_balanced|1|1|0|0|
|all_balanced|2|1|0|0|
|all_balanced|3|1/2|1/2|0|
|all_balanced|4|1/4|3/4|0|
|all_balanced|5|1/4|1/4|1/2|
|all_balanced|6|1/6|1/6|2/3|
|share|1|1|0|0|
|share|2|1|0|0|
|share|3|1/2|1/2|0|
|share|4|1/2|1/2|0|
|share|5|1/4|1/4|1/2|
|share|6|1/4|1/4|1/2|
|oracle|1|1|0|0|
|oracle|2|1|0|0|
|oracle|3|1/2|1/2|0|
|oracle|4|1/2|1/2|0|
|oracle|5|1/3|1/3|1/3|
|oracle|6|1/3|1/3|1/3|
|D|1|1|0|0|
|D|2|1|0|0|
|D|3|1/2|1/2|0|
|D|4|1/2|1/2|0|
|D|5|1/3|1/3|1/3|
|D|6|1/3|1/3|1/3|

All 60 recorded snapshots in `E16V_BUFFERS.csv` agree with the source. Balanced/share retain1,024 presentation samples per block (2,048 per completed family); all-entry buffers show F1=9,408, F2=24,000, F3=5,328 distinct entries. M logs1,024/2,048/3,072/4,096/5,120 after blocks 1–5. Original T has no buffer fields. No arm persists a block 6 buffer count: source saves the receipt before its final buffer mutation, and only the following stage save persists earlier mutations. Missing counters cannot be reconstructed as measurements.

Uniform→share F1 exposure during the two F2 blocks is 1/2→1/2 and 1/3→1/2; F2 exposure during F3 is 1/4→1/4 and 1/5→1/4. The first large-loss block is unchanged in both transitions. Relative to balanced, share changes F1 during F2 block2 from 1/4 to1/2. The claim that it doubled exposure throughout the loss-bearing period is false. Balanced/all-entry have identical expected family shares; their comparison changes support/coverage and repetition per item, not just a biological mechanism.

Verdict: DISCREPANCY in exposure interpretation. Realized draw frequencies: NOT_MEASURED.

## 4. Same-stream assertions and remaining differences

|Comparison|Matching source construction|Remaining differences / limits|
|---|---|---|
|T_oracle vs D|Identical family_entries, entry_stream seeds, seen-family order, joint_stream(seed+si), batch length and ground-truth render|D usesBP/.3/noStore; oracle useslocal trace/.25/Store. Data construction identical, actual runtime stream hashes absent.|
|M vs T_uniform|Same Random(seed+9), generators, rng.sample buffer admission/draws, ground-truth rerender; model outputs never construct training labels|M usesBP/.25/noStore; T useslocal trace/.25/Store. Store changes scoring, not these training batches. No actual historical batch digests.|
|Helpers|update/forward/trace/Store paths do not consume the Python replay RNG; render receives explicit rng; entry_stream owns its own Random|new_net seedsTorch only; no model-output-dependent branch alters data selection. AST call inventory in E16V_SOURCE_CALLS.json; current pinned source is not an immutable source-at-run attestation.|

The phrase “exactly the learner's batches” is supported by deterministic source construction, not by recorded per-batch hashes or an independent execution. This is stronger than merely similar distributions but weaker than an audited historical stream transcript. Local-vs-M training compares the backward rule under the same source-generated inputs/labels and optimizer settings; final accuracy also includes the Store difference. Store gain is6 and capacity65,536 exact context keys; do not confuse it with the replay buffer.

Verdict: CONFIRMED_WITH_LIMITS at source-construction level; exact historical stream identity NOT_ATTESTED.

## 5. Cell ablation, every arm and seed

All scores below are exact full-operation answers at k12 with the cell silenced. Silencing happens in the cell pathway; it does not disable the window pathway or the Store. No no-Store ablation was run for this audit.

|Arm|Seed|add97|sub97|max97|min97|add73|sub73|max73|min73|Any>10%|
|---|---|---|---|---|---|---|---|---|---|---|
|T|73129|0.75|0.75|4.50|4.75|0.75|1.00|7.25|2.25|False|
|T|93683|0.75|1.50|6.25|12.00|0.75|1.25|10.00|13.25|True|
|T|116089|3.00|1.75|2.50|6.50|2.00|1.00|9.50|9.25|False|
|D|73129|1.25|1.50|3.50|5.00|1.50|1.50|7.00|6.00|False|
|D|93683|1.25|0.50|2.25|4.25|1.00|0.25|3.25|2.50|False|
|D|116089|0.00|0.75|0.25|0.25|1.00|1.25|0.50|0.00|False|
|M|73129|1.00|0.25|7.00|9.00|0.75|2.25|9.25|9.00|False|
|M|93683|1.75|0.75|4.25|8.50|1.25|1.25|8.00|12.00|True|
|M|116089|0.50|0.25|1.00|1.50|1.00|1.00|1.75|2.00|False|
|T_oracle|73129|0.75|0.75|5.25|5.75|1.25|0.75|6.25|7.50|False|
|T_oracle|93683|1.25|1.25|15.25|19.00|1.75|1.25|23.00|18.00|True|
|T_oracle|116089|0.75|1.75|3.00|7.50|1.25|2.00|14.50|8.75|True|
|T_balanced|73129|0.50|1.50|2.25|2.25|1.50|1.00|4.00|1.75|False|
|T_balanced|93683|1.00|1.50|11.00|10.25|1.00|1.25|18.25|9.50|True|
|T_balanced|116089|1.75|2.50|3.75|12.50|1.75|0.75|13.75|16.25|True|
|T_all_balanced|73129|1.00|1.00|3.25|4.75|2.00|1.75|7.75|6.25|False|
|T_all_balanced|93683|1.25|1.00|2.75|6.00|2.00|0.75|7.25|5.75|False|
|T_all_balanced|116089|0.75|0.75|12.00|24.50|1.50|2.00|31.50|32.25|True|
|T_share|73129|1.00|1.75|2.75|5.25|2.00|1.50|3.50|7.00|False|
|T_share|93683|1.00|1.50|6.00|4.50|1.50|1.50|9.00|2.00|False|
|T_share|116089|1.75|2.25|1.50|3.00|2.00|1.25|4.00|5.75|False|

Silenced composition scores below are final-answer / full-scratchpad percentages, separately for all held pairs. They are not interchangeable with complete-response scores.

|Arm|Seed|+-|-+|><|<>|
|---|---|---|---|---|---|
|T|73129|0.6667/0.0000|0.3333/0.0000|7.0000/0.0000|7.0000/1.5000|
|T|93683|1.8333/0.1667|0.6667/0.0000|12.8333/0.1667|11.8333/1.1667|
|T|116089|1.3333/0.0000|0.1667/0.0000|12.5000/2.0000|15.1667/2.3333|
|D|73129|1.0000/0.0000|0.8333/0.0000|5.1667/0.0000|2.8333/0.0000|
|D|93683|1.1667/0.0000|0.8333/0.0000|6.5000/0.3333|4.0000/0.1667|
|D|116089|1.0000/0.1667|0.8333/0.1667|19.6667/3.3333|18.5000/2.6667|
|M|73129|0.5000/0.0000|2.0000/0.0000|7.3333/0.0000|3.6667/0.0000|
|M|93683|0.8333/0.0000|0.8333/0.0000|12.1667/0.0000|8.0000/0.1667|
|M|116089|0.6667/0.0000|1.3333/0.1667|23.0000/4.1667|22.1667/4.0000|
|T_oracle|73129|0.6667/0.1667|0.8333/0.0000|7.3333/0.1667|6.6667/0.5000|
|T_oracle|93683|1.1667/0.0000|1.1667/0.0000|15.3333/0.8333|17.8333/0.3333|
|T_oracle|116089|1.0000/0.0000|1.0000/0.0000|13.1667/1.5000|11.5000/0.1667|
|T_balanced|73129|1.0000/0.0000|0.6667/0.0000|4.5000/0.1667|6.5000/0.3333|
|T_balanced|93683|0.8333/0.0000|1.0000/0.0000|11.3333/0.1667|13.6667/1.5000|
|T_balanced|116089|1.5000/0.0000|0.6667/0.0000|23.8333/3.6667|23.6667/3.5000|
|T_all_balanced|73129|0.6667/0.0000|1.8333/0.0000|9.3333/0.3333|7.8333/0.1667|
|T_all_balanced|93683|0.5000/0.0000|0.5000/0.0000|2.3333/0.0000|6.1667/0.0000|
|T_all_balanced|116089|0.6667/0.0000|0.0000/0.0000|26.5000/4.0000|18.1667/2.3333|
|T_share|73129|0.8333/0.0000|1.0000/0.0000|5.6667/0.5000|8.6667/0.8333|
|T_share|93683|0.6667/0.0000|0.8333/0.0000|7.6667/0.8333|7.5000/0.6667|
|T_share|116089|1.1667/0.0000|1.5000/0.0000|22.1667/4.8333|24.3333/4.6667|

Verdict: DISCREPANCY. 18 of 168 operation/arm/seed scores exceed10%. Largest is32.25% (T_all_balanced,116089,min73); its max73 is31.5%. Low aggregate means hide these exceptions. The original T F1≥30pp cell-importance prediction still passes all 3 seeds; this does not prove all retained performance originates exclusively in the cell.

## 6. Deviations and reproducibility provenance

|Item|Finding|
|---|---|
|Standing memory births|All 3 e11_dev_M_{seed}.pt absent from pinned repository tree. Local availability/hashes are in E16V_PROVENANCE.json, not evidence that they are the exact historical run inputs.|
|Dry receipts|9 files exist in archive_e16_dry/, including T,D,M,oracle,balanced,all-entry,share and2 reservoir dry controls. They record4 updates/block; do not claim dry files are missing.|
|D settings|All 3 D stages[0].settings.wd=.25, but source actually creates D AdamW with .3. The settings dictionary reports shared E.WD, not actual arm decay.|
|Other settings|All 21 receipts record fresh/scan/6144/.25, consistent with local and M source. OldT/D omit replay16 field; later arms record it. W10,H512,B64 inferred from pinned helper defaults/assertions, not runtime hash/version attestation.|
|Source versions|Pinned exp16_integrated.py already includes later reservoir code. Earlier receipts omit fields now emitted; cannot pretend this exact later blob generated every earlier receipt. Helpers are pinned for this static audit.|
|Missing replay telemetry|No actual per-batch digest, realized family draw count, target accuracy or historical Store occupancy/lookup-hit counter. Buffer sizes support intended eligibility, not actual exposure or an isolated cause.|
|Compute provenance|Receipt processCPU excludes final evaluation and is not walltime; this audit does not certify quoted walltimes or invent RSS. No scientific compute performed.|
|Immutable inputs|79 snapshot source/report/receipt blobs match pinned git objects; SHA256/byte counts are in E16V_INPUT_MANIFEST.json.|

Verdict: DISCREPANCY / incomplete independent-replication bundle. These gaps do not erase the observed receipt outcomes; they limit claims of independent reproducibility and unique causal explanation.

## 7. Added check: composition coverage and self-generated target errors

This check reads the existing E16W batch archives; it does not run a learner. All 3 uniform streams match their compressed-file hashes and original canonical full-batch digests, with57,344 batches per seed. The truth calculator also agrees with all262,144 newly presented ground-truth composition targets per seed (786,432 checks total). All uniform replay prompts match an independent replay-RNG reconstruction exactly. Both arms' reconstructed stage draw counts match their recorded counters.

**New finding:** during each late mod73 stage,13.59–21.40% of uniform replay's full six-byte composition targets are mathematically wrong;11.29–18.88% have a wrong final answer. These are targets actually passed to update, not merely a model's untrained outputs. Every late stage still samples only the same1,024 distinct composition prompts in each seed (4.27% of24,000 taught triples across12 taught operator pairs). Increasing share increases repetition of that support; it does not broaden it.

C_share saved aggregate exposures, malformed-output counters and buffer membership, but did not save its generated targets. Its exact correctness cannot be recovered without executing a network, which this assignment forbids. We reconstruct its prompt selection with the original Python RNG/admission ordering and verify every stage draw count. This supports the coverage calculation conditionally on that source, but is not an actual target transcript. Malformed=0 does not imply mathematically correct.

|Arm|Seed|Stage|Composition replay count|Cumulative distinct|Wrong full target %|Wrong final target %|Held final %|
|---|---|---|---|---|---|---|---|
|C_uni|73129|compose|0|0|NO_REPLAY|NO_REPLAY|80.9167|
|C_uni|73129|add73|39235|1024|19.1003|16.4063|39.2500|
|C_uni|73129|sub73|32951|1024|19.7323|16.9191|35.3333|
|C_uni|73129|max73|27995|1024|19.5964|16.7387|33.0833|
|C_uni|73129|min73|24357|1024|20.2693|17.1983|35.3333|
|C_share|73129|compose|0|0|NOT_RECORDED|NOT_RECORDED|80.9167|
|C_share|73129|add73|98304|1024|NOT_RECORDED|NOT_RECORDED|43.8333|
|C_share|73129|sub73|98304|1024|NOT_RECORDED|NOT_RECORDED|39.5000|
|C_share|73129|max73|98304|1024|NOT_RECORDED|NOT_RECORDED|41.7083|
|C_share|73129|min73|98304|1024|NOT_RECORDED|NOT_RECORDED|41.5833|
|C_uni|93683|compose|0|0|NO_REPLAY|NO_REPLAY|77.6667|
|C_uni|93683|add73|39146|1024|20.2856|18.3263|36.5417|
|C_uni|93683|sub73|32824|1024|20.7866|18.6540|32.9167|
|C_uni|93683|max73|28338|1024|20.3155|18.1311|33.6667|
|C_uni|93683|min73|24635|1024|21.3964|18.8796|31.2500|
|C_share|93683|compose|0|0|NOT_RECORDED|NOT_RECORDED|77.6667|
|C_share|93683|add73|98304|1024|NOT_RECORDED|NOT_RECORDED|41.0833|
|C_share|93683|sub73|98304|1024|NOT_RECORDED|NOT_RECORDED|36.8750|
|C_share|93683|max73|98304|1024|NOT_RECORDED|NOT_RECORDED|41.9583|
|C_share|93683|min73|98304|1024|NOT_RECORDED|NOT_RECORDED|42.1250|
|C_uni|116089|compose|0|0|NO_REPLAY|NO_REPLAY|85.3750|
|C_uni|116089|add73|39622|1024|13.5884|11.2917|43.3333|
|C_uni|116089|sub73|32854|1024|14.3301|11.9742|38.6667|
|C_uni|116089|max73|28163|1024|14.4977|11.8808|36.5000|
|C_uni|116089|min73|24491|1024|14.8789|12.3270|37.2917|
|C_share|116089|compose|0|0|NOT_RECORDED|NOT_RECORDED|85.3750|
|C_share|116089|add73|98304|1024|NOT_RECORDED|NOT_RECORDED|49.5000|
|C_share|116089|sub73|98304|1024|NOT_RECORDED|NOT_RECORDED|44.6667|
|C_share|116089|max73|98304|1024|NOT_RECORDED|NOT_RECORDED|46.0000|
|C_share|116089|min73|98304|1024|NOT_RECORDED|NOT_RECORDED|45.4583|

Verdict: coverage and wrong-target measurements delivered for uniform; coverage reconstructed with counter corroboration for share, whose target error is NOT_RECORDED. This challenges a coverage-only explanation for E16W. It does not isolate whether wrong replay targets initiate forgetting, result from forgetting, or amplify it. E16 cell replay re-renders ground-truth answers, so these window-net error rates cannot be transferred to E16/D/D2. Full54-row accounting is in E16V_REPLAY_COVERAGE_ERRORS.csv; evidence is in E16V_REPLAY_AUDIT.json.

Read-only batch analysis cost: two file passes, 48.026 process-CPU seconds and 53.792 summed wall seconds, including the validation pass. Other reporting/Git overhead and peakRSS were not measured. No scientific training or inference was performed. E16V_REPLAY_AUDIT_HISTORY.json preserves both measurements.

## Scope of the next assigned part

E16-WR is a separate authorized compute task. It is NOT_RUN by this source-only verification. Earliest start16:30EST on8November2026, and only after process inspection confirms no exp16_integrated.py workers. The capacity test will use9 stage strata and model-generated replay targets, unlike E16-R's3 families and recomputed ground-truth targets. AlgorithmR retains duplicate presentations, so4,096 slots do not guarantee full operation coverage. No E16-R/E16-F worker was altered. This audit has not launched any training.
