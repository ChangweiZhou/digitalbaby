# V177 three-cycle exercise

**Outcome: REFERENCE_NOT_QUALIFIED.** Both same-graph BPTT references failed the frozen feasibility thresholds. BS/BR candidate training was therefore not executed. This is not evidence that local credit failed.

| World | Held bits/token | Gain vs UNI | Accuracy | Exact8 | Post-newline EOS |
|---|---:|---:|---:|---:|---:|
|202611027711|2.9174|0.1724|21.13%|0.00%|0.00%|
|202611027712|3.0326|0.0573|14.54%|0.00%|40.62%|

Required in each world: gain≥0.75 bits/target and unmasked single-next-byte accuracy≥35%. Exact-eight own continuation was0/704 in both. Newline prediction accuracy was0% in both; early EOS was0.30% and0%, so this is not the immediate-EOS collapse seen in V176. EOS-after-newline performance still raises the predeclared warning.

Cycle1 verified formulas, topology and eligibility; cycle2 ran the four fixed five-epoch pilots; cycle3 executed both frozen ten-epoch BP references, registered frozen-state diagnostics and the stopping/revision decision. This is three engineering design/audit/test/revise passes, not three independent confirmations.

The isolated-module BS gradient matched BPTT to relative error3.69e-16; connected BS differed by27.5%, as expected. Whole-conditional-roster forecast251.5 CPU seconds including1.5× and reserve passed the7,200-second cap. Learning stayed within2GiB checks; actual byte/target counts and startup uncertainty are disclosed. No GPU was used.

Actual learner fitting:640 pilot +1,280 main BP +24 technical records =1,944 records/188,568 target applications. Training-only counting references additionally examined128 unique records with three countermodels (384 model-record applications/37,248 reference updates); these are separate reference costs. Numerical/gradient and read-only inference/rollout work is separately recorded in DIAGNOSTICS.json and CONTRACT_CHECKS.json. No candidate weights or BP checkpoint/moments were transferred to a local learner.

The trained graph depends on recurrent/cross-module pathways under the registered lesions, but lesions also shift state distributions and do not prove a clean lag representation. BS raw hidden-gradient cosine at BP checkpoints was0.803/0.943; BR was−0.465/0.129. High applied-step cosines use shared BP optimizer history in frozen shadow computations and cannot be treated as BR training alignment or success.

W1 validation improved gradually through epoch10; W2 remained near the marginal baseline. The fixed dose has not established a trainable ceiling. This experiment cannot distinguish inadequate optimization/data dose from limitations of the chosen graph; do not declare recurrence or local learning generally incapable.

**Audit limitations:** source/data were locked prospectively before main fitting, but the pilot source snapshot was late. A timing-only C1 forward used main held data with an untrained pilot birth and discarded scores; no parameter/configuration was chosen from them. These limitations are preserved in FINAL_AUDIT.json rather than presenting this as sealed independent confirmation.

Original design, gates and failed main results remain unchanged. See REVISED_DESIGN.md for documentation corrections and the separately prospective feasibility question. No arithmetic, persistent A→B→A learning or native Full151 improvement is claimed.
