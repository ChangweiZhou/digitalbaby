# R_center physical shared-bank deletion

**Decision: DO NOT ADOPT: one or more prespecified reliability gates failed.**

Complete: 96 paired worlds, 192 world-arm receipts, 768 lives. Three design/audit/trial/revision cycles preceded science.

| Final taught stratum | Eight-store accuracy | Four-store accuracy | Four-store minus no-stage-write |
|---|---:|---:|---:|
| old | 94.57% | 94.57% | +70.01 pp |
| new | 94.69% | 94.69% | +70.51 pp |
| revision | 40.89% | 41.28% | +39.97 pp |

The table uses all scored cues and is descriptive. Primary decisions use the six exact bounds in SUMMARY.json; all three noninferiority and all three absolute-accuracy gates must pass.

CONTENT_4 physically has four independently born stores and no shared bank. The eight-store parent was not modified. All saved private states and values match between arms on every paired branch/checkpoint.

Claims are E1 only. Every scored association was taught; an exact-key table can solve it. No E3, autonomous output, teacher-free learning, or long-context claim is made. State count falls from eight to four; measured runtime/RSS do not necessarily fall by half.

| Gate | Loss upper bound | Accuracy lower bound | NI / absolute pass |
|---|---:|---:|---|
| old | 4.86% | 85.22% | True / False |
| new | 4.86% | 81.13% | True / False |
| revision | 4.86% | 27.80% | True / False |
