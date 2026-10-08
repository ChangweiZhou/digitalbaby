# Byte arithmetic evolution V2

Outcome: **No qualified mixed-arithmetic learner in this bounded search**.

Search: 1024 candidates × 128 generations, 2 new matched development worlds/generation, 16 sealed worlds.
World master seed: 2026092701. The inspected pilot uses a different master seed and contributes no worlds.
Every question was visible newline + ASCII expression ending '='. The learner received a scalar 0/−1…−5 consequence after each training first response; exact answers remained evaluator-only. Tests were read-only.
Development selection and extinction are not confirmation. Later-stage exploration cannot compensate for a failed earlier sealed gate.
Evidence level: E3 within this synthetic arithmetic family only if sealed novel-expression performance improves causally over the matched no-write life. No delay/intervening-learning probe was run, so this experiment cannot establish a persistent learning core.
Exposure audit: each training expression's answer affects only its own post-choice grade; canonical expressions and all spacing variants are excluded from the scored novel set. Every scored choice precedes feedback on that expression. The genotype is inherited; synaptic updates begin afresh in each independent life.
Recitation countermodel: exact and canonicalized expression-key tables have zero scored matches. Constant-zero and zero-shot arms expose output frequency or inherited-genotype shortcuts; the same-genome no-write arm controls sensory exposure and action sampling. A simple arithmetic feature rule learned during the life remains a legitimate E3 mechanism.

Family leaders in the final development generation:
- flat: exact 0.047, causal gain +0.047, q90 37, state 132612 bytes; survived False.
- recurrent: exact 0.055, causal gain +0.023, q90 38, state 52116 bytes; survived False.
- compartment: exact 0.070, causal gain +0.039, q90 42, state 52116 bytes; survived False.
- topology: exact 0.055, causal gain +0.023, q90 42, state 52116 bytes; survived False.

Survival in the final generation: flat 0/256, recurrent 0/256, compartment 0/256, topology 0/256.
Family extinction/reseed events over the search: flat 83, recurrent 82, compartment 81, topology 79.

## Stage 1 sealed confirmation

| Unique training questions/world | Learned exact | No-write exact | Zero-shot exact | q90 error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.077 | 0.077 | 0.077 | 56 |
| 8 | 0.060 | 0.077 | 0.077 | 56 |
| 32 | 0.032 | 0.077 | 0.077 | 55 |
| 128 | 0.016 | 0.077 | 0.077 | 51 |
| 512 | 0.013 | 0.077 | 0.077 | 44 |

Per-signature exact: * 0.003, + 0.007, - 0.014, / 0.028.
Controls: constant-zero 0.077; NON-FLY symbolic built-in parser 1.000; exact-expression cache has zero novel matches.
Acceptance: exact=False, causal_gain_over_no_write_and_zero_shot=False, q90=False, all_signatures=False.

## Stage 2 sealed confirmation

| Unique training questions/world | Learned exact | No-write exact | Zero-shot exact | q90 error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.045 | 0.045 | 0.045 | 32 |
| 8 | 0.048 | 0.045 | 0.045 | 32 |
| 32 | 0.050 | 0.045 | 0.045 | 32 |
| 128 | 0.059 | 0.045 | 0.045 | 31 |
| 512 | 0.035 | 0.045 | 0.045 | 27 |

Per-signature exact: *+ 0.021, *- 0.000, */ 0.031, +* 0.042, +- 0.062, +/ 0.062, -* 0.010, -+ 0.094, -/ 0.021, /* 0.021, /+ 0.021, /- 0.031.
Controls: constant-zero 0.045; NON-FLY symbolic built-in parser 1.000; exact-expression cache has zero novel matches.
Acceptance: exact=False, causal_gain_over_no_write_and_zero_shot=False, q90=False.

## Stage 3 sealed confirmation

| Unique training questions/world | Learned exact | No-write exact | Zero-shot exact | q90 error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.032 | 0.032 | 0.032 | 44 |
| 8 | 0.030 | 0.032 | 0.032 | 44 |
| 32 | 0.029 | 0.032 | 0.032 | 44 |
| 128 | 0.040 | 0.032 | 0.032 | 42 |
| 512 | 0.031 | 0.032 | 0.032 | 41 |

Per-signature exact: *+-/ 0.021, *+/- 0.021, *-+/ 0.000, *-/+ 0.042, */+- 0.042, */-+ 0.062, +*-/ 0.000, +*/- 0.062, +-*/ 0.083, +-/* 0.021, +/*- 0.042, +/-* 0.021, -*+/ 0.000, -*/+ 0.021, -+*/ 0.062, -+/* 0.021, -/*+ 0.042, -/+* 0.021, /*+- 0.021, /*-+ 0.042, /+*- 0.021, /+-* 0.042, /-*+ 0.021, /-+* 0.021.
Controls: constant-zero 0.032; NON-FLY symbolic built-in parser 1.000; exact-expression cache has zero novel matches.
Acceptance: exact=False, causal_gain_over_no_write_and_zero_shot=False, q90=False, all_signatures=False.

The output is one integer selected from −128…128 and serialized by a wrapper. Autonomous output bytes, natural-world arithmetic, and 24-hour persistence remain untested. A positive E3 gate does not establish biological identity or broad NLP competence.
