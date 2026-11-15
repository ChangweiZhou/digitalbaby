# E31 stage 1b, agreement check (worker B, 14 November 2026)

**What was compared:** worker B's stage-1b dry receipts (`E31_DRY_1B_B.json`, 05cd5263) against worker A's (`claude/sleepy-knuth-q96dd1`, 1ea99869). Seed 73129, `E11_STAGE_UPDATES=256`. I read worker A's branch only after both sets were pushed.

| Arm | Worker A `tensors_sha256` | Worker B | λ, A | λ, B | By ruling 1 |
|---|---|---|---|---|---|
| Ks | bc853613cc4f1e7a | bc853613cc4f1e7a | — | — | **agree**: equal hashes |
| P2a | a06e0e1be09050e7 | 1d5ecdff9f0145d1 | 22.306798493 | 22.306799227 | **agree**: gradients within 1e-6 (below) |
| P2b | 4aee5256821832d1 | 508eb4a9608a7d81 | 4.4613596986 | 4.4613598454 | the same functions as P2a, other strength |
| P2c | d55d41df5823332a | 7b024e8ebe7d6858 | 0.89227193973 | 0.89227196907 | the same functions as P2a, other strength |

**Why the P2 hashes differ.**
- Worker A recomputes the gate with one recurrent product per step and takes the norms in float32.
- Worker B uses one product over all steps and takes the norms in double precision.
- The λ values differ by 3.3e-8 relative, and float rounding compounds into different hashes.

**Ruling 1's test** (`E31_1B_P2_CROSSRUN_B.json`): on the life's first batch at the E26 S birth weights of seed 73129, each worker's term function was run on the same forward states.

| Quantity | Result |
|---|---|
| Task loss | bit-equal |
| Term value | bit-equal (0.027786217629909515) |
| Term gradient, largest relative difference over the four update-gate tensors | 3.2e-7 |
| Total P2a gradient over every parameter, each worker's own λ | 5.1e-8 |

All are within the 1e-6 of ruling 1. **Stage 1b agrees, so stage 2 starts** (addendum 2).

**Worker B's stage-2 lives:** K, P2a and P2c, seeds 73129, 93683 and 116089. Ks is not run at full length (Mr. Xu, 03:51 UTC).
