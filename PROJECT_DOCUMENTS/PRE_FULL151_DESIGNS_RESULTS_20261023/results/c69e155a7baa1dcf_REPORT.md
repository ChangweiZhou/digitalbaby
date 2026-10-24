# Can the fly's own route keep byte order and learn from it? — report

**Date:** 25 September 2026.
**Design:** `DESIGN.md`, fixed before the run; its hash is in `RESULTS.json`.
**Scale:** 3 fresh worlds × 5 arms, 6.5 minutes.

## Verdict

**No, not yet.** Neither fly-route arm passes V9's ORDER gate in any world:
- FE0 → 88 PN → native PN→KC → all 5,177 KCs → plastic outputs;
- the same route with TRACE88 as the front end.

V9's engineered route passes in all three worlds, so the task, scorer and gate work.

**Where order is lost, from the diagnostic:**
- **The input organ keeps the most recent byte but largely loses the byte before it.** That is the order information.
- **The online learning rule uses only part of what the KCs do carry.**

## Gate results

Chance macro accuracy is 1/16 = 0.0625. The target-slot last-byte Markov baseline scores 4.00 bits, and the gate needs ≤ 3.00 bits.

| Arm | World | Target bits | Macro accuracy | Swap penalty (bits) | Gate |
|---|---|---:|---:|---:|---|
| FLY_FE0 | 201 / 202 / 203 | 4.24 / 4.23 / 4.24 | 0.181 / 0.188 / 0.178 | +0.04 / +0.05 / +0.04 | fail ×3 |
| FLY_TRACE88 | 201 / 202 / 203 | 4.73 / 4.73 / 4.73 | 0.110 / 0.106 / 0.104 | +0.18 / +0.17 / +0.18 | fail ×3 |
| Plasticity-off twins | all | 8.00 | 0.000 | 0 | — |
| V9_POSITIVE (engineered pair cells) | 201 / 202 / 203 | 0.19 / 0.20 / 0.19 | 1.000 ×3 | +8.2 / +8.1 / +8.0 | **pass ×3** |

**Which gate conditions each fly arm meets:**
- **Both arms:**
  - beat their plasticity-off twin;
  - fail accuracy ≥ 0.65 with ≤ 2 bits;
  - fail the requirement of 1 bit better than the last-byte baseline.
- **Swap penalty:** TRACE88 meets the ≥ 0.1-bit requirement, so it does use some order. FE0 does not.

**Both fly arms score worse than 4.00 bits** even though they are above chance on accuracy. Bits above 4.00 are worse than a uniform guess over the 16 possible targets: the online output still spreads probability onto bytes that never occur there.

## Diagnostic: where order is lost (world 201)

Offline linear decoders were trained on 16,384 training target slots and tested on the 4,096 held-out ones. Accuracy is 16-way, with chance 0.0625.

| Decoded from | FE0 | TRACE88 |
|---|---:|---:|
| PN (88): second symbol b, the most recent byte | 0.867 | 0.997 |
| PN (88): first symbol a, the byte before | **0.215** | **0.347** |
| KC (5,177): b | 0.818 | 0.965 |
| KC (5,177): a | 0.380 | 0.456 |
| KC (5,177): target y, which needs a and b in order | **0.483** | **0.469** |
| *Online arm, target (from the table above)* | *0.18* | *0.11* |
| NC: y with shuffled labels, from KC | 0.066 | 0.061 |
| PC: a and b from V9's engineered cells | 1.00 / 1.00 | 1.00 / 1.00 |

1. **The input organ is the main bottleneck.** At PN level, the most recent byte is well represented, but the byte before it much less so (FE0 0.215, TRACE88 0.347). The information that distinguishes (a,b) from (b,a) is therefore mostly already gone when the signal reaches the fly's KCs.
2. **The native KC expansion helps rather than hurts.** a is more decodable from KCs than from PNs (0.380 vs 0.215, and 0.456 vs 0.347).
3. **The online rule is a second limit.** An ideal linear readout of the same KCs reaches 0.47–0.48 on the target, while V9's online rule, untuned, reaches only 0.11–0.18. Even the ideal readout stays below the gate's 0.65. Fixing the learning rule alone would not be enough.

## Checks

All pass:
- scoring never changed parameters, and every target was scored before it was seen (V9's scorer assertions);
- training and test coverage were exact;
- the fly-route inputs are exactly the 5,177 native KCs;
- the plasticity-off twins stay at zero.

## Limits

- This covers two-byte order only (the ORDER task), with V9's learning rate unchanged and a single training pass.
- The diagnostic ran on one world, and was fixed in advance to run only for failing arms.
- Identity, natural text and teaching were out of scope, as designed.

## Files

- `DESIGN.md`
- `order_flyroute.py` (6.5 min: `OPENBLAS_NUM_THREADS=1 <V88 venv python> order_flyroute.py`)
- `RESULTS.json`
- `run.log`
- this report
