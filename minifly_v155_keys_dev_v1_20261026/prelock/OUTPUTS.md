PRE-LOCK SYNTHETIC CHECKS (no text; not outcomes; run unlaunched before LOCK.json, about 15 CPU-minutes in total)

# Pre-lock synthetic checks of the V155 references (26 October 2026)
These scripts ran on synthetic receptor traces only, before the lock. They read their birth keys from the scratchpad copy made from the native model. They motivated design §12.9. Outputs are copied from the session log (held-out or calibration class bits; lower is better; 20 classes, uniform = 4.32).

## 1. Code parity (synthetic traces, 500 positions)
`codes.codes` and `codes.code_one` against `FastMemory.encode`: 0 mismatches each; mean 183.6 active KCs.

## 2. Teacher task: targets from a fixed random readout of the code of hidden teacher keys
Teacher keys use the same wiring with the birth weights permuted (native case), or new wiring (dense case).

| Set-up (native wiring unless stated) | Calibration bits, last epoch | Code overlap with teacher |
|---|---:|---:|
| Student initialised at the teacher keys, keys frozen (20 epochs) | 2.034 | 1.000 |
| Birth keys, frozen | 2.554 best (2.633 last) | 0.189 |
| Learned keys, ReLU stand-in or straight-through sigmoid, Adam key lr 1e-2 / 3e-2 (20 epochs) | 2.438 best | 0.194–0.198 |
| Learned keys, ReLU stand-in, Adam lr 1e-3 (10 epochs) | 2.796 (frozen 2.807) | 0.195 |
| Learned keys, SGD momentum 0.9, lr 0.1 (10 epochs) | 2.900 (frozen 2.903) | 0.190 |
| Dense learned keys, Adam lr 1e-3 (10 epochs) | 3.823 (frozen 2.969) | 0.038 |
| Free graded hidden layer, native mask (8 epochs, held-out) | 2.681 | – |
| Free graded hidden layer, dense (8 epochs, held-out) | 2.674 | – |

Reading:
* Back-propagation through the binary top-k layer recovers at most about 0.1 of the 0.5 bits of known headroom (2.554 → 2.438, against 2.034 with the teacher keys).
* Free graded layers train, but on this task they do worse than the frozen binary code.
* Neither is a sound gate, so the references became descriptive (design §12.9).
