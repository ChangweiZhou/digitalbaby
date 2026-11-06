# Proposed V223 — three engineering cycles

**2026-11-05 · Engineering complete.** The reference can acquire the repeated records on the continuous byte interface. The result is from one development world and eight taught records.

1. **Interface and reference.** The archived October 17 harness reached 100% taught and 5% held on world 202610143901, birth seed 0, after 16,384 presentations. Its original BEGIN/REPLY and response-only protocol is separate from the new continuous stream. The new dense cell matched PyTorch's GRUCell to 1.11e-16. The byte/EOS fixture checked response weights, genuine EOS, and rejection of premature EOS. The paired sparse masks have the same trainable counts.
2. **Credit parity.** Five float64 fixtures cover dense, sparse connected, isolated and inherited KC routes. Maximum forward versus reverse gradient error was 2.66e-15, directional finite-difference error 3.82e-09, and maximum state or parameter discrepancy after three updates 1.32e-14.
3. **Acquisition and resume.** Each BPTT arm saw the same 27,392 symbols through 256 presentations per record, with continuous state, 4× response weighting, and own answer plus EOS scoring. Checkpoint continuation was bit-identical for all five BPTT arms and the separate exact-forward KC fixture.

| Arm at 256 presentations per record | Taught arithmetic | Taught facts |
|---|---:|---:|
| dense64 | 4/4 | 4/4 |
| connected32 | 4/4 | 4/4 |
| connected64 | 4/4 | 4/4 |
| isolated32 | 4/4 | 4/4 |
| kc32 | 0/4 | 2/4 |

The direct-input arms acquired all eight records, including the isolated GRU. This fixture therefore establishes a learnable interface and cell, with no observed final connectivity advantage between the matched width-32 sparse arms. The matched KC32 arm acquired fewer records on this seed. The full three-world acquisition screen, held arithmetic, and continuous B/A′ lifetime remain pending. Exact forward credit has numerical and resume evidence here. The 256-presentation acquisition table uses reverse-mode diagnostics.

| Profile | Forward symbols/CPU second | BPTT symbols/CPU second |
|---|---:|---:|
| connected 32 | 8,346 | 6,629 |
| connected 64 | 3,996 | 6,828 |
| dense 64 | 2,202 | 6,153 |

The profiles are single-thread measurements over 256 training symbols per arm. The engineering process peaked at 312 MiB RSS, including Torch and the float64 parity fixtures. Use a sustained selected-graph profile to allocate the main roster. Projected credit remains a design option.

Run with NumPy 2.4.0, SciPy 1.16.3, and Torch 2.9.1. Keep the V221 engineering directory alongside this folder as `minifly-reentry-credit-design` for the sensor assets and world generator.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -B qualify.py
python -B historical/diag.py --world 202610143901 --seed 0 --regime online --steps 16384 --log-every 4096 --save HISTORICAL.json
```

The receipts include raw native outputs at each checkpoint, exposure hashes, exact gradient errors, and timing samples. The archived harness is retained separately. [Historical source](https://github.com/ChangweiZhou/digitalbaby/blob/9dfd12e2f7dfedce679e5a0732e9c1ac95478374/addition_difficulty_diagnosis_dev_20261017/diag.py)
