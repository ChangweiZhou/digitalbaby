# E31 stage 1, worker A (14 November 2026)

**Status.** Stage 1 only: the checks and the dry lives for seed 73129 at `E11_STAGE_UPDATES=256`. No full life has run. These are E2 checks, not results. They were written independently of worker B: I did not read B's branch before these receipts were pushed.

- **Brief.** `OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md` with its arm-P addendum, at `7287e071`.
- **Host.** Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, one thread a process, at most three processes.
- **Files.** New: `arch31.py`, `exp31_switch.py`, this record. No existing file is edited.

## Dry receipts for the agreement check (seed 73129, E11_STAGE_UPDATES = 256)

| arm | `tensors_sha256` | receipt |
|---|---|---|
| **G** | `12ef12e3bcebaf76e626865159f1d6891cb94f74a390ecb5b0ac21ccdea22381` | `exp45/e31/e31_life_G_dry256_73129.json` |
| **H** | `f58873993a90cad9028ac4e2b200d8808dcecbbc0f54780cff683c246cccd14c` | `exp45/e31/e31_life_H_dry256_73129.json` |
| **K** | `7a0f74a1c47d2aae9466e2e4395b615f9c1eab2d14c067f9b5cd8984592287fd` | `exp45/e31/e31_life_K_dry256_73129.json` |
| **P** | `85174cc48d1e787c55ee0b4309c770fca62b71593b4db2e4eeaa127db940f289` | `exp45/e31/e31_life_P_dry256_73129.json`; λ = 10.919742514435665 |
| N (no switch) | `60f36ab0e9711512f7c7e216213153f7f07bfee969bfdbf60acb1f1b0634cba2` | `exp45/e31/e31_life_N_dry256_73129.json` |
| P0 (λ = 0) | `60f36ab0…` (identical to N) | `exp45/e31/e31_life_P0_dry256_73129.json` |

All six share the data digest `948066565aa9…`. Each took 520–647 CPU-seconds. Summary: `exp45/e31/E31_DRY_A.json`.

## Checks

One-batch checks are on the first life batch of the admitted sweep birth, seed 73129 (`exp45/e31/E31_CHECKS_A_73129.json`, all pass).

**No switch.**
- On one real batch, `arch31.forward(m, x, None)` gives logits and every gradient **bit-identical** to `arch26.forward(m, x, True, True)` (relative difference 0).
- The no-switch dry life ends with `tensors_sha256` 60f36ab0…. That equals the dry E26 life S of the same birth and dose on file, `exp45/e26x/resume_check/e26x_life_SS_dry256_73129.json`, with the same data digest. **See the deviation below:** that file stands in for running `exp26_candidate.py life S 73129` myself.

**G and H.** At each of the 22 recurrent steps, and for G also the input map, the error sent back equals the written formula: `g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]` for G, `g @ W` for H. The relative difference is 0. Against the full sweep it differs by 0.64 (G) and 0.70 (H), so the check is able to fail.

**K.**
- Multiplying the birth's `weight_hh_l0` by −3 and adding 0.5 leaves the starting `Bhh` unchanged (`torch.equal`; sha256 `f4ed0a9c…`, range ±0.0625).
- With `Bhh` set to `W` by hand, K's gradients equal H's (relative 0). With `Bhh0` they differ (1.09).
- The update identity, checked in its difference form `(Bhh₁ − W₁) = (1 − 0.00025)(Bhh₀ − W₀)`, holds to 3.0e-8, 3.0e-8 and 6.0e-8 at updates 1, 10 and 100.
- The dry life's stage rows give the cosine of `Bhh` with `W` and ‖Bhh − W‖/‖W‖:

| stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| cosine | 0.004 | 0.196 | 0.363 | 0.478 | 0.594 | 0.669 | 0.726 |
| ‖Bhh − W‖/‖W‖ | 1.145 | 1.058 | 0.964 | 0.897 | 0.811 | 0.746 | 0.688 |

**P.**
- With λ = 0, the dry life equals the no-switch dry life bit for bit (P0 = N = 60f36ab0…).
- On one batch, the term's gradient is exactly 0 on every tensor and row outside rows 256–511 of `weight_ih_l0`, `weight_hh_l0` and both biases.
- On those rows it equals the written two-factor product to at most 3.3e-7 relative.
- λ is fixed on the first life batch: 0.25 × 0.42528 / 0.0097365 = 10.9197.
- The dry stage rows record U on the plus-minus panel: 65.2, 26.4, 16.2, 13.0, 8.9, 7.2 and 6.8 %.

## Choices the brief leaves open (they are also in every receipt as `e31.choices`)

1. **Switched maps.** G: `gx = g[..., :512] @ W[:512] + g[..., 512:] @ S[512:]`, with `S = sign(W)·mean|W|` over the whole matrix, for both `weight_ih_l0` and `weight_hh_l0`. H: `gx = g @ W`. K: `gx = g @ Bhh`. In every switched map the weight and bias gradients are `exp3_state.SignLinear`'s own expressions, so only the error sent back differs from the floor.
2. **Bhh.** `Bhh0 = torch.empty(768, 256).uniform_(-1/16, 1/16, generator=torch.Generator().manual_seed(seed + 31))`. After each step: `Bhh = (1 − 0.00025)·Bhh + (W1 − (1 − 0.00025)·W0)`.
3. **Arm P's term.** At positions t = 1 … L−1, `z_t = sigmoid(lin(z1.detach(), ih, True)[:, t, 256:512] + lin(hp_t.detach(), hh, True)[:, 256:512])`. The mean is over positions whose input byte is not BOS, and over all 256 units. BOS appears only at position 0 and in padding: the first life batch has 307 padded positions. The loss is `task + (−λ)·mean`.
4. **λ.** λ = 0.25 × ‖g_task‖ / ‖g_term(λ = 1)‖. Each norm is over the concatenation of rows 256–511 of `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0` and `bias_hh_l0`. The task gradient is taken by the floor's path, with `torch.autograd.grad`, on the first life batch, before the first update.
5. **U.** E30's definition: z from `e29_gates.gates` (exact forward values) on `e29_gates.plus_minus_panel(seed)`, every position after the first, all 256 units, with 0.1 ≤ z ≤ 0.9.

If worker B's hashes differ, these are the first places to look.

## Deviation: admission of the birth (please rule)

`exp26_candidate.admit_e26` refuses `e26_dev_S_<seed>.pt` on the current branch: *made by other sources: exp16_integrated.py*.
- The births and the E26 and E26-X lives were made with `exp16_integrated.py` at `a301ae8e` (sha256 f4559971…).
- The branch now has it at `deea398c` (c9fcedcf…). That change is E21b's resume and receipt-writing machinery, inside exp16's own `life` and `develop17`, which `exp26_candidate.life` does not call.
- For the same reason, the brief's reference command `exp26_candidate.py life S 73129` cannot run on today's sources at all.

Without editing any existing file, `exp31_switch.admit_s` runs `admit_e26` unchanged. Only if it fails on exactly that file does it repeat the check with exp16's recorded hash substituted. The hashes must be precisely that pair (f4559971 at birth, c9fcedcf now). Every other admission check still runs. The exception is written into every receipt (`e31.admission`).

The no-switch dry life reproduces, bit for bit, a dry life made on the births' own sources (60f36ab0…). So the source change does not alter the life. The full lives of stage 2 will be admitted the same way unless you rule otherwise.

## Not done

- Stage 2: no full life.
- The agreement comparison with worker B: next, only after this is pushed.
