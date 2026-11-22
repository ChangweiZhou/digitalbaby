# Plan: net8's step in C (net8c), 22 November 2026

Written before the code. An engineering change only: no new learner, no new result claim. Development level; no gate, seal or
outside review is claimed.

**Source.** `NEWNET/net8.py`, `net9.py`, `net7.c`/`net7.py`/`check_net7.py` on `claude/magical-albattani-sei232` at `dbc8f661`;
work on `claude/newnet-slowc-sei232`. Requested by the NEWNET lead session (owner's line of work): make the slow stage fast
without changing what it computes.

## What is ported

`net8c.Tissue` is a subclass of `net8.Tissue`: the constructor is net8's own (same arguments, same random draws, so the same
strengths at birth and all the same attributes). Only `step` and `_learn` are replaced, and only when the configuration is the one
net9 uses:

- `fb=none`, `gran > 0`, `k=0`, `rec=0`, `send_y=0`, `norm_rec=0` (any `tau_max`, `center`, `out_var`, `eta_o`, `rule`, `n_values`,
  `fbm=teach|yesno`, learn on/off).

Any other setting: the numpy step of net8 is used unchanged (a flag `self.c` says which).

In C (`net8c.c`, built with `gcc -O3 -march=native -shared -fPIC` like net7), per byte:
1. learning from the last answer (when learn and a byte was read before): mismatch e (teach, gated rule, or yes/no on `gave`),
   the running variance `vy`, the step of the output strengths of the active expansion cells `A[:, act] += eta_o/|act| e`;
2. the tissue: candidate and gate from the present byte's line, the F sources' states one byte before, the constants (and `Wd`
   of the byte before); new state h; running mean m and adapted output y;
3. the expansion: input of each granule cell from its g_fan tissue cells (y scaled by sqrt(vy)) and the byte's line; the g_k
   share with the largest input fire (z); O = A z over the firing cells.

Kept in Python: the bookkeeping net8 keeps (`yprev`, `yprev_byte`, `hprev_byte`, `t`, `prev_byte`, `byte`). The arrays that
net9's `get_state`/`set_state` swap (h, y, z, O, m, ...) are read from the object's attributes at every call, so a state put back
by rehearsal is used as numpy would use it.

**Not maintained in C: the traces `E`.** With `fb=none` and an expansion, net8 updates them but never reads them (its `_learn`
returns before any use). net8c leaves them at zero. Nothing that net8 or net9 outputs depends on them. Said here so that no one
reads `E` from a net8c tissue.

## How it is checked (`check_net8c.py`)

net8 (numpy) and net8c side by side, same seed, same bytes, at every step:
- the answer (argmax of O) must be identical;
- h, y, m, z, O, vy and A must agree to within 1e-9 (max absolute difference reported; "exact" said only if every one is 0).

Cases (N=1024, gran=8192, tau_max=150, center=1000, out_var=1000, eta_o=0.1, fb=none): tinyshakespeare and the context stream
(t13's A stream), 3,000 bytes each, in teach and in yes/no (gave = the answer, as net9 with eps=0 would give), n_values 32, 64,
128, with learn on; and a learn-off stretch after learning. A case passes only if all steps agree.

Then net9 gets `slow_c=1` (default 0: nothing changes) and `t13_life8.py net=9 f_mode=order eta_f=0.2 f_M=16 life=ctx` is run
with and without it: the printed numbers must be the same. A difference would be reported, not hidden.

**Timing.** ms a byte of the slow stage alone, numpy against C, learn on, OMP_NUM_THREADS=1; and the wall time of the t13 run
with and without slow_c.

## Not done

No other net8 option is ported; no other file than `net9.py` (the option only) is edited; no experiment is run beyond the checks.

## Outcome (entered 22 November 2026, after the runs; code at `b11761b1`)

**Check (`OMP_NUM_THREADS=1 python3 check_net8c.py`, 3,000 bytes + 7 rehearsal swaps of 40 bytes + 500 bytes learning off).**
All 12 cases (text and context stream x teach and yes/no x n_values 32, 64, 128): answers identical at every step. States not
exactly equal: largest |diff| over all steps and cases h 2.2e-15, y 2.2e-15, m 8.9e-16, O 1.3e-15, vy 7.6e-17, A 2.1e-17, z 0
(the same granule cells fired at every step). All within 1e-9. The fallback (fb=kp, no expansion) uses the numpy step.

**t13 (`t13_life8.py net=9 f_mode=order eta_f=0.2 f_M=16 life=ctx`), with and without `slow_c=1`: the same printed numbers.**

| run | tests A / B after A | after B | after the return | wall, numpy slow stage | wall, slow_c=1 |
|---|---|---|---|---|---|
| teach (the required test) | 1.000 / 0.017 | 1.000 / 0.343 | 1.000 / 0.340 | 114 s | 81 s |
| fbm=yesno (eps 0.1) | 0.000 / 0.009 | 0.000 / 0.215 | 0.000 / 0.214 | 113 s | 80 s |
| teach, rho=0.05 (rehearsal: get/set_state) | 1.000 / 0.018 | 1.000 / 0.343 | 1.000 / 0.339 | 104 s | 71 s |

(the first two pairs ran 4 at a time on 4 cores, the third 2 at a time; wall times are indicative only.)

**Speed of the slow stage alone** (V=128, N=1024, gran=8192, tinyshakespeare, 2,000 bytes, one process, OMP_NUM_THREADS=1):

| | teach, learning | teach, off | yes/no, learning | yes/no, off |
|---|---|---|---|---|
| net8 (numpy) | 1.385 ms | 0.770 ms | 1.522 ms | 0.701 ms |
| net8c | 0.180 ms | 0.158 ms | 0.185 ms | 0.156 ms |

About 7.5-8x with learning, 4.5x without. In net9 the whole t13 run is about 30% faster (114 s to 81 s): the rest of its time is the fast
stage (net7) and Python, not touched here.

**Left in numpy:** every other option of net8 (fb=kp/random/exact, rec, k, norm_rec, send_y, gran=0) and the traces E (not
kept by net8c; never read in this configuration).
