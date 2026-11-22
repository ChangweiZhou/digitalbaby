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
