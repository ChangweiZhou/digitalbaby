# NET13 plan (written 23 Nov 2026, before any full-size run of net13)

Branch `claude/newnet-net13-sei232` from `claude/magical-albattani-sei232` @ 32db8c13. Development level, one machine; no gate, seal or
outside review is claimed. New files only: `net13.py`, `check_net13.py`, `net13_runs/`, this plan, `NET13_REPORT.md`.

## Task 1 - consolidation
net13 = net10 (K8) + lag cells W=8 + direct readout + gated relay + persistent rule cells (rc=64, rc_k=3, rc_tau=8, rc_vig=0.4,
rc_eta=0.02, added to the relay gate as rc_gate=1). No other option of net12 / net12_rc.
Check: with seed 1 net13 must reproduce net12_rc's full-size t23 result (cue=1 seed 1: succ at end 0.82, bits 777,024, text
0.35 -> 0.24; rc_runs/full_res.jsonl, run A1). Short check done before this plan: `check_net13.py 300` (3,600 bytes, net13 and net12_rc
in separate processes): max |difference of summed outputs| = 0, 0 answers differing.

## Task 2 - rc_all=1: the rule cells gate every readout
Each output compartment c (16 context compartments, 16 holders' compartments, the direct readout; 33 in all) is summed with a gain
g_c = 1 + sum_r Gc[c, r] s_r, s the rule cells' normalised activity (the same s that sets the relay gate). Gc learns by the relay
gate's rule with the compartment in place of a lag line: Gc[c, r] += eta_r / ncomp * (e . O_c) * s_r (e the shared mismatch, O_c the
compartment's output). Gc starts at 0 (g = 1: rc_all=1 starts as rc_all=0). The compartments' own strengths keep their rules.
No new parameter: eta_r, ncomp and the rule cells are those of the relay.

Idea: the text is lost because the rule phases rewrite the compartments the text uses; if the rule cells (which separate the text
from each rule stream) can turn a compartment's gain down where it does harm and up where it helps, the text's readout is less
overwritten in effect.

Runs: t23_life_rules.py net=13 struct=K8 cue=1 seed=1, with rc_all=0 and rc_all=1 (about 27 min each, in parallel); seed 2 for both
only if rc_all=1 passes at seed 1.

**Prediction** (rc_all=1, seed 1): text probe at the end >= 0.30 (after text ~0.35; i.e. clearly above the ~69% kept), with succ at
the end not worse than 0.82, copy/rev at the end not worse than 1.00 (allowing 0.01 for probe noise: >= 0.81 and >= 0.99), and total
bits not worse than 777,024.
**Falsifier:** text probe at the end <= 0.26.
Between 0.26 and 0.30, or text met but another criterion missed: partial, reported as such.
