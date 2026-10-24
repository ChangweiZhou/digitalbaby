# Step 4 result: exact-support arbitration is inactive at ordinary byte timing

The complete unlabeled E0 census finished and its receipt arithmetic and
source hashes passed a separate audit. Among all **325** unordered pairs of
distinct uppercase ASCII bytes, swapped in the inherited 12-byte cue frame
at the inherited **30/14 s** spacing, the FE0 KC sparse supports differed in
**325/325** cases. Mean FE0 support Hamming distance was **14.29** KCs;
minimum was **2**. The temporal bridge produced a mean distance of **80.97**;
minimum was **14**.

Therefore a router that invokes the temporal Full151 branch only when the
two FE0 candidate codes have *identical sparse supports* would never invoke
it on this complete ordinary-timed roster. This does not prove FE0 can use
every difference for learning, nor that the temporal differences carry the
right value relation. It does rule out that specific exact-support gate as a
useful next behavioral test here. Fourteen pairs have FE0 distance at most 4,
and 37 at most 6, but these small nonzero distances are **not** licensed
thresholds: switching on them could override a useful FE0 value signal.

No labels, teaching, learned-value outcomes or answer choices were used; no
behavioral science run was launched. The next narrow test should first ask
whether the protected temporal branch can itself learn a prespecified
order-dependent value relation under native teaching, alongside the FE0
control and a matched no-write branch. Only if that is positive should a
task-independent arbitration rule be frozen and tested on fresh worlds.

See [DESIGN.md](DESIGN.md), [results/RESULTS.json](results/RESULTS.json), and
[census.py](census.py). The audit checks receipt arithmetic and hashes; it
does not independently recalculate KC activations.
