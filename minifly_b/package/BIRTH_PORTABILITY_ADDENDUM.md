# V2 birth-portability correction — applies to both packages

The original remote packages are superseded. Both Muse and Claude Code reported
that a raw Full151 newborn built on Linux has the prescribed Q digest but a
different PN→KC `B` data digest. The source-of-record Mac, using Python 3.11.5,
NumPy 2.2.6, SciPy 1.14.1 and Numba 0.61.2, **reproduces** the original B
digest `32a3726c…` with seed `3287462915`; the recorded digest is not stale.
The birth formula combines seeded Gaussian jitter with a numeric exponential,
whose exact output may differ across platforms. The precise remote divergence
is not yet proved; preserve the remote `raw_B_sha256` as forensic evidence.

`FULL151_CANONICAL_B.npz` holds the source-of-record newborn CSR matrix as
`data`, `indices`, `indptr`, and `shape`. It has **no learned state** and comes
from no science world. `FULL151_BIRTH_FINGERPRINT.json` records Q and 30 other
newborn arrays/scalars. Both files are covered by the zip checksum manifest.

## Required remote gate

1. Discard the old zip and extract the V2 zip into a fresh directory. Install
   `requirements.txt`, then run `python verify_bundle.py`. It checks every
   file, constructs a raw newborn, demands exact equality for all **non-B**
   birth arrays/scalars and the B CSR topology, installs the canonical B
   weights, checks the original B/Q digests, and builds the technical fixture.
   It runs **zero** trajectories.
2. Save the verifier JSON, including `raw_B_sha256` and `raw_B_match`, in the
   package's technical audit. If any non-B field or B topology differs, **stop**
   and report the field; do not weaken or bypass the check.
3. In the new package-local runner, instantiate **every** Full151 store via
   `portable_birth.canonical_fresh_native()`. It returns `(model,
   raw_b_digest)`. Do this before any FE event, teaching, topology/weight edit,
   or clone. Log both raw and canonical digests and verify that all four stores
   in every candidate/control/branch start with canonical B. A candidate may
   then change B only by its predeclared P/T/S mechanism. The fixed historical
   `REFERENCE_SOURCE` files must not be edited.
4. Re-run all technical qualification with the canonical birth policy, and
   include `portable_birth.py`, the anchor files and this addendum in the new
   `SOURCE_LOCK.json`. Only after that may the new science roster start. No
   previously committed science exists in either package.

The historical `bytecore.compute_source_lock()` and `t2_graph.build_pair()`
construct an unanchored raw newborn and can still fail on Linux. They are
reference implementations, not the package-local gate. Port their required
hash checks into the new runner **after** applying the canonical birth anchor;
do not delete the hash check or suppress the failure. Every graph donor and
static/random control must derive from the same canonical B. The new lock must
record both the parent source hashes and the anchor-file hashes.

Using each remote host's different raw B as a separate baseline is **not**
accepted. Pairing removes some sampling noise, but a mechanism's effect can
depend on the initial weights. This matters especially for P/T/S, which
directly edit the PN→KC substrate. The shared 26-contrast family assumes the
same defined birth state and fixture across both packages.

This anchor fixes **birth bytes**, not every later floating-point operation.
Report remote hardware, library versions, the raw B digest, technical-life
receipt hashes and any cross-host numerical differences; do not claim
bit-for-bit trajectory portability from this check alone.
