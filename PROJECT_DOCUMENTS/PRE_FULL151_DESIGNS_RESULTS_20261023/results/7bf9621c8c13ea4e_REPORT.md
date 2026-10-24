# Full151 bridge step 3a — protected route technical result

The protected dual-route prototype passed its E0 technical qualification on
fresh engineering world 9399. Across the complete 144-record old-teaching
schedule, the FE0 branch's **full state digest matched a standalone FE0 model
after every record** in both native-write W and no-write N conditions.
Immediate read-only FE0 values also matched exactly (maximum error 0), as did
all clocks. Predictor histories and weights agreed between branches after
every record; read-only probes left both live states unchanged. The temporal
branch's KC code differed from the protected FE0 code on all 12 taught and
six held-out cue orientations under both W and N.

The two Full151 native fast/slow/adaptation arrays do not share memory; a
targeted mutation of the temporal bank left the FE0 full-state digest
unchanged. Each branch currently duplicates about 2.88 MB of V9.2 predictor
weights and 0.17 MB of mutable Full151 native arrays. Sharing the predictor
without changing its behavior is a later resource optimization, not part of
this qualification.

This establishes a way to expose temporal byte information **without altering
the known FE0 value route**. It does not supply a fused answer policy or show
that the temporal branch learns a useful order relation. The next scientific
gate must first construct a task-relevant FE0 alias and a fixed routing rule,
then test behavior on new worlds with no-prior-write and exact-key controls.
No behavioral science worlds were run for this prototype. See [DESIGN.md](DESIGN.md),
[technical.py](technical.py), and [results/TECHNICAL.json](results/TECHNICAL.json).
