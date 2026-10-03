# Cycle 2 — audit and revision

The native brain snapshot was inspected before designing serialization. It
omits CONTENT.visible; using the inherited save() directly would silently lose
the private address after restore. Revision: store and validate this suffix for
each of the four private stores. Also persist pending prediction, cue cursor,
required newline, explicit clock and write receipt, beyond native snapshots.

The loader creates eight independent canonical newborns, validates all fixed
parameters and source bytes, restores mutable arrays with matching shape/dtype,
and checks the resulting full state digest. Serialization uses allow_pickle=False,
atomic replacement and bounded decompressed size. Checksums detect accidental
corruption; they are not cryptographic signatures from a trusted publisher.

Acceptance includes fresh-process restoration and rejection of recomputed-seal
but wrong-source/shape/cursor metadata, rather than checksum tests alone.
