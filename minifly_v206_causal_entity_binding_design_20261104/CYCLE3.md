# Cycle 3

Refine. Run the inherited predictor, answer generator, and episodic writer on two reduced worlds. Restore query history after each probe, so paired questions cannot influence each other. The only learned-state writes permitted at evaluation are the inherited position-triggered destination writes.

Audit. Use one restored checkpoint, the same story bytes across arms, exact hit/follow counts, frozen-state hashes and own-newline scoring. Dry-generate the fixed official roster and verify the receipt archive.

Test. All 32 reduced world/length/arm cases completed using the same restored checkpoint. Across the two engineering worlds, final exact own-newline counts out of 16 per length were:

| Arm | 0 bytes | 16 bytes | 40 bytes | 64 bytes |
|---|---:|---:|---:|---:|
| RECENT32 | 16 | 3 | 3 | 3 |
| RECENT96 | 16 | 16 | 15 | 8 |
| SPLIT-RECENT | 16 | 3 | 3 | 4 |
| SPLIT-ANCHOR | 16 | 16 | 16 | 16 |

The address contrast produced a favorable engineering signal for the anchor. These worlds use opaque neutral byte padding, supplied write positions and four stories per world. They do not support a promotion verdict. The frozen-state checks, write counts, receipt hashes and ZIP readback passed. Qualification consumed one historical training replay for the checkpoint, 20 historical evaluation stories and 128 new story-arm applications. The eight official worlds were dry-generated only. A production reducer, shortcut scoring, exact cost census and official launch checks remain pending.
