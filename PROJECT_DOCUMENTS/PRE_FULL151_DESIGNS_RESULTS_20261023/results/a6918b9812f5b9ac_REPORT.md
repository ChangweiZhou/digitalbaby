# Step 7 result: H4's exact-cue gain does not survive a shifted input key

The fixed read-only census completed on all **24 new worlds** (9701–9724),
with no labels, teaching calls, or native writes. Its independent audit passed.
The prespecified decision is **`E2_NOT_ELIGIBLE_FROM_THIS_CENSUS`** for the
H4 `SHIFT_RIGHT` presentation. H4's six-way nearest-canonical-cue score was
**24.3%** (world-level two-sided 95% Student-t interval **17.4–31.3%**),
far below the fixed 75% mean and >50% lower-bound access gate. Six-way
chance is 16.7%. The shifted form changed H4's actual four-byte feature key
for every tested cue.

| Changed form | FE0 nearest own | Old Q nearest own | H4 nearest own | H4 same four-byte key |
|---|---:|---:|---:|---:|
| Prefix edited, final four bytes unchanged | 55.8% | 97.2% | **100.0%** | 100% |
| Ordered bytes moved right one position | **74.7%** | 32.6% | **24.3%** | 0% |
| Ordered bytes moved left one position | **67.4%** | 29.3% | **19.8%** | 0% |

The strong H4 prefix score is deliberately **not** E2 evidence. The prefix
edit leaves its exact four-byte feature rows unchanged, so a fixed
canonicalized-key lookup could identify the cue. When the ordered bytes move
right, H4 still sees both within its four-byte window, but its fixed hash
rekeys them: the mean KC Hamming distance to the *same* canonical cue is
**151.2**, versus **155.3** to its reversal. It offers little identity
continuity. Moving left pushes the first ordered byte out of that window,
and H4's score drops further. FE0's right-shift score is **74.7%**
(95% interval **68.1–81.2%**), showing considerably more access under this
specific perturbation. That is a representation measurement, not proof FE0
would retrieve a learned answer after a changed presentation.

This resolves the question left by Step 6: the H4 projection can give
Full151 excellent E1 acquisition on a repeated exact byte frame, but its
four-byte feature hash is a brittle address for changed frames. The result
does not imply all possible H4 presentations fail, nor that FE0 has solved
E2. It does mean a new H4 E2 science run on this shifted form is not justified
by the prespecified access gate. The appropriate next decision is to retain
FE0 as the working route for this input family and, if changed-presentation
retrieval is still a priority, design a fresh FE0 E2 test with a no-prior-
teaching control. Another H4 hash or Full151 memory-topology search is not
supported by this access result.

**Evidence level: E0 only.** The frozen technical fixture checked cue length,
ordered-byte identity, feature-window change, read-only state purity, and
independent FE0 code parity. The audit checked all 24 receipts, source and
cue hashes, packed KC code populations, six-candidate distances, tie-aware
nearest scores, roster and intervals without replaying a learning trajectory.
Packed-code distance and all-tie arithmetic checks passed. The compact
readout receipts occupy approximately 2.6 MB; no event stream was saved.

Artifacts: [design](DESIGN.md), [census runner](step7_access.py),
[audit](audit.py), [summary](results/SUMMARY.json),
[source lock](results/SOURCE_LOCK.json), and
[technical receipt](results/TECHNICAL.json).
