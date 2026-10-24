# Step 8: FE0 retrieval after a one-position byte shift

This is a bounded E2 test of the *existing* FE0→Full151 native value path.
Step 7 found that FE0's read-only code recognizes a one-position right shift
far better than the H4 projection, but code geometry is not learned-value
retrieval. We leave Full151, FE0, the V9.2 byte predictor, and all teaching
rules unchanged. No H4, Q mixing, router, or output head participates.

## Claim, exposure and countermodels

Draw three random uppercase bytes and balanced cyclic order labels by the
frozen V9.3 rule in each world. Teach all six ordered cues in the Step 5
canonical 12-byte frame (`8 spaces || x y || 2 spaces`) for 12 bouts, 72
records, at the inherited byte and record times. Every canonical cue is
directly reinforced 12 times. The outcome byte `!` and newline follow each
cue, and the source `Teach(label)` occurs immediately after the outcome
byte. The only test variant is `9 spaces || x y || 1 space`, also 12 bytes.
It is never placed in a teaching record or shown to the live learner at a
probe. Both branches see identical canonical bytes, labels, outcome bytes,
timing, and online predictor updates. W permits native old-teaching writes;
N calls the same Teach with native writes disabled.

Read-only clones probe canonical and shifted cues before teaching and once
after the complete 72-record schedule. The first shifted choice occurs on
those clones **before any shifted feedback or reteaching**. The evaluator
supplies the two alternatives; the learner does not locate an answer point
or emit a byte. For each of three reversed pairs, score neutral value minus
aversive value; a positive margin is correct and a tie earns 0.5.

An exact raw 12-byte cue table has no reinforced shifted entry. A table
which parses out and canonicalizes the same ordered `x y` pair can solve
every shifted comparison; this test is therefore **E2 changed presentation,
not E3 unseen relation**. Each symbol occurs equally often in both roles
and labels, so a fixed symbol/position-frequency shortcut has no stable
answer. The W−N branch comparison asks whether prior native writes caused
the shifted score, rather than sensory exposure or predictor updates alone.

## Frozen run and decision

Technical world **9799** is excluded. Science worlds are **9801–9832**,
exactly 32 new worlds; all run with no interim selection. The world is the
statistical unit. The primary bout-12 shifted W−N choice effect passes only
if its two-sided 95% Student-t lower bound is >0. Two guards are conjunctive:
canonical bout-12 W−N lower bound >0, and shifted W absolute lower bound
>0.5. For a perfect 32/32 absolute world score, replace the zero-width
Student-t interval with the two-sided exact binomial lower bound; any other
zero-variance case is unresolved. This n is an economical qualification,
not a precision-powered broad promotion. Failure or a wide interval is
reported as such, with no tuning or automatic larger run.

The technical fixture checks cue framing, no shifted exposure, W/N birth
equality, Step 5 FE0 state parity after every record, predictor/clock/write
parity, read-only probe purity and complete teaching count. One receipt per
world is atomically committed. A scientific exception writes a failure
receipt and stops without replacement or automatic restart. Completed
worlds are never rerun; caffeinate stays attached to the active runner.
An independent reducer validates source hashes, world roster, schedule and
cue hashes, exposure counts, raw value-to-margin-to-choice arithmetic and
the fixed intervals without replaying the native trajectories. The report
states E2 at most; no arithmetic, autonomous output, or persistent-core
claim follows from this immediate synthetic test.
