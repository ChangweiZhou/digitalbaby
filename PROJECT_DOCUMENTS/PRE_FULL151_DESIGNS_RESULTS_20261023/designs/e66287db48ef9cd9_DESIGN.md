# Output Step 1: four fixed answer-byte channels using inherited Full151 plasticity

This is a prospective **E1 learned-action diagnostic**, not arithmetic and not
a claim that unmodified Full151 already emits bytes. It asks whether four
copies of the inherited native fast/slow learner, sharing the same fixed
FE0/KC input and clock but maintaining separate writable state, can make a
pre-feedback one-byte choice. The output policy is a fixed argmax over four
channel values, mapped internally to ASCII `0`–`3`. The evaluator neither
enumerates candidate answer strings nor prints a selected integer. Copying
the *entire* native learner per byte is a new, expensive architecture fork;
it is only a mechanistic feasibility test for future local compartments.

In each world, draw eight distinct uppercase bytes and group them into four
two-byte cues. Draw one permutation of answer bytes `0`–`3`, assigning one
answer to each cue. A cue is exactly `9 spaces || xy || '='` (12 bytes). For
12 bouts, each cue appears once per bout in a randomized order: 48 records.
The visible `=` ends the question. The model emits one answer byte from its
four channels **before** seeing the teacher. Then the environment sends the
correct digit byte and newline; a 135-second passive tail completes the
inherited 165-second record. The answer digit is a privileged teacher during
training, not self-supervision from an unlabeled world. At a scored probe,
the teacher byte, grade, and all writes are absent. Each cue and answer is
directly taught, so exact-key lookup can pass; no E2/E3 claim follows.

Every class copy sees the same input bytes and time. The true W branch gives
native event `p=0` to the teacher-indicated class and `p=1` to the other
three. N receives the **same** events and bytes with native writes disabled.
P receives the same bytes and number of writes, but deliberately routes the
teacher to `(teacher_class+1) mod 4`. The `p` direction was chosen on one
explicitly exploratory engineering world (11000); its 4/4 post-teaching
readout is excluded. This direction is a new output teaching policy, not an
inherited four-class Full151 function. Each branch has the same sensory
history, feedback count, class count, clock, and fixed output alphabet.

Before each teacher byte, log the branch's actual emitted byte. Read-only
clones separately probe the four fixed cues before training and after all
48 records. Save full class values and output bytes, plus clock, teacher,
source, and state receipts. A no-write branch should remain at its identical
birth-class tie; a misrouted teacher should learn another mapping if native
credit is causal. Source files and acceptance code are hashed before science.
No completed trajectory is rerun after a failure.

Technical world **11099** is excluded. Science worlds are **11101–11132**,
all 32 run without interim selection. The world is the statistical unit.
The E1 gate requires W absolute choice accuracy to have a two-sided 95%
world-level lower bound `> 0.75`, using Student-t unless all 32 worlds score
exactly 1, in which case use the exact binomial lower bound. It also
requires both mean paired effects W−N and W−P to be positive **and** the
two-sided exact binomial lower bound on the proportion of worlds where W
strictly beats each comparator to exceed 0.5. Ties count as non-wins. The
paired mean and its Student-t interval are reported descriptively when
variance is nonzero; a zero-variance difference cannot manufacture a
zero-width effect interval. This is a small feasibility qualification, not
a powered estimate of general byte-generation ability.

The technical preflight checks the byte and event timeline, first-choice
before-teacher ordering, FE0 equality across class copies, birth equality,
read-only clone purity, teach/write counts, and normal Full151 clock
progress. The independent reducer verifies source hashes, world and cue
rosters, teacher/record hashes, all output bytes, class-value argmax,
accuracy arithmetic, interval arithmetic, and receipt completeness without
replaying science trajectories. Technical failures stop before science;
scientific failures stop without automatic restart. `caffeinate` remains
attached to the science runner until it exits.

Even if this gate passes, it only proves directly taught one-byte recitation
with four independent native learner copies. It says nothing about two-digit
emission, expression parsing, changed-presentation retrieval, unseen sums,
retention, or spontaneous answer timing. Those must be separate gates.
