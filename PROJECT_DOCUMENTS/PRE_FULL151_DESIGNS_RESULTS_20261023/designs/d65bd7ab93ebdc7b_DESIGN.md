# Step 5: Can the protected temporal Full151 branch learn ordered-byte value?

The protected dual route passed E0 state isolation, while an exact-FE0-support
router was found inactive at ordinary byte timing. This experiment asks only
whether the **separate temporal Full151 branch** can acquire an order-dependent
value association under native teaching. It does not fuse branch outputs,
choose a router, test unseen combinations, or claim arithmetic. The endpoint
is **E1**: all six scored ordered cues are explicitly taught. Exact cue
recitation is a sufficient countermodel, so a positive result is a component
gate, not progress to E3 on its own.

## Fixed input, exposure and controls

Use fresh 16 worlds 9501–9516, technical world 9499. In each world, draw three
distinct uppercase bytes with the source V9.3 `order_world` rule. The three
cyclic orientations are aversive and the reverse orientations neutral, so
reward is balanced by byte in both positions. Teach all six orientations for
12 bouts in the inherited deterministic shuffle: **72 records**. Each cue is
eight visible spaces, the two ordered bytes, then two visible spaces (12 cue
bytes total); the outcome byte is always `!`, followed by newline. The two
spaces after the ordered bytes make this a new fixed **two-byte-delay**
condition. Every byte arrives at the inherited 30/14 s spacing, record starts
are 165 s apart, and `Teach` follows the outcome byte at that time.

Two protected dual learners per world receive identical streams: W permits
native value writes in **both** independent Full151 branches; N calls the same
`Teach` with writes disabled. Both byte predictors update online and must
remain identical across W/N and FE0/temporal branches. The protected FE0
route and temporal route are each probed read-only before teaching, after six
bouts, and after 12 bouts. For each of the three reversed pairs, choice is
correct when `value(neutral) − value(aversive) > 0`; ties score 0.5. The
environment supplies both alternatives, so no autonomous action is tested.

A separate two-cue **simple FE0 positive control** uses the inherited V9.3
simple context and 12 presentations per cue in a fresh matched W/N FE0
branch for the same world. Its taught choice at bout 12 must show learning
for the order assay to be considered sensitive. This control is separate from
the 72-record order learner; no teaching leaks between them.

## Decision and limits

The world is the unit; run all 16 worlds without interim selection. The
component gate is descriptive but fixed: at bout 12, the temporal route's
W−N choice must have a two-sided 95% Student-t lower bound above zero, and
its W choice lower bound must exceed chance 0.5. The simple FE0 W−N control
must also have a positive lower bound. If the control fails, classify the
assay as insensitive; if the temporal gate fails, report failure or
unresolved, without changing dose or suffix on these worlds. Report FE0 and
temporal absolute W/N choices, margins, pre/bout6 trajectories, and KC code
separation. Multiple trajectory diagnostics are not separate claims.

Technical world 9499 must reproduce a standalone FE0 path exactly after
every order record and at readout, verify predictor/clock/write parity and
probe purity, and validate all world fixtures before science. A scientific
exception writes a failure receipt and stops without automatic restart or
world replacement. Completed receipts are never rerun. Caffeinate remains
attached to the active runner. A separate receipt reducer checks source
hashes, roster, exposure counts, scoring arithmetic, and intervals; it does
not independently rerun the native synapse dynamics.
