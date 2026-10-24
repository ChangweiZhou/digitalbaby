# Full151 bridge, step 1: dose versus predictor drift

This is a small **engineering diagnosis**, not an E3 confirmation or a search
over Full151 memory mechanisms. The previous bridge mixed a normalized V9.2
next-byte distribution into FE0 PN at gain 0.5 while the predictor continued
to update. It reached KC but largely erased the immediate taught-pair
write benefit. This step varies only the bridge dose and whether the V9.2
predictor updates on the 144 old teaching records. No parent code, checkpoint,
or completed world is changed or rerun.

## Fixed roster and arms

Engineering parity world: 9099. Exploratory worlds: 9101–9108, chosen before
looking at outcomes and disjoint from earlier bridge and V9.9 worlds. Use the
unchanged V9.8 P/N generator. Each condition has matched native-write W and
no-native-write N branches with identical byte, label, clock and predictor
exposure. Conditions are FE0 with predictor online, and bridge gains 0.05 and
0.5, each with predictor online or frozen. The frozen condition still advances
the same sensory and history state; only predictive synapse updates are off.

The primary diagnostic is **immediate taught-pair** W−N choice, an E1 measure.
The three never-reinforced pairings are also scored before feedback as a
secondary E3 diagnostic. Exact taught-pair recitation could pass E1, so this
step alone cannot promote a reusable learning core. A fixed comparator receives
two alternatives and emits L/R; it is not autonomous answer generation.

The FE0 gain-zero path must reproduce the frozen factorial runner's immediate
pair margins exactly on engineering world 9099. The 0.5-online path must also
reproduce that runner. W/N predictor weights must agree within each online
condition, frozen weights must stay at their birth digest, and all clocks and
write counts must agree. No science-world outcome is inspected before these
technical gates pass. A failure stops the run and writes an error receipt.

The eight exploratory worlds select no scientific winner. They ask whether a
smaller dose or stopping online predictor drift restores immediate teaching
while retaining any measurable non-FE0 KC contribution. If a candidate looks
promising, it needs a separately frozen fresh-world behavioral confirmation;
if not, the next step is a structurally isolated temporal channel. Do not tune
another gain on these eight worlds or recycle them as confirmation data.

The learner is exposed to six taught P/N combinations in both orientations;
the three scored held-out combinations and their labels are absent from all
teaching records. An exact raw or canonical pair cache cannot answer those
held-out combinations. A learned one-bit symbol class can. FE0 is the known
positive control and W−N isolates prior native teaching under matched input.
The V9.2 predictor is an engineered next-byte learner whose text checkpoint
is inherited unchanged; this experiment cannot attribute an effect to
biological byte prediction.
