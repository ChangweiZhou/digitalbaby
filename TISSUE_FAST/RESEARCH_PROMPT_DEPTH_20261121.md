# Research prompt: carrying information ten or more connections deep in a network of local learners (21 November 2026)

For a deep-research tool. Self-contained; written in the field's standard terms. It names no project or person.

## 1. What we are building

A recurrent network of units that organises itself under local plasticity rules. It reads a raw byte stream through an input layer, emits bytes through an output layer, and receives feedback on its own output (at minimum correct/incorrect; the correct byte is allowed). It learns over one continuous life without resets or task labels. Units have random positions in a unit square and connect only to units within a fixed radius (about 94 candidates within reach; 2,048 units; 256 input lines, one per byte value). There is no layer structure: "depth" means the number of connections on the shortest path from an input line to a unit.

**The problem.** Information about the present input byte must reach units at least ten connections from the input lines and still be decodable there. In our tissue it is decodable at one and two connections and not at four.

## 2. Constraints: an admissibility checklist for every method you report

Mark each **yes**, **partly** or **no**, with one line of reason. **[stated]** items are firm; **[open]** items are undecided, and we want to know what the literature implies for them.

- **A1 [stated]. No single backward pass over the network.** Error information may be relayed unit to unit over connections. No unit is given an answer computed for it from outside.
- **A2 [stated]. Modulatory (chemical) signals are regional.** A unit can release one into its neighbourhood and nearby units read it in their own way. None reaches every unit.
- **A3 [stated]. Connection strength is local**, decided jointly by the units it joins, any of which can set it to zero. A connection may run through intermediate units. **[open]** Whether one parameter shared by a whole class of units counts as local.
- **A4 [stated]. No long window of past input, or full history, anywhere in the learner.** A small store of episodes used for rehearsal (as in hippocampal replay) is allowed.
- **A5 [stated]. One continuous life:** learning never stops, and memory forms by itself.
- **A6 [stated]. Bounded units and connections.** Each unit makes and withdraws its own connections; the fewer parameters fixed in advance, the better.
- **A7 [stated]. Feedback.** The input and output ends learn too. Dedicated units receive the error signal and pass it on, and how it is passed on is learned.
- **[open] Fixed architecture.** Fixed structure given at birth (unit types, long-range axons for a fraction of units, distance-dependent wiring) is acceptable in principle, as evolution provides; report it separately from what is learned.

## 3. What we have measured (development level, unpublished; for calibration)

All runs: 16,384 bytes, three random seeds. Decodability is the nearest-centroid score for the present byte from the outputs of units at a given distance; chance is 0.022 and our threshold is 0.11.

1. **Static weights.** In a version whose weights do not change, the present byte explains nearly all of a unit's output variance out to 6 connections, but the mean output amplitude falls about 4.5-fold per connection (0.0046, 0.00088, 0.00022, 0.00005 at 1 to 4 connections). It falls below anything a readout or a unit can use.
2. **Our current best version (plastic):**
   - each unit's total incoming excitatory and inhibitory strength is held constant (3/4 and 1/4, rescaled multiplicatively);
   - new connections are made at a fixed slow rate and kept only if correlated with the receiving unit's activity during a trial period;
   - about 18 connections per unit;
   - decodability 0.89 to 0.99 at 1 connection, 0.57 to 0.76 at 2, and 0.02 to 0.08 at 4.
3. **A lower activity target hurt depth.** Lowering the target activity of a threshold-adaptation rule made activity sparser and lowered decodability at 2 connections (0.58 to 0.34).
4. **Cases that reached deeper, all in variants that did not learn their strengths:**
   - With a bounded per-unit gain that drives each unit's mean output toward 7/8 of the mean output of its sources, decodability was above threshold at 1, 2 and 4 connections.
   - Very few connections per unit (about 5), with total strength conserved and strengths changing by spike order (pre before post grows, post before pre shrinks), was above threshold at every distance from 1 to 11.
   - A fraction of units (1 in 8) with long axons, reaching eight times farther, carried the present byte to the output side.
5. **Earlier literature reading told us:**
   - SORN-type networks need STDP, synaptic normalisation, structural plasticity, intrinsic plasticity and inhibitory STDP together for stable activity and heavy-tailed weights.
   - No self-organising local-rule network has published character-level text results.

## 4. Questions

For each question, give numbers, network sizes and the exact measure, not only qualitative claims.

**Q1. Signal propagation depth in recurrent and layered spiking or rate networks.**
1. What determines how far a stimulus-specific signal survives: synfire chains; balanced excitation and inhibition; criticality and the branching ratio; gain or homeostatic control; dendritic nonlinearity and coincidence detection; heavy-tailed (log-normal) weights; sparse, strong connections?
2. Give published numbers for depth (number of synaptic stages, or "layers" in feedforward embeddings of recurrent networks) at which a stimulus remains decodable, and for the conditions under which it is lost.
3. Cover Diesmann, Gewaltig and Aertsen 1999 and later synfire-chain work; van Rossum, Turrigiano and Nelson 2002 on rate propagation through layers; Kumar, Rotter and Aertsen 2008/2010 on rate versus synchrony propagation; Vogels and Abbott 2005 and 2009 on gating and signal transmission in balanced networks; Teramae, Tsubo and Fukai 2012 on log-normal weights and reliable transmission; Beggs and Plenz 2003 and later work on criticality and information transmission; and others you find.

**Q2. Local learning rules that create or protect deep propagation.**
1. Which local rules form chains or pathways that carry a signal many stages deep? Cover STDP chain formation (Fiete et al. 2010; Jun and Jin 2007), structural plasticity forming chains, homeostatic gain or intrinsic plasticity, synaptic scaling, inhibitory plasticity, and heterosynaptic plasticity.
2. Which of these were shown in networks that also keep learning (A5), and with what depth?
3. Does multiplicative normalisation of a unit's total input cause per-stage attenuation? What is known to counter it (additive normalisation, heavy tails, competition that concentrates weight on a few inputs, gain)?

**Q3. Fan-in, sparsity and depth.**
1. What is the relation between the number of inputs per unit, the distribution of their strengths, and how far a signal travels?
2. What fan-in do cortical neurons have, and how many strong inputs are needed to drive one? Cover the few-strong-synapses literature (for example Lefort et al. 2009; Song et al. 2005; Buzsáki and Mizuseki 2014).
3. What does this imply for a network where each unit has 5 to 30 connections?

**Q4. Long-range connections and small-world structure.**
1. In networks wired by distance, how much do a fraction of long-range connections reduce the path length and increase how far information travels?
2. Give published fractions and effects (for example Watts and Strogatz 1998 for path length; cortical long-range axon fractions; models of cortical columns with long-range pyramidal axons).
3. Is long-range structure given at birth biologically standard (A6 and the fixed-architecture item)?

**Q5. Depth from learning rather than wiring.**
1. In deep networks trained with local rules (target propagation, predictive coding, forward-forward, feedback alignment, Hebbian layer-wise learning), how many layers have been trained, with what loss of performance per added layer, and with what mechanisms (normalisation, residual paths, layer-wise objectives)?
2. Which of these can be cast as a recurrent tissue without layers (A1 to A7)?

**Q6. Measures.**
1. How do published studies measure "information at depth d" in recurrent networks: decoders, mutual information, Fisher information, the share of variance explained?
2. What are the pitfalls: a decoder that reads the input from residues; correlated noise; a signal that is present but too small for downstream units?

## 5. What to return

For each question:

1. **Direct answers**, two to eight bullets, each with its source.
2. **A table** with columns: authors, year, venue; network type and size; mechanism; the depth reached, its measure and number; constraints A1 to A7 (yes/partly/no); code available (yes/no, link).
3. **A verification flag** on each claim: **verified** (read in the source) or **recalled** (from memory). Correct any names, years or venues above that are wrong.
4. **Searched for and not found**, stated explicitly.
5. **One to three experiments of about one CPU-day each** on a 2,048-unit tissue as described, each with what result would count against it.
6. **The strongest argument against the mechanism you rank highest.**

At the end:

1. **A ranked list of the five mechanisms** most likely to carry a byte ten or more connections deep in our tissue while it keeps learning. For each give: what it changes, its cost, the risk it brings (runaway activity, loss of selectivity, forgetting), and the cheapest test.
2. **A note on which of our measured findings in section 3** match, contradict or are explained by published work.
