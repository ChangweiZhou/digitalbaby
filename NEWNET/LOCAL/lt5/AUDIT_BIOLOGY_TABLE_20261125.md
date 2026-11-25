# Audit of the lt5 biology table (25 Nov 2026)

An independent audit of the eight-row table that assigns each lt5 component a biological principle and a known model. The auditor wrote no code and ran no experiment. The sources read were:
- `WHAT_A_NEURON_HAS_20261118.md`, sections 1–19; sections 7–19 govern;
- the NEWNET requirements checklist and the process rulings in memory;
- `lt5.c` and `lt5.py` (the defaults);
- `checks/checks_out.txt`, which holds a check run that the auditor did not repeat.

## How the references are marked

Every reference below is cited from memory. None was checked against the publisher during this session.

- **No mark:** the auditor is confident of the authors, year and venue.
- **[U]:** the auditor is unsure of at least one detail (volume, pages, exact title or year). Check these before anyone quotes them.

No reference was made up. Where the auditor knows a line of work exists but cannot name a paper, the text says so and gives no citation.

## 0. Two problems that cut across the table

### 0.1 The table never says how long a byte lasts

lt5 counts time in bytes, and every constant in `lt5.py` is in bytes:
- the cell time constants run from 1 to 16 bytes (`tau_lo`, `tau_hi`);
- the eligibility trace lasts 4 bytes (`tau_el`);
- the rate average lasts 500 bytes (`tau_r`).

The table counts time in milliseconds and seconds. The two cannot be compared until the design fixes how many model milliseconds one byte lasts. Natural reading runs at roughly 25–35 characters per second, so a plausible choice is about 30–40 ms per byte. The auditor's own figure, not a measured one.

At that rate:

| Biological process | Its time scale | The same span in bytes |
|---|---|---|
| Axonal and synaptic delay | about 1–20 ms (cortico-cortical delays occasionally longer) | at most about one byte |
| Membrane time constant | 10–30 ms | under one byte |
| STDP window | about ±20 ms | within one byte |
| Short-term facilitation | about 0.5–1.5 s | about 15–45 bytes |
| Behavioural-timescale plasticity and eligibility traces | about 1–several s | about 30–100+ bytes |
| Persistent activity, slow cation currents, time cells | seconds to tens of seconds | about 30–500+ bytes |

The consequence is the main finding of this audit. Row 1 cannot produce "small passages from different places" tens or hundreds of bytes back, because conduction delays are too short. lt5's present passages reach back as far as 195 bytes, through the lags 192–195. In biology, access over seconds comes from slow processes (rows 2 and 8 and the time-cell literature), not from delay lines.

### 0.2 The current check record

`checks/checks_out.txt` records the following. The auditor did not rerun any of it.
- **c4 (memory beyond the window):** FAIL.
- **c5 (relay ablation):** FAIL. The memory task costs 1.3253 nats with the relay and 1.3374 without it. English costs 3.4228 with the relay and 3.4804 without it. The relay is worth about 0.01–0.06 nats per byte.
- **c6 (cost):** FAIL. Cost per byte grows with the number of cells with exponent 2.03, and late bytes cost 4.05 times as much as early ones.
- **c2:** the largest tissue fan-in is 41, against the owner's allowance of "tens of thousands". There are 138,900 relay-reading synapses (T_RD), about four times the 33,006 tissue excitatory synapses.

Any row that adds work per synapse makes the c6 failure worse unless it is computed lazily. Section 4 says how.

---

## 1. Verdict per row

### Row 1. Reading the past: per-cell delays and time constants, with myelin plasticity

**Verdict: needs change. Part of it is wrong on time scale.**

**The biology.**
- Conduction delays really do vary between axons and are an established basis for temporal codes over milliseconds:
  - the Jeffress delay-line model (Jeffress 1948, *J Comp Physiol Psychol* 41:35);
  - its confirmation in the barn owl (Carr & Konishi 1990, *J Neurosci* 10:3227 [U]);
  - polychronous groups in cortex models (Izhikevich 2006, *Neural Comput* 18:245).
- Activity-dependent myelination exists:
  - Gibson et al. 2014, *Science* 344:1252304;
  - McKenzie et al. 2014, *Science* 346:318;
  - review: Fields 2015, *Nat Rev Neurosci* 16:756 [U];
  - a model of its effect on timing: Pajevic, Basser & Fields 2014, *Neuroscience* 276:135 [U].
- The auditor knows of no evidence that delays of seconds exist, or that myelin plasticity shifts timing by more than milliseconds. Myelin changes over days to weeks.

**What the row gets wrong.** "Small passages from different places" (s.8) means a cell sees bytes from tens to hundreds of bytes ago. Delays of milliseconds cannot deliver that (section 0.1). The known mechanisms for seconds-scale access to the past are different:
- **Time cells and sequential activity:**
  - Pastalkova et al. 2008, *Science* 321:1322;
  - MacDonald, Lepage, Eden & Eichenbaum 2011, *Neuron* 71:737.
- **A bank of leaky traces with log-spaced time constants**, which gives a scale-invariant memory of the past, as in the "Laplace / inverse-Laplace" model:
  - Shankar & Howard 2012, *Neural Comput* 24:134 [U];
  - Howard et al. 2014, *J Neurosci* 34:4692 [U].
- **Slow intrinsic currents:** graded persistent activity in entorhinal cortex (Egorov et al. 2002, *Nature* 420:173).

**What follows from the principle.** The owner's rule R4 already gives each cell its own time constant. The biologically grounded way to "read separate passages" is:
- input-side cells with heterogeneous slow time constants spread over a log scale;
- a learned conjunction (row 4) that picks a time slice out of those traces.

A genuine delay is still useful in two places:
- per synapse, within one byte, where STDP can select it (row 4);
- in a declared sensory buffer, which ruling 3 of 10 Nov allows. lt5's history ring is such a buffer.

If the buffer is kept, say so: it is an allowed sensory buffer, not "axonal delay". Then make the lags per synapse rather than one global lag table. Today `h->lag[l]` is global, and every input line of group l has the same lag.

**Owner's rules.**
- Per-cell rather than global receptive fields agree with s.8. The present global lag set does not, and the table is right to move away from it.
- Hidden counts:
  - the number of passages per cell;
  - the passage lengths;
  - the buffer length HL;
  - KIN (the copies of each input line).

  Make the passages emerge: give each input synapse its own delay drawn from a broad distribution, and let structural plasticity (row 6) keep the useful ones. A cell's set of passages is then the set of delays it keeps. The bounds of the delay distribution are scales, not counts. HL is a memory size; the owner treats window length as a search dimension, so it may remain one.

**Cost.**
- A per-synapse delay needs a spike-arrival queue: a ring of buckets indexed by arrival time, as in NEST. This costs O(1) per synaptic event (Morrison et al. 2005, *Neural Comput* 17:1776 [U]; Brette et al. 2007, *J Comput Neurosci* 23:349).
- The buffer costs O(HL) memory and nothing per cell.
- Slow per-cell traces are already lazy in lt5 (`trace()`, `catch_up()`).

**Functional consequence.**
- Log-spaced traces give a fading memory that can cover 40+ bytes with few cells.
- For linear networks, total memory capacity is bounded by the number of units, and normal (non-orthogonal) networks do worse (Ganguli, Huh & Sompolinsky 2008, *PNAS* 105:18970).
- Whether a local rule can then *use* information from 40+ bytes ago is unknown. lt5 already has direct access through lags 192–195, and c4 still fails. Access to the past is probably not lt5's present bottleneck.

### Row 2. Working memory over seconds: short-term facilitation and depression

**Verdict: sound in principle; the wording needs change.**

**The biology.**
- The Tsodyks–Markram model of dynamic synapses:
  - Tsodyks & Markram 1997, *PNAS* 94:719;
  - Markram, Wang & Tsodyks 1998, *PNAS* 95:5323;
  - Tsodyks, Pawelzik & Markram 1998, *Neural Comput* 10:821.
- Facilitation-based working memory: Mongillo, Barak & Tsodyks 2008, *Science* 319:1543. The auditor recalls τ_F ≈ 1.5 s and τ_D ≈ 0.2 s there [U].
- Facilitating synapses are common between pyramidal cells in prefrontal cortex: Wang et al. 2006, *Nat Neurosci* 9:534 [U].
- Review: Zucker & Regehr 2002, *Annu Rev Physiol* 64:355.
- Persistent firing: Goldman-Rakic 1995, *Neuron* 14:477; Wang 2001, *Trends Neurosci* 24:455.
- Slow currents: Egorov et al. 2002, as above.
- Dynamic synapses in a reservoir: Maass, Natschläger & Markram 2002, *Neural Comput* 14:2531.

**What the row gets wrong.** "Memory exists naturally rather than as an added device" overstates the case.
- In the Tsodyks–Markram form, u and x depend only on the presynaptic spike history. For the target-independent version, that is one state per sending cell: a fading record of which cells fired recently, which lt5's per-cell trace `xl` already partly is.
- Markram et al. 1998 show that facilitation and depression can differ by target, which needs state per synapse.
- In the Mongillo model the memory is silent. It needs periodic reactivation (population spikes) to be read out, and its capacity is a few items. On capacity: Mi, Katkov & Tsodyks 2017, *Neuron* 93:323 [U].

Short-term plasticity makes a store available. It does not decide what is stored, and it does not make the store usable; reading it still has to be learned (row 4).

**Owner's rules.**
- The STP constants are time constants and a release probability, so they are rates, not counts. Allowed.
- Release probability is presynaptic. Together with postsynaptic plasticity this fits s.16, where strength is set by both ends.

**Cost.**
- Per sending cell: O(1) per spike, using the exact exponential relaxation between spikes.
- Per synapse: the state is updated only during delivery, which already visits every out-synapse. Cost is O(active synaptic events), with two extra floats per synapse.
- Both are fully lazy.

**Functional consequence.**
- At about 35 ms per byte, τ_F ≈ 1–1.5 s covers about 30–45 bytes. This is the one row whose biology naturally matches the 40-byte target.
- Expect a capacity of a few items, not a store of byte identities.

**Change.**
- State which form is used (per sending cell or per synapse).
- Drop the claim that memory then "exists naturally".
- Pair the row with persistent and slow currents (row 8, or a per-cell slow current) and with a readout learned through row 4.

### Row 3. Propagation within one byte: continuous time, delayed events, no fixed S and H

**Verdict: sound as a direction; needs change on cost and on hidden counts.**

**The biology.** Continuous time and spikes carried with delays are correct, and the owner's own s.1.5 says so. Exact or event-driven simulation of spiking networks with delays is standard:
- Brette et al. 2007, as above;
- Morrison, Straube, Plesser & Diesmann 2007, *Neural Comput* 19:47 [U].

**Hidden counts.**
- "How many stages a byte's time allows" is the byte duration divided by the typical delay. That ratio is the old S in disguise. It is acceptable under the owner's rule as a time constant (the world's clock rate), but the table should name it.
- Removing H makes the feedback relay take real time. Feedback then reaches far tissue later, so the eligibility trace must outlast the relay latency. The eligibility time constant and the depth of the relay become coupled (see row 7).

**Cost.**
- A bucketed ring of arrival times (integer delays in sub-byte ticks) costs O(1) per event. A heap costs O(log E), which keeps the total within n log n.
- The danger is runaway cascades. In continuous time nothing caps the number of spikes within a byte except inhibition and refractoriness.
- A guard such as "at most k events per byte" would be a hard count and is excluded. The guard must therefore be a mechanism:
  - refractoriness (lt5 sets `h[i] = 0` and lets a cell fire once per byte, which is itself a hidden refractory rule);
  - fast inhibition;
  - the activity set points.
- Cost per byte is then about (spikes per byte) × (mean fan-out). With fan-out in the tens of thousands, spike delivery dominates, as it does in NEST.

**Functional consequence.**
- Delays within a byte let STDP learn order within a byte, which matters little for single-byte events.
- The large consequence is on feedback timing, not on memory.

**Change.** Keep the row. State:
- the byte duration as a constant;
- a refractory period per cell;
- that the cascade is bounded by inhibition and refractoriness, not by a count.

### Row 4. Synaptic learning: STDP with a neuromodulator-gated eligibility trace

**Verdict: needs change.** The principle is right. The time scale and the claimed replacement are wrong.

**The biology.**
- STDP experiments:
  - Markram, Lübke, Frotscher & Sakmann 1997, *Science* 275:213;
  - Bi & Poo 1998, *J Neurosci* 18:10464.
- STDP theory:
  - Gerstner, Kempter, van Hemmen & Wagner 1996, *Nature* 383:76;
  - Song, Miller & Abbott 2000, *Nat Neurosci* 3:919;
  - van Rossum, Bi & Turrigiano 2000, *J Neurosci* 20:8812 (soft bounds, stability).
- Triplet and voltage rules:
  - Pfister & Gerstner 2006, *J Neurosci* 26:9673;
  - Clopath, Büsing, Vasilaki & Gerstner 2010, *Nat Neurosci* 13:344.
- Three-factor rules:
  - Izhikevich 2007, *Cereb Cortex* 17:2443;
  - Frémaux & Gerstner 2016, *Front Neural Circuits* 9:85;
  - Gerstner, Lehmann, Liakoni, Corneil & Brea 2018, *Front Neural Circuits* 12:53.
- Experimental eligibility windows of about 1–10 s:
  - Yagishita et al. 2014, *Science* 345:1616;
  - He et al. 2015, *Neuron* 88:528 [U];
  - Brzosko, Schultz & Paulsen 2015, *eLife* 4:e09685 [U];
  - behavioural-timescale plasticity: Bittner et al. 2017, *Science* 357:1033.
- Theory of reward-modulated STDP:
  - Legenstein, Pecevski & Maass 2008, *PLoS Comput Biol* 4:e1000180 [U];
  - Frémaux, Sprekeler & Gerstner 2010, *J Neurosci* 30:13326 [U].

**What the row gets wrong.**
1. **Time scale.** A ±20 ms STDP window covers less than one byte, so the timing part of STDP only orders spikes within a byte. Associations across bytes, and credit that arrives at the next byte or later, need the seconds-long eligibility of Gerstner 2018 and Yagishita 2014, or the plateau-gated window of Bittner 2017. lt5's `tau_el` = 4 bytes is roughly 140 ms at 35 ms per byte. That is short of the biological 1–10 s, and short of the relay latency once row 3 is in place.
2. **Requirements for the third factor that the table omits.**
   - A reward-modulated rule works only if the third factor is a prediction error, that is, reward minus expected reward (Frémaux et al. 2010; Schultz, Dayan & Montague 1997, *Science* 275:1593). lt5 meets this with `m - eav`.
   - The rule needs variability in firing to correlate with. Without stochastic exploration the gradient estimate disappears:
     - Seung 2003, *Neuron* 40:1063 [U];
     - Fiete & Seung 2006, *Phys Rev Lett* 97:048104 [U].
   - With one scalar shared by N cells, the variance grows with N and learning slows roughly in proportion (Werfel, Xie & Seung 2005, *Neural Comput* 17:2699 [U]).

   The table assumes the eligibility trace plus a neuromodulator solves credit assignment. It solves the timing gap. It does not solve the spatial credit problem at scale.
3. **"Replacing the current feature rule and normalization."**
   - The current feature rule, `eh * x * (1 - w)` for an input active in this or the previous byte when the cell fires, is already a soft-bounded causal Hebbian rule over a one-byte window. STDP would refine it, not replace it.
   - The normalization by active inputs (`step / (na + 1)`) is on the output-cell rule, not the tissue rule. STDP does not take its place. Row 5 also claims to replace it (see section 2).

**Owner's rules.**
- The two-factor part uses only what is present at the synapse: both ends (s.16). Allowed.
- A third factor delivered locally (s.9) is allowed.
- The eligibility is per synapse and local. Allowed.
- Nothing is sent back along a forward connection (s.4.1). Allowed, provided the third factor arrives on separate synapses or a separate compartment, as lt5 does.

**Cost.**
- Standard lazy STDP keeps a trace per neuron, not per synapse (Morrison, Diesmann & Gerstner 2008, *Biol Cybern* 98:459).
- To keep the cost at O(active synaptic events), update a synapse only when its sending cell spikes, reading the receiving cell's spike history. That is NEST's archiving scheme.
- lt5 instead loops over *all* incoming synapses of every cell that fires (`forward`, lines 313–320) and of every cell that receives the third factor (lines 262–268). That is O(fan-in) per spike. With fan-in in the tens of thousands, that loop dominates.
- Neuromodulated eligibility can also be lazy. NEST's dopamine-modulated STDP synapse keeps the modulator's spike history in a "volume transmitter" and integrates the weight in closed form at the next spike from the sending cell (Potjans, Morrison & Diesmann 2010, *Front Comput Neurosci* 4:141 [U]).

**Functional consequence.**
- With a seconds-long eligibility and a local prediction-error signal, learning on a delay of 1–2 bytes is plausible.
- How well it delivers credit through several stages of tissue depends on the feedback pathway (row 7), not on this row.

### Row 5. Stability of total input: synaptic scaling

**Verdict: needs change.** The biology is right. It conflicts with s.16, and it does not do the job the table assigns it.

**The biology.**
- Turrigiano, Leslie, Desai, Rutherford & Nelson 1998, *Nature* 391:892;
- Turrigiano & Nelson 2004, *Nat Rev Neurosci* 5:97 [U];
- Turrigiano 2008, *Cell* 135:422.

Scaling is multiplicative, set by the receiving cell, and slow (hours to days).

**What the row gets wrong.**
1. **Owner's s.16.** A connection's strength is set by both of its ends. Synaptic scaling is set by the receiving cell alone. Either:
   - pair it with presynaptic homeostatic plasticity, an established phenomenon (for example the work of G. W. Davis on the fly neuromuscular junction; the auditor cannot give an exact citation, so it is unverified); or
   - write the effective weight as `a_pre * g_post * w_ij`, with each factor kept by its own cell.
2. **It does not replace the normalization.**
   - The 1/(na+1) in lt5's output rule is a learning-rate normalization, the same as normalized least mean squares. It stabilizes the error-correction step whatever the number of inputs.
   - Synaptic scaling regulates the firing rate on a time scale far slower than one learning step. It cannot stop a single large step from overshooting.
   - The biological counterpart of a fast, total-conserving adjustment is heterosynaptic plasticity, or conservation of synaptic resources (Chistiakova, Bannon, Bazhenov & Volgushev 2014, *Neuroscientist* 20:483 [U]).
   - SORN, which combines STDP, intrinsic plasticity, synaptic normalization and structural plasticity, uses explicit normalization of the summed input (Lazar, Pipa & Triesch 2009, *Front Comput Neurosci* 3:23).
3. **Too slow to stabilize Hebbian growth on its own.** This is the "temporal paradox" (Zenke, Gerstner & Ganguli 2017, *Curr Opin Neurobiol* 43:166 [U]). A fast rate detector is needed (Zenke, Hennequin & Gerstner 2013, *PLoS Comput Biol* 9:e1003330).

**Cost.** A per-cell scale factor g_i:
- the effective weight is g_i × w_ij;
- a Hebbian step adds Δ/g_i to w_ij;
- pruning compares g_i × w_ij with the threshold;
- each update is O(1), with no pass over the inputs.

The cost is fine.

**Functional consequence.** Scaling stabilizes rates over long times and keeps relative weights, and with them stored patterns. It does nothing for memory over 40 bytes or for credit.

### Row 6. Making and withdrawing connections: Butz–van Ooyen structural plasticity

**Verdict: needs change.** The named model withdraws connections at random, which breaks s.19 on its own.

**The biology and the model.**
- Butz & van Ooyen 2013, *PLoS Comput Biol* 9:e1003259 (rules for forming spines and boutons);
- review: Butz, Wörgötter & van Ooyen 2009, *Brain Res Rev* 60:287 [U];
- implementation in NEST: Diaz-Pier, Naveau, Butz-Ostendorf & Morrison 2016, *Front Neuroanat* 10:57 [U];
- experimental spine turnover and learning:
  - Holtmaat & Svoboda 2009, *Nat Rev Neurosci* 10:647;
  - Xu et al. 2009, *Nature* 462:915;
  - Yang, Pan & Gan 2009, *Nature* 462:920;
  - Hayashi-Takagi et al. 2015, *Nature* 525:333 [U].

In Butz–van Ooyen:
- the numbers of axonal and dendritic elements grow or shrink as a Gaussian-shaped function of calcium, a proxy for the rate, relative to a set point;
- free elements pair at random, weighted by distance;
- when elements retract, a synapse is deleted *at random* among the cell's synapses.

**What the row gets wrong.**
1. **Owner's s.19.** A cell withdraws a connection over which it keeps receiving what is of no use to it. Butz–van Ooyen withdraws by the cell's overall rate, not by the usefulness of the connection. Combine it with weight- or use-dependent survival:
   - Deger, Helias, Rotter & Diesmann 2012, *PLoS Comput Biol* 8:e1002689 [U];
   - Fauth, Wörgötter & Tetzlaff 2015, *PLoS Comput Biol* 11:e1004031 [U];
   - Knoblauch & Sommer 2016, *Front Neuroanat* 10:63 [U].

   lt5's present rule (strength decays by a price; withdraw when below w_min) is a crude use-dependent survival rule and is closer to s.19 than Butz–van Ooyen alone. The right model keeps formation by need (Butz–van Ooyen) and removal by low weight or use (Deger, Fauth).
2. **Output cells have no activity set point.** lt5 makes their synapses by recent error (`g_acc * eav`), not by need. Butz–van Ooyen does not cover this. The biological counterpart is formation and stabilization of spines driven by learning or reward (Yagishita 2014; Xu 2009; Hayashi-Takagi 2015). This needs its own row.
3. **A fourth homeostat.** Butz–van Ooyen servos the rate to a set point, as do rows 5 and 8 and lt5's threshold homeostasis (section 2).

**Owner's rules.**
- Formation needs a free element on *both* ends (a bouton and a spine). That agrees closely with s.16.
- No hard cap; the count emerges. Agrees.
- The growth constants (ν, η, ε) are rates. Agrees.
- Pairing within a fixed neighbourhood: use a distance kernel cut off at r. Agrees.
- Very large fan-in is possible, because fan-in settles where (drive needed)/(weight × rate of the sending cell) balances.

**Cost.**
- Naive pairing over all free elements is O(N²) per update. Rinke, Butz-Ostendorf, Hermanns, Naveau & Wolf 2018, *J Parallel Distrib Comput* 120 [U], use a Barnes–Hut tree to reach O(N log N).
- Within a fixed neighbourhood, lt5's grid bins with random sampling are O(1) per trial. Keep per-bin pools of free elements.
- The real difficulty: Butz–van Ooyen lets *silent* cells grow the most. A silent cell receives no events, so an event-driven tissue never touches it. Two ways out:
  - give each cell a wake-up event at the closed-form time when its element count next crosses an integer, kept in a calendar queue at O(log N) per event;
  - or add spontaneous firing or release, which is biological.

  Either costs time proportional to the number of cells, not to the active part. That is irreducible for any mechanism that rescues silent cells, and the design should say so.
- lt5's sweep every 2,000 bytes (line 343) is a global scan, which PERSISTENT_CORE_REQUIREMENTS asks to avoid. Pruning can be made lazy at delivery: `spread()` already visits each out-synapse with `weff()`, but it never cuts.

**Functional consequence.**
- The row lets fan-in grow to what is useful and prunes what carries noise.
- It conveys no information or credit by itself.
- Measured in lt5: 3.29M synapses made against 3.07M cut. The churn is about 10 times the standing population, which suggests the present price withdraws much of what it makes before it can prove useful.

### Row 7. Feedback: neuromodulatory (dopamine) cells release right/wrong locally and are read through receptors

**Verdict: wrong as a biological identification. The design itself (feedback cells plus a learned relay) should stay.**

**Why dopamine is the wrong model.**
1. **Dopamine is not local.** Dopamine cells sit in the midbrain (VTA and SNc) and project across most of the forebrain. Each release site acts over micrometres:
   - Rice & Cragg 2008, *Brain Res Rev* 58:303 [U];
   - Liu, Kershberg, Wang, Schneeberger & Kaeser 2018, *Cell* 172:706 [U].

   The cells, however, are a long-range broadcast system, which s.9 (no signal reaches every cell) and checklist item 1 (no global connections) exclude.
2. **Dopamine carries a scalar reward prediction error** (Schultz et al. 1997), not "the next byte was wrong at value v".
   - With 128 values, yes/no carries at most about one bit per byte.
   - lt5's sites already carry value-specific + and − signals: `ans - mean sigm(z)` per value. That is the "right answer arrives as input" form of s.3.3 and s.14, which a scalar would throw away.
   - "Each cell reads it with its own receptors" is good biology: D1 and D2 receptors have opposite effects on plasticity (Shen, Flajolet, Greengard & Surmeier 2008, *Science* 321:848). It fits s.7.2.
3. **Better-matched models** for "dedicated cells receive the answer and pass it on over learned connections":
   - **Climbing fibres from the inferior olive in the cerebellum.** Dedicated comparator cells receive mismatch signals and deliver a teaching signal to Purkinje cells. Plasticity of the parallel-fibre synapses has eligibility timing tuned to the feedback delay:
     - Marr 1969, *J Physiol* 202:437;
     - Albus 1971, *Math Biosci* 10:25;
     - Ito 2001, *Physiol Rev* 81:1143;
     - Suvrathan, Payne & Raymond 2016, *Neuron* 92:959 [U].
   - **Prediction-error neurons in cortex, built from ordinary excitatory and inhibitory synapses.**
     - Keller & Mrsic-Flogel 2018, *Neuron* 100:424.
     - These can be learned by inhibitory plasticity (Hertäg & Sprekeler 2020, *eLife* 9:e57541 [U]).
     - lt5's feedback site computes `ans - mean sigm(zout)` with a hand-written formula. It could instead be a positive error cell (excited by the answer line, inhibited by the output cells of its value) and a negative error cell (the reverse), with Vogels-style balancing. That would put the comparator inside the tissue's own rules.
   - **Top-down feedback onto a separate dendritic compartment.** Feedback arrives on apical dendrites and gates plasticity through bursts or plateaus. This is the "separate part of the cell" of s.7.1:
     - Larkum, Zhu & Sakmann 1999, *Nature* 398:338;
     - Larkum 2013, *Trends Neurosci* 36:141;
     - Guerguiev, Lillicrap & Richards 2017, *eLife* 6:e22901;
     - Sacramento, Costa, Bengio & Senn 2018, *NeurIPS*;
     - Payeur, Guerguiev, Zenke, Richards & Naud 2021, *Nat Neurosci* 24:1010 [U].
   - **Learned or random feedback pathways for credit:**
     - feedback alignment: Lillicrap, Cownden, Tweed & Akerman 2016, *Nat Commun* 7:13276;
     - e-prop, which combines eligibility traces with learning signals over feedback weights and is the closest known algorithm to lt5's feedback path: Bellec et al. 2020, *Nat Commun* 11:3625;
     - attention-gated reinforcement learning: AGREL, Roelfsema & van Ooyen 2005, *Neural Comput* 17:2176 [U]; AuGMEnT, Rombouts, Bohte & Roelfsema 2015, *PLoS Comput Biol* 11:e1004060 [U]; review, Roelfsema & Holtmaat 2018, *Nat Rev Neurosci* 19:166;
     - learned feedback without weight transport: Kolen & Pollack 1994 [U, a conference paper]; Akrout et al. 2019, *NeurIPS*.

     All of these fall inside rulings 4, 5 and 7 (learned feedback; Kolen–Pollack; anything short of backprop across the whole network).

**Owner's rules.**
- Local release (s.9) and per-cell reading (s.7.2) agree.
- Only feedback cells read answer lines; lt5 audits this. Agrees.
- A relay on separate kinds of cells and synapses agrees with s.4.1. If row 3 merges relay and tissue cells into one continuous-time population, that guarantee is lost. Keep the relay kinds separate.

**Cost.**
- Release into a region: store each release in the history of its grid bin. A cell reads it when it is next touched, as NEST's volume transmitter does. O(1) per release plus O(1) per touched cell.
- lt5's present relay loops over every T_RD out-synapse of each feedback and relay cell. There are 138,900 such synapses, they only grow under the `(1 - w)` rule, and they are a likely contributor to the c6 exponent (an inference; not profiled).

**Functional consequence.**
- e-prop and feedback alignment learn in small recurrent and spiking networks.
- In lt5 the relay is worth about 0.01–0.06 nats per byte (c5).
- Nobody has shown that purely local, learned feedback in a spiking tissue delivers useful credit several stages deep at language-model scale. That is an open question, not an engineering detail.

### Row 8. Activity: intrinsic plasticity plus inhibitory plasticity

**Verdict: sound. Name the models, and give the four homeostats distinct roles.**

**The biology and the models.**
- Intrinsic plasticity:
  - Desai, Rutherford & Turrigiano 1999, *Nat Neurosci* 2:515;
  - Stemmler & Koch 1999, *Nat Neurosci* 2:521 [U];
  - Triesch 2007, *Neural Comput* 19:885 [U];
  - SORN (Lazar et al. 2009).
- Inhibitory plasticity: Vogels, Sprekeler, Zenke, Clopath & Gerstner 2011, *Science* 334:1569. The rule is symmetric iSTDP plus a depression term on each presynaptic spike, set by a target rate ρ0.

**lt5's present form.**
- Its rule `ei * trace(pre) * (rate - rho) / rho` at each spike of the receiving cell is a rate-based stand-in for the Vogels rule. It is close in spirit but not the model.
- Its threshold homeostasis is a form of intrinsic plasticity, with two engineering clamps:
  - thresholds clamped to [0.02, 20];
  - the relative deviation clamped to [−1, 1].

**Owner's rules.**
- Set points are rates; no firing share. Agrees.
- The Vogels rule uses pre and post activity, so both ends (s.16). Agrees.
- Dale's principle (the inhibitory sign stays fixed) holds.

**Cost.** Both are lazy: the threshold per cell is already lazy, and iSTDP can be made lazy like row 4.

**Functional consequence.** These rows keep the tissue in a working regime of balanced, irregular firing. They do not carry memory or credit.

---

## 2. Rows that conflict with each other

1. **Rows 1 and 3, on time scale.** Row 3 needs delays short compared with a byte, so that several stages fit in one. Row 1 needs delays tens to hundreds of bytes long. One delay distribution cannot do both. Resolve it as described under row 1:
   - delays work within a byte;
   - a declared sensory buffer and slow per-cell traces cover the reach into the past.
2. **Rows 4 and 5 both claim to replace "the normalization".** Neither one replaces the error-step normalization in the output rule (row 5, point 2). Assign it explicitly: either a heterosynaptic or resource-conservation rule, or keep it, labelled as engineering.
3. **Rows 5, 6 and 8, plus lt5's threshold, are four homeostats on one variable.** Synaptic scaling, Butz–van Ooyen growth, intrinsic plasticity and inhibitory plasticity all drive the rate toward the same ρ.
   - With similar speeds and the same set point they fight or become redundant, and which one wins decides the fan-in, the weights and the thresholds.
   - Assign separate time scales and separate jobs (compare Zenke et al. 2017):
     - inhibitory plasticity: fast, the balance of excitation and inhibition;
     - intrinsic plasticity: medium, the rate;
     - synaptic scaling: slow, the total input;
     - structural plasticity: slowest, the count.
4. **Row 3 and row 7.** Without H, relay latency grows with distance and depth. The eligibility time constant in row 4 must then exceed the slowest path of the relay, so tau_el is set by the architecture rather than drawn freely.
5. **Row 6 with lt5's uniform price λ, against the goal of retention.**
   - Every synapse decays by λ = 2e-4 per byte, a time constant of 5,000 bytes, and is cut at w_min.
   - Anything not reinforced within about 1/λ × ln(w/w_min) bytes is erased. That is about 11,000 bytes for w = 0.1.
   - The project's goal is retention under fixed memory (goal statement, 16 Nov). A uniform forgetting price works directly against it (see section 3, item 1).

## 3. Components missing from the table

1. **Consolidation (s.2.5).** The owner's page says a synaptic change is at first easily undone and later becomes lasting, depending on its history. No row covers this, and it is the component that most directly decides persistent learning. Known models:
   - synaptic tagging and capture: Frey & Morris 1997, *Nature* 385:533; Redondo & Morris 2011, *Nat Rev Neurosci* 12:17;
   - a tag–trigger–consolidation model: Clopath, Ziegler, Vasilaki, Büsing & Gerstner 2008, *PLoS Comput Biol* 4:e1000248 [U];
   - cascade models: Fusi, Drew & Abbott 2005, *Neuron* 45:599;
   - a multi-variable synaptic model: Benna & Fusi 2016, *Nat Neurosci* 19:1697;
   - on synaptic volatility: Mongillo, Rumpel & Loewenstein 2017, *Curr Opin Neurobiol* 46:7 [U].

   These replace lt5's single uniform price with a cascade of variables per synapse, each slower than the last. They are lazy: each variable relaxes in closed form between touches.
2. **The episodic store and replay (allowed 22 Nov), and the checklist's "dreaming/replay phases" and "thinking time".**
   - Hippocampal replay: Wilson & McNaughton 1994, *Science* 265:676.
   - Complementary learning systems: McClelland, McNaughton & O'Reilly 1995, *Psychol Rev* 102:419; Kumaran, Hassabis & McClelland 2016, *Trends Cogn Sci* 20:512.

   Nothing in lt5 or in the table provides this.
3. **The output end as part of the tissue.** s.10 says the output end learns how to give out. lt5 has value-tagged output cells, but the byte is "emitted" only by the experimenter's instrument, a softmax over all output cells.
   - There is no row for selection or competition among output cells. Within a neighbourhood that would be local winner-take-all through inhibition; a basal-ganglia-like selection is the biological analogue.
   - There is no row for how output synapses form (see row 6, point 2).
4. **Feedback cells as cells.** The comparator at each site is a hand formula, not a cell with synapses (see row 7, point 3).
5. **Dendritic compartments.** lt5 has two compartments in effect:
   - a distal predictive part (T_DIS, with `g_d` added to the drive);
   - a feedback-reading part (T_RD feeding `m`).

   The table has no row for them. The distal rule is close to HTM temporal memory (Hawkins & Ahmad 2016, *Front Neural Circuits* 10:23): grow a distal synapse when a cell fires unpredicted, depress it when predicted and silent. Related dendritic-prediction learning: Urbanczik & Senn 2014, *Neuron* 81:521. The table should derive both compartments from the dendritic literature (Larkum; Bittner; Urbanczik & Senn).
6. **A source of variability for exploration.** Any three-factor rule needs it (row 4, point 2). Biologically this is stochastic release or channel noise (Seung 2003 [U]). There is no row and no mechanism.
7. **Surprise and uncertainty modulators.** Acetylcholine and noradrenaline set learning rates by expected and unexpected uncertainty (Yu & Dayan 2005, *Neuron* 46:681). This is relevant to continual learning on overlapping streams. Optional.
8. **The input end learning.** s.10 says the input end learns too. lt5's input lines are fixed one-hot lines, KIN copies per byte value per lag. No row covers this.
9. **The instrument.** `p = softmax(beta * D_v / KOUT + c_v)` learns c_v by log loss.
   - A per-value bias learned by log loss is a unigram model sitting inside the readout. The reported nats per byte therefore include a unigram component that the tissue did not produce.
   - The owner rejected unigram baselines (24 Nov), so the score should not quietly contain one.
   - Fix c_v at zero, or report the score with and without c_v. The `biasonly` diagnostic still includes c_v.

## 4. Engineering patches in lt5.c that no row covers

The line numbers refer to `lt5.c` as read.

1. **Micro-steps and hops.** S = 3 micro-steps and H = 2 relay hops; H is capped at 6 because the `mark` encoding is `T*8 + hop` (lines 145, 245). These are counts. Row 3 covers S and H, but the cap is a code artifact.
2. **Input.** A global lag set (`PASSAGES`), KIN copies of each one-hot line, and 7-bit bytes (`x & 127`, V = 128).
   - The last of these drops the high bit of every UTF-8 byte, so multibyte (for example Chinese) text is aliased.
   - This is a data decision that no row records.
3. **The distal prediction rule** (lines 286–287, 298, 321–331):
   - p is clamped at 1;
   - the gain is `g_d`;
   - a synapse grows when p < 0.5, with probability `p_dis`;
   - the updates are `+ed*(1-p)` and `-ed*p`.

   This is HTM-like, not in the table.
4. **The feedback-site comparator** (lines 202–210): `ans - mean sigm(zout)`, split into + and −, gated by a threshold `eps`. The site reads the drive of the output cells, including their bias b, rather than their firing.
5. **The output-cell rule** (lines 216–228):
   - the error is divided by the number of sites;
   - the bias learns at `eta_b * e`;
   - the step is divided by (active inputs + 1);
   - weights are clipped at 0 or above;
   - the running error average uses hard-coded 0.98 and 0.02;
   - acceptance is `g_acc * eav`;
   - there are `tr_out` growth trials;
   - new synapses start at 2 × w_min or 3 × w_min.
6. **The rule for synapses that read feedback** (line 239): `eta_q * sf * xi * (1 - w)` only potentiates. Only the price shrinks it. A new one is made with probability `trace(i)`.
7. **The relay** (lines 253–258): accumulation clamped at 1; its own Hebbian rule `eta_r`; relay cells accept connections by need.
8. **The tissue third factor** (line 263): `m - eav` with a hard-coded 0.01 average; weights clipped to [0, 1]. The same `eav` array also holds the output cells' error average.
9. **Price and pruning.**
   - Every synapse decays by the same price λ.
   - It is withdrawn below `w_min`.
   - There is a global sweep every 2,000 bytes (line 343), a mandatory global scan.
   - `spread()` never prunes.
10. **Homeostasis details** (lines 94–100):
    - the threshold is clamped to [0.02, 20];
    - the relative deviation is clamped to [−1, 1];
    - the catch-up uses the current rate for the whole gap d rather than integrating it;
    - a leak of −30 is used for τ ≤ 1;
    - `tau_el` is set to `max(tau, tel)`.
11. **Firing and recurrence.**
    - Firing resets the potential to 0, and a cell fires at most once per byte (a hidden refractory rule).
    - Recurrence from the previous byte is injected only at micro-step 0.
12. **Offer routing** (line 336): `roll % 3` splits offers equally among target kinds. This is a fixed allocation ratio that works like a count. There is also `p_off`.
13. **O(fan-in) loops that do not follow the active part:**
    - `connected()` is a linear scan;
    - the count of active inputs at output cells (line 221);
    - the eligibility loop over all incoming synapses of each fired cell (lines 315–316);
    - the third-factor loop over all incoming synapses (line 266).

    With the fan-in the owner allows, these dominate. They should be pre-spike-driven or hashed.
14. **Neighbour sampling.** It samples a bin uniformly, then a cell uniformly within that bin. This over-samples cells in sparse bins, so neighbour choice is slightly non-uniform. Minor.
15. **The instrument's learned `beta` and `c_v`** (section 3, item 9).

## 5. Corrected table

| Component | Biological principle | Known model and references | Implementation consistent with the owner's rules and the cost rule |
|---|---|---|---|
| Time base | Bytes arrive at the world's rate; all biological constants are converted at a declared rate in milliseconds per byte | — | Declare the model milliseconds per byte (for example about 35). Convert every constant. Name the byte-duration-to-delay ratio as a time constant, not a stage count. |
| Reading the past | Seconds-scale access comes from slow processes: heterogeneous slow traces, slow currents, time cells. Delays act within a byte. | Shankar & Howard 2012 [U]; MacDonald et al. 2011; Egorov et al. 2002; Izhikevich 2006 (delays) | Per-cell time constants drawn log-spaced. Per-synapse delays drawn from a broad distribution; the set kept by row 6 forms each cell's passages, so no count of passages is set. Any long buffer is declared a sensory buffer (ruling of 10 Nov), with lags per synapse, not a global table. Lazy traces; arrival ring. |
| Working memory over seconds | Short-term facilitation and depression; persistent and slow currents | Tsodyks & Markram 1997; Markram et al. 1998; Mongillo et al. 2008 | Tsodyks–Markram u and x per sending cell, or per synapse if they are to depend on the target. Exact relaxation between spikes; updated only on delivery. Read through learned synapses. Expect a capacity of a few items. |
| Propagation | Continuous time, delayed spikes, refractoriness, fast inhibition | Brette et al. 2007; Morrison et al. 2005 [U] | Event queue (bucketed ring at O(1), or a heap at O(log n)). The cascade is bounded by refractoriness and inhibition, never by a count of events. |
| Synaptic learning | STDP within a byte; three-factor learning with seconds-long eligibility across bytes; the third factor is a prediction error; exploration from stochastic firing | Bi & Poo 1998; Markram et al. 1997; Izhikevich 2007; Frémaux & Gerstner 2016; Gerstner et al. 2018; Yagishita et al. 2014; Bittner et al. 2017 | STDP updated lazily at each spike of the sending cell, from the receiving cell's spike history (Morrison et al. 2008). Eligibility per synapse with closed-form decay, and τ_el longer than the slowest relay path. Third factor = what the cell reads minus its running average, applied lazily (Potjans et al. 2010 [U]). Add a declared noise source. |
| Error-step normalization | Heterosynaptic plasticity / conservation of synaptic resources (if it is kept) | Chistiakova et al. 2014 [U] | Either derive it from that model or keep 1/(na+1), labelled as engineering. Not replaced by synaptic scaling. |
| Total input (slow) | Synaptic scaling (postsynaptic) together with presynaptic homeostasis, so that strength is set by both ends (s.16) | Turrigiano et al. 1998; Turrigiano 2008; presynaptic homeostasis (G. W. Davis, unverified) | Effective weight = a_pre × g_post × w. Each factor is O(1) and lazy. Slowest of the homeostats except structural plasticity. |
| Making and withdrawing connections | Formation by need of both ends; removal of connections of little use; spine formation and stabilization driven by learning | Butz & van Ooyen 2013 (formation); Deger et al. 2012 [U] and Fauth et al. 2015 [U] (removal by weight or use); Xu et al. 2009; Yang et al. 2009; Yagishita et al. 2014 | Free elements on both ends, paired within a neighbourhood from per-bin pools (O(1)). Removal by low effective weight or use, checked lazily at delivery. No global sweep. Wake-up events (or spontaneous activity) for silent cells, at a cost proportional to cells × time, stated as such. Output synapses form by local error or third factor. |
| Consolidation (missing) | Early, labile changes become lasting according to their history (s.2.5) | Frey & Morris 1997; Clopath et al. 2008 [U]; Fusi et al. 2005; Benna & Fusi 2016 | Several variables per synapse (cascade), each relaxing in closed form between touches. Replaces the uniform price λ as the forgetting mechanism. |
| Feedback comparator (missing as cells) | Prediction-error neurons built from the answer input and inhibition by the output cells; cerebellar comparator as precedent | Keller & Mrsic-Flogel 2018; Hertäg & Sprekeler 2020 [U]; Marr 1969; Albus 1971; Ito 2001 | Positive and negative error cells as ordinary cells: excited by the answer line, inhibited by output cells of their value (and the reverse), balanced by Vogels plasticity. Only they read the answer lines. |
| Feedback relay and reading | Learned feedback pathways onto a separate dendritic compartment; plasticity gated by plateaus or bursts; local modulators read through each cell's receptors | Bellec et al. 2020 (e-prop); Lillicrap et al. 2016; Roelfsema & Holtmaat 2018; Guerguiev et al. 2017; Payeur et al. 2021 [U]; Larkum 2013; Shen et al. 2008 | Keep separate relay kinds (s.4.1). The feedback compartment carries the learned signal. Relay synapses learn by a local, both-ends rule that can also depress. Releases are stored per grid bin and read lazily. Not "dopamine". |
| Activity | Inhibitory plasticity (fast balance); intrinsic plasticity (rate) | Vogels et al. 2011; Desai et al. 1999; Triesch 2007 [U]; Lazar et al. 2009 | The Vogels rule itself, event-driven. Intrinsic plasticity on the threshold. Distinct time scales from scaling and structural plasticity. |
| Dendritic prediction | Distal and apical compartments predict and gate somatic firing | Hawkins & Ahmad 2016; Urbanczik & Senn 2014; Larkum et al. 1999 | Derive lt5's distal rule from one of these instead of the present hand formula. |
| Output end (missing) | Local competition and selection among output cells | Local winner-take-all through inhibition (no single reference) | Output cells inhibit each other within a neighbourhood, and the emitted byte comes from the tissue. The instrument only reads it. |
| Replay (missing) | Rehearsal during rest; complementary learning systems | Wilson & McNaughton 1994; McClelland et al. 1995; Kumaran et al. 2016 | A small episodic store replayed through the same local rules during the "dream" phases on the checklist. |
| Instrument | (not biology) | — | No learned per-value bias c_v, or report the score both with and without it. |

## 6. Most important changes, ranked

1. **Fix the time base, and with it the conflict between rows 1 and 3.** Delays of milliseconds cannot read passages tens to hundreds of bytes back. Reach into the past must come from slow traces, short-term plasticity and slow currents, or from a declared sensory buffer.
2. **Add consolidation (s.2.5) and retire the uniform decay price λ as the forgetting mechanism.** With a 5,000-byte decay on every synapse, retention under fixed memory, the project's stated goal, is ruled out by design.
3. **Re-derive the feedback row.**
   - Feedback is not dopamine.
   - Comparator cells should be built from ordinary synapses.
   - The relay should be a learned pathway onto a separate compartment, in the line of e-prop, feedback alignment and AGREL.
   - Keep the value-specific signal that s.14 asks for, not a scalar right/wrong.
4. **Give eligibility seconds, not 4 bytes, and add exploration noise.** Without them a three-factor rule cannot bridge the delay of the answer and the relay, or estimate a direction.
5. **Add use-based removal to structural plasticity.** Butz–van Ooyen alone withdraws at random, against s.19. Remove the global 2,000-byte sweep.
6. **Separate the four homeostats by time scale and role.** Make synaptic scaling a two-ended factor (s.16). Assign the error-step normalization explicitly.
7. **Make every rule pre-spike-driven and lazy.**
   - Remove the O(fan-in) loops (section 4, item 13).
   - Bound the growth of feedback-reading synapses.
   - The measured cost exponent of 2.03 fails the cost rule already, before any row is added.
8. **Remove the learned unigram bias c_v from the instrument, or report the score both with and without it.**

## 7. Claims the auditor could not verify

- **Citations marked [U].** Their details (volume, pages, year, sometimes the exact title) are from memory and unchecked: Carr & Konishi 1990; Fields 2015; Pajevic et al. 2014; Shankar & Howard 2012; Howard et al. 2014; Wang et al. 2006; Mi et al. 2017; Morrison et al. 2005 and 2007; He et al. 2015; Brzosko et al. 2015; Legenstein et al. 2008; Frémaux et al. 2010; Seung 2003; Fiete & Seung 2006; Werfel et al. 2005; Potjans et al. 2010; Turrigiano & Nelson 2004; Chistiakova et al. 2014; Zenke et al. 2017; Butz et al. 2009; Diaz-Pier et al. 2016; Rinke et al. 2018 (the page numbers especially); Hayashi-Takagi et al. 2015; Deger et al. 2012; Fauth et al. 2015; Knoblauch & Sommer 2016; Rice & Cragg 2008; Liu et al. 2018; Suvrathan et al. 2016; Hertäg & Sprekeler 2020; Payeur et al. 2021; Roelfsema & van Ooyen 2005; Rombouts et al. 2015; Kolen & Pollack 1994; Stemmler & Koch 1999; Triesch 2007; Clopath et al. 2008; Mongillo et al. 2017.
- **Presynaptic homeostatic plasticity (G. W. Davis).** No precise citation could be given.
- **The time constants in Mongillo et al. 2008** (τ_F ≈ 1.5 s, τ_D ≈ 0.2 s) are recalled, not checked.
- **The figure of about 35 ms per byte** is the auditor's estimate from reading speed. The design must choose its own.
- **That the 138,900 feedback-reading synapses contribute to the c6 cost exponent** is an inference from the code, not a profile.
- **All check results** (c2, c4, c5, c6) come from `checks/checks_out.txt` as found. The auditor did not rerun them.
- **The state of the field.** Whether any local three-factor scheme with learned feedback delivers useful credit several stages deep at language scale is, as far as the auditor knows, not established anywhere.
