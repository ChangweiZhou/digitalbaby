# Fly mushroom body architecture distillation

Script version: `1.0.0`

## The architecture in one paragraph

The mushroom body is a **parallel multi-stream associative memory** with three
independently modulated lobes. Each lobe has its own Kenyon cell (KC) population,
output neurons (MBONs), and dopaminergic teaching neurons (DANs). Within each lobe,
specific (MBON, DAN) pairs share overlapping KC axon segments, forming **compartments**
where plasticity is locally modulated. MBONs form a **recurrent processing layer**
with mixed excitatory/inhibitory interactions before feeding back to DANs. A single
giant APL neuron enforces ~5% KC activation sparsity across all lobes. The three lobes
differ in DAN density, valence bias, and cross-lobe coupling, suggesting they handle
different temporal and motivational aspects of memory.

## Three parallel learning streams

### gamma lobe

- **KCs**: 2055
- **MBONs**: 39 neurons (15 types)
  - By neurotransmitter: acetylcholine: 6, gaba: 8, glutamate: 4, unknown: 21
- **DANs**: 132 neurons (8 types)
  - By family: PAM: 124, PPL: 8
- **DAN density**: 0.064 DANs per KC
- **KC MBON partners**: median 3, max 14
- **KC DAN partners**: median 1, max 11
- Largest lobe. Heavy PPL (punishment) DAN innervation. Rich inhibitory MBON recurrence (GABA, glutamate). Bidirectionally coupled with alpha-prime/beta-prime. Likely handles aversive and short-term memory.

### ab lobe

- **KCs**: 1457
- **MBONs**: 25 neurons (8 types)
  - By neurotransmitter: acetylcholine: 13, glutamate: 8, unknown: 4
- **DANs**: 61 neurons (7 types)
  - By family: PAM: 58, PPL: 3
- **DAN density**: 0.042 DANs per KC
- **KC MBON partners**: median 3, max 13
- **KC DAN partners**: median 0, max 6
- Mid-size, most isolated lobe (minimal cross-lobe connections). Sparse DAN targeting (median 0 DAN partners per KC). No GABA MBONs. Strong glutamate self-inhibition. Likely handles consolidated long-term memory.

### a'b' lobe

- **KCs**: 607
- **MBONs**: 40 neurons (12 types)
  - By neurotransmitter: acetylcholine: 19, gaba: 12, glutamate: 4, unknown: 5
- **DANs**: 100 neurons (9 types)
  - By family: PAM: 98, PPL: 2
- **DAN density**: 0.165 DANs per KC
- **KC MBON partners**: median 6, max 14
- **KC DAN partners**: median 2, max 8
- Smallest KC population but highest DAN density per KC. Cleanest compartment structure (PAM05, PAM06 reward DANs). Strong bidirectional coupling with gamma lobe. Likely handles appetitive and recent memory.

## Lobe-level connectivity

### MBON to MBON recurrence (synapses)

| from / to | gamma | ab | a'b' |
|---|---|---|---|
| gamma | 5350 | 353 | 2205 |
| ab | 143 | 2988 | 126 |
| a'b' | 771 | 130 | 3071 |

Key pattern: gamma and alpha-prime/beta-prime are bidirectionally coupled
(gamma->a'b': 2205, a'b'->gamma: 771),
while alpha/beta is largely isolated (max cross-lobe: 143).

### MBON to DAN feedback (synapses)

| MBON / DAN | gamma | ab | a'b' |
|---|---|---|---|
| gamma | 701 | 44 | 314 |
| ab | 12 | 332 | 43 |
| a'b' | 183 | 56 | 326 |

Feedback is predominantly intra-lobe: DANs receive teaching signals
from MBONs in the same lobe they modulate.

### DAN to DAN recurrence (synapses)

| from / to | gamma | ab | a'b' |
|---|---|---|---|
| gamma | 690 | 6 | 27 |
| ab | 6 | 283 | 8 |
| a'b' | 29 | 5 | 388 |

DAN recurrence is almost entirely intra-lobe (single-digit cross-lobe).
This creates independent teaching-signal dynamics within each lobe.

## Compartment structure

A compartment is a (MBON type, DAN type) pair that shares overlapping
KC axon territory within the same lobe. Listed by Jaccard overlap:

| Lobe | MBON | MBON NT | DAN | DAN family | Jaccard | KC overlap | MBON->DAN | DAN->MBON |
|---|---|---|---|---|---:|---:|---:|---:|
| gamma | MBON12 | acetylcholine | PPL103 | PPL | 0.501 | 522 | 9 | 89 |
| a'b' | MBON26 | ? | PAM05 | PAM | 0.478 | 128 | 2 | 53 |
| a'b' | MBON03 | glutamate | PAM05 | PAM | 0.475 | 208 | 12 | 69 |
| gamma | MBON09 | gaba | PPL103 | PPL | 0.443 | 443 | 3 | 0 |
| a'b' | MBON03 | glutamate | PAM06 | PAM | 0.420 | 185 | 8 | 120 |
| gamma | MBON11 | gaba | PPL103 | PPL | 0.412 | 554 | 3 | 0 |
| gamma | MBON01 | glutamate | PPL103 | PPL | 0.391 | 366 | 4 | 0 |
| a'b' | MBON31 | ? | PAM05 | PAM | 0.352 | 114 | 13 | 0 |
| gamma | MBON20 | gaba | PPL103 | PPL | 0.302 | 281 | 3 | 1 |
| ab | MBON02 | glutamate | PAM04 | PAM | 0.294 | 110 | 38 | 37 |
| a'b' | MBON13 | acetylcholine | PAM06 | PAM | 0.283 | 127 | 2 | 0 |
| a'b' | MBON04 | glutamate | PAM05 | PAM | 0.278 | 82 | 20 | 30 |
| gamma | MBON01 | glutamate | PAM08 | PAM | 0.270 | 175 | 14 | 5 |
| gamma | MBON30 | ? | PPL103 | PPL | 0.262 | 228 | 1 | 5 |
| gamma | MBON01 | glutamate | PAM01 | PAM | 0.252 | 153 | 58 | 47 |
| gamma | MBON05 | glutamate | PPL103 | PPL | 0.242 | 213 | 7 | 1 |
| gamma | MBON22 | ? | PPL103 | PPL | 0.237 | 308 | 4 | 0 |
| gamma | MBON09 | gaba | PAM08 | PAM | 0.226 | 178 | 6 | 5 |
| a'b' | MBON13 | acetylcholine | PAM05 | PAM | 0.214 | 105 | 3 | 0 |
| gamma | MBON12 | acetylcholine | PAM08 | PAM | 0.203 | 183 | 9 | 0 |
| gamma | MBON05 | glutamate | PAM08 | PAM | 0.200 | 102 | 34 | 6 |
| a'b' | MBON17 | acetylcholine | PAM06 | PAM | 0.200 | 62 | 0 | 0 |
| ab | MBON06 | glutamate | PAM04 | PAM | 0.194 | 104 | 0 | 2 |
| gamma | MBON09 | gaba | PAM01 | PAM | 0.188 | 143 | 6 | 1 |
| gamma | MBON20 | gaba | PAM08 | PAM | 0.187 | 115 | 1 | 0 |
| a'b' | MBON04 | glutamate | PAM06 | PAM | 0.187 | 56 | 17 | 10 |
| a'b' | MBON16 | acetylcholine | PAM05 | PAM | 0.179 | 56 | 0 | 0 |
| gamma | MBON32 | ? | PPL103 | PPL | 0.174 | 143 | 1 | 13 |
| gamma | MBON11 | gaba | PAM08 | PAM | 0.170 | 206 | 9 | 1 |
| gamma | MBON21 | ? | PAM08 | PAM | 0.169 | 70 | 30 | 15 |
| gamma | MBON11 | gaba | PPL101 | PPL | 0.168 | 199 | 18 | 140 |
| gamma | MBON29 | ? | PAM01 | PAM | 0.168 | 56 | 6 | 13 |
| gamma | MBON09 | gaba | PPL101 | PPL | 0.167 | 133 | 2 | 1 |
| gamma | MBON21 | ? | PPL103 | PPL | 0.165 | 136 | 0 | 0 |
| gamma | MBON30 | ? | PAM08 | PAM | 0.164 | 87 | 25 | 3 |
| gamma | MBON20 | gaba | PPL101 | PPL | 0.162 | 97 | 12 | 2 |
| a'b' | MBON28 | acetylcholine | PAM05 | PAM | 0.162 | 43 | 0 | 0 |
| ab | MBON02 | glutamate | PAM10 | PAM | 0.160 | 65 | 43 | 0 |
| gamma | MBON12 | acetylcholine | PPL101 | PPL | 0.159 | 144 | 3 | 2 |
| gamma | MBON30 | ? | PPL101 | PPL | 0.159 | 80 | 6 | 1 |
| gamma | MBON20 | gaba | PAM01 | PAM | 0.156 | 90 | 0 | 1 |
| ab | MBON06 | glutamate | PAM10 | PAM | 0.155 | 84 | 18 | 21 |
| gamma | MBON05 | glutamate | PAM01 | PAM | 0.153 | 73 | 6 | 1 |
| ab | MBON24 | ? | PAM04 | PAM | 0.153 | 29 | 4 | 4 |
| a'b' | MBON10 | gaba | PAM05 | PAM | 0.153 | 39 | 3 | 0 |
| gamma | MBON12 | acetylcholine | PAM01 | PAM | 0.152 | 135 | 25 | 2 |

The compartment structure reveals the actual mechanism of local plasticity:
- The DAN innervates a specific segment of KC axons within a lobe.
- The MBON reads from an overlapping segment of the same KC axons.
- Dopamine release from the DAN modifies KC->MBON synaptic strength
  within that shared segment.
- Different compartments can be independently modulated by different DANs.

### Compartment density by lobe

- **gamma**: 29 pairs involving 11/15 MBON types and 4/8 DAN types
- **ab**: 5 pairs involving 3/8 MBON types and 2/7 DAN types
- **a'b'**: 12 pairs involving 9/12 MBON types and 2/9 DAN types

## MBON recurrent processing

MBONs form a recurrent network that processes the KC readout before
feeding back to DANs. This layer implements:

1. **Within-lobe competition** (glutamate and GABA MBONs inhibit other MBONs)
2. **Within-lobe amplification** (ACh MBONs excite other MBONs)
3. **Cross-lobe coordination** (gamma and alpha-prime/beta-prime exchange signals)

Top MBON->MBON pathways by synapse count:

| Source type | NT | Sign | Source lobe | Target lobe | Synapses | Cross-lobe |
|---|---|---|---|---|---:|---|
| MBON06 | glut | inhibitory | ab | ab | 800 | no |
| MBON01 | glut | inhibitory | gamma | gamma | 726 | no |
| MBON02 | glut | inhibitory | ab | ab | 719 | no |
| MBON05 | glut | inhibitory | gamma | gamma | 703 | no |
| MBON11 | gaba | unknown | gamma | gamma | 656 | no |
| MBON09 | gaba | unknown | gamma | gamma | 567 | no |
| MBON14 | acet | excitatory | ab | ab | 561 | no |
| MBON03 | glut | inhibitory | a'b' | a'b' | 555 | no |
| MBON03 | glut | inhibitory | a'b' | gamma | 532 | yes |
| MBON12 | acet | excitatory | gamma | gamma | 393 | no |
| MBON20 | gaba | unknown | gamma | gamma | 367 | no |
| MBON01 | glut | inhibitory | gamma | a'b' | 354 | yes |
| MBON13 | acet | excitatory | a'b' | a'b' | 352 | no |
| MBON04 | glut | inhibitory | a'b' | a'b' | 278 | no |
| MBON18 | acet | excitatory | ab | ab | 261 | no |
| MBON12 | acet | excitatory | gamma | a'b' | 236 | yes |
| MBON19 | acet | excitatory | ab | ab | 219 | no |
| MBON17 | acet | excitatory | a'b' | a'b' | 205 | no |
| MBON11 | gaba | unknown | gamma | ab | 184 | yes |
| MBON07 | glut | inhibitory | ab | ab | 142 | no |

## Global modulation: APL and DPM

### APL (1 neuron)

Giant lateral inhibitor that enforces KC sparsity (~5% activation).
Bidirectional connections with ALL lobes:

- gamma: KC->APL=4580, APL->KC=5247
- ab: KC->APL=4690, APL->KC=5061
- a'b': KC->APL=1307, APL->KC=1764

### DPM (2 neurons)

Global state modulator. Primarily receives from KCs:

- gamma: KC->DPM=491, DPM->KC=60
- ab: KC->DPM=517, DPM->KC=34
- a'b': KC->DPM=249, DPM->KC=14

## The design pattern (abstract)

The mushroom body persistent learning architecture has six components:

### 1. Sparse distributed representation (KC layer)

- ~4500 units with ~5% activation (enforced by global lateral inhibition)
- Three independent subpopulations ("lobes") with different sizes
- Each input activates a sparse, mostly non-overlapping pattern
- Sparsity is the PRIMARY anti-interference mechanism

### 2. Parallel readout streams (MBON layer)

- Multiple readout neurons per lobe, each reading from a different KC subset
- Mixed neurotransmitter types within each lobe:
  - Excitatory (ACh): drive downstream behavior
  - Inhibitory (GABA, glutamate): suppress competing readouts
- Readout neurons form a RECURRENT network (not a simple feedforward layer)

### 3. Compartmented teaching signals (DAN layer)

- Each DAN type innervates a specific KC axon segment within one lobe
- A (DAN, MBON) pair that shares KC territory forms a "compartment"
- Plasticity at KC->MBON synapses is modulated ONLY by the local DAN
- Two DAN families with asymmetric distribution:
  - PAM (reward): 280 neurons, distributed across all lobes
  - PPL (punishment): 21 neurons, concentrated in gamma lobe

### 4. MBON->DAN feedback (teaching circuit)

- MBONs feed back to DANs, creating a closed loop
- Feedback is primarily INTRA-LOBE: each lobe has its own feedback circuit
- The recurrent MBON layer processes the readout before it reaches DANs
- This allows the system to compute PREDICTION ERRORS, not just relay outcomes

### 5. Cross-lobe coupling (memory coordination)

- gamma and alpha-prime/beta-prime are bidirectionally coupled via MBONs
- alpha/beta is relatively isolated from the other two lobes
- This allows recent (gamma) and appetitive (alpha-prime/beta-prime)
  memories to interact while long-term (alpha/beta) memories remain stable

### 6. Global sparsity enforcement (APL)

- Single giant neuron with massive bidirectional KC connections in ALL lobes
- Enforces population-level sparsity constraint on KC activation
- This is NOT a plasticity mechanism -- it is a representation constraint
  that makes all downstream plasticity less prone to interference

## What this means for ELM persistent learning

The fly does NOT protect old memories by gating individual synapses.
It reduces interference through three orthogonal mechanisms:

1. **Sparse representation**: ~5% KC activation ensures low pattern overlap.
   New patterns mostly use different synapses than old ones.

2. **Parallel streams**: Different memory types use different lobes.
   Aversive/short-term (gamma), appetitive/recent (alpha-prime/beta-prime),
   and consolidated (alpha/beta) memories cannot interfere because they
   use physically separate synapses.

3. **Compartmented modulation**: Within a lobe, a DAN only modifies
   KC->MBON synapses in its compartment. Synapses in other compartments
   of the same lobe are not touched. This is structural write protection
   at the compartment level, not the synapse level.

The V20 approach of output-column gain modulation misses all three:
- It does not enforce hidden-layer sparsity
- It uses one readout stream, not parallel streams
- It modulates output columns, not input-specific compartments

### Concrete ELM implementation targets

1. **k-winner-take-all on hidden layer**: Replace unbounded activation with
   a sparsity constraint (top k of N units active). This is the APL analog.

2. **Multiple parallel readout heads**: Each head reads from a different
   subset of hidden units (or all units with different learned projections).
   This is the lobe analog.

3. **Compartmented updates**: Each teaching signal modifies only the synapses
   between its designated hidden-unit subset and its designated output.
   Other synapses in the same head are left untouched. This is the
   DAN compartment analog.

4. **Recurrent readout competition**: Before computing the teaching signal,
   readout heads interact through lateral inhibition/excitation.
   This is the MBON recurrence analog.

The order of priority for implementation is 1 > 3 > 2 > 4.
Sparsity alone may provide significant interference reduction.

## What this analysis does NOT claim

- The lobe functional assignments (aversive/appetitive/consolidated) are
  hypotheses from Drosophila literature, not derived from this connectome data.
- Neurotransmitter sign assignments (ACh=excitatory, GABA=inhibitory,
  glutamate=inhibitory) are standard but context-dependent.
- Compartment pairings are based on KC overlap (Jaccard >= 0.15),
  not on direct observation of co-localized synaptic modulation.
- The three anti-interference mechanisms are identified from structure,
  not validated by a learning simulation.
- This is an architectural description, not a complete model.

## Files

- `ARCHITECTURE.json`: Machine-readable architecture specification
- `COMPARTMENTS.csv`: All compartment pairs with overlap metrics
- `LOBE_PROFILES.json`: Detailed per-lobe population statistics
- `LOBE_CONNECTIVITY.json`: Lobe-level synapse count matrices
- `MBON_RECURRENCE.csv`: Significant MBON->MBON pathways
- `KC_PARTNER_STATS.json`: Per-lobe KC connectivity statistics
- `APL_DPM.json`: Global modulator connectivity