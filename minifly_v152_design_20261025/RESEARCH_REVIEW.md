# V152 — what the FLM repository contributes

This is a source-level review at pinned commits, not an independent training reproduction. The linked repository helped specify the mechanism comparison; no FLM checkpoint, tokenizer, graph, trained representation or task data is imported into a MiniFly candidate.

## Findings used

**Language training versus local credit.** The inspected language trainer explicitly uses cross-entropy, backward propagation and AdamW. That path cannot be cited as a successful native/local-learning implementation. Its tokenized language interface and optimization/data budget differ from this project's byte/feedback setting. [Source](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/flm/language_train.py).

**Temporal state as a separable mechanism.** The rate core has fast and slow states plus trained input/output mappings. V152 borrows the idea of isolating temporal computation and credit, not its exact graph or lexical system. The numerical constants, smaller graph, raw-byte interface and C16 port in the proposed comparator are explicitly different. [Source](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/flm/model.py).

**Approximate eligibility.** The local implementation documents that it drops cross-neuron temporal paths. It retains neuron-local temporal derivatives and uses instantaneous output derivatives. V152 specifies analytic local updates rather than silently treating an autograd-assisted implementation as entirely biological; BP remains an independent diagnostic condition. [Source](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/flm/local_learning.py).

**Ablation interpretation.** The published language-core report shows a much larger loss difference when all temporal state is removed than when lateral recurrence alone is removed. Those are exploratory fitted-system contrasts on that corpus, not percentages of intelligence, a topology theorem or proof that recurrence will solve addition. We do not reproduce its numerical language ranking in our success criteria. [Report](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/docs/LANGUAGE-CORE-RESULTS.md).

**Local-learning evidence is task-specific.** The separate paper compares delayed cue association with engineered sensory/action channels. It does not establish the same local method on arithmetic or the full language setup. V152 therefore treats its own BP reference as unproven until bounded acquisition qualification succeeds. [Paper](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/papers/local-learning.tex).

**V149 is not a matched gap estimate.** Its development readout used 40 KB training, a wide regularization search and an evaluator-side optimizer. The native V150 text dose was about 2 KB per corpus. A difference between those scores cannot be assigned entirely to native credit. Existing V149 results motivate caution about input usability, without authorizing another wide readout sweep here. [Development record](https://github.com/ChangweiZhou/digitalbaby/blob/f73b7ff91b2da554f83cfc8e7f18c98a4b8a91a9/minifly_v149_input_ceiling_dev_20261025/cycles/CYCLE2.md).

## Corrections to the proposed bridge

- “BPTT upper reference” means an exact gradient comparator for the same objective and model. A particular BP training run is not a mathematical upper bound on achievable accuracy and is not guaranteed to learn at this dose.
- Gradients can be compared in shared parameter coordinates inside the recurrent model. They cannot be meaningfully subtracted from native fast/slow state updates with different dimensions and semantics.
- Output bias can express arbitrary symbol frequencies, but a fixed bias alone cannot greedily emit a nonempty varying sequence and EOS. Complete-reply/position-frequency controls are stronger than simply beating a bias-only NLL baseline.
- A new trained input mapping, output decoder and temporal-credit rule should not all change in one native arm. Native input and plasticity remain pinned; the generic bias is separated. The independent comparator tests its own U/edge credit and cannot inherit the native lineage.
- A fixed random core followed by a trained decoder could blur the ELM exclusion. V152 contains no such training route. Its head-only counterfactual is a single update at a learned state, not a reservoir model.
- “Gradually replace BP and arrive at native MiniFly” is not an established integration plan. A future port must identify actual native neurons, signals, update state and readout. No integrated result is claimed in advance.

## Research boundary

V152 uses addition acquisition, interference and sealed never-taught relations. It does not copy FLM's natural-language objective, BPE, anatomical claims, Transformer baseline, global credit into a candidate, or unsupported language-to-knowledge inference. Its local reference is biologically motivated computation, not certified synaptic biology. The native arms remain the only possible native improvement evidence, and even favorable development results do not authorize adoption.

Sources were read through the GitHub connector after ordinary browser/terminal retrieval failed. Local research cache text is retained for provenance and excluded from a later publication payload. No fetched code was executed.

Current-main check: the fetched model.py, language_train.py and local_learning.py contents match the pinned versions used above. The current README also describes KC selection previews and larger engineering configurations with pending studies. Therefore the older selected language graph must not be generalized into a claim that no FLM variant includes KCs. These ongoing variants do not supply completed MiniFly integration evidence. [Current repository README](https://github.com/Kuberwastaken/flm/blob/main/README.md).

## How the proposed outcomes change the next decision

| Observed pattern | Interpretation allowed |
|---|---|
| Native bias improves loss, but frequency/position controls explain exact answers | Better calibration; no native generalization claim |
| Native hybrid improves retained unseen exact answers with prior-teaching and path controls | Native development evidence worth separate confirmation |
| R_BP acquires answers, R_LE fails | Approximate credit is inadequate in that reference at this dose; native failure still has other possible causes |
| R_BP and R_LE learn, R_LE0 fails | Eligibility history helps that reference; a native port remains unimplemented |
| All three recurrent methods learn similarly | This short task does not demonstrate a need for the proposed temporal-credit approximation |
| The selected BP reference cannot acquire taught answers | Stop at REFERENCE_NOT_READY; do not infer a fundamental limitation of local learning |

The native comparison preserves an old-rule control. Its N_A versus N_2B contrast changes objective and amplitude together, while the N_2B_Q versus N_2B contrast adds only q. The gradient comparison changes credit computation inside one reference architecture. These contrasts must not be combined into a single claim that one mechanism explains every historical MiniFly failure.
