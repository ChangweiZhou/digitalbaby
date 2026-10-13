# Generic byte input and output

The immediate goal is a memory-driven interface that accepts arbitrary byte payloads and generates variable-length byte sequences. This removes the arithmetic-specific input parser, the restricted answer alphabet and the single-byte output slot. Later pretraining, Q&A and mixed arithmetic will use this interface; their task accuracy is not an additional condition for completing this stage.

This is the user's explicit priority of 12 October 2026. Existing addition results remain valid within their published scope. The generic implementation earns its own evidence and does not inherit an older core's arithmetic or Full151 claims.

## Required behavior

1. Every integer from 0 through 255 is payload. NUL, LF, spaces, `=` and invalid UTF-8 must remain valid binary data. UTF-8 is an external text encoding, not a restriction on the core.
2. Sequence start, reply start and sequence end are independent control events. EOS has identifier 256 in the output vocabulary; it is not a reserved payload byte. The same declared framing is available during training and inference. No correct answer, output length, task class or evaluator-only metadata may enter the core at inference.
3. Input arrives incrementally in arbitrary chunk sizes. The interface accepts streams longer than its finite context. Its context policy and overflow behavior are explicit. Within that context, all bytes and their order participate; the old last-four-visible-byte alias is removed. Finite context still allows collisions and discards older information according to the declared policy.
4. Before learning from an observed token, the core predicts using the pre-arrival state. The actually received byte or actually observed sequence boundary supplies the update target. Prompt-only admission with learning disabled changes working context but no persistent learning state.
5. The model chooses each output from 256 bytes plus EOS. Its own prior output advances working context. It receives neither a teacher-forced answer prefix nor the expected answer length. A generation budget produces an explicit `TRUNCATED` outcome if EOS was not selected.
6. Generation freezes persistent learned state. Guesses may influence subsequent generation through working context, but cannot teach themselves as observed truth. Snapshot and restore cover learned state, working context, counters and any pending prediction.
7. Output depends on the declared operative memory. A separate answer table, arithmetic calculator, pretrained helper or output renderer cannot supply the scored answers. Fixed-size plastic arrays may implement an engineered associative baseline, with that provenance disclosed.

## Initial engineering baseline

Use one shared sparse address computation and a vector-valued plastic byte memory, rather than 257 independent copies of Full151. The address is fixed and label-free; the learned output state is bounded independently of training history. State size, active rows, output vocabulary, context window and event costs are recorded together.

A context-hashed associative baseline is allowed for this IO stage. It can demonstrate acquisition and retained sequence recall, not relation extraction or useful natural-language understanding. Its input hashing may lose similarity, and collisions remain possible. This implementation is an engineered replacement of the restricted IO path, not a claim that an unchanged Full151 learned unrestricted output.

The existing addition core remains the frozen specialist reference. There is no claim that its 31 scalar parameters are automatically transferred into categorical byte memory, or that its previous generalisation result holds in the new generic implementation. Future arithmetic work must use this generic surface and identify the learning mechanism that produces answers.

## Complete verification set

| Check | Required evidence |
| --- | --- |
| Binary coverage | Every payload byte can be admitted and emitted; embedded LF, NUL and `=` do not reset or parse a record |
| Own sequence output | Random context-to-random-answer fixtures include empty, one-byte and multi-byte answers, raw binary and UTF-8; score exact payload and selected EOS |
| Context and stream | Short contexts with the same final four bytes and different earlier bytes remain distinguishable; input exceeding the declared window is accepted with explicit overflow accounting |
| Stream parity | Different chunkings of identical observations produce identical learned and working states |
| Causal memory | Matched no-write observations and erased learned state do not reproduce learned sequence recall; the encoder holds no answer cache |
| Retention | Score old sequences after simulated idle time and a disjoint new cohort; count new learning as well |
| Inference freeze | Long generation, normal EOS and truncation leave all learned weights and learned auxiliary statistics unchanged |
| State recovery | Snapshot and restored execution reproduce continuation, including an outstanding pre-arrival prediction if that API is public |
| Cost | Report persistent and temporary state and cost per admitted byte, query and output; distinguish production work from audit hashing |

All conditions are assessed together. Technical fixtures are E0 interface qualification plus E1 acquisition checks; they are not fresh scientific confirmation worlds. After implementation and independent audit, one prospective bounded end-to-end confirmation can test retained sequence recall on fresh fixtures. Failures remain attached to this IO contract; do not substitute unrelated technical or learning goals.

## Exposure and attribution

Training may expose the exact context-answer fixtures and observed EOS events. Test generation starts with the prompt only and receives no answer or answer length. Every output byte and EOS is scored, with no inference learning. Exact-key and canonicalised-key lookup can solve these taught fixtures, so the behavioral evidence remains E1. No E2, E3 or general persistent-core promotion follows from passing this stage.

The primary causal comparison disables all persistent updates while preserving the same bytes and control events. Erasure is an additional state-attribution check. Stable engineered weights retaining information across idle time are not evidence of biological consolidation. A frozen scientific extension must fix sample counts, exposure, cost budget and stopping conditions before inspecting its outcomes; technical test results cannot retrospectively set those gates.

## Completion boundary

Completion means arbitrary byte format, variable-length stream admission, memory-driven autoregressive output and model-selected EOS operate reliably within the declared context and capacity. It does not mean unlimited historical memory, a universal task solver or successful pretraining and Q&A. These boundaries define the present request without raising its goal after a pass.
