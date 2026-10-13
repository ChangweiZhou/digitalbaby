# Generic byte memory interface

This engineered prototype accepts all 256 payload bytes, learns from observations and generates its own variable-length byte continuation and EOS. It removes the fixed arithmetic parser and single-byte answer slot. It is a fixed-capacity E0/E1 interface reference, not an unchanged Full151 implementation or a new arithmetic-generalisation result.

## Observe and generate

```python
from generic_byte_io import ByteCore

core = ByteCore()
prompt = b"arbitrary\x00binary\ncontext="
answer = "多字节回答".encode("utf-8") + b"\xff"

# Actual observed experience; observe() returns the prediction BEFORE update.
core.begin_sequence()
core.observe_bytes(prompt)
core.start_reply()                       # explicit framing, no answer length
core.observe_bytes(answer)
core.observe_end()                       # actually observed sequence ending

# Query: the prompt supplies no persistent updates; guesses do not self-teach.
core.begin_sequence()
core.observe_bytes(prompt, learn=False)
core.start_reply()
output = core.generate(max_bytes=512)
assert output.status == "ENDED"
assert output.payload == answer
```

Ordinary continuation learning uses `begin_sequence()`, repeated `observe(byte)`, and an observed `observe_end()`; it does not require `start_reply()`, correctness labels or an arithmetic teacher API. Each returned prediction includes 257 probabilities: payload bytes 0–255 and EOS at index 256. NUL, LF, `=` and invalid UTF-8 remain data. Inference and training must use the same declared framing. Calling `begin_sequence()` does not manufacture a training target for an unobserved previous ending.

`TRUNCATED` means the byte budget was exhausted without model-selected EOS. The working context remains open, so generation can continue. A final EOS decision is allowed at the exact payload cap, including a zero-byte cap. `rest(seconds)` advances simulated elapsed time; this stable engineered bank has no time decay or consolidation mechanism. Checkpoints preserve learned state, working context, counters and simulated clock. `prediction()` is a read-only pre-arrival query and creates no pending state; `observe()` computes its own pre-arrival prediction before any update.

## Memory and limits

One vector-valued plastic bank stores 4096 × 257 float32 weights. A fixed, label-free ordered-context hash selects 16 active rows. Categorical output and observed-byte gradients read and update these same weights. There is no learned key-answer dictionary, calculator or separately trained predictor hidden behind the output.

The context ring retains 4096 events, including framing and generated events. Longer streams are accepted, oldest events are discarded, and `context_overflow_events` reports each overflow. This is a finite context policy, not unlimited recall. Exact hashes do not deliberately preserve semantic similarity; collision/interference remain possible. Successful taught-sequence retrieval therefore supports E1, not E3.

The arrays occupy 4,218,880 bytes: 4,210,688 learned weights and 8,192 context bytes. Python metadata/object overhead is additional, and counters use Python integers. Production event work is `O(C + kV)` with context bound `C`, 16 active rows `k` and 257 output categories `V`. Production does not scan all 4096 memory rows. Address hashing scans the current bounded context; output buffering takes `O(max_bytes)`. Checkpoint and audit digest operations scan the full bank and are separate from production. These are transparent prototype costs, not a demonstration of giant-scale pretraining efficiency.

The old [addition core](../addition_core_v2_support_v1_20261011/README.md) remains the specialist reference. Its previous arithmetic evidence is unchanged; its scalar parameters are not silently transplanted into this generic memory. Future core replacements can use this byte surface, but must demonstrate their own behavioral reuse and arithmetic performance.

## Verification

- [Engineering report](REPORT.md)
- [Interface contract](../PROJECT_DOCUMENTS/GENERIC_BYTE_IO_CONTRACT_20261012.md)
- [Final engineering audit](FINAL_ENGINEERING_AUDIT.json)

15 unit tests, 10 independent adversarial checks, and the complete eight-fixture engineering acceptance passed. The acceptance spec was saved before outcomes and sources were locked before execution. Frozen earlier sources were not modified. This stage used no formal scientific confirmation or background experiment.
