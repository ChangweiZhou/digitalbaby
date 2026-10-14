# Cycle 1 — mathematical and byte-interface qualification

Goal unchanged: learn arithmetic from feedback without supplied arithmetic rules, and gain unseen-case ability per sample and compute.

Implement the declared GRU, tied processing, categorical confidence update and deferred encoder learning. Before any arithmetic test, independently finite-difference the encoder using the **captured pre-update readout for each target**. Test both processing depths, frozen/no-write state, full/diagonal stability, arbitrary byte relabelling, pending-gradient checkpoints, inference purity and abort semantics. These are technical fixtures, not science or evidence of arithmetic ability.

Audit concerns: target enters only after prediction; EOS is an observed event, not an inferred label; internal zero-input steps must not mutate the persistent prefix; byte IDs have no numerical meaning. Long-stream gradients are truncated, with evicted targets disclosed; optimizer mean divides by all permitted targets in the frame, including dropped ones. No hidden arithmetic helpers or uncharged replay.
