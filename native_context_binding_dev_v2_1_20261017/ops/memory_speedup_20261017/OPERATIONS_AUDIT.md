# Operations-only memory admission adjustment

User explicitly requested examining overly conservative admission and accelerating without changing frozen science. DEV_A used a1,563,410,432-byte uniform worker reserve. DEV_B used2,535,936,000 bytes,1.25×the largest measured DEV_A worker peak, uniformly including low-memory NATIVE. The observed DEV_A NATIVE maximum was485,621,760 bytes.

The new dispatcher uses per-configuration observed peak RSS×1.20, reserves remaining growth against each live worker RSS in the same units, keeps512MiB OS headroom and blocks new admission when macOS pressure is not normal. RSS and physical-footprint measures are captured separately and not mixed in the admission equation. Concurrency remains capped at4. Measurements affect only operations, never candidate selection or learning.

No frozen source file is edited. Current worker PIDs are adopted unchanged. The previous dispatcher is deliberately retired after an atomic handoff; no signal is sent to a scientific worker during normal handoff, no trajectory is replayed, and the original session elapsed time is carried. Caffeinate moves to the new dispatcher. Frozen DEV_B worker, auditor, manifest, learner, source identities and analysis pipeline remain operative. All existing accepted receipts/checkpoints are preserved.

Qualification executes synthetic admission/process fixtures only, with zero native records. Normal and warning pressure, insufficient headroom, process exit, accepted receipts, no duplicate counting and time carry are checked before launch.
