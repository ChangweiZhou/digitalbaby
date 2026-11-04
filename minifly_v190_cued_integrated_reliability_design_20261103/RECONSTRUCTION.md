# V190-R1 source reconstruction

The original Round 4 scripts were unavailable. These files implement a fresh development test of the V190 proposal. They do not reproduce Round 4's exact model, learned states or scores.

The relation fixture uses an explicit raw-byte frame with two cyclic symbols. It identifies the first two registered symbols in a bounded raw history. This is task-interface structure. The registry is sorted by byte, and only observed transitions establish adjacency. Fixed features include both learned successor rows and two directed successor activations. TD step size is 0.5, readout steps are fixed in source, and signed maps freeze after eight laps. The task tags are environmental. All relation routes are read-only comparisons on a common trained expert trajectory. No previous model state is imported.

The mixed-stream panel uses a fixed held prose carrier per world, repeated four times per query. It measures the routing contrast with equal prose exposure. Its bits are not a broad natural-text measure. The preflight uses a reduced full-path fixture, and resource admission combines measured fixture resources with a full-roster count and conservative scaling. The full run enforces physical process caps at checkpoints. This is a local development experiment without an independent seal.
