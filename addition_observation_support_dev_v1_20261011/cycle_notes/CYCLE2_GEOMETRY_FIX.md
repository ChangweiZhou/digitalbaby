# Cycle 2 geometry fixture correction

Failed technical preflight: benchmark_fixtures.check_mask expects hashable tuple edges; DESIGN_CONTRACT JSON supplies list edges. Converted JSON rows to tuples at the geometry-only call. No core, update equation, gate setting, samples, endpoint or selection threshold changed.

Preserved original source snapshot, traceback and status in cycles/cycle2_geometry_fixture_failure. Zero full jobs committed in the failed attempt; no DEV/CONF science. Local synthetic tests and checkpoint fixtures from the attempt are technical expenditure, not accepted full jobs. Repeat only this technical cycle under the corrected test adapter.

Second geometry preflight caught the undirected edge (2,0) representation: check_mask requires canonical (0,2). Normalize geometry edges using orbit(min,max) at this environment-only witness; never canonicalize the learner support keys. Original failed attempt preserved in cycles/cycle2_canonical_geometry_failure, zero full jobs committed. The actual triangle-tail teaching edges and learner behavior are unchanged. A direct geometry-only check now runs before another full cycle invocation.
