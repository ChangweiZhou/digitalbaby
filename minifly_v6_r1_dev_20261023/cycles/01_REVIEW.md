# Cycle 1 — design, audit, test, revision

Implemented the declared stochastic recurrent sensory state and original nine-compartment native values. No task parser or separate answer head is imported by the learner.

Pretest code audit corrected one-tape training probes to three fixed tapes and removed a redundant second native snapshot in queries. Preserved initial source in `01_INITIAL_CORE.py`.

The actual technical run passed local score finite differences (maximum error 1.08e-9), enumerable delayed-credit expectation (7.63e-12), Gaussian score direction, byte codec, native signed self influence on six real recurrent frames, causal target timing, readonly queries, checkpoint restoration and NO_OLD disabling both plasticities. It then FAILED when evaluator `import data` resolved to an inherited module after the native bootstrap modified sys.path. Two full synthetic records and one partial write fixture were executed; zero addition records. Log and partial results are preserved. This is a packaging/import failure, not a learning result.

Revision: give the evaluator a unique module name `v6_data`; keep it outside the core dependency closure. Rerun the bounded technical suite as cycle 2, charge both attempts, and run no addition until qualification passes. Native parameters, warmup and credit rule are unchanged.
