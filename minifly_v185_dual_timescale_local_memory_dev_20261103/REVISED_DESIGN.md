# V185 qualification revision

The test run exposed an admission problem in the readout witness. Preserve the registered sensor, projection, data seed, roster, writer rules, gates and resource limits. Replace the all-at-once multiclass constraint matrix with a bounded active-set witness.

Start from a small deterministic set of competitor inequalities. Solve that restricted problem, evaluate the logits on all 240 saved feature rows, and add the worst violated competitor for each violating row. Track the active matrix allocation and remaining budget before each solve. Reuse the same labeled feature table and charge every subsequent target evaluation. Run the solver in an isolated process with a measured memory guard. Stop at the existing limits with an unresolved ceiling if no checked solution is available. A solver timeout has no bearing on mathematical feasibility.

A positive certificate requires deterministic float32 own-byte decoding of all 80 exact responses with EOS and a recorded margin check against all 257 classes. Save aggregate witness results and keep its weights separate from the candidate. Complete the conflict, release, zero-direction and small-step fixtures, then test actual checkpoint serialization and profile the full roster before requesting scientific launch. The scientific experiment still requires separate authorization.
