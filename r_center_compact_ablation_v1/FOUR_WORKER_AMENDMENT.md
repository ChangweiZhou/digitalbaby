# User-authorized four-worker execution amendment

Before any science trajectory started, the user asked for four processes because
no other experiment was running. Host inventory: 12 physical/logical cores,
19,327,352,832 bytes RAM. Each worker still uses one BLAS/Numba thread. Change
SPEC.workers from 2 to 4; the execution wording in PROTOCOL.md follows it.
All executable files, world IDs, fixture, arms, labels, records, clocks,
statistical family, sample size, acceptance gates and resource caps are unchanged.

The three original cycle receipts remain immutable and truthfully retain their
original source identity. They are NOT relabelled as trials on different bytes.
The prior source lock, spec, protocol and qualification are retained under
results/pre_four_worker_amendment. The new lock records the actual final files.
QUALIFICATION.json explicitly links the unchanged scientific/executable evidence
and the additional concurrency test, rather than silently repairing a stale lock.

Four fresh child processes ran together on engineering world 490103: two copies
of each physical configuration, 64 records each. Every saved prediction, actual
write call and final private state exactly matched the earlier qualified
uninterrupted history. Wall time 4.98 seconds; sum of individual recorded peak RSS
944,078,848 bytes. That short-history sum is not claimed as the full science
memory peak: full-life single-worker qualification measured about 425 MB and
319 MB, implying around 1.49 GB for two of each configuration before supervisor
overhead. The supervisor retains a 1 GiB cap per worker and the same job deadlines.

Measured full-life worker estimate stays about 9.62 CPU hours. Four workers imply
an ideal wall estimate near 2.4 hours; thermal and parallel efficiency are measured
operationally, not guaranteed. No additional science sample, mechanism, gate or
retry was introduced. Generalization remains outside adoption gates.
