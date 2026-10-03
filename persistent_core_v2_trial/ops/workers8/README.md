# Authorized eight-worker dispatch

The user requested increasing the active experiment from four to eight CPU workers. This is an execution amendment only. The scientific identity remains `6f0d7a765df41dcc51b8fbc45a5ce5a8278726e34a9162351b652101fc72daf7`. No frozen top-level scientific source or SPEC was edited.

The adapter retains the original supervisor logic with four documented changes: worker count, dispatch capacity, preserved original science start time, and separate worker log filenames. Its independent source lock and exact dispatch diff are stored here. It calls the unchanged worker and retains the same roster, endpoints, guards, deadlines and resource budgets.

Eight concurrent short development workers (world 500103; two replicas per arm) passed the independent receipt auditor and matched the prior qualified 64-record scientific histories and final states exactly. This is an E0 execution check, not an additional scientific finding or proof of a twofold speedup.

The original dispatcher was paused while its active workers finished whole jobs. Sixteen committed science jobs were retained byte-for-byte, with zero interrupted or repeated learner trajectories and no partial checkpoints. The intentional dispatcher SIGTERM record is preserved under `transition/AUTHORIZED_SUPERVISOR_STOP.json`; it is not a scientific failure. The new supervisor is detached and has its own attached caffeinate process.

Resource interpretation: the first sixteen jobs used the original four-worker execution; subsequent jobs use eight workers. The short concurrency qualification overlapped the final draining job. CPU and wall costs include these actual execution conditions. Existing timing values were not rewritten or normalized, and completed jobs will not be rerun for timing. This execution change must be considered when interpreting compute-efficiency contrasts.

Scientific source lock and the original three-cycle qualification remain authoritative. Eight-worker monitoring reads `results/LAUNCH.json` and `results/supervisor8.log`. The experiment still totals 768 jobs, 2304 lives and 96 paired worlds.
