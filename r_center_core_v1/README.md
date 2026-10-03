# R_center engineering core v1

First engineering delivery after the R3 observed-outcome ablation. This package
provides a usable byte→memory→byte component, not a new mechanism search or a
claim of general intelligence. Three design/audit/trial/revision cycles are in
`cycles/`; machine-readable acceptance receipts and logs are in `results/`.

## Frozen behavior

Two banks, four independently canonical-born Full151 stores each. The shared
bank uses FE0 and native opponent teaching. The private bank uses the most
recent four non-space bytes and centered signed teaching `c=0, s=.25-y`.
The received outcome byte constructs `y`. The model emits ASCII `0`–`3` via
the frozen scaled-sum argmax before that byte arrives; ties select the first
channel. No hidden target map, world, stage, or branch enters the core.

This is the tested 12-byte cue protocol: cue → prediction → observed outcome
at the same scheduled slot → newline. It does not supply a general sequence
parser, learned stopping or open-ended text generation. CONTENT falls back to
FE0 below four visible bytes; long strings sharing their last four visible
bytes collide. The timing and alphabet deliberately match the upstream study.

## Use

Use Python 3.11 with `pip install -r requirements.txt`. The acceptance host used
Python 3.11.5, NumPy 2.2.6, SciPy 1.14.1 and Numba 0.61.2. All project model
assets are shipped; no historical experiment results are needed to operate it.
The native canonical-birth check rejects incompatible non-B state; new hosts
must qualify before scientific use. Exact parity is established on the recorded
host, not promised across arbitrary floating-point implementations.

From this directory:

```python
from centered_core import CenteredCore
core = CenteredCore()
dt = 30 / 14
for i, b in enumerate(b"        1+2="):
    core.feed(b, i * dt)
guess = core.predict(12 * dt)  # emitted is an ASCII byte, chosen before teaching
core.observe_outcome(ord("3"), 12 * dt)
core.feed(10, 13 * dt)
core.flush(165)
core.rest(86400)              # simulated learner time, not a wall-clock sleep
core.save("model.checkpoint.npz")
restored = CenteredCore.load("model.checkpoint.npz")
```

The example demonstrates API use; one teaching record does not guarantee a
correct arithmetic answer. Labels remain taught associations, not addition.
`observe_outcome(..., learn=False)` preserves the observable byte/time history
and disables native and signed writes on all eight stores. Prediction changes
the adapter's pending state, but does not modify any bank. Use `clone()` for a
disposable read-only probe; mutable state is independent of the continuing life.

The JSONL adapter accepts only declared byte events:

```sh
python -m centered_core --save model.checkpoint.npz < events.jsonl > output.jsonl
python -m centered_core --load model.checkpoint.npz --save model.checkpoint.npz < continuation.jsonl
python scripts/verify_release.py
```

Events are `{"op":"feed","byte":49,"t":0}`, `{"op":"predict","t":...}`,
`{"op":"observe","byte":51,"t":...,"learn":true}`, `{"op":"flush","t":...}`
or `{"op":"rest","seconds":86400}`. A checkpoint may include an incomplete
cue or pending prediction. Missing/stale source locks, corrupt checkpoint
fields, backwards time and illegal event sequences fail explicitly.

## Qualification and delivery

```sh
python scripts/run_qualification.py
python scripts/build_bundle.py
```

Qualification is bounded to engineering fixtures 390101–390103. It compares
the centered adapter to a documented, independently adapted original reference
on the same histories; it never executes upstream science worlds 290001–290064.
It includes native write-off tests, clone isolation, checkpoint tampering,
fresh-process recovery and clean-directory deployment. On macOS caffeinate is
attached to the qualification PID. Logs/status persist; the qualification
launcher neither retries nor starts science. Historical vendor runners are
retained as provenance, not engineering entry points. For an unattended technical run, use `nohup` with
the qualification command and direct its log to `results/`.

The source lock covers this NEW engineering version, including executables,
reference tests and numeric assets. `scripts/seal_release.py` is developer
tooling, not an automatic repair for a failed gate. Source changes require a
new version, a deliberate new lock and qualification. The upstream public
archive retained its historical lock despite disclosed privacy redactions;
that archive is preserved and is not relabeled as executable under this lock.

The inherited scientific result is E1 taught-cue acquisition/retention. The
current acceptance is E0 engineering equivalence. Neither establishes E3
generalization, teaching-free learning, or autonomous multi-byte output.
