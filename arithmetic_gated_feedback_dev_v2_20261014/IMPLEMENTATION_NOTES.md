# GATED-FEEDBACK V2 implementation

This directory implements the shipped 2026-10-14 design. DESIGN.json is the
unchanged historical design, not a live completion record. AUTHORIZATION.json
permits implementation and three technical cycles only. DEV, CONFIRM and GitHub
publication require separate authorization.

The operative learner imports only the Python standard library, NumPy and Torch.
It receives raw prompt bytes, actual observed output bytes or None, an actual
correctness bit, the public feedback phase and an explicit write permission.
Arithmetic parsing, groups, operations and held labels remain environment-side.
The GRU is a clean implementation of KEEP's existing scalar equations. The LSTM
is the standard four-gate cell with no peepholes or prompt cache. Neither has
attention, a Transformer, an ELM cell, numerical output targets or arithmetic
features. legacy_keep_reference.py is a test-only extraction of the historical
scalar GRU and KEEP wrapper; no inactive cell architecture is imported or ported.
The frozen copied worlds.py and runner_helpers.py preserve the old data and
countermodel algorithms. independent_truth.py contains only the prior separate
AST evaluator/group checker, never used by the learner.

G62 retains 124 recurrent inference scalars (memory and hidden state). Unlike
KEEP's necessary 64-scalar decoder context, it does not cache the pre-REPLY state
through generation. Its detached critic feature is reused from a visible-answer
CE forward, or reconstructed from the original prompt before any actor update
when no answer is visible. That reconstruction is charged to production CPU and
transition counters. Training graphs, full vocabulary trace caches, RMS moments,
candidate copies and archives are additional memory, not recurrent state; they
are not claimed to fit the 124-scalar figure. KEEP has 128 recurrent/context
scalars. All model memory and peak process RSS are separately reported.

Before the first test, numerical roundoff tolerances are fixed at 1e-12 for the
empirical trace KL's nonnegativity and 0.01-nat boundary; candidate loss tolerance
is 1e-12*(1+abs(old_loss)). These are floating-point tolerances, not effect gates.
SAFE tries exactly the eight declared scales. The cached old baseline stays
fixed. Rejection restores actor parameters, moments and actor step; the actual
reward still updates only the detached critic. It offers no expected-trajectory
KL or global performance guarantee. Generic numerical fixtures independently
check these properties; the receipt auditor verifies every logged decision and
the stored complete states, without replaying native trajectories.

Actual record applications count each separately executed history. Shared warm
prefixes are executed once per prefix, never charged seven times. Query output
generation and SAFE recomputation are counted and timed; taught-item diagnostic
CPU is reported separately from the production learning/held-query clock.
Wall time, process CPU, audit cost and retained archives are separate fields.
All seven terminal histories are correlated, not seven independent lives.

Cycle 1 uses generic byte mapping and implementation fixtures. Cycle 2 uses
the one predeclared short-dose world. Cycle 3 uses the two predeclared full-dose
worlds. Revisions are recorded with source snapshots and preserved failures.
No held-arithmetic effect tuning or replacement of failed native worlds is
permitted. Technical results are descriptive, never formal adoption evidence.

Cycle-2 review: the two short-dose jobs and all 21 independent tamper cases
passed. The report now explicitly distinguishes its temporary short-dose clock
from the full-dose cycle-3 Cstar; the earlier report/source remain archived.
No model, learning rule, data, dose, candidate or effect gate changed. The generic
cycle-1 acquisition fixture supplies a synthetic constant zero critic bit while
testing supervised CE acquisition; it is not an actual-reward scientific world.
Every native cycle-2/3 reward is independently checked against the model's own
emitted bytes and EOS. Their ledger/source/state audits remain separate.
