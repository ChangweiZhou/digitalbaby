# Cycle 2: exact finite-horizon dynamics and shared geometry

Prospective design, after cycle 1 gate failure. Candidate and gate thresholds remain unchanged. No new candidate or parameter sweep; no full run may occur after a failed necessary gate.

## Small test scope
Use development worlds 300001,300002, identical 192-record input histories, 96 old and 96 new. Maintain causal baselines chronologically. Compute 32 teacher-basis output banks, born zero; old and new item identities are evaluator-only indices used for teacher Jacobian injection, never input to geometry. At each actual teacher time, decay W by exp(-dt/86400), then W += .25*(e_item-W phi) outer phi / max(1,||phi||²). This represents exact linear dependence on each of the 32 item targets. Save response operator L=Phi_query W^T at old_end, old_day, new_end, final. Native gain .4237781016501581 multiplies the separately reported score forecasts, not target formation fractions.

For the old-grid old-target 16×16 operator L_old, report Q_int^T L_old Q_int; minimum eigenvalue of its symmetric part, every singular value, Frobenius residual to identity, actual target-aligned gain and F/G/H component signed projection. Also report new-teacher interference. Formation gates remain minimum symmetric gain>=.20 old_end and>=.02 final. This is a uniform signed quadratic-gain condition, not a claim that singular values predict learning by themselves. Run separate direct four-channel bank updates, with literal clipping at +/-16, and independently compare predicted outputs. If any direct clipping occurs the unclipped Jacobian is insufficient; declare that forecast invalid rather than ignore it. The algorithm includes nonstationary feature histories and chronological updates rather than powering a final-snapshot Gram matrix.

## Shared/relation geometry (audit-strengthened exact-address exclusion)
Use all 30 ordered distinct pairs of symbols A,B,C,D,E,F in original 12-byte relation-cue format. Labels are not used in input encoding. Source-only canonical FE0 comparator: instantiate immutable native birth once and compute actual sparse KC encode_sparse vectors from the FE0 sensor on each same cue and timing, resetting sensor per panel clone. Do not read any Package A results. Birth and candidate histories are explicitly different architectures, reported as such; this is a source geometry diagnostic, not trained relation behavior.

At old_end and new_end for both development histories, form centered trace-normalized Gram G=HKH/tr(HKH), H=I-11^T/30. Construct X=[sender one-hot,receiver one-hot], shared projector P onto col(HX), using a numeric rank tolerance 1e-10. Shared mass=tr(PG). Define R=G-H/29 to remove the isotropic exact-address component. Require:
- candidate shared mass >=.50 times native FE0 shared mass
- nonzero native and candidate residual norms (>1e-12)
- off-diagonal residual-kernel cosine >=.50
- candidate ||R||F/||G||F >=.50 times native ||R||F/||G||F
Report an exact-address identity-kernel negative control; it must fail residual criteria. Shared geometry preservation is necessary, not proof of held-out relation transfer. A full scientific claim would additionally require a structured held-out relation task; this preflight will not claim such a benefit.

## Prospective prediction boundary
Predicted added-bank response tensors/margins from these equations are exact conditional forecasts for the bank. Total native-plus-bank held-out-world accuracy cannot be claimed from an observed final native state. Existing native final scores may be used only as clearly labeled development diagnostics. Unless an independent native-score forecast with prespecified error tolerance exists, record NO QUANTITATIVE THEORY PREDICTION for total prospective accuracy and block full launch. All unopened Package A arms receive that same designation.

Budget: one thread, 180 CPU seconds, 1 GiB RSS, 50 MB new output; no Full151 learned life, only one canonical source encoding birth. Stop on cap breach and keep failure artifact.

Audit clarification before execution: direct four-channel versus teacher-Jacobian max absolute tolerance=1e-10. Include query-read clock decay exactly at checkpoint+12*DT and all B/F/G/H cross-space block norms, plus full-target signed gain. Zero centered kernel trace (<1e-15), undefined ratios, or nonfinite values fail the relevant preservation gate. No tolerance is selected after seeing replay results.
