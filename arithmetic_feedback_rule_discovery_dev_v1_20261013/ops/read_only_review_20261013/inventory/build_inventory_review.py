"""Read-only document census; no learner import, fitting or trajectories."""
from pathlib import Path
import json
from collections import Counter

R = Path(__file__).resolve().parents[4]
I = R / 'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007'
O = Path(__file__).resolve().parent
paths = {
    'historical108': I/'FULL_REVIEW_POST_U034_20261011/HISTORICAL_108_COMPARISON.json',
    'catalogue136': I/'FULL_REVIEW_POST_U034_20261011/CATALOGUE_136_COMPARISON.json',
    'versions254': I/'FULL_REVIEW_POST_U034_20261011/VERSION_254_COMPARISON.json',
    'subset71': I/'AUTONOMOUS_71_SCREEN.json',
    'first16': I/'FULL_REVIEW_POST_U034_20261011/FIRST16_COMPARISON.json',
}
data = {k: json.loads(p.read_text()) for k,p in paths.items()}
expected = dict(historical108=108,catalogue136=136,versions254=254,subset71=71,first16=16)
assert {k:len(v) for k,v in data.items()} == expected

candidate_notes = {
 'M001': 'Generic expressive cell reference. RAW uses a small GRU inspired by this direction, not the exact private-state ELM version. Its technical failure does not exclude the family.',
 'M007': 'Sparse GRU-like historical source missing decisive comparison. RAW is a dense 64-state GRU and cannot be treated as a scientific test of this exact historical version.',
 'M008': 'RG-LRU source named in ledger; local ELM5 notebook not recovered in this review. Different recurrence is not first priority without evidence of recurrence-specific failure.',
 'M022': 'Frozen/live and stable/flexible coordinates are relevant if current saved states show coordinate drift. Do not inherit historical retention improvements as arithmetic gains.',
 'M023': 'Dense transport/anchor transport has negative or partial precedent. It adds global work; avoid claiming it is untested or a default solution.',
 'M024': 'V23 adapter_retained/recycled supplies a concrete learned nonlinear downstream address alternative. Engineering backprop acceptable here; prior locality disqualification is not efficacy evidence for this task.',
 'M025': 'Composite learned operator mixtures exist in ledger, but arithmetic-specific operator choices or numeric targets would violate current task. Only generic learned modules are eligible.',
 'M027': 'Hebbian and residual fast matrices already tested, with weak historical tradeoffs. A new raw-byte version is not excluded, but direct context/readout acquisition evidence should precede adding fast/slow stores.',
 'M028': 'Hopfield/episodic working memory reference. Exact retrieval can repair E1 but cannot establish held-out arithmetic laws; not a stand-alone next primary.',
 'M031': 'Global optimizer/engineering reference. Coherent categorical loss and synchronized encoder/head updates can be compared without arithmetic priors; do not claim a biological mechanism from this.',
 'M033': 'Ordinary error-based output learning is a necessary baseline; compare like-for-like byte classifier objectives and transient state.',
 'M034': 'Cumulative RLS is meaningful in a fixed feature basis. RAW learned-basis P is explicitly not an exact fixed-basis posterior; frozen RAW already fails acquisition, so drift alone is insufficient.',
 'M048': 'Fixed/inferred bank variants already exist. More stores or routing is not the first remedy unless saved-state evidence shows shared interference after accurate encoding.',
 'M050': 'Reservoir/recent/residual-priority replay already tested elsewhere. Re-encoding observed raw records under current encoder is a distinct moving-coordinate version, but pay memory and compute and preserve equal external samples.',
 'M058': 'Full covariance already appears in RAW. Repeating full-vs-diagonal is not a new mechanism and did not fix current acquisition.',
 'M059': 'Lag96 binds byte identity to chronological position. A generic position-preserving context skip is relevant to prompt access and introduces no arithmetic rules. Historical exact bits need not be copied; arbitrary-byte identity embeddings avoid numeric semantics.',
 'M061': 'Residual/reference alignment matters if a rapidly moving base is used. A pure readout or exact bank that recalls taught cues remains E1, not rule learning.',
 'M077': 'Positive-pair invariance requires relations inferred from observed feedback only. No evaluator-provided arithmetic equivalence or exchange/inverse labels are allowed.',
 'M096': 'Local eligibility is a credit-assignment option, but current analytic BPTT gradient tests pass. It should not replace a coherent generic learning baseline solely for biological naming.',
 'M106': 'Stable/slow reference or compensation is a distinct coordinate-management hypothesis. Needs matched resource controls and direct drift evidence; unrelated retention results do not prove raw arithmetic benefit.',
}

def row_id(row): return row.get('id',row.get('dedup_id'))
def label(row):
    o=row.get('original',row)
    return o.get('mechanism',o.get('name',row_id(row)))
def prior(row):
    o=row.get('original',row)
    return row.get('review_status',row.get('broad_core_status',row.get('budget_status',row.get('priority_status',o.get('budget_status','preserved')))))
def note(kind,row):
    ident=row_id(row)
    if kind=='historical108':
        return candidate_notes.get(ident,'Original exact evidence retained. Not promoted, erased, or family-excluded by RAW technical fixtures; no direct present-task benefit established by this document review.')
    if kind=='versions254':
        if any(t in ident for t in ('adapter_retained','adapter_recycled','local_turnover')):
            return 'Concrete downstream-address version; relevant nonlinear interface precedent. Original matched parent and locality limits retained; not raw arithmetic confirmation.'
        if ident.startswith('V25/') and any(t in ident for t in ('dual_','split_')):
            return 'Concrete coordinate/stability version; conditionally relevant to moving features. Not a present-task positive result.'
        if any(t in ident for t in ('replay','REPLAY','EXTRA')):
            return 'Already tested replay/extra-update version. Do not relabel as new; raw re-encoding would need explicit new operation and equal compute controls.'
        if any(t in ident for t in ('TOPO','SUPPORT-DIR','V30_DIRECTION/D','V30_DIRECTION/P')):
            return 'Exact existing disposition preserved. No resurrection by a renamed dose, and no whole-family exclusion.'
        return 'Original version, parameterization, parent/context and verdict retained. Historical/nonmatched evidence does not establish current raw-byte four-operation acquisition or E3 efficiency.'
    return 'Overlapping inventory view retained in full. Preserve exact version/recipe scope; not a unique additional candidate and not newly excluded by this review.'

review = {
 'scope':'Full document census and source-level acquisition relevance; no science, training, replay or frozen-file edits.',
 'date':'2026-10-13', 'counts':expected,
 'count_warning':'These views overlap; neither 585 total rows nor 254 version records is a count of distinct surviving mechanisms.',
 'all_original_rows_preserved':True,
 'scientific_survivor_count':None,
 'survivor_count_reason':'No binary survivor flag, different tasks and validity levels, repeated baselines/controls, and many undefined or missing comparisons.',
 'views':{}
}
lines=['# Full inventory review for RAW-RULE acquisition failure','',
       'Read-only review, 13 October 2026. Every original row and version parameter record is preserved in the JSON companion. This is a prospective relevance view, not a reranking of historical science.','',
       '| View | Rows | Counting boundary |','|---|---:|---|',
       '| Historical mechanisms |108| M001–M108 |',
       '| Mechanism/variant catalogue |136| C001–C136; overlapping views |',
       '| Actual program/version records |254| Includes controls, repeat anchors, compounds and contexts |',
       '| Older deduplicated shortlist |71| Subset, never the complete inventory |',
       '| First-round named operations |16| Separate experiment-view roster |','',
       '**No scientifically defensible single survivor count follows from these files.** The register ends at U034; recent four-operation, ADAPT-2, VALUE-INTERACT and RAW-RULE results are supplementary evidence, not secretly inserted rows in the old 254 count.','',
       '## Recommended acquisition choices','',
       '1. Coherent categorical encoder/head credit assignment (M031/M033) is a generic engineering change; compare it with the present alternating squared-error/RLS recipe using the same bytes, hidden width and feedback count. It supplies no arithmetic laws.','',
       '2. Generic chronological context access and a learned nonlinear readout/address interface (M059/M024) are concrete older mechanisms omitted by treating all recurrence as one choice. Input identity/order should reach the answer decisions; do not inject digit values, operand roles or operation IDs.','',
       '3. Coordinate management (M022/M106), or bounded replay with raw re-encoding (M050), is conditional on drift evidence. The frozen RAW controls also fail acquisition, so learned-coordinate drift cannot be the only explanation.','',
       'Protection, release, metaplasticity, extra banks and more timescales are lower priority while acquisition itself fails. This is task-specific scheduling, not a scientific claim that those families are dead.','',
       '## Full row-by-row preservation and current relevance','']
for kind,rows in data.items():
    ids=[row_id(r) for r in rows];assert len(set(ids))==len(ids)
    augmented=[{'id':row_id(r),'original_review_row':r,'raw_rule_acquisition_review':note(kind,r),'new_run_authorized':False,'new_family_wide_exclusion':False} for r in rows]
    review['views'][kind]={'source':str(paths[kind]),'rows':augmented,'prior_status_counts':dict(Counter(prior(r) for r in rows))}
    lines += [f'### {kind} ({len(rows)} rows)','', '| ID | Original entry/status | Current review |','|---|---|---|']
    for r in rows:
        title=str(label(r)).replace('|','/');n=note(kind,r).replace('|','/')
        lines += [f'| {row_id(r)} | {title}; `{prior(r)}` | {n} |']
    lines += ['']
(O/'FULL_INVENTORY_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
(O/'FULL_INVENTORY_REVIEW.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'counts':expected,'review_rows':sum(expected.values()),'scientific_records':0,'written':['FULL_INVENTORY_REVIEW.json','FULL_INVENTORY_REVIEW.md']}))
