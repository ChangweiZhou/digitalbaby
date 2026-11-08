"""Build a source-preserving mechanism re-screen. Does not train or change models."""
from pathlib import Path
import collections, datetime, hashlib, html, json, re, shutil, zipfile

R = Path(__file__).resolve().parent
S = R / 'sources'
REPO = Path('/Users/pencilbard/Documents/second_try')
DOWNLOADS = Path('/Users/pencilbard/Downloads')
ANCHOR = '7ff786619cc9d5833ce0308219dd5eb85cd11449'

# These are relevance decisions, not retrospective experimental success gates.
# Each original row remains separate even when several share a computation.
FAMILY_ROWS = [
('F01','Local error-correcting writes and outcome signals','KEEP COMPONENT',
 'Output-local correction is already central to the present learner. Molecular names do not create additional computational capabilities.',
 'E45 E1-E9; E12-E14. The target/error interface and development versus lifetime supervision must stay explicit.',
 'Discuss only a concrete error-information or write-normalization defect; compare the same observed targets and applied update budget.'),
('F02','Eligibility traces and delayed synaptic credit','KEEP BOUNDED POSITIVE',
 'Temporal credit is supported on diagonal state and delayed discrete actions. Neither establishes general recurrent credit or reasoning.',
 'Second sweep diagonal_trace and multi_action_credit; E11. Exact diagonal derivatives require a matching learning signal; sign feedback makes the complete learner approximate.',
 'A later extension should address a measured horizon/coupling failure, with a true-gradient reference on the same forward model.'),
('F03','Separate working-memory write maintain and read gates','IN PROGRESS E15',
 'Controlling storage and controlling use can differ. This directly concerns interference and selective reuse.',
 'Second sweep separate_wm_gates has 87.1-97.7% versus 57.4-82.0% tied, but active gate capacity differs. E15 C now tests a split-gate cell.',
 'Read E15 before duplicating it. Attribute any gain carefully: extra gates, extra active parameters and different developed births travel together.'),
('F04','Learnable multiscale or oscillatory working state','DISCUSS NEXT CONDITIONAL',
 'A plausible way to replace the fixed input window and support variable delays; long-state use must be learned rather than supplied as task positions.',
 'E11 still uses a ten-byte window plus a diagonal cell. The second sweep holds one cue across a doubled gap, not general stream computation.',
 'Change state dynamics alone; compare matched window/state capacity and test unseen gap lengths plus intervening learning. Arbitrary logarithmic decays are not a proven optimal history code.'),
('F05','Shared recurrent internal computation','DISCUSS LATER',
 'Repeated learned transformations can add computational depth without hand-coding the operation. This is more directly about computation than protecting memory.',
 'Both thalamic sweep proxies failed to resolve the question. The second fixture teaches only same-operator compositions and holds cross-operator ones, retaining a factor confound.',
 'If a remaining step-depth failure warrants it, compare shared recurrent passes with work-matched untied/feedforward passes on de-confounded relations. Do not infer a biological algorithm from reverberation alone.'),
('F06','Dendritic targets and adaptive feedback for learning','DISCUSS CONDITIONAL',
 'Different mechanisms include random feedback, sign feedback, learned feedback and compartmental targets. They must not be treated as interchangeable.',
 'Sign feedback is positive in E45/E7. Adaptive apical feedback did not beat fixed feedback in the second classification proxy. E14 diagnostics are unfinished; observed raw alignment is already high in one seed.',
 'Use learned feedback only for a demonstrated remaining credit problem. Include calibrated sign feedback at the same decay and distinguish raw gradients from Adam and decay steps.'),
('F07','Pattern-separated fast memory and a shared slow learner','DISCUSS LATER',
 'Preserving individual episodes and extracting reusable structure are different jobs. Pattern separation/completion is more specific than merely having two learning rates.',
 'E8 demonstrates replay-supported retention; its exact prompt store is not a demonstrated biological pattern-separating memory. The second sparse_fast_slow proxy gives only modest old-rule scores.',
 'Only if the current replay/store becomes limiting: compare a bounded learned sparse store with raw uniform replay and a single store, charging all memory and teacher information. Avoid task-ID banks masquerading as learned separation.'),
('F08','Experience replay and its timing sampling and consolidation','KEEP CURRENT REFERENCE',
 'The current strong result is preserving useful observed experience, not a special sleep order or automatic benefit from downscaling.',
 'E9/E12/E13: replay matters in these histories; recency weighting harms early operations; retention buffer gains saturate around 1024 prompts/stage. The second sleep order/scaling proxy adds no advantage.',
 'Use uniform observed-experience replay as reference. Discuss another schedule only for a specific measured deficit; match exposures, update count, decay and replay storage.'),
('F09','Activity normalization sparsity and homeostasis','RESERVE',
 'Can regulate representational load or saturation, but stronger forgetting/downscaling is not intrinsically useful.',
 'Second sweep APL-on and off tie; sleep scaling adds no gain. E13/E14 identify excessive optimizer decay as harmful in these models.',
 'Reopen for observed activity collapse, saturation or capacity limits; match coding fraction, active parameters and post-optimizer write energy.'),
('F10','Synaptic tagging capture commitment and metaplasticity','RESERVE',
 'Several molecular names reduce to local tags, consolidation resources or multiple weight timescales. They are not independent reasoning modules.',
 'Older 56/136/254 inventories preserve different recipes. No result here establishes their benefit for the current learned composer after calibration.',
 'Require a retention/plasticity tradeoff that replay and calibration do not resolve. Compare correction and new acquisition as well as old retention, with random or write-matched consolidation.'),
('F11','Selective forgetting and reopening obsolete memories','RESERVE FOR REVISION',
 'Appropriate for replacing obsolete knowledge, not a general cure for a model already erasing valid knowledge.',
 'Many historical active_forgetting/Rac1/DAMB variants exist; their exact implementations and signal provenance differ.',
 'Reopen on genuine old-to-corrected labels. Protect still-valid cases and match reduced write/decay controls. Do not treat all versions as scientifically excluded.'),
('F12','Latent memory retrieval and reinstatement','DIAGNOSTIC RESERVE',
 'Storage versus accessibility is a useful distinction, not evidence that every lost answer remains stored.',
 'E12 savings test found no large erased-versus-birth relearning advantage in its tested conditions; this does not establish absence of every residual representation.',
 'Use only when behavior and a source-grounded readout disagree. Reminders and fitted probes are diagnostics, not autonomous recall.'),
('F13','Structured connectivity and genuine structural plasticity','DISCUSS LATER CONDITIONAL',
 'True edge growth/pruning can differ from masking weights, redistributing support or changing a routing table.',
 'The older catalogs contain sparse bridges, hierarchical graphs, expanders and growth/stabilization variants. Most historical memory results do not test current arithmetic computation.',
 'After a useful computation has a measured capacity/interference limit, compare fixed and plastic adjacency with matched active edges/work and preserved transfer. Do not launch a fresh topology census.'),
('F14','Role filler binding contextual routing and copying','OPEN CAPABILITY QUESTION',
 'Learning which operation/operand belongs where can matter for systematic reuse. Attention and task-specific parsing are distinct solutions with distinct permissions.',
 'Second operator_binding executor is perfect with correct prompts but rewrite exactness is zero; E7 balanced combinations work, while E14 E8 has a weak carry combination.',
 'Separate rule execution, first rewrite, intermediate output and second rewrite. Keep attention as the authorized reference; a key/value content lookup does not become attention-free by renaming it.'),
('F15','Ring grid geometry and supplied vector operators','REFERENCE OR EMBODIED RESERVE',
 'Structured transformations can compose by construction. This is useful as an inductive-bias/control result, but arithmetic supplied in the wiring is not learned arithmetic.',
 'Second grid_graph_transfer supplies a torus and phase transport; E10 supplies ring shifts and positional parsing. E15 D tests local-rule interface preservation.',
 'Use to bound what good interfaces could buy. For learned reasoning, require acquisition of relevant transforms and transfer beyond the supplied geometry. Do not duplicate E15.'),
('F16','Native central-complex heading and supplied-goal steering','KEEP EMBODIED RESULT',
 'The fitted extracted projection subset supports heading and steering; it is not the whole fly brain or a general reasoning network.',
 'FIT_OFFICIAL_REPORT: native combined 80/96 versus fitted rewired 20-24/96 on fresh panels; 15 fitted scalars and a specified subset of projections.',
 'Retain as embodied foundation. Visual memory plasticity and learned goals remain separate from a supplied goal and a fitted heading loop.'),
('F17','Displacement accumulation in native navigation circuitry','PARK CURRENT PACKAGE',
 'A world-centric velocity representation is not yet a time-integrated displacement representation. Finding the accumulator is a real remaining biological/computational question.',
 'PI follow-up with correct heading clamp still has native return magnitude 0.710-0.717 of outbound distance. Lu et al. identify hDeltaB world-centric velocity and distinguish subsequent temporal integration.',
 'Respect the prescribed stop. Discuss a new anatomically/physiologically supported accumulator only when its dynamics and readout are specified; gain tuning alone is not evidence it exists.'),
('F18','Mushroom-body compartments and MBON DAN feedback','MEMORY RESERVE',
 'Addressed associative updates and reversal are different from reusable multi-step computation. Feedback, postsynaptic plasticity and sparse input mechanisms remain distinct variants.',
 'Second MB proxy addressed arms 70.7-79.3% versus DAN-permuted 46.9-54.7%; APL on/off tie. The proxy uses supplied context addressing, not extracted plasticity masks.',
 'Discuss a concrete memory/reversal deficit with actual teaching access. A compartmental association win cannot establish learned arithmetic or the complete biological feedback circuit.'),
('F19','Cerebellar adaptive filtering and temporal bases','SPECIALIZED RESERVE',
 'Potentially useful for calibration and timing. The second sweep implements a nonlinear feature-regression task, not an actual sequence-timing circuit.',
 'Its source uses independent two-dimensional points and a sinusoidal target, Gaussian features versus affine/scrambled features; RMSE 0.144-0.201 versus about 0.73-0.84.',
 'Keep the numerical feature-basis positive, without promoting a cerebellar timing or reasoning claim. Revisit for a genuine temporal/sensorimotor calibration need.'),
('F20','Actor critic and delayed action credit','KEEP FOR FUTURE AGENCY',
 'Strongly relevant to learning actions from delayed outcomes; it does not provide a working state or solve novel compositions by itself.',
 'First sweep delayed choice trace 64/64; second four-action trace 100%, immediate 0.8-24.6%, shifted 0-19.9%. Each cue has an independent action mapping; trajectory combinations can repeat from training.',
 'Use when an agent must select actions. Distinguish delay credit from exploration and planning; later match cue/trajectory novelty and outcome access.'),
('F21','Motivational metabolic circadian and stress gates','LOW CURRENT PRIORITY',
 'Biological state can tune learning rates or access. Named transmitters and pathologies do not specify a missing computational capability.',
 'The surveys include octopamine, sugar state, arousal, stress/age and circadian entries. Their biological evidence is not a current MiniFly learning advantage.',
 'Reopen only for a specified resource/state signal and a measured context-dependent learning failure; no literal metabolism or aging module is needed for current toy reasoning.'),
('F22','Molecular implementation details','ABSTRACT DO NOT MULTIPLY MODULES',
 'CREB, cAMP, transcription, local translation and epigenetic names can instantiate learning, tagging or consolidation; duplicating them as software modules is not justified.',
 'The 42-row fly survey and 56-label handoff overlap heavily. Individual biological evidence grades were not independently re-certified in this computational audit.',
 'Retain each source label and map to a computational rule. A molecule becomes a software hypothesis only when its distinct state/update adds a falsifiable capability.'),
('F23','Episodic capacity banks slots and exact caches','CAPACITY REFERENCE',
 'More isolated storage can improve recall without improving transfer. Exact hashing and addressed slots change geometry and information access.',
 'Historical HASH/bank lines; E8 exact prompt storage/replay. Synthetic capacity gains do not establish learned reasoning.',
 'Reopen for measured store capacity/cost. Keep stored answers, keys, router bytes and addressing assumptions in the ledger; content-addressed retrieval is an attention reference under the current boundary.'),
('F24','Write allocation importance uncertainty and resource budgets','RESERVE',
 'Controls where/how strongly learning changes weights. Many protection schemes can win simply by writing less.',
 'Older importance, availability, uncertainty, quotas and subspace variants have different histories; exact outcomes remain in the original rows.',
 'Only after the calibrated learner has an identified destructive-write problem; match applied updates and charge acquisition/correction losses. Do not infer a family-wide impossibility from old screens.'),
('F25','Worked examples scaffolding and curriculum','TRAINING INTERVENTION',
 'Can alter the information provided or factor coverage. These are not automatically neural circuit mechanisms.',
 'Prior recurrent/worked-example runs and E5-E8. Balanced operator-factor coverage materially changes conclusions; task-specific scratchpad formats remain supplied.',
 'Discuss with matched observed information and examples. Keep teacher-supplied decomposition distinct from model-learned decomposition.'),
('F26','Neurogenesis intrinsic excitability and AIS remodeling','LOW CURRENT PRIORITY',
 'Growth changes capacity; threshold/channel changes affect state and gain. Neither is a specific reasoner by itself.',
 'Human-neuron survey includes these topics but does not supply a current MiniFly rule or demonstrate general learning gains.',
 'Map growth to a capacity-matched structural comparison and excitability to a state/gain comparison if a measured failure warrants one; do not import biological evidence labels as software efficacy.'),
('F27','Electrical coupling and intercellular communication','LOW CURRENT PRIORITY',
 'A real physical communication mechanism needs a distinct computational role before becoming a software module.',
 'Fly surveys mention innexins, gap junctions and Arc-like transfer. These do not establish a missing MiniFly communication channel.',
 'Discuss only when coupling dynamics or signal transmission is the demonstrated limit; compare a simple matched coupling model first.'),
('F28','Baselines lesions readouts and engineering controls','REFERENCE NOT NEW MECHANISM',
 'Controls and evaluator readouts are valuable but should not inflate the count of new mechanisms.',
 'Historical variants include ordinary/slow/frozen/shuffled/oracle/read-authority controls; the 254 count is experimental recipes, not 254 unique circuit motifs.',
 'Preserve exact recipes and provenance. A better evaluator does not become the learner\'s native capability.'),
('F29','Learned input representations invariance and sensory plasticity','DISCUSS CONDITIONAL',
 'A reusable representation should preserve task-relevant distinctions while ignoring nuisance renderings. Representation quality differs from the ability to use it.',
 'Old front-end/Oja/contrast/drift catalogs; current E11 fresh-render replay changes nuisance exposure using domain-aware task parsing.',
 'If a learned rule fails on new symbols/renderings, compare frozen and locally learned encoders with unchanged rule/answer information. Concept-cell observations are an outcome to explain, not a supplied encoder.'),
('F30','Exploration metacognition and self regulation','LATER AGENCY QUESTION',
 'Autonomous selection of experience is a future capability, distinct from learning a supplied stream.',
 'Human-learning audit discusses motivation, metacognition and social learning as educational interventions, often assuming a human-facing platform rather than a neural learner.',
 'Reopen after a useful online learner exists and active information gathering has a measurable value. Do not add UI gamification or an external planner as a purported brain mechanism.'),
('F31','Developmental meta learning or evolved local rules','LONGER TERM DISCUSSION',
 'Could reduce task-specific engineering, but it is a development/training intervention whose lifetime rule still needs qualification.',
 'Older evolved-local-rule and modern meta-learning suggestions exist; no result inspected here establishes a general learned local rule across domains.',
 'Use multiple held task families and keep developmental optimization costs separate. Avoid an unconstrained mechanism search before the present learner has a diagnosed limit.'),
]
families = {x[0]:dict(id=x[0],name=x[1],decision=x[2],reason=x[3],evidence=x[4],reopen_condition=x[5]) for x in FAMILY_ROWS}
rows=[]
def add(source, eid, name, family, locator, original, caveat=''):
    assert family in families
    rows.append(dict(id=source+':'+eid,source=source,source_entry=eid,name=name,
                     family=family,secondary_families=[],locator=locator,original=original,
                     decision=families[family]['decision'],reason=families[family]['reason'],
                     relevance_scope='Family-level computational relevance; not a new efficacy replication of this exact recipe.',
                     exact_entry_caveat=caveat))

doc_names=[
 'Deep Research Backing Conversation for session None',
 'Biological Circuit Mechanisms Literature Survey',
 'Fly Brain Learning Mechanisms',
 'Fly Brain Learning Mechanisms_ Evidence Map and Verification Synthesis',
 'Neuronal Human Learning Mechanisms Research',
 'Audit of “Minifly” Learning Mechanisms (Human-Learning Perspective)',
]
docs=[json.loads((S/(x+'.json')).read_text()) for x in doc_names]
circuit_families=['F07','F03','F06','F19','F20','F15','F16','F18','F08','F05','F14']
starts=list(range(0,121,11))
for j,start in enumerate(starts):
    end=starts[j+1] if j<10 else next(i for i,b in enumerate(docs[0]) if b['kind']=='table')
    text='\n'.join(b['text'] for b in docs[0][start:end] if b['kind']=='paragraph')
    add('CIRCUIT_A',f'{j+1:02}',text.split('. ')[0],circuit_families[j],f'paragraph blocks {start+1}-{end}',text)
starts=[i for i,b in enumerate(docs[1]) if b.get('style')=='Heading2' and b.get('text','').startswith('Circuit ')]
assert len(starts)==11
for j,start in enumerate(starts):
    end=starts[j+1] if j<10 else next(i for i,b in enumerate(docs[1][start+1:],start+1) if b.get('style')=='Heading2')
    text='\n'.join(b['text'] for b in docs[1][start:end] if b['kind']=='paragraph')
    add('CIRCUIT_B',f'{j+1:02}',docs[1][start]['text'],circuit_families[j],f'Circuit {j+1}',text)

fly_map=['F01','F29','F18','F21','F11','F18','F18','F18','F09','F21','F20','F27','F22','F22','F08','F08','F10','F10','F27','F11','F11','F21','F09','F16','F20','F16','F29','F21','F20','F24','F13','F10','F10','F11','F21','F12','F08','F08','F22','F21','F29','F18']
table=next(b['rows'] for b in docs[2] if b['kind']=='table')
assert len(table)==43
for i,row in enumerate(table[1:]):add('FLY42',f'{i+1:02}',row[1],fly_map[i],f'matrix row {i+1}',dict(zip(table[0],row)))
top_map=['F01','F22','F18','F18','F10','F11','F11','F10','F01','F02','F08','F18','F09','F21','F18','F18','F29','F20','F16','F08']
tables=[b['rows'] for b in docs[3] if b['kind']=='table']
for i,row in enumerate(tables[0][1:]):
    caveat='This physiological observation is not itself an extra algorithm.' if i==11 else ''
    if i==18:caveat='The existing fitted heading/steering result does not demonstrate this rutabaga-dependent visual-memory plasticity.'
    add('FLY_TOP20',f'{i+1:02}',row[1],top_map[i],f'ranked row {i+1}',dict(zip(tables[0][0],row)),caveat)
fm=['F22','F01','F18','F18','F10','F11','F27','F18','F18','F21']
for i,row in enumerate(tables[1][1:]):add('FLY_FAMILIES10',f'{i+1:02}',row[0],fm[i],f'family row {i+1}',dict(zip(tables[1][0],row)),'A summary family, not an additional distinct mechanism count.')
human=['F01','F01','F01','F01','F02','F02','F21','F29','F02','F26','F26','F26']
table=next(b['rows'] for b in docs[4] if b['kind']=='table')
for i,row in enumerate(table[1:]):
    note='Human concept-cell association and rodent CA1 BTSP are different evidence; do not infer one local mechanism from the other.' if i==8 else ''
    add('HUMAN12',f'{i+1:02}',' / '.join(x for x in row[:2] if x),human[i],f'matrix row {i+1}',dict(zip(table[0],row)),note)
edu=['F08','F08','F08','F01','F25','F25','F30','F30','F30','F25','F29','F01','F25','F25']
table=next(b['rows'] for b in docs[5] if b['kind']=='table')
for i,row in enumerate(table[1:]):add('EDUCATION14',f'{i+1:02}',row[0],edu[i],f'table 1 row {i+1}',dict(zip(table[0],row)),'Evidence for human instruction is not a demonstrated neural learning mechanism in MiniFly.')

pruned=DOWNLOADS/'MiniFly_50plus_Mechanisms_Pruned_Codex_Summary_2026-09-26.md'
shutil.copyfile(pruned,S/pruned.name)
txt=pruned.read_text();part=txt[txt.index('# 7. Original 56-entry'):txt.index('# 8. Exact historical')]
tables56=[];table=[]
for line in part.splitlines():
    if re.match(r'^\|\s*\d+\s*\|',line):table.append([x.strip() for x in line.strip('|').split('|')])
    elif table:tables56.append(table);table=[]
if table:tables56.append(table)
assert [len(x) for x in tables56]==[31,25]
map25=['F10','F01','F22','F10','F10','F18','F01','F11','F21','F21','F11','F11','F12','F08','F08','F21','F10','F10','F11','F22','F18','F21','F21','F29','F28']
for k,table in enumerate(tables56):
    for i,row in enumerate(table):add('HANDOFF56',f'{k+1}.{i+1:02}',row[1],fly_map[i] if k==0 else map25[i],f'section 7.{k+1} row {i+1}',row)

cat=json.loads((S/'PROJECT_DOCUMENTS__MECHANISM_SCREEN_20261007__CATALOGUE_136_SCREEN.json').read_text())
mapping={}
def m(f,nums):
    for n in nums:assert n not in mapping;mapping[n]=f
m('F04',[1,6,10]);m('F29',[2,3,4,5,7,12,13,14,15,16,20,21,22]);m('F14',[8,9,11]);m('F09',[17,18,19,*range(41,50)])
m('F13',range(23,41));m('F02',[50,51,52,77]);m('F24',[53,*range(57,71),122,123,124]);m('F10',[54,120]);m('F31',[55]);m('F13',[56])
m('F01',[71,72,73,74,75,76,78,79]);m('F28',[80,90,91,*range(131,137)]);m('F23',[81,82,83,84,87,88,89]);m('F07',[85,86,121])
m('F11',range(92,104));m('F12',range(104,111));m('F08',range(111,120));m('F21',range(125,131))
assert set(mapping)==set(range(1,137))
for i,row in enumerate(cat,1):add('CATALOG136',f'C{i:03}',row['name'],mapping[i],f'original line {row["source_line"]}',row)

hist=json.loads((S/'PROJECT_DOCUMENTS__FULL151_VERSION_CENSUS_20261023__MECHANISMS_254_REVIEW.json').read_text())['rows']
def historical_family(row):
    ident=row['id'];name=row['original_register_row'].get('name','').lower()
    if ident.startswith('V30_DIRECTION'):return 'F29'
    if ident.startswith('V31_UNCERTAINTY'):return 'F24'
    if ident.startswith('SAMPLE_GEN'):return 'F08' if 'replay' in name else 'F29'
    if ident.startswith(('FEEDBACK_EFFICIENCY','OBSERVATION_SUPPORT','SUPPORT_DIRECTION')):return 'F01'
    if any(s in name for s in ['shuffled','scrambled','frozen','ordinary','historical_ridge','repeat_current']):return 'F28'
    if any(s in name for s in ['retrieve_hard','retrieve_soft','context_']):return 'F14'
    if 'silent_' in name:return 'F12'
    if any(s in name for s in ['active_forgetting','surprise_release','bayes_reopening']):return 'F11'
    if any(s in name for s in ['surprise_immediate','surprise_smoothed']):return 'F24'
    if any(s in name for s in ['cascade']):return 'F10'
    if any(s in name for s in ['replay']):return 'F08'
    if any(s in name for s in ['fast_slow','fastweights','feedback_consolidation','dual_','split_']):return 'F07'
    if any(s in name for s in ['sparse96','sparse384']):return 'F09'
    if any(s in name for s in ['allocate','allocation','importance','availability','bayes_stable','projection','turnover','adapter_','stable','recycled']):return 'F24'
    if any(s in name for s in ['bank','expansion']):return 'F23'
    if any(s in name for s in ['target_','reward_','e3_local','local_base','local_slow','spill']):return 'F01'
    if any(s in name for s in ['structured_','v21_input','v22_joint','e3_','minimal_']):return 'F29'
    raise ValueError('Unmapped historical recipe '+ident)
for row in hist:
    add('HISTORY254',row['id'],row['original_register_row'].get('name',row['id']),historical_family(row),'historical recipe '+row['id'],row,
        'This audit preserves the old result/provenance and assigns relevance; it does not rerun or re-certify the historical result. Control labels do not constitute new candidate mechanisms.')

# Executive Summary is another 11-row circuit rendition, not another 11 unique ideas.
execnames=['Hippocampus-neocortex','PFC-BG working-memory gating','Cortical feedback','Cerebellum','BG actor-critic','Grid and head-direction attractors','Insect central complex','Mushroom-body output loop','Sleep consolidation','Thalamocortical recurrence','Attention routing']
exec_text=(S/'Executive Summary.txt').read_text().replace('\f','')
pattern=r'(?m)^(?:Hippocampus[–-]Neocortex|Prefrontal Cortex \+ Basal Ganglia|Cortical Feedback \(|Cerebellum \(|Basal Ganglia as Actor|Entorhinal Grid Cells|Insect Central Complex|Insect Mushroom.Body Output Loop|Sleep.Dependent Consolidation\.|Thalamocortical Loops|Attention \(Learned Routing)'
exec_starts=[x.start() for x in re.finditer(pattern,exec_text)]
assert len(exec_starts)==11
for i,n in enumerate(execnames):
    end=exec_starts[i+1] if i<10 else len(exec_text)
    add('EXECUTIVE11',f'{i+1:02}',n,circuit_families[i],f'circuit section {i+1}',exec_text[exec_starts[i]:end],
        'Duplicate circuit rendition. Parameter counts differ between source renditions and are not a measured model census.')
for row in rows:
    if row['id'] in ('CIRCUIT_A:07','CIRCUIT_B:07','EXECUTIVE11:07'):
        row['secondary_families']=['F17']
        row['decision']='KEEP HEADING STEERING; PARK PI PACKAGE'
        row['exact_entry_caveat']+=' Heading/steering and displacement accumulation have different verdicts. F17 covers the stopped PI package; a connectome does not identify a validated lifetime goal-memory delta rule.'

expected={'CIRCUIT_A':11,'CIRCUIT_B':11,'FLY42':42,'FLY_TOP20':20,'FLY_FAMILIES10':10,'HUMAN12':12,'EDUCATION14':14,'HANDOFF56':56,'CATALOG136':136,'HISTORY254':254,'EXECUTIVE11':11}
counts=dict(collections.Counter(r['source'] for r in rows));assert counts==expected,(counts,expected)
assert len({r['id'] for r in rows})==len(rows)==577
assert all(r['original'] and r['reason'] and r['locator'] and r['family'] in families for r in rows)
allhash=[]
for x in sorted(S.rglob('*')):
    if x.is_file():allhash.append(dict(path=str(x.relative_to(R)),bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()))
original_names=[DOWNLOADS/(n+'.docx') for n in doc_names]+[DOWNLOADS/'Executive Summary.pdf',pruned]
original_names += [REPO/'output/MiniFly_Mechanism_Gaps_Colab.ipynb',REPO/'output/minifly-mechanism-gaps/screen_receipts.zip',DOWNLOADS/'MiniFly_outputs_receipts.zip',REPO/'output/minifly-mechanism-gaps/SCREEN_REPORT.md',REPO/'output/minifly-connectome-navigation/PI_FOLLOWUP_REPORT.md']
original_names += [REPO/'MOD97_LOCAL_RULE_20261106'/n for n in ('E9_REPORT.md','E12_REPORT.md','E13_REPORT.md')]
for x in original_names:allhash.append(dict(original_path=str(x),bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()))
e14=json.loads((REPO/'MOD97_LOCAL_RULE_20261106/exp45/E14_STATUS.json').read_text())
payload=dict(title='MiniFly mechanism catalog re-screen memo 01',
 reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository='ChangweiZhou/digitalbaby',branch='codex/minifly-consolidated-20261023',source_commit=ANCHOR,
 scope='Source-preserving computational re-screen; source/receipt inspection and selected primary-paper verification; no new scientific runs and no independent replication.',
 source_row_counts=counts,total_source_rows=len(rows),distinct_review_families=len(families),
 count_warning='577 is overlapping source rows, not 577 distinct mechanisms. The 254 historical rows are experimental recipes. Family grouping is for relevance, not proof of identical algorithms.',
 incomplete_source={'document':doc_names[3]+'.docx','claimed_catalog_entries':70,'available_ranked_rows':20,'available_family_rows':10,'missing_link':'sandbox:/mnt/data/fly_brain_learning_mechanisms_deep_research.md','consequence':'Cannot claim that all 70 individually specified entries were inspected. Family examples are preserved but do not reconstruct the missing full list.'},
 e14_status_at_capture=dict(status=e14['status'],complete=len(e14['completed']),planned=46,failed_parts=e14['failed_parts']),
 protected_files='No active experiment source/lock/checkpoint or prior result was changed.',families=list(families.values()),rows=rows)
payload['source_registry']={
 'CIRCUIT_A':dict(original=str(DOWNLOADS/(doc_names[0]+'.docx')),snapshot='sources/'+doc_names[0]+'.json'),
 'CIRCUIT_B':dict(original=str(DOWNLOADS/(doc_names[1]+'.docx')),snapshot='sources/'+doc_names[1]+'.json'),
 'FLY42':dict(original=str(DOWNLOADS/(doc_names[2]+'.docx')),snapshot='sources/'+doc_names[2]+'.json'),
 'FLY_TOP20':dict(original=str(DOWNLOADS/(doc_names[3]+'.docx')),snapshot='sources/'+doc_names[3]+'.json'),
 'FLY_FAMILIES10':dict(original=str(DOWNLOADS/(doc_names[3]+'.docx')),snapshot='sources/'+doc_names[3]+'.json'),
 'HUMAN12':dict(original=str(DOWNLOADS/(doc_names[4]+'.docx')),snapshot='sources/'+doc_names[4]+'.json'),
 'EDUCATION14':dict(original=str(DOWNLOADS/(doc_names[5]+'.docx')),snapshot='sources/'+doc_names[5]+'.json'),
 'HANDOFF56':dict(original=str(pruned),snapshot='sources/'+pruned.name),
 'CATALOG136':dict(repository_path='PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/CATALOGUE_136_SCREEN.json',snapshot='sources/PROJECT_DOCUMENTS__MECHANISM_SCREEN_20261007__CATALOGUE_136_SCREEN.json'),
 'HISTORY254':dict(repository_path='PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/MECHANISMS_254_REVIEW.json',snapshot='sources/PROJECT_DOCUMENTS__FULL151_VERSION_CENSUS_20261023__MECHANISMS_254_REVIEW.json'),
 'EXECUTIVE11':dict(original=str(DOWNLOADS/'Executive Summary.pdf'),snapshot='sources/Executive Summary.txt')}
(R/'MECHANISM_RESCREEN.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
(R/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(source_commit=ANCHOR,files=allhash),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),families=len(families),source_counts=counts,e14=payload['e14_status_at_capture']),indent=2))

def table_html():
    data=json.dumps([dict(id=r['id'],source=r['source'],name=r['name'],family=r['family'],secondary_families=r['secondary_families'],decision=r['decision'],reason=r['reason'],locator=r['locator'],caveat=r['exact_entry_caveat'],original=r['original']) for r in rows],ensure_ascii=False).replace('</','<\\/')
    famdata=json.dumps(families,ensure_ascii=False).replace('</','<\\/')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MiniFly mechanism re-screen memo 01</title>
<style>body{font:16px/1.5 system-ui,sans-serif;margin:0;color:#142438;background:#f3f6f9}main{max-width:1350px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 12px}p{max-width:950px}nav,aside{background:white;padding:18px;border:1px solid #dce3eb;border-radius:10px;margin:18px 0}.filters{display:flex;flex-wrap:wrap;gap:10px}input,select{font:inherit;padding:10px;border:1px solid #aab7c6;border-radius:6px}input{flex:1;min-width:240px}select{max-width:100%}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:left;border-bottom:1px solid #dce3eb;vertical-align:top}th{background:#e6edf5}td:first-child{font-size:13px;width:140px}td:nth-child(2){width:26%}.badge{display:inline-block;font-size:12px;border-radius:4px;background:#edf2f7;padding:3px 7px;margin-bottom:5px}button{font:inherit;cursor:pointer;border:1px solid #9dacbe;background:white;border-radius:6px;padding:7px 10px}details{margin-top:8px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace;background:#f6f8fa;padding:12px;max-height:440px;overflow:auto}small{color:#526278}#count{margin:14px 0}.family{margin:0 0 14px}a{color:#125ea7}@media(max-width:650px){main{padding:15px}table{font-size:14px}th,td{padding:8px}td:first-child{width:85px}}</style>
<main><h1>MiniFly mechanism re-screen memo 01</h1><p><b>577 overlapping source rows · 31 review families.</b> All available catalog rows retain an explicit disposition and original text. This is a relevance review, not 577 new mechanisms or a new experiment.</p>
<aside><b>Reading the filter.</b> “Discuss” means an idea survives for a specific future question; it does not authorize a run. “Reserve” is low current priority, not a family-wide scientific rejection. “In progress E15” means do not duplicate Claude’s current work.<br><br><b>Source gap:</b> the ChatGPT fly document names a 70-entry supplement at an unavailable sandbox link. Only its 20 ranked rows and 10 family summaries are available here. The full 70-entry supplement was not inspected.<br><br><a href="MINIFLY_MECHANISM_RESCREEN_MEMO_01_20261107.md">Read the numbered memo</a> · <a href="MECHANISM_RESCREEN.json">Full source-preserving JSON</a> · <a href="SOURCE_MANIFEST.json">Source hashes</a></aside>
<nav><div class="filters"><input id="q" aria-label="Search all catalog text" placeholder="Search mechanism, molecule, circuit, original evidence…"><select id="src" aria-label="Source"><option value="">All source lists</option></select><select id="fam" aria-label="Family"><option value="">All review families</option></select><select id="decision" aria-label="Decision"><option value="">All dispositions</option></select><button id="reset">Reset</button></div><div id="count" aria-live="polite"></div><div id="family"></div></nav>
<table><thead><tr><th>Original entry</th><th>Mechanism and disposition</th><th>Reason and source detail</th></tr></thead><tbody id="body"></tbody></table></main><script>
const DATA=__DATA__, FAMILIES=__FAMS__;
const E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for (const [id,values] of [['src',[...new Set(DATA.map(r=>r.source))]],['fam',Object.keys(FAMILIES)],['decision',[...new Set(DATA.map(r=>r.decision))]]]) for (const x of values){let o=document.createElement('option');o.value=x;o.textContent=id==='fam'?x+' '+FAMILIES[x].name:x;document.getElementById(id).append(o)}
const SEARCH=DATA.map(r=>JSON.stringify(r).toLowerCase());
function render(){const q=document.getElementById('q').value.toLowerCase(),src=document.getElementById('src').value,fam=document.getElementById('fam').value,d=document.getElementById('decision').value;const a=DATA.filter((r,i)=>(!q||SEARCH[i].includes(q))&&(!src||r.source===src)&&(!fam||r.family===fam||r.secondary_families.includes(fam))&&(!d||r.decision===d));document.getElementById('count').textContent=a.length+' / '+DATA.length+' source rows shown';const f=FAMILIES[fam];document.getElementById('family').innerHTML=f?'<div class="family"><b>'+E(f.name)+'</b><p>'+E(f.evidence)+'</p><p><b>When to reopen:</b> '+E(f.reopen_condition)+'</p></div>':'';document.getElementById('body').innerHTML=a.map(r=>'<tr><td>'+E(r.id)+'<br><small>'+E(r.locator)+'</small></td><td><span class="badge">'+E(r.decision)+'</span><br><b>'+E(r.name)+'</b><br><small>'+E(r.family)+' '+E(FAMILIES[r.family].name)+'</small></td><td>'+E(r.reason)+(r.caveat?'<p><small>'+E(r.caveat)+'</small></p>':'')+'<details><summary>Original entry and evidence</summary><pre>'+E(typeof r.original==='string'?r.original:JSON.stringify(r.original,null,2))+'</pre></details></td></tr>').join('')}
for(const id of ['q','src','fam','decision'])document.getElementById(id).addEventListener(id==='q'?'input':'change',render);document.getElementById('reset').onclick=()=>{for(const id of ['q','src','fam','decision'])document.getElementById(id).value='';render()};render();
</script></html>'''.replace('__DATA__',data).replace('__FAMS__',famdata)
(R/'mechanism-review.html').write_text(table_html())
