"""V41 targeted read/admission split. Inject M (V39), C (Claude V40), np.

No evaluator label, cohort, target marker, exposure count or checkpoint enters a
learner method. Step boundaries are complete records; state contains the cursor,
history, observations, generator outcomes and every learned/encoder/RNG object.
READ-HYBRID shares SLOTS' physical memory. It changes only fallback prediction.
FULL-HYBRID changes that fallback plus the frozen prediction controlling admission.
"""
import collections
import hashlib
import json
import math
import pickle

ARMS = ('BASE', 'SUPER', 'SLOTS', 'READ-HYBRID', 'FULL-HYBRID')


def _canonical_hash(obj):
    """Complete deterministic hash, preserving mapping order used for tie breaks."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, np.ndarray):
            h.update(b'array'); h.update(x.dtype.str.encode()); h.update(str(x.shape).encode())
            h.update(x.tobytes(order='C'))
        elif isinstance(x, np.generic):
            visit(x.item())
        elif isinstance(x, np.random.Generator):
            h.update(b'generator'); visit(x.bit_generator.state)
        elif isinstance(x, dict):
            h.update(b'dict'); h.update(str(len(x)).encode())
            for k,v in x.items(): visit(k); visit(v)
        elif isinstance(x, (tuple,list)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for v in x: visit(v)
        elif isinstance(x, (set,frozenset)):
            h.update(b'set')
            for v in sorted(x): visit(v)
        elif hasattr(x, '__dict__'):
            h.update(type(x).__name__.encode()); visit(vars(x))
        else:
            data=repr(x).encode(); h.update(str(len(data)).encode()+b':'+data)
    visit(obj)
    return h.hexdigest()


def _learners(state):
    return (state['encoder'], state['base'], state['live'], state['memories'])


def _layer_scores(layer, window):
    # Same numerical calculation as C.SuperposedLayer.scores, without ops mutation.
    ix, v = layer.encode(window)
    return v @ layer.output[ix]


def _read(state, window):
    z,b = state['encoder'].encode(bytes(window))
    base_scores = state['base'].scores(z,b)
    super_scores = _layer_scores(state['live'], window)
    fallback = super_scores if float(np.linalg.norm(super_scores)) > 1e-9 else base_scores
    slots = state['memories']['SLOTS'].lookup(window, base_scores)
    full = state['memories']['FULL-HYBRID'].lookup(window, fallback)
    scores = {'BASE':base_scores, 'SUPER':super_scores, 'SLOTS':slots['scores'],
              'READ-HYBRID':slots['scores'] if slots['authority'] else fallback,
              'FULL-HYBRID':full['scores']}
    return scores, {'SLOTS':slots, 'FULL-HYBRID':full}, (z,b)


def _stratum(packet):
    if packet['exact']: return 'exact_' + packet['type']
    if packet['authority']: return 'supported_slot'
    if packet['id'] is None: return 'no_candidate'
    return 'tentative_' + packet['type']


def _add_group(groups, key, arm, loss, correct):
    g=groups.setdefault(key, {}).setdefault(arm, {'n':0,'loss_sum':0.,'correct':0})
    g['n']+=1; g['loss_sum']+=float(loss); g['correct']+=int(correct)


def _probe(state, name, kind, ids, aliases=(0,1), revised=False):
    # Pickle contains full encoders, fixed weights, counters, containers and RNG.
    before=pickle.dumps(_learners(state), protocol=5)
    stream=state['stream']
    for i in ids:
        y=int(stream['revisions'][i] if revised and i in stream['revisions']
              else stream['labels'][i])
        for alias in aliases:
            c=M.context(stream['keys'][i],alias)
            scored, packets, _ = _read(state,c)
            for arm,s in scored.items():
                state['probes'].append({'checkpoint':name,'kind':kind,'arm':arm,
                     'key_id':int(i),'alias':int(alias),'accuracy':int(np.argmax(s)==y),
                     'bpb':float(M.loss(s,y)),
                     'authority':None if arm in ('BASE','SUPER') else
                            bool(packets['FULL-HYBRID' if arm=='FULL-HYBRID' else 'SLOTS']['authority'])})
    if pickle.dumps(_learners(state),protocol=5)!=before:
        raise AssertionError('Full learner state changed during a probe: '+name)
    state['probe_invariance_checks']+=1


def init_state(seed,cfg):
    stream=C.make_stream(int(seed),keys_per_wave=int(cfg['keys_per_wave']),repeats=int(cfg['repeats']))
    checkpoints={}
    for cp in stream['checkpoints']: checkpoints.setdefault(cp['record_count'],[]).append(cp)
    state={'seed':int(seed),'stream':stream,'checkpoints':checkpoints,'cursor':0,
      'history':bytearray(),'encoder':M.Encoder(seed,'last_fd'),'base':M.Learner('full'),
      'live':C.SuperposedLayer(seed),
      'memories':{a:M.Memory('TRACE-CORR',seed,trace_cap=int(cfg.get('trace_cap',512)),
                            slot_cap=int(cfg.get('slot_cap',384)))
                  for a in ('SLOTS','FULL-HYBRID')},
      'groups':{},'probes':[],'events':{},'checkpoint_memory':[],
      'admission_diagnostics':{},'read_diagnostics':{},'probe_invariance_checks':0,
      'done':False}
    return state


def _checkpoint(state,cp):
    stream=state['stream'];name=cp['name'];k=stream['keys_per_wave']
    if cp['type']=='wave':
        wi=ord(cp['wave'])-65
        _probe(state,name,'untouched_A',stream['untouched'],aliases=(0,1))
        _probe(state,name,'untouched_A_heldout',stream['untouched'],aliases=(2,))
        _probe(state,name,'current',range(wi*k,(wi+1)*k),aliases=(0,1))
        _probe(state,name,'current_heldout',range(wi*k,(wi+1)*k),aliases=(2,))
    else:
        _probe(state,name,'revision',stream['revised'],aliases=(0,1,2),revised=True)
    for a,m in state['memories'].items():
        c=m.cost()
        state['checkpoint_memory'].append({'checkpoint':name,'arm':a,
           **{k:int(c[k]) for k in ('traces','slots','anchors','semantic_bytes')},
           'clock':int(m.clock),'serial':int(m.serial)})


def step(state,cfg,max_units=64):
    if max_units<1: raise ValueError('max_units must be positive')
    if state['done']: return True
    rows=state['stream']['rows']
    stop=min(len(rows),state['cursor']+int(max_units))
    while state['cursor']<stop:
        number=state['cursor']+1;row=rows[number-1]
        stage=row['wave']+'/'+row['kind']
        for pos,y in enumerate(row['raw']):
            scored,packets,(z,b)=_read(state,state['history'])
            cat='target' if pos==C.CONTEXT_BYTES else 'nontarget'
            lp={a:float(M.loss(s,y)) for a,s in scored.items()}
            correct={a:int(np.argmax(s)==y) for a,s in scored.items()}
            for arm in ARMS:
                for group in ('whole',cat,stage+'/'+cat):
                    _add_group(state['groups'],group,arm,lp[arm],correct[arm])
                ref=packets['FULL-HYBRID' if arm=='FULL-HYBRID' else 'SLOTS']
                if arm not in ('BASE','SUPER'):
                    _add_group(state['groups'],'authority/'+_stratum(ref)+'/'+cat,
                               arm,lp[arm],correct[arm])
                # Common strata allow paired contrasts without post-treatment rebinning.
                _add_group(state['groups'],'slots_state/'+_stratum(packets['SLOTS'])+'/'+cat,
                           arm,lp[arm],correct[arm])
            rd=state['read_diagnostics'].setdefault(cat,{'n':0,'read_changed':0,
                  'read_helped':0,'read_hurt':0,'slot_authority':0,'full_authority':0,
                  'read_minus_slots_loss_sum':0.,'full_minus_read_loss_sum':0.,
                  'full_minus_slots_loss_sum':0.})
            rd['n']+=1; rd['read_changed']+=int(not np.array_equal(scored['SLOTS'],scored['READ-HYBRID']))
            rd['read_helped']+=int(lp['READ-HYBRID']<lp['SLOTS']-1e-12)
            rd['read_hurt']+=int(lp['READ-HYBRID']>lp['SLOTS']+1e-12)
            rd['slot_authority']+=int(packets['SLOTS']['authority'])
            rd['full_authority']+=int(packets['FULL-HYBRID']['authority'])
            rd['read_minus_slots_loss_sum']+=lp['READ-HYBRID']-lp['SLOTS']
            rd['full_minus_read_loss_sum']+=lp['FULL-HYBRID']-lp['READ-HYBRID']
            rd['full_minus_slots_loss_sum']+=lp['FULL-HYBRID']-lp['SLOTS']
            # Diagnostic only: frozen baseline correctness and layer correctness.
            # This describes admission eligibility; it does not alter any action.
            fb_top=packets['FULL-HYBRID']['base_top']
            d=state['admission_diagnostics'].setdefault(stage+'/'+cat,collections.Counter())
            d['n']+=1
            d['base_wrong_layer_correct']+=int(packets['SLOTS']['base_top']!=y and fb_top==y)
            d['base_correct_layer_wrong']+=int(packets['SLOTS']['base_top']==y and fb_top!=y)
            d['slots_no_exact_base_wrong']+=int(not packets['SLOTS']['exact'] and packets['SLOTS']['base_top']!=y)
            d['full_no_exact_fallback_wrong']+=int(not packets['FULL-HYBRID']['exact'] and fb_top!=y)
            state['base'].observe(z,b,y)
            state['live'].observe(state['history'],y)
            for arm,memory in state['memories'].items():
                ev=memory.observe(packets[arm],y)
                dst=state['events'].setdefault(arm,{}).setdefault(stage+'/'+cat,collections.Counter())
                for e in ev:
                    label=e['event']
                    if label in ('trace_removal','slot_eviction'):label+=':'+e.get('reason','capacity')
                    dst[label]+=1
            state['history'].append(y)
            if len(state['history'])>C.CONTEXT_BYTES:del state['history'][0]
            if pos==C.CONTEXT_BYTES and row['kind']=='acquisition' and row['exposure'] in (1,2):
                _probe(state,'E'+str(row['exposure']),'emergence',[row['key_id']],aliases=(2,))
        state['cursor']=number
        for cp in state['checkpoints'].get(number,[]): _checkpoint(state,cp)
    state['done']=state['cursor']==len(rows)
    return state['done']


def _mean_probe(probes,arm,checkpoint,kind,ids=None,aliases=None):
    p=[x['accuracy'] for x in probes if x['arm']==arm and x['checkpoint']==checkpoint
       and x['kind']==kind and (ids is None or x['key_id'] in ids)
       and (aliases is None or x['alias'] in aliases)]
    return float(np.mean(p)) if p else None


def summarize(state,cfg):
    if not state['done']: raise ValueError('Cannot summarize incomplete hybrid state')
    groups={key:{arm:{'n':g['n'],'bpb':g['loss_sum']/g['n'],
                        'accuracy':g['correct']/g['n'],'loss_sum':g['loss_sum']}
                   for arm,g in arms.items()} for key,arms in state['groups'].items()}
    stream=state['stream'];probes=state['probes']; k=stream['keys_per_wave']
    metrics={}
    for arm in ARMS:
        good=[]
        for i in stream['untouched']:
            e1=_mean_probe(probes,arm,'E1','emergence',ids=[i],aliases=[2])
            delays=[_mean_probe(probes,arm,cp,'untouched_A_heldout',ids=[i],aliases=[2])
                    for cp in ('after_B','after_C','after_D')]
            good.append(int(e1==1 and all(x==1 for x in delays)))
        metrics[arm]={
          'whole_bpb':groups['whole'][arm]['bpb'],
          'target_bpb':groups['target'][arm]['bpb'],
          'nontarget_bpb':groups['nontarget'][arm]['bpb'],
          'E1_all':_mean_probe(probes,arm,'E1','emergence'),
          'E1_A':_mean_probe(probes,arm,'E1','emergence',ids=range(k)),
          'E1_later':_mean_probe(probes,arm,'E1','emergence',ids=range(k,4*k)),
          'E2_all':_mean_probe(probes,arm,'E2','emergence'),
          'after_A_acquisition':_mean_probe(probes,arm,'after_A','current'),
          'after_D_acquisition':_mean_probe(probes,arm,'after_D','current'),
          'A_after_D_seen':_mean_probe(probes,arm,'after_D','untouched_A'),
          'A_after_D_heldout':_mean_probe(probes,arm,'after_D','untouched_A_heldout'),
          'same_identity_E1_all_delays':float(np.mean(good)),
          'revision_4_heldout':_mean_probe(probes,arm,'revision_4','revision',aliases=[2]),
          'revision_8_seen':_mean_probe(probes,arm,'revision_8','revision',aliases=[0,1])}
    costs={}
    for arm in ARMS:
        out={'slot_store_bytes':0,'slot_peak_bytes':0,'superposed_store_bytes':0,
             'superposed_fixed_code_bytes':0,'contains_operation_proxy':False}
        if arm in ('SLOTS','READ-HYBRID','FULL-HYBRID'):
            m=state['memories']['FULL-HYBRID' if arm=='FULL-HYBRID' else 'SLOTS']
            out['slot_store_bytes']=m.cost()['semantic_bytes'];out['slot_peak_bytes']=m.peak['semantic_bytes']
        if arm in ('SUPER','READ-HYBRID','FULL-HYBRID'):
            layer=state['live'];out['superposed_store_bytes']=int(layer.output.nbytes)
            out['superposed_fixed_code_bytes']=int(layer.contacts.nbytes+layer.weights.nbytes)
        costs[arm]=out
    return {'experiment':'hybrid_read_admission_split','seed':state['seed'],
       'keys':len(stream['keys']),'keys_per_wave':k,'repeats':stream['repeats'],
       'records':len(stream['rows']),'byte_events':len(stream['raw']),
       'unique_complete_records':len({r['raw'] for r in stream['rows']}),
       'unique_target_contexts':len({r['raw'][:C.CONTEXT_BYTES] for r in stream['rows']}),
       'target_presentations':len(stream['rows']),
       'stream_sha256':stream['raw_sha256'],'untouched_ids':[int(i) for i in stream['untouched']],
       'groups':groups,'metrics_by_arm':metrics,
       'metrics':[{'arm':arm,'metric':metric,'value':float(value)}
                  for arm,values in metrics.items() for metric,value in values.items()
                  if value is not None],
       'probes':probes,'costs':costs,
       'events':state['events'],'checkpoint_memory':state['checkpoint_memory'],
       'admission_diagnostics':state['admission_diagnostics'],'read_diagnostics':state['read_diagnostics'],
       'probe_invariance_checks':state['probe_invariance_checks'],
       'learner_full_state_hash':_canonical_hash(_learners(state)),
       'physical_slot_states':2,'read_hybrid_write_state':'SLOTS (same object by construction)',
       'limitations':['No clutter tail; targeted read/admission closure at one fixed load.',
          'Four seeds provide exploratory precision, not a broad qualification claim.',
          'FULL minus READ includes the entire downstream consequence of different admission histories.',
          'Shared scoring paths do not establish standalone runtime.',
          'Memory parameters are inherited; this is not a reduced-neuron architecture.']}


def self_test():
    cfg={'keys_per_wave':4,'repeats':2,'trace_cap':512,'slot_cap':384}
    s=init_state(419901,cfg)
    # Repeated scores and encoder use must leave absolutely all learner fields unchanged.
    before=pickle.dumps(_learners(s),protocol=5)
    _read(s,b''); _read(s,M.context(b'ABC123',2))
    assert pickle.dumps(_learners(s),protocol=5)==before
    assert len(s['memories'])==2 and 'READ-HYBRID' not in s['memories']
    # Functional score is numerically identical to the inherited scoring calculation.
    old=pickle.loads(pickle.dumps(s['live'],protocol=5))
    assert np.array_equal(_layer_scores(s['live'],b'ABC123'),old.scores(b'ABC123'))
    assert s['live'].ops==0 and old.ops>0
    # Compare the two deployable paths directly against original Claude Branches.
    # Legacy scoring counters differ because its read mutates ops; scientific states agree.
    ref=C.Branches(419901,freeze_after=0)
    equiv=init_state(419901,cfg)
    hist=bytearray()
    for record,row in enumerate(equiv['stream']['rows'][:2],1):
        for y in row['raw']:
            scored,packets,(z,b)=_read(equiv,hist)
            legacy_scores,legacy_packets=ref.read(hist,scored['BASE'])
            assert np.array_equal(scored['SLOTS'],legacy_scores['SLOTS'])
            assert np.array_equal(scored['FULL-HYBRID'],legacy_scores['HYBRID'])
            assert np.array_equal(scored['SUPER'],legacy_scores['SUPER'])
            equiv['base'].observe(z,b,y);equiv['live'].observe(hist,y)
            for arm,m in equiv['memories'].items():m.observe(packets[arm],y)
            ref.write(hist,scored['BASE'],legacy_packets,y,record)
            hist.append(y)
            if len(hist)>C.CONTEXT_BYTES:del hist[0]
    assert equiv['base'].state_hash()==ref.base.state_hash()
    assert equiv['memories']['SLOTS'].state_hash()==ref.memories['SLOTS'].state_hash()
    assert equiv['memories']['FULL-HYBRID'].state_hash()==ref.memories['HYBRID'].state_hash()
    assert np.array_equal(equiv['live'].output,ref.live.output)
    step(s,cfg,3);assert s['cursor']==3
    resumed=pickle.loads(pickle.dumps(s,protocol=5))
    while not step(s,cfg,7):pass
    while not step(resumed,cfg,5):pass
    a=summarize(s,cfg);b=summarize(resumed,cfg)
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    before=pickle.dumps(s,protocol=5)
    assert step(s,cfg,1)
    assert pickle.dumps(s,protocol=5)==before
    # Observers cannot affect learning: run the same stream with probe output disabled.
    plain=init_state(419901,cfg)
    global _probe
    original_probe=_probe
    try:
        _probe=lambda *args,**kwargs:None
        while not step(plain,cfg,9):pass
    finally:_probe=original_probe
    assert _canonical_hash(_learners(plain))==_canonical_hash(_learners(s))
    # No-read substitution on an authoritative exact trace must reproduce SLOTS.
    observed=init_state(419902,cfg)
    ctx=M.context(b'ABC123',0);packet=observed['memories']['SLOTS'].lookup(ctx,np.zeros(256))
    observed['memories']['SLOTS'].observe(packet,ord('a'))
    scored,packets,_=_read(observed,ctx)
    assert packets['SLOTS']['authority']
    assert np.array_equal(scored['SLOTS'],scored['READ-HYBRID'])
    assert np.array_equal(scored['FULL-HYBRID'],np.zeros(256))
    return {'pure_read':True,'functional_super_score_equivalence':True,
      'claude_deployable_paths_historical_equivalence':True,
      'two_slot_states_only':True,'record_budget_obeyed':True,'resume_complete_equality':True,
      'completed_noop':True,'observer_invariance':True,'exact_authority_preserved':True,
      'probe_checks':s['probe_invariance_checks'],'engineering_records':len(s['stream']['rows'])}
