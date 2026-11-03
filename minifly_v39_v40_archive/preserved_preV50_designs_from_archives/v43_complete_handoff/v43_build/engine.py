"""V43 shared physical trajectory and strictly causal read-only arbitration.

The runner injects the unchanged historical modules M, C, H, NumPy np and O
(offline controls). No workload metadata is passed to any learner method.
"""
import collections
import hashlib
import math
import pickle
import time

ARMS = ('SUPER', 'READ-HYBRID', 'SILENT-HARD', 'ANCHOR-ONLY', 'FIXED-50',
        'GLOBAL-BAYES', 'LOCAL-BAYES-0', 'LOCAL-BAYES-E2')
LN2 = math.log(2.)


def logistic_bits(e):
    """Stable logistic; scientific evidence itself is never clipped."""
    x = float(e) * LN2
    if x >= 0:
        z = math.exp(-x)
        return 1. / (1. + z)
    z = math.exp(x)
    return z / (1. + z)


def _distribution(scores):
    logits = np.asarray(scores, dtype=np.float64) / M.TEMP
    maximum = float(np.max(logits))
    logp = logits - (maximum + math.log(float(np.exp(logits-maximum).sum())))
    p = np.exp(logp)
    if not np.isfinite(logp).all() or not np.isfinite(p).all():
        raise FloatingPointError('Nonfinite expert distribution')
    return p, logp


def _mix(f, s, w):
    if w == 0.: return f
    if w == 1.: return s
    logp = np.logaddexp(math.log1p(-w)+f[1], math.log(w)+s[1])
    return np.exp(logp), logp


def _physical(state):
    return (state['encoder'], state['base'], state['live'], state['memory'],
            state['history'])


def _array_digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).digest()


def _walk_arrays(obj, action):
    if isinstance(obj, np.ndarray): action(obj)
    elif isinstance(obj, dict):
        for value in obj.values(): _walk_arrays(value, action)
    elif isinstance(obj, (tuple, list)):
        for value in obj: _walk_arrays(value, action)
    elif hasattr(obj, '__dict__'):
        _walk_arrays(vars(obj), action)


def _freeze(state):
    _walk_arrays(_physical(state), lambda a: a.setflags(write=False))


def _new_hash_cache(state):
    """Execution-only Merkle leaves; not part of the physical learner."""
    _freeze(state)
    cache = {'partitioned': {}, 'immutable': {}}
    for name, array in (('base.weights', state['base'].weights),
                        ('base.inverse', state['base'].inverse),
                        ('live.output', state['live'].output)):
        if array is not None:
            cache['partitioned'][name] = {'array': array,
                'leaves': [_array_digest(row) for row in array]}
    state['_hash_cache'] = cache
    return cache


def _restore_write_buffers(state):
    """Protocol-5 readonly arrays can restore over immutable byte buffers.

    Re-own only arrays that the native learner writes in place. Values and
    arithmetic stay identical; storage ownership is execution metadata.
    """
    changed = False
    for owner, name in ((state['base'], 'weights'), (state['base'], 'inverse'),
                        (state['base'], 'writes'), (state['live'], 'output')):
        array = getattr(owner, name)
        if array is not None and not array.flags.owndata:
            setattr(owner, name, np.array(array, copy=True, order='C'))
            changed = True
    return changed


def _refresh_hash_cache(state, bank, fast_indices):
    cache = state['_hash_cache']
    for name, array, indices in (('base.weights', state['base'].weights, (bank,)),
                                ('base.inverse', state['base'].inverse, (bank,)),
                                ('live.output', state['live'].output, fast_indices)):
        if array is None: continue
        entry = cache['partitioned'][name]
        if entry['array'] is not array: raise AssertionError('Unexpected physical array replacement')
        for index in indices: entry['leaves'][int(index)] = _array_digest(array[int(index)])
    _freeze(state)


def physical_hash(state, full=False):
    """Complete Merkle digest, including every field of the physical learner.

    All physical buffers are write-protected during reads. Only authorized
    shared writes can unprotect the base and fast output; their affected leaves
    are refreshed immediately. Every record independently recomputes all leaves.
    Dynamic trace/slot arrays and all counters/container/RNG fields are always
    hashed directly. Immutable encoder and fast-front-end leaves are cached.
    """
    cache = state.get('_hash_cache')
    if cache is None:
        cache = _new_hash_cache(state)
    h = hashlib.sha256()
    def visit(x, path):
        if isinstance(x, np.ndarray):
            if x.flags.writeable: raise AssertionError('Writable physical buffer during read: '+path)
            h.update(b'array'); h.update(x.dtype.str.encode()); h.update(repr(x.shape).encode())
            if path in cache['partitioned']:
                entry = cache['partitioned'][path]
                if entry['array'] is not x: raise AssertionError('Physical array identity changed during read')
                leaves = [_array_digest(row) for row in x] if full else entry['leaves']
                digest = hashlib.sha256(b''.join(leaves)).digest()
            elif path.startswith(('encoder.', 'live.encoder.', 'memory.enc.')) or path in ('live.contacts', 'live.weights'):
                entry = cache['immutable'].get(path)
                if full or entry is None or entry['array'] is not x:
                    digest = _array_digest(x)
                    if not full: cache['immutable'][path] = {'array': x, 'digest': digest}
                else: digest = entry['digest']
            else: digest = _array_digest(x)
            h.update(digest)
        elif isinstance(x, np.generic): visit(x.item(), path)
        elif isinstance(x, np.random.Generator):
            h.update(b'generator'); visit(x.bit_generator.state, path+'.rng')
        elif isinstance(x, dict):
            h.update(b'dict'); h.update(str(len(x)).encode())
            for k, v in x.items():
                visit(k, path+'.key'); visit(v, path+'.'+str(k))
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for i, v in enumerate(x): visit(v, path+'.'+str(i))
        elif isinstance(x, (set, frozenset)):
            h.update(b'set')
            for v in sorted(x): visit(v, path+'.member')
        elif hasattr(x, '__dict__'):
            h.update(type(x).__name__.encode()); visit(vars(x), path)
        else:
            data = repr(x).encode(); h.update(str(len(data)).encode()+b':'+data)
    for key in ('encoder', 'base', 'live', 'memory', 'history'):
        h.update(key.encode()); visit(state[key], key)
    return h.hexdigest()


def _verify_full_hash(state, count=True):
    if physical_hash(state) != physical_hash(state, full=True):
        raise AssertionError('Complete independently recomputed Merkle hash differs')
    if count: state['audits']['full_hash_recomputations'] += 1


def physical_components(state):
    return {key: H._canonical_hash(state[key])
            for key in ('encoder', 'base', 'live', 'memory', 'history')}


def _route(packet):
    if packet['exact']:
        return 'A' if packet['type'] == 'slot' else 'T'
    if packet['authority']:
        assert packet['type'] == 'slot' and packet['support'] >= 2
        return 'G'
    return 'F'


def _expert_read(state, window, diagnostics=False):
    """All returned arrays/distributions precede the actual byte."""
    z, bank = state['encoder'].encode(bytes(window))
    base_scores = state['base'].scores(z, bank)
    ix, v = state['live'].encode(window)
    fast_scores = v @ state['live'].output[ix]
    packet = state['memory'].lookup(window, base_scores)
    route = _route(packet)
    base, fast = _distribution(base_scores), _distribution(fast_scores)
    memory = state['memory']
    item = (memory.slots[packet['id']] if packet['type'] == 'slot' else
            memory.traces[packet['id']] if packet['type'] == 'trace' else None)
    candidate = _distribution(item['value']) if item is not None else None
    slow = candidate if route in ('A', 'T', 'G') else fast
    uid = int(packet['id']) if packet['type'] == 'slot' else None
    ledger = state['ledgers'].get(uid) if uid is not None else None
    if uid is not None and ledger is None:
        raise AssertionError('Missing generation ledger')
    weights = {'GLOBAL-BAYES': logistic_bits(state['global_E']),
               'LOCAL-BAYES-0': logistic_bits(ledger['E0']) if ledger else .5,
               'LOCAL-BAYES-E2': logistic_bits(ledger['E2']) if ledger else .5}
    historical_fallback = fast if float(np.linalg.norm(fast_scores)) > 1e-9 else base
    policies = {'SUPER': fast,
                'READ-HYBRID': slow if route in ('A', 'T', 'G') else historical_fallback}
    primary = slow if route == 'A' else fast
    policies['SILENT-HARD'] = slow if route == 'G' else primary
    policies['ANCHOR-ONLY'] = primary
    policies['FIXED-50'] = _mix(fast, slow, .5) if route == 'G' else primary
    for arm in ('GLOBAL-BAYES', 'LOCAL-BAYES-0', 'LOCAL-BAYES-E2'):
        policies[arm] = _mix(fast, slow, weights[arm]) if route == 'G' else primary
    features = None
    if diagnostics and route in ('A', 'T', 'G'):
        now = memory.clock
        # Native indices are monotone generation identifiers. Trace ids occupy
        # the same monotone namespace but are explicitly distinguished by route.
        object_uid = uid if uid is not None else int(item['id'])
        top2sim = None
        if route == 'G':
            address = memory.address(packet['c'])
            candidates = []
            if memory.slots:
                sums = np.asarray([x['sum'] for x in memory.slots.values()])
                norms = np.linalg.norm(sums, axis=1, keepdims=True)
                normalized = np.divide(sums, norms, out=np.zeros_like(sums), where=norms != 0)
                candidates.extend((normalized @ address).tolist())
            if memory.traces:
                candidates.extend((memory.trace_matrix() @ address).tolist())
            if len(candidates) >= 2:
                top2sim = float(np.partition(np.asarray(candidates), -2)[-2])
        fp, sp = fast[0], slow[0]
        fm = np.partition(fp, -2)[-2:]; sm = np.partition(sp, -2)[-2:]
        midpoint = .5 * (fp + sp)
        # xlogx masks handle an underflowed probability without inventing mass.
        def entropy(p, lp):
            nz = p > 0
            return float(-np.dot(p[nz], lp[nz]) / LN2)
        good_f = fp > 0; good_s = sp > 0
        js = .5 * (float(np.dot(fp[good_f], fast[1][good_f]-np.log(midpoint[good_f]))) +
                    float(np.dot(sp[good_s], slow[1][good_s]-np.log(midpoint[good_s])))) / LN2
        features = {'route': route, 'slot_uid': object_uid,
            'generation_uid': object_uid, 'support': int(packet['support']),
            'slot_age': int(now-item['created']), 'time_since_last_use': int(now-item['last']),
            'prior_similarity_use_count': int(ledger['uses']) if ledger else 0,
            'similarity_top1': float(packet['sim']), 'similarity_top2': top2sim,
            'similarity_margin': float(packet['sim']-top2sim) if top2sim is not None else None,
            'fast_top1': int(np.argmax(fp)), 'slow_top1': int(np.argmax(sp)),
            'fast_slow_agree': bool(np.argmax(fp) == np.argmax(sp)),
            'fast_entropy': entropy(fp, fast[1]), 'slow_entropy': entropy(sp, slow[1]),
            'fast_margin': float(fm[-1]-fm[-2]), 'slow_margin': float(sm[-1]-sm[-2]),
            'js_divergence': js, 'local_E_before': float(ledger['E2']) if ledger else 0.,
            'local_0_E_before': float(ledger['E0']) if ledger else 0.,
            'global_E_before': float(state['global_E']),
            'local_w_before': weights['LOCAL-BAYES-E2'],
            'local_0_w_before': weights['LOCAL-BAYES-0'],
            'global_w_before': weights['GLOBAL-BAYES'],
            'revision_age': int(now-ledger['last_revision']) if ledger and ledger['was_revised'] else None,
            'was_revised': bool(ledger['was_revised']) if ledger else False}
    return {'packet': packet, 'route': route, 'z': z, 'bank': bank, 'fast_indices': ix,
            'base': base, 'fast': fast, 'slow': slow, 'candidate': candidate,
            'policies': policies, 'features': features, 'uid': uid}


def _add(groups, group, arm, loss, correct=None):
    g = groups.setdefault(group, {}).setdefault(arm, {'n': 0, 'loss_sum': 0., 'correct': 0})
    g['n'] += 1; g['loss_sum'] += float(loss)
    if correct is None: g['correct'] = None
    elif g['correct'] is not None: g['correct'] += int(correct)


def init_state(seed, cfg):
    load = int(cfg['load'])
    if load % 4: raise ValueError('Load must contain four equally sized waves')
    stream = C.make_stream(int(seed), keys_per_wave=load//4, repeats=int(cfg.get('repeats', 8)))
    checkpoints = {}
    for cp in stream['checkpoints']:
        checkpoints.setdefault(cp['record_count'], []).append(cp)
    state = {'seed': int(seed), 'load': load, 'stream': stream, 'checkpoints': checkpoints,
             'cursor': 0, 'history': bytearray(), 'encoder': M.Encoder(seed, 'last_fd'),
             'base': M.Learner('full'), 'live': C.SuperposedLayer(seed),
             'memory': M.Memory('TRACE-CORR', seed, trace_cap=int(cfg.get('trace_cap', 512)),
                                slot_cap=int(cfg.get('slot_cap', 384))),
             'ledgers': {}, 'global_E': 0., 'groups': {}, 'probes': [],
             'authority_events': [], 'physical_events': collections.Counter(),
             'checkpoint_memory': [], 'route_counts': collections.Counter(),
             'audits': {'firewall_checks': 0, 'probe_checks': 0, 'ledger_births': 0,
                        'ledger_evictions': 0, 'revision_resets': 0,
                        'exact_slot_contradictions': 0, 'prepromotion_captures': 0,
                        'raw_fast_fallback_differences': 0,
                        'nonfinite_predictions': 0, 'policy_state_divergences': 0,
                        'full_hash_recomputations': 0},
             'ledger_lifecycle': [], 'done': False}
    state['initial_physical_hash'] = physical_hash(state)
    return state


def _probe(state, name, kind, ids, aliases=(0, 1), revised=False):
    before = physical_hash(state)
    virtual_before = pickle.dumps((state['ledgers'], state['global_E']), protocol=5)
    stream = state['stream']
    for ident in ids:
        y = int(stream['revisions'][ident] if revised and ident in stream['revisions']
                else stream['labels'][ident])
        for alias in aliases:
            read = _expert_read(state, M.context(stream['keys'][ident], alias))
            packet = read['packet']
            for arm, (p, lp) in read['policies'].items():
                state['probes'].append({'checkpoint': name, 'kind': kind, 'arm': arm,
                    'key_id': int(ident), 'alias': int(alias), 'accuracy': int(np.argmax(p) == y),
                    'bpb': float(-lp[y]/LN2), 'route': read['route'],
                    'physical_slots_accuracy': int(np.argmax(packet['scores']) == y),
                    'physical_slow_accuracy': int(np.argmax(read['slow'][0]) == y)
                        if read['route'] in ('A', 'G') else None,
                    'physical_slot_uid': read['uid'], 'support': int(packet['support']),
                    'local_w_before': logistic_bits(state['ledgers'][read['uid']]['E2'])
                        if read['uid'] is not None else None})
    if physical_hash(state) != before:
        raise AssertionError('Physical state mutated by evaluation probe: ' + name)
    if pickle.dumps((state['ledgers'], state['global_E']), protocol=5) != virtual_before:
        raise AssertionError('Virtual state mutated by evaluation probe: ' + name)
    state['audits']['probe_checks'] += 1


def _checkpoint(state, cp):
    stream = state['stream']; name = cp['name']; k = stream['keys_per_wave']
    if cp['type'] == 'wave':
        wi = ord(cp['wave']) - 65
        _probe(state, name, 'untouched_A', stream['untouched'], aliases=(0, 1))
        _probe(state, name, 'untouched_A_heldout', stream['untouched'], aliases=(2,))
        _probe(state, name, 'current', range(wi*k, (wi+1)*k), aliases=(0, 1))
        _probe(state, name, 'current_heldout', range(wi*k, (wi+1)*k), aliases=(2,))
        if name == 'after_D':
            _probe(state, name, 'revision_retained', stream['revised'], aliases=(0, 1, 2), revised=True)
    else:
        _probe(state, name, 'revision', stream['revised'], aliases=(0, 1, 2), revised=True)
    c = state['memory'].cost()
    state['checkpoint_memory'].append({'checkpoint': name,
        **{k: int(c[k]) for k in ('traces', 'slots', 'anchors', 'semantic_bytes')},
        'clock': int(state['memory'].clock), 'serial': int(state['memory'].serial),
        'physical_hash': physical_hash(state), 'components': physical_components(state),
        'ledger_count': len(state['ledgers']), 'global_E': float(state['global_E'])})


def _lifecycle(state, old_ids, prior_uid, prior_top, trace_delta, packet, y, physical_events):
    """Virtual bookkeeping after the single common physical update."""
    memory = state['memory']; ledgers = state['ledgers']; now = memory.clock
    resident = set(memory.slots)
    for uid in sorted(old_ids-resident):
        del ledgers[uid]
        state['audits']['ledger_evictions'] += 1
        state['ledger_lifecycle'].append({'time': now, 'event': 'eviction', 'uid': uid})
    for uid in sorted(resident-old_ids):
        if trace_delta is None or not any(e['event'] == 'promotion' for e in physical_events):
            raise AssertionError('New mature generation without frozen E2 trace evidence')
        ledgers[uid] = {'E0': 0., 'E2': float(trace_delta), 'uses': 0,
                        'birth': now, 'last_revision': now, 'was_revised': False}
        state['audits']['ledger_births'] += 1
        state['audits']['prepromotion_captures'] += 1
        state['ledger_lifecycle'].append({'time': now, 'event': 'birth', 'uid': uid,
                                          'E2_seed': float(trace_delta)})
    if prior_uid is not None and packet['exact'] and prior_uid in resident:
        new_top = int(np.argmax(memory.slots[prior_uid]['value']))
        if prior_top != y: state['audits']['exact_slot_contradictions'] += 1
        if new_top != prior_top:
            ledger = ledgers[prior_uid]
            ledger['E0'] = 0.; ledger['E2'] = 0.
            ledger['last_revision'] = now; ledger['was_revised'] = True
            # Prior-use count concerns all G reads of this generation. A value
            # revision resets evidence but does not create a new slot generation.
            state['audits']['revision_resets'] += 1
            state['ledger_lifecycle'].append({'time': now, 'event': 'revision_reset',
                'uid': prior_uid, 'old_top': prior_top, 'new_top': new_top})
    if set(ledgers) != resident:
        raise AssertionError('Ledger generation namespace is not resident-only')


def _record_evidence(state, read, actual):
    """Post-outcome evidence only; deliberately separate from read construction."""
    if read['route'] != 'G': return
    delta = float((read['slow'][1][actual]-read['fast'][1][actual])/LN2)
    ledger = state['ledgers'][read['uid']]
    ledger['E0'] += delta; ledger['E2'] += delta; ledger['uses'] += 1
    state['global_E'] += delta


def step(state, cfg, max_units=16):
    if max_units < 1: raise ValueError('max_units must be positive')
    if state['done']: return True
    # NumPy pickle restores writable arrays. This one entry-boundary operation
    # restores guards before the first pre-read hash; hashes inside a read never
    # silently repair guard failures.
    restored = _restore_write_buffers(state)
    if restored or state['base'].weights.flags.writeable or state['live'].output.flags.writeable:
        _new_hash_cache(state)
        _verify_full_hash(state, count=False)
    rows = state['stream']['rows']
    stop = min(len(rows), state['cursor']+int(max_units))
    while state['cursor'] < stop:
        number = state['cursor']+1; row = rows[number-1]
        stage = row['wave']+'/'+row['kind']
        for pos, y in enumerate(row['raw']):
            before = physical_hash(state)
            read = _expert_read(state, state['history'], diagnostics=True)
            # No y or evaluator metadata has been passed to read/policy functions.
            tops = {arm: int(np.argmax(p)) for arm, (p, lp) in read['policies'].items()}
            if physical_hash(state) != before:
                raise AssertionError('Policy scoring mutated complete physical state')
            state['audits']['firewall_checks'] += 1
            route = read['route']; packet = read['packet']
            category = 'target' if pos == C.CONTEXT_BYTES else 'nontarget'
            state['route_counts'][route] += 1
            # Outcome becomes available only now to scoring and virtual evidence.
            losses = {arm: float(-lp[y]/LN2) for arm, (p, lp) in read['policies'].items()}
            for arm in ARMS:
                for group in ('whole', category, 'route/'+route, stage+'/'+category):
                    _add(state['groups'], group, arm, losses[arm], tops[arm] == y)
            if route == 'F' and losses['SUPER'] != losses['READ-HYBRID']:
                state['audits']['raw_fast_fallback_differences'] += 1
            slow_delta = float((read['slow'][1][y]-read['fast'][1][y])/LN2)
            if read['features'] is not None:
                event = read['features']
                event.update({'seed': state['seed'], 'load': state['load'],
                    'event_index': int(state['memory'].clock), 'physical_time': int(state['memory'].clock),
                    'category': category, 'true_byte': int(y),
                    'pB_true': float(read['base'][0][y]), 'pF_true': float(read['fast'][0][y]),
                    'pS_true': float(read['slow'][0][y]),
                    'logpF_true': float(read['fast'][1][y]), 'logpS_true': float(read['slow'][1][y]),
                    'slow_minus_fast_loss': -slow_delta, 'local_loss': losses['LOCAL-BAYES-E2'],
                    'policy_losses': losses, 'policy_correct': {arm: int(tops[arm] == y) for arm in ARMS}})
                state['authority_events'].append(event)
            _record_evidence(state, read, y)
            # Capture any tentative trace, including route F. Promotion eligibility
            # is still decided solely by the frozen native Memory.observe packet.
            trace_delta = (float((read['candidate'][1][y]-read['fast'][1][y])/LN2)
                           if packet['type'] == 'trace' and not packet['exact'] else None)
            memory = state['memory']; old_ids = set(memory.slots)
            prior_uid = read['uid']; prior_top = packet['candidate_top']
            # The only write window. Encoder and address buffers remain readonly.
            state['base'].weights.setflags(write=True)
            if state['base'].inverse is not None: state['base'].inverse.setflags(write=True)
            state['base'].writes.setflags(write=True)
            state['live'].output.setflags(write=True)
            state['base'].observe(read['z'], read['bank'], y)
            state['live'].observe(state['history'], y)
            physical_events = memory.observe(packet, y)
            for ev in physical_events:
                label = ev['event']
                if label in ('trace_removal', 'slot_eviction'): label += ':'+ev.get('reason', 'capacity')
                state['physical_events'][label] += 1
            _lifecycle(state, old_ids, prior_uid, prior_top, trace_delta, packet, y, physical_events)
            _refresh_hash_cache(state, read['bank'], read['fast_indices'])
            state['history'].append(y)
            if len(state['history']) > C.CONTEXT_BYTES: del state['history'][0]
            if pos == C.CONTEXT_BYTES and row['kind'] == 'acquisition' and row['exposure'] in (1, 2):
                _probe(state, 'E'+str(row['exposure']), 'emergence', [row['key_id']], aliases=(2,))
        _verify_full_hash(state)
        state['cursor'] = number
        for cp in state['checkpoints'].get(number, []): _checkpoint(state, cp)
    state['done'] = state['cursor'] == len(rows)
    return state['done']


def _probe_mean(probes, arm, checkpoint, kind, ids=None, aliases=None):
    values = [p['accuracy'] for p in probes if p['arm'] == arm and p['checkpoint'] == checkpoint
              and p['kind'] == kind and (ids is None or p['key_id'] in ids)
              and (aliases is None or p['alias'] in aliases)]
    return float(np.mean(values)) if values else None


def summarize(state, cfg):
    if not state['done']: raise ValueError('Cannot summarize an incomplete trajectory')
    groups = pickle.loads(pickle.dumps(state['groups'], protocol=5))
    # Reporting annotations are attached to copies. Summarization is read-only
    # with respect to the saved physical trajectory and online evidence history.
    events = [dict(e) for e in state['authority_events']]
    gevents = [e for e in events if e['route'] == 'G']
    offline = O.evaluate(gevents, seed=state['seed'], load=state['load'])
    sl = offline['shuffled_losses']; perm = offline['permutable_mask']
    eo = offline['event_oracle_losses']; so = offline['slot_oracle_losses']
    mapping_columns = ('shuffle_source_event_index', 'shuffle_age_bin', 'shuffle_support_bin',
                       'shuffle_prior_use_bin', 'shuffle_permutable', 'shuffled_loss',
                       'event_oracle_loss', 'slot_oracle_loss')
    for event in events:
        for column in mapping_columns: event[column] = None
    for j, event in enumerate(gevents):
        source = offline['permutation_source_indices'][j]
        support_bin, age_bin, use_bin = offline['stratum_keys'][j]
        event.update({'shuffle_source_event_index': int(gevents[source]['event_index']),
            'shuffle_age_bin': int(age_bin), 'shuffle_support_bin': ('2', '3-4', '5+').index(support_bin),
            'shuffle_prior_use_bin': ('0-1', '2-3', '4+').index(use_bin), 'shuffle_permutable': bool(perm[j]),
            'shuffled_loss': float(sl[j]), 'event_oracle_loss': float(eo[j]),
            'slot_oracle_loss': float(so[j])})
        for arm, value in (('SHUFFLED-ALPHA', sl[j]), ('EVENT-ORACLE', eo[j]), ('SLOT-ORACLE', so[j])):
            _add(groups, 'route/G', arm, value)
        if perm[j]:
            for arm in ARMS: _add(groups, 'G_permutable', arm, event['policy_losses'][arm], event['policy_correct'][arm])
            _add(groups, 'G_permutable', 'SHUFFLED-ALPHA', sl[j])
    for category in ('whole', 'target', 'nontarget'):
        original = groups[category]['LOCAL-BAYES-E2']
        delta = math.fsum(sl[j]-event['local_loss'] for j, event in enumerate(gevents)
                          if category == 'whole' or event['category'] == category)
        groups[category]['SHUFFLED-ALPHA'] = {'n': original['n'],
            'loss_sum': original['loss_sum']+delta, 'correct': None}
    for arms in groups.values():
        for g in arms.values():
            g['bpb'] = g['loss_sum']/g['n'] if g['n'] else None
            g['accuracy'] = g['correct']/g['n'] if g['n'] and g['correct'] is not None else None
    probes = state['probes']; stream = state['stream']; k = stream['keys_per_wave']
    metrics = {}
    hardloss = groups.get('route/G', {}).get('SILENT-HARD', {}).get('loss_sum', 0.)
    oracleloss = math.fsum(eo); headroom = hardloss-oracleloss
    for arm in ARMS+('SHUFFLED-ALPHA', 'EVENT-ORACLE', 'SLOT-ORACLE'):
        m = {}
        for group, key in (('whole', 'whole'), ('target', 'target'), ('nontarget', 'nontarget'),
                           ('route/G', 'G'), ('route/A', 'A'), ('route/T', 'T'), ('G_permutable', 'G_permutable')):
            if arm in groups.get(group, {}):
                g = groups[group][arm]; m[key+'_bpb'] = g['bpb']
                if g['accuracy'] is not None: m[key+'_accuracy'] = g['accuracy']
        if arm in groups.get('route/G', {}):
            m['oracle_capture'] = (hardloss-groups['route/G'][arm]['loss_sum'])/headroom if headroom > 1e-12 else None
        if arm in ARMS:
            m.update({'E1_all': _probe_mean(probes, arm, 'E1', 'emergence'),
                'E2_all': _probe_mean(probes, arm, 'E2', 'emergence'),
                'E1_A': _probe_mean(probes, arm, 'E1', 'emergence', ids=range(k)),
                'E1_later': _probe_mean(probes, arm, 'E1', 'emergence', ids=range(k, 4*k)),
                'A_after_D_seen': _probe_mean(probes, arm, 'after_D', 'untouched_A'),
                'A_after_D_heldout': _probe_mean(probes, arm, 'after_D', 'untouched_A_heldout'),
                'after_D_current_seen': _probe_mean(probes, arm, 'after_D', 'current'),
                'after_D_current_heldout': _probe_mean(probes, arm, 'after_D', 'current_heldout'),
                'revision_after_D_seen': _probe_mean(probes, arm, 'after_D', 'revision_retained', aliases=(0, 1)),
                'revision_after_D_heldout': _probe_mean(probes, arm, 'after_D', 'revision_retained', aliases=(2,))})
            for r in (1, 2, 4, 8):
                for suffix, aliases in (('seen', (0, 1)), ('heldout', (2,)), ('exact', (0,)), ('known', (1,))):
                    m[f'revision_{r}_{suffix}'] = _probe_mean(probes, arm, f'revision_{r}', 'revision', aliases=aliases)
            good = []
            for ident in stream['untouched']:
                v = [_probe_mean(probes, arm, 'E1', 'emergence', ids=[ident], aliases=(2,))]
                v.extend(_probe_mean(probes, arm, cp, 'untouched_A_heldout', ids=[ident], aliases=(2,))
                         for cp in ('after_B', 'after_C', 'after_D'))
                good.append(int(all(x == 1 for x in v)))
            m['same_identity_E1_all_delays'] = float(np.mean(good)) if good else None
        metrics[arm] = m
    acquisition = []
    for cp in ('after_A', 'after_B', 'after_C', 'after_D'):
        rows = [p for p in probes if p['arm'] == 'SILENT-HARD' and p['checkpoint'] == cp and p['kind'] == 'current']
        mature = [p for p in rows if p['route'] in ('A', 'G')]
        acquisition.append({'checkpoint': cp, 'population_n': len(rows),
            'population_correct': sum(p['physical_slots_accuracy'] for p in rows),
            'population_accuracy': float(np.mean([p['physical_slots_accuracy'] for p in rows])) if rows else None,
            'mature_n': len(mature), 'mature_correct': sum(p['physical_slow_accuracy'] for p in mature),
            'mature_accuracy': float(np.mean([p['physical_slow_accuracy'] for p in mature])) if mature else None,
            'mature_coverage': len(mature)/len(rows) if rows else None,
            'mature_success_over_population': sum(p['physical_slow_accuracy'] for p in mature)/len(rows) if rows else None})
    audits = dict(state['audits'])
    audits.update({'route_counts': {r: int(state['route_counts'][r]) for r in 'ATGF'},
        'route_exclusivity_pass': sum(state['route_counts'].values()) == len(stream['raw']),
        'physical_acquisition_current': acquisition,
        'shuffled': offline['audit'], 'oracle_headroom_sum': headroom,
        'oracle_headroom_per_G_event': headroom/len(gevents) if gevents else None,
        'full_state_hash_algorithm': 'complete SHA256 Merkle hash; readonly buffers; refreshed shared-write leaves; full independent recomputation each record',
        'all_slot_ledgers_resident': set(state['ledgers']) == set(state['memory'].slots),
        'E2_seed_captures_equal_promotions': state['audits']['prepromotion_captures'] == state['physical_events']['promotion']})
    authority_summary = {}
    for arm in ARMS:
        if arm in ('SUPER', 'ANCHOR-ONLY'): weights = [0.]*len(gevents)
        elif arm in ('READ-HYBRID', 'SILENT-HARD'): weights = [1.]*len(gevents)
        elif arm == 'FIXED-50': weights = [.5]*len(gevents)
        else:
            field = {'GLOBAL-BAYES': 'global_w_before', 'LOCAL-BAYES-0': 'local_0_w_before',
                     'LOCAL-BAYES-E2': 'local_w_before'}[arm]
            weights = [e[field] for e in gevents]
        mass = math.fsum(weights)
        authority_summary[arm] = {'G_authority_mass': mass,
            'G_mean_authority': mass/len(gevents) if gevents else None,
            'G_events': len(gevents),
            'mature_exact_authority_events': int(state['route_counts']['A']) if arm != 'SUPER' else 0,
            'probationary_exact_authority_events': int(state['route_counts']['T']) if arm == 'READ-HYBRID' else 0}
    return {'schema_version': 1, 'experiment': 'V43_fast_slow_authority', 'seed': state['seed'],
        'load': state['load'], 'keys_per_wave': k, 'repeats': stream['repeats'],
        'records': len(stream['rows']), 'byte_events': len(stream['raw']),
        'physical_trajectories': 1, 'physical_base_updates': len(stream['raw']),
        'physical_fast_updates': int(state['live'].updates), 'physical_slow_updates': int(state['memory'].clock),
        'target_presentations': len(stream['rows']),
        'unique_complete_records': len({r['raw'] for r in stream['rows']}),
        'unique_target_contexts': len({r['raw'][:C.CONTEXT_BYTES] for r in stream['rows']}),
        'stream_sha256': stream['raw_sha256'], 'groups': groups, 'metrics_by_arm': metrics,
        'probes': probes, 'authority_events': events,
        'offline_mapping_columns': list(mapping_columns),
        'offline_mapping_definition': 'G-only flat authority-event columns; source is the physical event_index; '
            'age bins are 0..3, support/use bins are 0..2; non-G columns are null. '
            'Each nonsingleton stratum is deranged; singleton sources remain themselves.',
        'slot_oracle': offline.get('slot_oracle_rows', []), 'authority_summary': authority_summary,
        'terminal_evidence': [{'slot_uid': int(uid), **ledger} for uid, ledger in state['ledgers'].items()],
        'physical_events': dict(state['physical_events']), 'checkpoint_memory': state['checkpoint_memory'],
        'ledger_lifecycle': state['ledger_lifecycle'], 'audits': audits,
        'final_physical_hash': physical_hash(state), 'final_physical_components': physical_components(state),
        'initial_physical_hash': state['initial_physical_hash'],
        'costs': {'slow': state['memory'].cost(), 'slow_peak': dict(state['memory'].peak),
                  'fast': state['live'].cost(), 'base': state['base'].costs(state['encoder']),
                  'virtual_ledger_count': len(state['ledgers']),
                  'virtual_ledgers_logical_float64_bytes': len(state['ledgers'])*16+8},
        'limitations': ['SHUFFLED-ALPHA is an offline loss-only control; no top-1 or probe predictions reconstructed.',
            'Slot-wide revision commit is operationalized as an exact-anchor update changing value argmax.',
            'Mature slot support is retained-anchor count, bounded at two by the inherited substrate.']}


def self_test():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    check('stable_logistic_extremes', logistic_bits(1e9) == 1. and logistic_bits(-1e9) == 0. and logistic_bits(0) == .5)
    scores = np.linspace(-.3, .7, 256)
    p, lp = _distribution(scores)
    check('historical_loss_equivalence', abs(-lp[17]/LN2-M.loss(scores, 17)) < 1e-12)
    check('mixture_normalized', abs(float(_mix((p, lp), _distribution(-scores), .4)[0].sum())-1.) < 1e-12)
    for expected, packet in [('A', {'exact': True, 'type': 'slot'}), ('T', {'exact': True, 'type': 'trace'}),
            ('G', {'exact': False, 'type': 'slot', 'authority': True, 'support': 2}),
            ('F', {'exact': False, 'type': 'trace', 'authority': False})]:
        check('exclusive_route_'+expected, _route(packet) == expected)
    cfg = {'load': 16, 'repeats': 2, 'trace_cap': 8, 'slot_cap': 2}
    st = init_state(943001, cfg); m = st['memory']
    c0 = M.context(b'ABCDEF', 0); c1 = M.context(b'ABCDEF', 1); y = ord('q')
    # Direct native trace→slot fixture uses the locked confirmation rule. Ledger
    # initialization gets the frozen pre-E2 trace likelihood, never a postupdate value.
    badbase = np.zeros(256); badbase[ord('r')] = 1.
    packet = m.lookup(c0, badbase); m.observe(packet, y)
    packet = m.lookup(c1, badbase); candidate = _distribution(m.traces[packet['id']]['value'])
    delta = (candidate[1][y]+math.log(256))/LN2
    old = set(m.slots); ev = m.observe(packet, y)
    _lifecycle(st, old, None, packet['candidate_top'], delta, packet, y, ev)
    uid = next(iter(m.slots)); check('pre_E2_promotion_capture', st['ledgers'][uid]['E2'] == delta)
    _freeze(st)
    before = physical_hash(st); virt = pickle.dumps(st['ledgers'])
    _expert_read(st, c0, diagnostics=True)
    check('policy_full_physical_firewall', physical_hash(st) == before and pickle.dumps(st['ledgers']) == virt)
    caught = False
    try: st['live'].output[0, 0] = 1.
    except ValueError: caught = True
    check('physical_readonly_guard_rejects_array_write', caught)
    check('cached_and_full_Merkle_agree', physical_hash(st) == physical_hash(st, full=True))
    old_evidence = pickle.dumps((st['ledgers'], st['global_E']), protocol=5)
    gread = _expert_read(st, M.context(b'ABCDEF', 2))
    check('G_prediction_uses_prior_evidence', gread['route'] == 'G' and
          pickle.dumps((st['ledgers'], st['global_E']), protocol=5) == old_evidence)
    old_e2 = st['ledgers'][uid]['E2']
    expected_delta = (gread['slow'][1][y]-gread['fast'][1][y])/LN2
    _record_evidence(st, gread, y)
    check('G_evidence_updates_only_after_outcome', st['ledgers'][uid]['E2'] == old_e2+expected_delta
          and st['ledgers'][uid]['uses'] == 1)
    after_g = pickle.dumps((st['ledgers'], st['global_E']), protocol=5)
    _record_evidence(st, _expert_read(st, c0), ord('z'))
    check('exact_anchor_excluded_from_evidence', pickle.dumps((st['ledgers'], st['global_E']), protocol=5) == after_g)
    st['ledgers'][uid]['E2'] = 5.; st['ledgers'][uid]['E0'] = 6.
    p0 = m.lookup(c0, badbase); pretop = p0['candidate_top']; old = set(m.slots)
    ev = m.observe(p0, ord('z'))
    _lifecycle(st, old, uid, pretop, None, p0, ord('z'), ev)
    check('exact_revision_reset', st['ledgers'][uid]['E2'] == 0. and st['ledgers'][uid]['E0'] == 0.)
    # Explicit eviction deletes generation evidence; native ids monotonically
    # advance even when capacity is reused. No tombstone is kept by arbitration.
    old = set(m.slots); ev = []; m._drop_slot(uid, ev)
    _lifecycle(st, old, None, None, None, {'exact': False}, y, ev)
    check('eviction_removes_ledger', uid not in st['ledgers'])
    for c in (c0, c1):
        packet = m.lookup(c, badbase)
        candidate = _distribution(m.traces[packet['id']]['value']) if packet['type'] == 'trace' else None
        delta2 = (candidate[1][y]+math.log(256))/LN2 if candidate else None
        old = set(m.slots); ev = m.observe(packet, y)
        _lifecycle(st, old, None, packet['candidate_top'], delta2, packet, y, ev)
    newuid = next(iter(m.slots))
    check('reconsolidation_fresh_monotone_generation', newuid > uid and st['ledgers'][newuid]['uses'] == 0)
    return {'passed': True, 'checks': checks, 'n': len(checks), 'offline': O.self_test()}


def engineering(seed, load, max_records=None, checkpoint_load=None,
                checkpoint_save=None, should_stop=None):
    """Independent uninstrumented native replay, excluding FULL-HYBRID branch."""
    cfg = {'load': int(load), 'keys_per_wave': int(load)//4, 'repeats': 8, 'trace_cap': 512, 'slot_cap': 384}
    saved = checkpoint_load() if checkpoint_load else None
    if saved:
        observed = saved['observed']; reference = saved['reference']
        reference_counts = saved['reference_counts']; packet_checks = saved['packet_checks']
        elapsed = saved['elapsed_seconds']
        _restore_write_buffers(observed); _new_hash_cache(observed); _verify_full_hash(observed, count=False)
    else:
        observed = init_state(seed, cfg); reference = init_state(seed, cfg)
        reference_counts = collections.Counter(); packet_checks = 0; elapsed = 0.
    _restore_write_buffers(reference)
    for array in (reference['base'].weights, reference['base'].inverse,
                  reference['base'].writes, reference['live'].output):
        if array is not None: array.setflags(write=True)
    rows = observed['stream']['rows']; limit = min(len(rows), max_records or len(rows))
    begin = time.monotonic()
    def save():
        if checkpoint_save:
            checkpoint_save({'observed': observed, 'reference': reference,
                'reference_counts': reference_counts, 'packet_checks': packet_checks,
                'elapsed_seconds': elapsed+time.monotonic()-begin})
    # Reference uses the native implementation with no V43 ledger/read policy.
    # Its empty virtual ledger is irrelevant: the reference reader below does not
    # call any V43 policy or ledger routine.
    for record in rows[observed['cursor']:limit]:
        for y in record['raw']:
            z, b = reference['encoder'].encode(bytes(reference['history']))
            bs = reference['base'].scores(z, b)
            packet = reference['memory'].lookup(reference['history'], bs)
            reference['base'].observe(z, b, y)
            reference['live'].observe(reference['history'], y)
            for ev in reference['memory'].observe(packet, y):
                label = ev['event']
                if label in ('trace_removal', 'slot_eviction'): label += ':'+ev.get('reason', 'capacity')
                reference_counts[label] += 1
            reference['history'].append(y)
            if len(reference['history']) > C.CONTEXT_BYTES: del reference['history'][0]
            packet_checks += 1
        step(observed, cfg, max_units=1)
        if physical_components(reference) != physical_components(observed):
            raise AssertionError('V41 physical reproduction failed at record '+str(observed['cursor']))
        if observed['cursor'] % 8 == 0:
            save()
            if should_stop and should_stop(): raise InterruptedError('Engineering checkpoint saved')
    save()
    if dict(reference_counts) != dict(observed['physical_events']):
        raise AssertionError('Physical event counts differ')
    terminal_checks = 0
    for ident in range(len(observed['stream']['keys'])):
        for alias in (0, 1, 2):
            c = M.context(observed['stream']['keys'][ident], alias)
            z, b = reference['encoder'].encode(c); bs = reference['base'].scores(z, b)
            packet = reference['memory'].lookup(c, bs)
            read = _expert_read(observed, c)
            for key in ('type', 'id', 'exact', 'sim', 'support', 'candidate_top', 'base_top', 'authority'):
                if packet[key] != read['packet'][key]: raise AssertionError('Terminal packet mismatch '+key)
            ix, v = reference['live'].encode(c)
            fast = _distribution(v @ reference['live'].output[ix])
            if not np.array_equal(fast[0], read['fast'][0]): raise AssertionError('Terminal fast mismatch')
            terminal_checks += 1
    return {'passed': True, 'seed': int(seed), 'load': int(load), 'records': limit,
        'full_length': limit == len(rows), 'byte_checks': packet_checks,
        'terminal_probe_checks': terminal_checks, 'components': physical_components(observed),
        'physical_events': dict(reference_counts), 'firewall_checks': observed['audits']['firewall_checks'],
        'probe_checks': observed['audits']['probe_checks'],
        'full_hash_recomputations': observed['audits']['full_hash_recomputations'],
        'elapsed_seconds': elapsed+time.monotonic()-begin}
