"""V82E phenotype: local allocation + total alpha gain + declared support expansion.

Genes 0-3 are the inherited V79E coordinate-local allocation constants.
Gene 4 (g_alpha_gain) scales the TOTAL alpha write, deliberately breaking the
alpha conservation that made V79E a pure redistribution family -- the family
V68's LP oracle already searched and found empty.
Gene 5 (g_support) grants alpha teaching to KCs that have alpha readout but no
direct DAN route (544 of them), modelling volume transmission.  This is a
DECLARED MODELLING CHANGE, not an anatomical finding.

Original V79E docstring follows.
Autonomous four-constant local allocation phenotype for bounded EvoFly.

Only the fast/slow split of the current native alpha write changes.  Every
coordinate uses its own pre-event adaptation, alpha-fast state, slow state and
the sign of its current native alpha update.  No task label, replay, evaluator
budget, cue identity or population statistics enter this rule.
"""
from common_evo import *
from model73 import advance73


def raw_event_scaled(m, dt, x=None, punishment=0., plastic=True, scale=1.):
    """model79.raw_event with an explicit total-alpha scale (gene g_alpha_gain)."""
    dt = float(dt); scale = float(scale)
    if not np.isfinite(dt) or dt < 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid duration or alpha scale')
    if x is None or not np.any(x):
        step_encoded(m, dt)
        return dict(dt=dt, active=False)
    x = np.asarray(x)
    a = m.adapt[x > 0]
    summary = dict(adaptation_mean=float(a.mean()), adaptation_min=float(a.min()),
                   adaptation_max=float(a.max()))
    k = m.kernel
    r = advance73(x, dt, float(punishment), m.Q, m.T, m.F, m.MM, m.k0,
                  m.fw0, m.fwd, m.ta, m.tg, k['fast_tau'], k['slow_tau'],
                  k['fraction'], m.fast, m.slow, m.adapt, 0, 0, m.strength,
                  m.Q4, m.weights, m.side, m.ei, m.ej, m.eq, m.share,
                  m.ET, m.R, m.R2, bool(plastic), scale)
    m.elapsed = np.float64(m.elapsed + dt)
    m.event_count += np.uint64(1)
    m.presentation_count += np.uint64(1)
    fast = r[3][:, 1].copy(); slow = r[4].copy()
    return dict(dt=dt, active=True, rawslow=slow, rawfast=fast,
                rawalpha=fast + slow, gamma_L1=float(abs(r[3][:, 0]).sum()),
                **summary)

GENE_NAMES = tuple(CFG['gene_names'])
IDX = {n: i for i, n in enumerate(GENE_NAMES)}


def gene(genes, name):
    return float(genome_array(genes)[IDX[name]])


def apply_phenotype(m, genes):
    """Apply every relaxed-domain gene to a fresh model instance.

    Each hook replaces a frozen array with a modified copy; none mutates shared
    state in place.  Returns an audit dict recorded alongside every candidate.
    Neutral genes reproduce the inherited model bit-for-bit.
    """
    g = genome_array(genes)
    audit = {}
    # --- connectivity magnitudes -------------------------------------------
    ts = 2. ** gene(g, 'g_teach_scale')
    if ts != 1.: m.T = np.asarray(m.T, float) * ts
    audit['teach_scale'] = ts
    rs = 2. ** gene(g, 'g_readout')
    if rs != 1.: m.Q = np.asarray(m.Q, float) * rs
    audit['readout_scale'] = rs
    # --- recurrent gains ----------------------------------------------------
    fs = 2. ** gene(g, 'g_feedback')
    if fs != 1.: m.F = np.asarray(m.F, float) * fs
    audit['feedback_scale'] = fs
    ms = 2. ** gene(g, 'g_mbon')
    if ms != 1.: m.MM = np.asarray(m.MM, float) * ms
    audit['mbon_scale'] = ms
    # --- baseline potentials ------------------------------------------------
    ks = 2. ** gene(g, 'g_k0')
    if ks != 1.: m.k0 = np.asarray(m.k0, float) * ks
    audit['k0_scale'] = ks
    # --- kernel timescales and native split ---------------------------------
    k = dict(m.kernel)
    k['fast_tau'] *= 2. ** gene(g, 'g_fast_tau')
    k['slow_tau'] *= 2. ** gene(g, 'g_slow_tau')
    f0 = k['fraction']
    z = np.log(f0 / (1. - f0)) + gene(g, 'g_fraction')
    k['fraction'] = float(1. / (1. + np.exp(-z)))
    if not (0. < k['fraction'] < 1.) or not np.isfinite([k['fast_tau'], k['slow_tau']]).all():
        raise ValueError('Kernel gene produced an invalid kernel')
    m.kernel = k
    audit.update(fast_tau=k['fast_tau'], slow_tau=k['slow_tau'], fraction=k['fraction'])
    # --- PN->KC weights and topology ----------------------------------------
    jitter = gene(g, 'g_pnkc_weight')
    rewire = gene(g, 'g_pnkc_rewire')
    if jitter > 0. or rewire > 0.:
        B = m.B.copy().tocsr()
        rng = np.random.default_rng(abs(hash(tuple(np.round(g, 6)))) % (2 ** 32))
        if jitter > 0.:
            B.data = B.data * np.exp(rng.normal(0., jitter, B.data.shape))
        if rewire > 0.:
            # Degree-preserving: permute each PN row's column indices among the
            # columns that row already uses elsewhere in the matrix is not well
            # defined for CSR, so permute the KC targets of a random subset of
            # rows.  Row and column counts of nonzeros per PN are preserved.
            for r in range(B.shape[0]):
                lo, hi = B.indptr[r], B.indptr[r + 1]
                if hi - lo < 2 or rng.random() >= rewire: continue
                perm = rng.permutation(hi - lo)
                B.data[lo:hi] = B.data[lo:hi][perm]
        m.B = B
    audit.update(pnkc_jitter=jitter, pnkc_rewire=rewire)
    # --- KC coding sparsity --------------------------------------------------
    audit['active_fraction'] = 0.05 * (2. ** gene(g, 'g_sparsity'))
    m.active_fraction = audit['active_fraction']
    # --- teaching support expansion -----------------------------------------
    audit['expanded_cells'] = expand_support(m, gene(g, 'g_support'))
    audit['alpha_scale'] = alpha_scale(g)
    return audit


def encode_sparse(m, p):
    """SmallFly.encode with a gene-controlled active fraction."""
    frac = float(getattr(m, 'active_fraction', 0.05))
    if not 0. < frac < 1.: raise ValueError('Active fraction outside (0,1)')
    p = np.asarray(p, float)
    raw = np.asarray(m.B.T @ (p[m.pn_type_index])).ravel()
    x = np.zeros(len(raw))
    for s in (0, 1):
        ix = np.flatnonzero((m.kc_side == s) & (raw > 0))
        n = min(len(ix), int(np.ceil(frac * np.sum(m.kc_side == s))))
        order = np.lexsort((ix, -raw[ix]))
        x[ix[order[:n]]] = 1.
    return x
BOX = CFG['gene_box']


def alpha_scale(genes):
    """Total alpha write multiplier; 1.0 at the reference gene value."""
    return float(2. ** float(genome_array(genes)[IDX['g_alpha_gain']]))


def support_mask(m):
    """KCs with alpha readout but no direct alpha teaching route."""
    aT = np.abs(np.asarray(m.T)[:, 2:]).sum(1)
    aQ = np.abs(np.asarray(m.Q)[:, 2:]).sum(1)
    return (aQ > 0) & (aT <= 0)


def expand_support(m, g_support):
    """Grant scaled alpha teaching rows to readout-only KCs. Returns n added."""
    g = float(g_support)
    if not (0. <= g <= 1.):
        raise ValueError('g_support outside [0,1]')
    if g == 0.:
        return 0
    cand = support_mask(m)
    if not cand.any():
        return 0
    T = np.array(m.T, dtype=float, copy=True)
    aT = np.abs(T[:, 2:]).sum(1)
    donor = aT > 0
    if not donor.any():
        raise ValueError('No reference alpha teaching rows to calibrate against')
    ref_mag = float(aT[donor].mean())
    aQ = np.abs(np.asarray(m.Q)[:, 2:])
    qsum = aQ[cand].sum(1, keepdims=True)
    share = np.divide(aQ[cand], qsum, out=np.zeros_like(aQ[cand]), where=qsum > 0)
    T[np.flatnonzero(cand)[:, None], [2, 3]] = g * ref_mag * share
    if not np.isfinite(T).all():
        raise FloatingPointError('Nonfinite expanded teaching matrix')
    m.T = T
    return int(cand.sum())


def genome_array(genome):
    if isinstance(genome, dict):
        values = genome['genes'] if 'genes' in genome else [genome[k] for k in GENE_NAMES]
    else:
        values = genome
    out = np.asarray(values, dtype=float)
    if out.shape != (len(GENE_NAMES),) or not np.isfinite(out).all():
        raise ValueError(f'{len(GENE_NAMES)} finite genes are required')
    for i, name in enumerate(GENE_NAMES):
        lo, hi = BOX[name]
        if out[i] < lo - 1e-12 or out[i] > hi + 1e-12:
            raise ValueError(f'Gene {name} outside its frozen box')
    return out.copy()


def allocation_fraction(genome, native_fraction, adaptation, fast, slow, rawalpha):
    """Coordinate-local bounded fraction, using only pre-event state."""
    genes = genome_array(genome)
    f0 = float(native_fraction)
    if not 0. < f0 < 1.:
        raise ValueError('Native allocation must be strictly between zero and one')
    scales = CFG.get('allocation_scales', {'fast': 1., 'slow': 1.})
    sf, ss = float(scales['fast']), float(scales['slow'])
    if not np.isfinite(sf + ss) or min(sf, ss) <= 0:
        raise ValueError('Frozen local state scales must be finite and positive')
    a, f, s, u = [np.asarray(v, dtype=float) for v in (adaptation, fast, slow, rawalpha)]
    if not (a.shape == f.shape == s.shape == u.shape) or not all(np.isfinite(v).all() for v in (a, f, s, u)):
        raise ValueError('Local state and update arrays must match and be finite')
    if np.any(a < -1e-12) or np.any(a > 1. + 1e-12):
        raise ValueError('Adaptation outside its inherited range')
    if not np.any(genes[:4]):
        return np.full_like(u, f0)
    sign = np.sign(u)
    z = (np.log(f0 / (1. - f0)) + genes[0] + genes[1] * (2. * a - 1.)
         + genes[2] * np.tanh(f / sf) * sign
         + genes[3] * np.tanh(s / ss) * sign)
    return 1. / (1. + np.exp(-z))


def empty_stats():
    return dict(active_events=0, pair_presentations=0, raw_alpha_L1=0.,
                raw_slow_L1=0., raw_alpha_fast_L1=0., raw_gamma_L1=0.,
                allocation_min=1., allocation_max=0.,
                max_split_error=0., max_state_patch_error=0., no_write_events=0)


class EvoLearner:
    """Inherited learned state plus four immutable genotype constants.

    The statistics below are evaluator instrumentation and are never read by
    the update rule. No additional persistent learned array is introduced.
    """
    def __init__(self, genome, prestate=None):
        self.genes = genome_array(genome)
        self.m = Fly() if prestate is None else prestate.clone()
        self.phenotype = apply_phenotype(self.m, self.genes)
        self.alpha_scale = self.phenotype['alpha_scale']
        self.expanded_cells = self.phenotype['expanded_cells']
        self.reader = FrozenReader(self.m.Q, READER)
        self.stats = empty_stats()

    def clone(self):
        out = copy.copy(self)
        out.genes = self.genes.copy()
        out.m = self.m.clone()
        for attr in ('T','Q','F','MM','k0','B'):
            setattr(out.m,attr,getattr(self.m,attr))
        out.m.kernel = dict(self.m.kernel)
        out.m.active_fraction = getattr(self.m,'active_fraction',0.05)
        out.phenotype = dict(self.phenotype)
        out.alpha_scale = self.alpha_scale
        out.expanded_cells = self.expanded_cells
        out.stats = empty_stats()
        return out

    def reset_stats(self):
        self.stats = empty_stats()

    def event(self, dt, x=None, punishment=0., plastic=True):
        """One inherited EVENT update, then substitute its local alpha split."""
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0 or not np.isfinite(punishment):
            raise ValueError('Invalid event duration or punishment')
        if x is None or not np.any(x):
            return raw_event_scaled(self.m, dt, scale=self.alpha_scale)
        x = np.asarray(x, dtype=float)
        if x.shape != self.m.adapt.shape or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError('Invalid encoded activity')
        # Temporary event copies, not extra learned memory or a trace bank.
        pre_a = self.m.adapt.copy()
        pre_fast = self.m.fast[:, 1].copy()
        pre_slow = self.m.slow.copy()
        record = raw_event_scaled(self.m, dt, x, punishment, bool(plastic), self.alpha_scale)
        u = record['rawalpha']
        f0 = self.m.kernel['fraction']
        fraction = allocation_fraction(self.genes, f0, pre_a, pre_fast, pre_slow, u)
        if not np.any(self.genes[:4]) or not plastic:
            # REF is bit-for-bit the inherited raw_event, including its split.
            actual_fast = record['rawfast']
            actual_slow = record['rawslow']
            patch_error = 0.
        else:
            actual_slow = fraction * u
            actual_fast = (1. - fraction) * u
            df = np.exp(-dt / self.m.kernel['fast_tau'])
            ds = np.exp(-dt / self.m.kernel['slow_tau'])
            self.m.fast[:, 1] += (actual_fast - record['rawfast']) * df
            self.m.slow += (actual_slow - record['rawslow']) * ds
            patch_error = max(float(abs(self.m.fast[:, 1] - (pre_fast + actual_fast) * df).max()),
                              float(abs(self.m.slow - (pre_slow + actual_slow) * ds).max()))
        split_error = float(abs(actual_fast + actual_slow - u).max())
        if split_error > 1e-10 or patch_error > 1e-8:
            raise AssertionError('Native alpha preservation certificate failed')
        if not (np.isfinite(self.m.fast).all() and np.isfinite(self.m.slow).all() and np.isfinite(self.m.adapt).all()):
            raise FloatingPointError('Nonfinite autonomous state')
        if np.any(actual_slow * u < 0) or np.any(actual_fast * u < 0):
            raise AssertionError('The allocation rule changed an alpha update sign')
        if np.any(actual_slow[u == 0] != 0) or np.any(actual_fast[u == 0] != 0):
            raise AssertionError('The allocation rule created unsupported plasticity')
        s = self.stats
        s['active_events'] += 1
        s['no_write_events'] += int(not plastic)
        s['raw_alpha_L1'] += float(abs(u).sum())
        s['raw_slow_L1'] += float(abs(actual_slow).sum())
        s['raw_alpha_fast_L1'] += float(abs(actual_fast).sum())
        s['raw_gamma_L1'] += record['gamma_L1']
        s['max_split_error'] = max(s['max_split_error'], split_error)
        s['max_state_patch_error'] = max(s['max_state_patch_error'], patch_error)
        active = u != 0
        if np.any(active):
            s['allocation_min'] = min(s['allocation_min'], float(fraction[active].min()))
            s['allocation_max'] = max(s['allocation_max'], float(fraction[active].max()))
        record.update(rawfast=actual_fast, rawslow=actual_slow,
                      allocation_min=float(fraction.min()), allocation_max=float(fraction.max()),
                      split_error=split_error, state_patch_error=patch_error)
        return record

    def learn_pair(self, codes, label, write=True):
        if int(label) != label or int(label) not in (0, 1):
            raise ValueError('Observed punishment must identify one of two presentations')
        if np.asarray(codes).shape != (2, len(self.m.adapt)):
            raise ValueError('Two encoded presentations are required')
        for i in range(2):
            self.event(30., codes[i], float(i == int(label)), bool(write))
            self.event(135.)
        self.stats['pair_presentations'] += 1

    def rest(self, seconds):
        self.event(seconds)

    def read(self, codes, roles, horizon=0., erase_alpha_fast=False):
        """Frozen read-only 5-second observer; evaluation labels stay here."""
        n = self.m.clone()
        if erase_alpha_fast:
            n.fast[:, 1] = 0.
        if horizon:
            n.mode = 'no_learning'
            n.step(float(horizon))
        dx = observed_activity(n, codes)
        actual = n.expression(dx)
        baseline = n.expression(dx, True)
        predicted = self.reader.predict(dx)
        direction = 2 * np.asarray(roles) - 1
        ca = (actual - baseline).mean(1)
        cr = (actual - predicted).mean(1)
        a = direction * (ca[::2] - ca[1::2])
        r = direction * (cr[::2] - cr[1::2])
        b = (a >= 1.) & (r >= self.reader.pair_threshold)
        return a, r, b

    def diagnostic(self):
        out = dict(self.stats)
        mass = out['raw_alpha_L1']
        out['mass_weighted_slow_fraction'] = out['raw_slow_L1'] / mass if mass else None
        if out['allocation_max'] == 0.:
            out['allocation_min'] = out['allocation_max'] = None
        out.update({('pheno_'+k):v for k,v in self.phenotype.items()})
        out['alpha_scale'] = float(self.alpha_scale)
        out['expanded_cells'] = int(self.expanded_cells)
        out['mutable_array_bytes'] = int(self.m.fast.nbytes + self.m.slow.nbytes + self.m.adapt.nbytes)
        out['mutable_state_bytes'] = out['mutable_array_bytes'] + 24  # elapsed and two counters
        return out
