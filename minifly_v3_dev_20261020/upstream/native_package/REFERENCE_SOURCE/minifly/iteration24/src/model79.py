"""V79 native raw-event instrumentation and bounded slow-vector matching.

These are evaluator interventions, not deployable learning rules. Raw updates
are recorded BEFORE the inherited event decay. No synthetic update is amplified.
"""
from common79 import *

REPLAY_ARMS = ('D_RAW', 'A_RAW', 'D_GLOBAL_LOW', 'A_GLOBAL_LOW',
               'D_ROUTE_SHARED', 'A_ROUTE_SHARED', 'NO_SLOW')


def raw_event(m, dt, x=None, punishment=0., plastic=True):
    """Advance the unchanged EVENT kernel and expose its pre-decay increments."""
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('Invalid duration')
    if x is None or not np.any(x):
        step_encoded(m, dt)
        return dict(dt=dt, active=False)
    x = np.asarray(x)
    active = x > 0
    a = m.adapt[active]
    summary = dict(adaptation_mean=float(a.mean()), adaptation_min=float(a.min()),
                   adaptation_max=float(a.max()))
    k = m.kernel
    r = advance73(x, dt, float(punishment), m.Q, m.T, m.F, m.MM, m.k0,
                  m.fw0, m.fwd, m.ta, m.tg, k['fast_tau'], k['slow_tau'],
                  k['fraction'], m.fast, m.slow, m.adapt, 0, 0, m.strength,
                  m.Q4, m.weights, m.side, m.ei, m.ej, m.eq, m.share,
                  m.ET, m.R, m.R2, bool(plastic), 1.)
    m.elapsed = np.float64(m.elapsed + dt)
    m.event_count += np.uint64(1)
    m.presentation_count += np.uint64(1)
    fast = r[3][:, 1].copy()
    slow = r[4].copy()
    return dict(dt=dt, active=True, rawslow=slow, rawfast=fast,
                rawalpha=fast + slow, gamma_L1=float(abs(r[3][:, 0]).sum()),
                **summary)


def native_pair_records(m, codes, label, subdivisions=1):
    """Two 30-s presentations with unchanged 135-s rests; optional active split."""
    subdivisions = int(subdivisions)
    if subdivisions < 1 or subdivisions not in (1, 3):
        raise ValueError('Only prespecified active subdivisions 1 or 3')
    records = []
    for presentation in range(2):
        for substep in range(subdivisions):
            q = raw_event(m, 30. / subdivisions, codes[presentation],
                          float(presentation == int(label)))
            q.update(presentation=presentation, substep=substep)
            records.append(q)
        q = raw_event(m, 135.)
        q.update(presentation=presentation, substep=-1)
        records.append(q)
    return records


def adaptation_shadow(m, seconds=900.):
    """Recover only adaptation, without advancing the donor clock or other state."""
    q = m.clone()
    q.adapt[:] = 1. - (1. - q.adapt) * np.exp(-float(seconds) / q.ta)
    return q


def native_adaptation_gap(m, seconds=900.):
    """V78 N001: real elapsed gap, slow decay/recovery; both fast states frozen."""
    before = m.fast.copy()
    step_encoded(m, float(seconds))
    m.fast[:] = before


def anatomical_groups(m):
    """The exact inherited hemisphere + alpha teaching row partition."""
    _, group = np.unique(np.column_stack((m.kc_side, m.T[:, 2:])),
                         axis=0, return_inverse=True)
    return group


def matched_vectors(dense, adapted, groups):
    """Global common-min dose and exact route/sign common-min dose.

    Opposite-sign or absent route buckets contribute zero to BOTH route-shared
    vectors. Their lost coverage is reported, never cross-route borrowed.
    """
    d = np.asarray(dense, dtype=float)
    a = np.asarray(adapted, dtype=float)
    if d.shape != a.shape or groups.shape != d.shape:
        raise ValueError('Mismatched vector shapes')
    if not np.isfinite(d).all() or not np.isfinite(a).all():
        raise ValueError('Nonfinite native write')
    md, ma = float(abs(d).sum()), float(abs(a).sum())
    common = min(md, ma)
    dg = d * (common / md) if md else np.zeros_like(d)
    ag = a * (common / ma) if ma else np.zeros_like(a)
    dr, ar = np.zeros_like(d), np.zeros_like(a)
    bucket_rows = []
    for group in np.unique(groups[(d != 0) | (a != 0)]):
        for sign in (-1, 1):
            di = (groups == group) & (np.sign(d) == sign)
            ai = (groups == group) & (np.sign(a) == sign)
            dm, am = float(abs(d[di]).sum()), float(abs(a[ai]).sum())
            if dm == am == 0:
                continue
            bm = min(dm, am)
            if dm:
                dr[di] = d[di] * (bm / dm)
            if am:
                ar[ai] = a[ai] * (bm / am)
            bucket_rows.append(dict(group=int(group), sign=sign,
                                    dense_mass=dm, adapted_mass=am,
                                    shared_mass=bm,
                                    unmatched_dense_mass=dm-bm,
                                    unmatched_adapted_mass=am-bm,
                                    zero_source=bool((dm == 0) != (am == 0)),
                                    error=abs(float(abs(dr[di]).sum()) -
                                              float(abs(ar[ai]).sum()))))
    v = dict(D_RAW=d.copy(), A_RAW=a.copy(), D_GLOBAL_LOW=dg,
             A_GLOBAL_LOW=ag, D_ROUTE_SHARED=dr, A_ROUTE_SHARED=ar,
             NO_SLOW=np.zeros_like(d))
    route_mass = float(abs(dr).sum())
    norm = float(np.linalg.norm(d) * np.linalg.norm(a))
    union = (d != 0) | (a != 0)
    geometry = dict(dense_raw_slow_L1=md, adapted_raw_slow_L1=ma,
                    common_global_L1=common, common_route_L1=route_mass,
                    cosine=float(d @ a / norm) if norm else np.nan,
                    support_jaccard=float(np.sum((d != 0) & (a != 0)) /
                                          np.sum(union)) if np.any(union) else 1.,
                    sign_flip_coordinates=int(np.sum(d * a < 0)),
                    sign_flip_dense_mass=float(abs(d[d*a < 0]).sum()),
                    sign_flip_adapted_mass=float(abs(a[d*a < 0]).sum()),
                    zero_mass_dense=bool(md == 0), zero_mass_adapted=bool(ma == 0),
                    route_dense_retained=route_mass / md if md else np.nan,
                    route_adapted_retained=route_mass / ma if ma else np.nan,
                    global_match_error=abs(float(abs(dg).sum()) - float(abs(ag).sum())),
                    route_match_error=abs(float(abs(dr).sum()) - float(abs(ar).sum())),
                    max_bucket_error=max([r['error'] for r in bucket_rows] or [0.]))
    tolerance = 1e-10 + 1e-12 * max(md, ma)
    for name, out in v.items():
        source = a if name.startswith('A_') else d
        if np.any(out[source == 0] != 0) or np.any(out * source < 0):
            raise AssertionError('Support/sign changed')
        if float(np.max(abs(out) - abs(source), initial=0.)) > tolerance:
            raise AssertionError('Low/shared control amplified a coordinate')
    if max(geometry['global_match_error'], geometry['route_match_error'],
           geometry['max_bucket_error']) > tolerance:
        raise AssertionError('Dose certificate failed')
    return v, geometry, bucket_rows


def accumulate_event(states, vectors, dt, slow_tau):
    """Each source vector is inserted BEFORE the event's inherited decay."""
    decay = np.exp(-float(dt) / slow_tau)
    for name in states:
        states[name][:] = (states[name] + vectors[name]) * decay


def accumulate_rest(states, dt, slow_tau):
    decay = np.exp(-float(dt) / slow_tau)
    for state in states.values():
        state *= decay


def wrap_model(m):
    """Use the unchanged inherited reader/probe implementation."""
    return ParentLearner('REF', prestate=m)
