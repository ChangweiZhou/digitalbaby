"""Batched float64 native equations, retaining signed W1-W0 write semantics.

Equivalence is checked against the pinned scalar native implementation, not
assumed from shared names. No byte parser or target decoder lives here.
"""
import math
import numpy as np
from numba import njit
import bootstrap
from readout import predict_drive


@njit(cache=True)
def event_batch(fast, slow, adapt, x, dt, coeff, plastic, Q, T, F, MM, k0,
                fw0, fwd, ta, tg, tf, ts, fraction, alpha_scale, genes, sf, ss):
    nc, n = slow.shape
    dg, df, ds, da = math.exp(-dt/tg), math.exp(-dt/tf), math.exp(-dt/ts), math.exp(-dt/ta)
    rm_all = np.zeros((nc, 4))
    applied = 0.0
    has_genes = np.any(genes[:4] != 0)
    for c in range(nc):
        dx = np.empty(n)
        end = np.empty(n)
        agg, inp = np.zeros(4), np.zeros(4)
        for i in range(n):
            end[i] = 1 - (1 - adapt[c, i]*math.exp(-.05*dt*x[i]))*da
            dx[i] = .5*(adapt[c, i]+end[i])*x[i]
            for j in range(4):
                agg[j] += Q[i, j]*dx[i]
                value = fast[c, i, 0] if j < 2 else fast[c, i, 1]+slow[c, i]
                inp[j] += Q[i, j]*dx[i]*(k0[3 if j < 2 else 5]+value)
        rm = np.empty(4)
        for j in range(2):
            rm[j] = min(max(inp[j]+35.2, 0), 71.66)-35.2
        for j in range(2):
            val = inp[j+2]
            for h in range(2):
                val += MM[h, j]*rm[h]
            rm[j+2] = min(max(val+11.2, 0), 31.16)-11.2
        rm_all[c] = rm
        p0, p1 = np.empty(4), np.empty(4)
        for j in range(4):
            d0 = k0[0 if j < 2 else 2]*agg[j]
            for h in range(4):
                d0 += F[h, j]*rm[h]
            p0[j] = fw0*d0
            p1[j] = fw0*d0+fwd*(27.85 if j < 2 else 11.38)
        for i in range(n):
            f0, f1, sl = fast[c, i, 0], fast[c, i, 1], slow[c, i]
            if plastic and coeff[c] != 0 and dx[i] != 0:
                g0 = (T[i, 0]*p0[0]+T[i, 1]*p0[1])*dx[i]*dt/90
                g1 = (T[i, 0]*p1[0]+T[i, 1]*p1[1])*dx[i]*dt/90
                a0 = (T[i, 2]*p0[2]+T[i, 3]*p0[3])*dx[i]*dt/90*alpha_scale
                a1 = (T[i, 2]*p1[2]+T[i, 3]*p1[3])*dx[i]*dt/90*alpha_scale
                q0, q1 = fraction, fraction
                if has_genes:
                    base = math.log(fraction/(1-fraction))+genes[0]+genes[1]*(2*adapt[c, i]-1)
                    influence = genes[2]*math.tanh(f1/sf)+genes[3]*math.tanh(sl/ss)
                    q0 = 1/(1+math.exp(-(base+influence*np.sign(a0))))
                    q1 = 1/(1+math.exp(-(base+influence*np.sign(a1))))
                fast[c, i, 0] = f0*dg+coeff[c]*((f0+g1)*dg-(f0+g0)*dg)
                fast[c, i, 1] = f1*df+coeff[c]*((f1+(1-q1)*a1)*df-(f1+(1-q0)*a0)*df)
                slow[c, i] = sl*ds+coeff[c]*((sl+q1*a1)*ds-(sl+q0*a0)*ds)
                applied += abs(coeff[c]*(a1-a0))
            else:
                fast[c, i, 0] = f0*dg
                fast[c, i, 1] = f1*df
                slow[c, i] = sl*ds
            adapt[c, i] = end[i]
    return rm_all, applied


@njit(cache=True)
def query_batch(fast, slow, adapt, x, dt, Q, reader_Q, MM, k0, ta, tg, tf, ts):
    nc, n = slow.shape
    dxagg, actual = np.zeros((nc, 4)), np.empty((nc, 2))
    dg, df, ds, da = math.exp(-dt/tg), math.exp(-dt/tf), math.exp(-dt/ts), math.exp(-dt/ta)
    obs = math.exp(-5/ta)
    for c in range(nc):
        inp = np.zeros(4)
        for i in range(n):
            if x[i] != 0:
                a = 1-(1-adapt[c, i])*da
                end = 1-(1-a*math.exp(-.25*x[i]))*obs
                dx = .5*(a+end)*x[i]
                for j in range(4):
                    dxagg[c, j] += dx*reader_Q[i, j]
                    value = fast[c, i, 0]*dg if j < 2 else fast[c, i, 1]*df+slow[c, i]*ds
                    inp[j] += dx*Q[i, j]*(k0[3 if j < 2 else 5]+value)
        gamma = np.empty(2)
        for j in range(2):
            gamma[j] = min(max(inp[j]+35.2, 0), 71.66)-35.2
        for j in range(2):
            val = inp[j+2]
            for h in range(2):
                val += gamma[h]*MM[h, j]
            actual[c, j] = min(max(val+11.2, 0), 31.16)-11.2
    return actual, dxagg


class NativeBank:
    def __init__(self, parent):
        flies = [c.fly for c in parent.bank]
        self.births = [c.birth for c in parent.bank]
        m = flies[0].m
        for key in ('Q', 'T', 'F', 'MM', 'k0', 'pn_type_index', 'kc_side'):
            value = np.array(getattr(m, key), copy=True)
            for f in flies[1:]:
                assert np.array_equal(value, getattr(f.m, key)), key
            value.flags.writeable = False
            setattr(self, key, value)
        self.fast = np.stack([f.m.fast.copy() for f in flies])
        self.slow = np.stack([f.m.slow.copy() for f in flies])
        self.adapt = np.stack([f.m.adapt.copy() for f in flies])
        self.genes = np.array(flies[0].genes, copy=True)
        self.B = m.B
        self.active_fraction = float(m.active_fraction)
        self.side_n = dict(parent.side_n)
        self.reader_head = flies[0].reader.head
        self.reader_Q = np.array(flies[0].reader.Q, copy=True)
        self.reader_Q.flags.writeable = False
        self.fw0, self.fwd, self.ta, self.tg = map(float, (m.fw0, m.fwd, m.ta, m.tg))
        self.tf, self.ts, self.fraction = (float(m.kernel[k]) for k in ('fast_tau', 'slow_tau', 'fraction'))
        self.alpha_scale = float(flies[0].alpha_scale)
        cfg = bootstrap.stores.bb.bc.native().model.CFG
        self.sf, self.ss = (float(cfg['allocation_scales'][k]) for k in ('fast', 'slow'))
        self.events, self.elapsed = 0, 0.

    def clone(self):
        import copy
        out = copy.copy(self)
        out.fast, out.slow, out.adapt = self.fast.copy(), self.slow.copy(), self.adapt.copy()
        return out

    def rest(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('invalid native time')
        self.fast[..., 0] *= math.exp(-dt/self.tg)
        self.fast[..., 1] *= math.exp(-dt/self.tf)
        self.slow *= math.exp(-dt/self.ts)
        self.adapt[:] = 1-(1-self.adapt)*math.exp(-dt/self.ta)
        self.elapsed += dt

    def event(self, x, dt, coefficients=None):
        allowed = coefficients is not None
        c = np.zeros(16) if coefficients is None else np.asarray(coefficients, dtype=float)
        if c.shape != (16,) or np.any(~np.isfinite(c)) or np.any(abs(c) > 1+1e-12):
            raise ValueError('invalid signed coefficients')
        _, applied = event_batch(self.fast, self.slow, self.adapt, x, dt, c, allowed,
                                self.Q, self.T, self.F, self.MM, self.k0, self.fw0, self.fwd,
                                self.ta, self.tg, self.tf, self.ts, self.fraction,
                                self.alpha_scale, self.genes, self.sf, self.ss)
        self.elapsed += dt
        self.events += 1
        return float(applied)

    def values(self, x, dt):
        actual, drive = query_batch(self.fast, self.slow, self.adapt, x, dt,
                                   self.Q, self.reader_Q, self.MM, self.k0, self.ta, self.tg, self.tf, self.ts)
        return (actual-predict_drive(drive, self.reader_head)).mean(1)

    def encode(self, p, theta):
        raw = np.asarray(self.B.T @ p[self.pn_type_index]).ravel()-theta
        x = np.zeros(len(raw))
        for side in (0, 1):
            ix = np.flatnonzero((self.kc_side == side) & (raw > 0))
            order = np.lexsort((ix, -raw[ix]))
            x[ix[order[:min(len(ix), self.side_n[side])]]] = 1.
        return x, raw
