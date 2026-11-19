"""Point 4 of the task: which cells hold many connections, from drive6.py's per-cell data (out/<label>_cells.npz). Standard numpy.

    python3 -I big_cells.py <label> [...]     # prints, and writes big/<label>.json

Non-feedback cells reached from the input lines in the last window are grouped by their number of connections (at most 8, 9 to 15,
16 or more). For each group: cells; distance from the input lines (median, and the share at d <= 2); the long mean of output M
(median); connections holding more than 1/128 of the cell's excitatory or inhibitory total (median, and the share of the cell's
connections they are); largest raw output in the window (median); the share of the output variance explained by the present byte
(median over cells with variance). The ten cells with the most connections are listed one by one.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def one(label):
    c = np.load(os.path.join(HERE, 'out', label + '_cells.npz'))
    deg, dist, isfb, M, strong, mx, eta = c['deg'], c['dist'], c['isfb'], c['M'], c['strong'], c['maxout'], c['eta_present']
    ok = (~isfb) & (dist > 0)
    groups = {'<=8': ok & (deg <= 8), '9-15': ok & (deg >= 9) & (deg <= 15), '>=16': ok & (deg >= 16)}
    out = dict(label=label, reached=int(ok.sum()), unreached=int(((~isfb) & (dist == 0)).sum()), groups={})
    for g, s in groups.items():
        if not s.any():
            out['groups'][g] = dict(cells=0); continue
        out['groups'][g] = dict(cells=int(s.sum()), dist_median=med(dist[s]), share_d_le_2=float((dist[s] <= 2).mean()), M_median=med(M[s]),
                                strong_median=med(strong[s]), strong_share_median=med(strong[s] / np.maximum(deg[s], 1)),
                                maxout_median=med(mx[s]), present_explained_median=med(eta[s]))
    top = np.argsort(-np.where(ok, deg, -1))[:10]
    out['top10'] = [dict(cell=int(i), deg=int(deg[i]), dist=int(dist[i]), strong=int(strong[i]), M=float(M[i]), maxout=float(mx[i]),
                         present_explained=(None if not np.isfinite(eta[i]) else float(eta[i]))) for i in top]
    os.makedirs(os.path.join(HERE, 'big'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'big', label + '.json'), 'w'), indent=1)
    print(f"== {label}: reached {out['reached']}, unreached {out['unreached']}")
    for g, r in out['groups'].items():
        print(f"  conns {g:5s} " + " ".join(f"{k} {('%.4g' % v) if isinstance(v, float) else v}" for k, v in r.items()))
    for r in out['top10']:
        print("   ", r)


if __name__ == '__main__':
    for lab in sys.argv[1:]:
        one(lab)
