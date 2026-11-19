# pb_check.py: pbstand/<label>.json against the reviewer's perbyte/<name>_linear.json (perbyte3.py), all groups, lags, readers and figures.
import sys, json
a = json.load(open(sys.argv[1]))['perbyte']; r = json.load(open(sys.argv[2]))['groups']
dmax = 0.0; n = 0
for g in r:
    for lag in r[g]['lags']:
        for rd in ('linear', 'centroid'):
            for k, v in r[g]['lags'][lag][rd].items():
                x = a[g]['lags'][lag][rd][k]
                if v != v and x != x: continue
                dmax = max(dmax, abs(x - v)); n += 1
    assert a[g]['cells'] == r[g]['cells'], (g, a[g]['cells'], r[g]['cells'])
print(sys.argv[1].split('/')[-1], 'figures compared', n, 'largest difference', dmax)
