# C<L(L) for window A2 of where.py's tissues: what every nearer byte gives (one-hot of the bytes at lags 0 .. L-1), the readings tool's
# c_lt, for lags 1..4 and the three seeds; the stream only (no tissue). usage: python3 where_clt.py [nbytes] -> where/CLT[_<nbytes>].json
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tis
import readings_tool as RD
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
out = {}
for s in (73129, 93683, 116089):
    b, _ = tis.ts.stream(s, NB)
    lo = NB - 4096
    out[str(s)] = {str(L): v["S"] for L, v in RD.c_lt(b[lo:NB], b, lo, (1, 2, 3, 4)).items()}
json.dump(out, open(os.path.join(HERE, 'where', 'CLT.json' if NB == 16384 else f'CLT_{NB}.json'), 'w'))
print(json.dumps(out))
