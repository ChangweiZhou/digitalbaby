"""Gate weights and gate values by update: saved weights only (plus a forward pass for gate values). No training."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import load_weights, comp_panels, forward, P, BOS
SEEDS = (73129, 93683, 116089); U = 256
FILES = {'birth S': 'e26/e26_dev_S_{s}', 'birth B': 'e26/e26_dev_B_{s}', 'life SS': 'e26/e26_life_S_{s}', 'life BS': 'e26x/e26x_life_BS_{s}',
         'life BB': 'e26/e26_life_B_{s}', 'life SB': 'e26x/e26x_life_SB_{s}'}
rows = {}
for tag, stem in FILES.items():
    for s in SEEDS:
        W, _ = load_weights(P + stem.format(s=s) + '.pt')
        hh, ih = W['rnn.weight_hh_l0'], W['rnn.weight_ih_l0']; bz = (W['rnn.bias_ih_l0'] + W['rnn.bias_hh_l0'])[U:2*U]
        blk = lambda M, k: np.abs(M[k*U:(k+1)*U]).mean()
        comp, _ = comp_panels(s); items = comp['+-'] + comp['-+']
        x = np.array([[BOS] + p + t[:-1] for p, t in items], dtype=np.int64)
        Z = forward(W, x, want_logits=False)[0][:, 1:, :]                      # update gate at every position after BOS
        mid = float(((Z > 0.1) & (Z < 0.9)).mean()); sat_hi = float((Z >= 0.99).mean()); sat_lo = float((Z <= 0.01).mean())
        rows[(tag, s)] = dict(hh=(blk(hh, 0), blk(hh, 1), blk(hh, 2), np.abs(hh).mean()), ih=(blk(ih, 0), blk(ih, 1), blk(ih, 2), np.abs(ih).mean()),
                              bz=(bz.mean(), np.abs(bz).mean()), mid=mid, hi=sat_hi, lo=sat_lo)
f3 = lambda v: ' / '.join(f'{x:.3f}' for x in v)
print('mean |weight| of the update-gate rows relative to the whole stacked matrix (1.0 = same scale as the sign path assumes); seeds 73129 / 93683 / 116089')
for tag in FILES:
    print(f"  {tag:8s} recurrent: z rows {f3([rows[(tag,s)]['hh'][1]/rows[(tag,s)]['hh'][3] for s in SEEDS])} | r rows {f3([rows[(tag,s)]['hh'][0]/rows[(tag,s)]['hh'][3] for s in SEEDS])} | n rows {f3([rows[(tag,s)]['hh'][2]/rows[(tag,s)]['hh'][3] for s in SEEDS])} | stacked mean {f3([rows[(tag,s)]['hh'][3] for s in SEEDS])}")
print()
for tag in FILES:
    print(f"  {tag:8s} input:     z rows {f3([rows[(tag,s)]['ih'][1]/rows[(tag,s)]['ih'][3] for s in SEEDS])} | r rows {f3([rows[(tag,s)]['ih'][0]/rows[(tag,s)]['ih'][3] for s in SEEDS])} | n rows {f3([rows[(tag,s)]['ih'][2]/rows[(tag,s)]['ih'][3] for s in SEEDS])} | stacked mean {f3([rows[(tag,s)]['ih'][3] for s in SEEDS])}")
print('\nupdate-gate bias: mean, and mean absolute value')
for tag in FILES:
    print(f"  {tag:8s} {f3([rows[(tag,s)]['bz'][0] for s in SEEDS])}  |  {f3([rows[(tag,s)]['bz'][1] for s in SEEDS])}")
print('\nupdate-gate values on untaught composition items: share between 0.1 and 0.9 | share at 0.99 or more | share at 0.01 or less')
for tag in FILES:
    print(f"  {tag:8s} {f3([rows[(tag,s)]['mid'] for s in SEEDS])}  |  {f3([rows[(tag,s)]['hi'] for s in SEEDS])}  |  {f3([rows[(tag,s)]['lo'] for s in SEEDS])}")
