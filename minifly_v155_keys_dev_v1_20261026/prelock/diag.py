import sys, time, numpy as np, torch
sys.path.insert(0, '/home/user/digitalbaby/minifly_v155_keys_dev_v1_20261026')
import refs, codes
from scipy import sparse
exec(open('teacher_ste.py').read().split("mode = sys.argv[1]")[0].split("import refs, codes", 1)[1].split("if 'ste' in sys.argv[1]:")[0])
S = '/tmp/claude-0/-home-user-digitalbaby/029f5b11-1f0a-5a9b-9105-3bb53d7ced9b/scratchpad/v155pre/'
birth = codes.load_birth(S + 'birth.npz'); pn, side, k = birth['pn'], birth['side'], birth['k']
B0 = birth['B'].tocoo()
print('birth weights: min', B0.data.min(), 'median', np.median(B0.data), 'max', B0.data.max())
rng = np.random.default_rng(5)
w = rng.permutation(B0.data); Bt = sparse.csr_matrix((w, (B0.row, B0.col)), shape=B0.shape)
def traces(n):
    P = rng.random((n, 88)) * (rng.random((n, 88)) < 0.25)
    return P / np.maximum(P.max(1, keepdims=True), 1e-12)
V = rng.normal(size=(B0.shape[1], 20))
def task(n):
    P = traces(n); X = codes.codes(P, Bt, pn, side, k); return P, 40 + np.asarray(X @ V).argmax(1)
Ptr, ytr = task(8192); Pca, yca = task(1024)
cls = refs.R.Classes(ytr)
Q = {'train': torch.from_numpy(Ptr[:, pn].astype(np.float32)), 'calib': torch.from_numpy(Pca[:, pn].astype(np.float32))}
c = {'train': torch.from_numpy(cls.encode(ytr)), 'calib': torch.from_numpy(cls.encode(yca))}
for fwd in ('ste', 'relu'):
  refs.Keys.forward = ste_forward if fwd == 'ste' else refs.Keys.__dict__.get('_orig_forward', refs.Keys.forward)
  for init, klr in (('teacher', 0.0), ('birth', 0.0), ('birth', 1e-2), ('birth', 3e-2)):
    if fwd == 'relu' and init == 'teacher': continue
    t0 = time.process_time()
    b = {**birth, 'B': Bt if init == 'teacher' else birth['B']}
    model = refs.Keys(b, False, cls.K, k)
    opt = torch.optim.Adam([{'params': [model.W], 'lr': klr}, {'params': model.out.parameters(), 'lr': 1e-3}])
    g = torch.Generator().manual_seed(0); curve = []
    for ep in range(20):
        model.train(); perm = torch.randperm(len(c['train']), generator=g)
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = torch.nn.functional.cross_entropy(model(Q['train'][idx]), c['train'][idx])
            opt.zero_grad(); loss.backward(); opt.step(); model.project()
        if ep % 4 == 3: curve.append(round(refs.class_bits(model, Q['calib'], c['calib']), 3))
    Bl = model.keys_csr()
    Xs = codes.codes(Pca, Bl, pn, side, k); Xt = codes.codes(Pca, Bt, pn, side, k)
    print(fwd, init, 'key lr', klr, curve, 'overlap', round(float(np.mean(np.asarray(Xs.multiply(Xt).sum(1)).ravel() / 184)), 3), 'cpu', round(time.process_time() - t0), flush=True)
