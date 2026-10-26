import sys, time, numpy as np, torch
sys.path.insert(0, '/home/user/digitalbaby/minifly_v155_keys_dev_v1_20261026')
import refs, codes
from scipy import sparse
S = '/tmp/claude-0/-home-user-digitalbaby/029f5b11-1f0a-5a9b-9105-3bb53d7ced9b/scratchpad/v155pre/'
birth = codes.load_birth(S + 'birth.npz'); pn, side, k = birth['pn'], birth['side'], birth['k']
B0 = birth['B'].tocoo()
rng = np.random.default_rng(5)
def teacher(dense):
    if not dense:   # same wiring, re-drawn weights (REF-N can reach it)
        w = rng.permutation(B0.data)
        return sparse.csr_matrix((w, (B0.row, B0.col)), shape=B0.shape)
    rows = rng.integers(0, B0.shape[0], B0.nnz)   # new wiring, same number of synapses per KC
    return sparse.csr_matrix((rng.permutation(B0.data), (rows, B0.col)), shape=B0.shape)
def traces(n):
    P = rng.random((n, 88)) * (rng.random((n, 88)) < 0.25)
    return P / np.maximum(P.max(1, keepdims=True), 1e-12)
V = rng.normal(size=(B0.shape[1], 20))
def task(Bt, n):
    P = traces(n); X = codes.codes(P, Bt, pn, side, k)
    return P, 40 + np.asarray(X @ V).argmax(1)
mode = sys.argv[1]
results = {}
for dense in (False, True):
    Bt = teacher(dense)
    Ptr, ytr = task(Bt, 8192); Pca, yca = task(Bt, 1024)
    cls = refs.R.Classes(ytr)
    Q = {'train': torch.from_numpy(Ptr[:, pn].astype(np.float32)), 'calib': torch.from_numpy(Pca[:, pn].astype(np.float32))}
    c = {'train': torch.from_numpy(cls.encode(ytr)), 'calib': torch.from_numpy(cls.encode(yca))}
    for keys_learn in (False, True):
        t0 = time.process_time()
        model = refs.Keys(birth, dense, cls.K, k)
        if mode == 'adam':
            groups = [{'params': [model.W], 'lr': 1e-3 if keys_learn else 0.0}, {'params': model.out.parameters(), 'lr': 1e-3}]
            opt = torch.optim.Adam(groups)
        elif mode == 'sgd':
            groups = [{'params': [model.W], 'lr': 0.1 if keys_learn else 0.0}, {'params': model.out.parameters(), 'lr': 0.1}]
            opt = torch.optim.SGD(groups, momentum=0.9)
        else:  # mixed: Adam readout, SGD keys (separate optimizers)
            opt = torch.optim.Adam(model.out.parameters(), lr=1e-3)
            optk = torch.optim.SGD([model.W], lr=float(mode.split(':')[1]) if keys_learn else 0.0, momentum=0.9)
        g = torch.Generator().manual_seed(0); curve = []
        for ep in range(10):
            model.train(); perm = torch.randperm(len(c['train']), generator=g)
            for i in range(0, len(perm), 256):
                idx = perm[i:i+256]
                loss = torch.nn.functional.cross_entropy(model(Q['train'][idx]), c['train'][idx])
                opt.zero_grad()
                if mode.startswith('mixed'): optk.zero_grad()
                loss.backward(); opt.step()
                if mode.startswith('mixed'): optk.step()
                model.project()
            curve.append(round(refs.class_bits(model, Q['calib'], c['calib']), 3))
        Bl = model.keys_csr()
        Xs = codes.codes(Pca, Bl, pn, side, k); Xt = codes.codes(Pca, Bt, pn, side, k); X0 = codes.codes(Pca, birth['B'], pn, side, k)
        ov = lambda A, Bm: float(np.mean(np.asarray(A.multiply(Bm).sum(1)).ravel() / 184))
        print(mode, 'dense' if dense else 'native', 'keys learn' if keys_learn else 'keys frozen', curve,
              'overlap with teacher code: learned', round(ov(Xs, Xt), 3), 'birth', round(ov(X0, Xt), 3), 'cpu', round(time.process_time()-t0, 1), flush=True)
