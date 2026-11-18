#!/usr/bin/env python3
"""Check of the brief of E55, first part, revision 3: the figures of CHECK_OF_TASK_E55_BRIEF_REVISION_3_20261118.md.

The script imports the script of the check of revision 1 (check_e55_brief.py beside this file, as `K`) and, through it, the reviewer's
verification script of E51's trial (as `V`). Every file of another branch is read with `git show` into memory; torch is not used and no code
of another branch is run. Nothing is written unless a folder for emulated weights is named (E55R3_SAVE).

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e55_brief_r3.py <part> [...]

Parts:
    blind_start    the start `blind`, written in numpy as the brief's `make_blind` and `load_start55` write it: whether the directions in use
                   contain the constant; the figures on which the brief's check 1 puts limits; the same figures on two starts that are wrong
    blind_check2   eight emulated updates of the arm `blind` (the gradient of the rows kept blind projected before the step, the whole rows
                   projected after it, the named entries set to zero): the figures of the brief's check 2 (a) and (d); the arm `shut` of
                   revision 3 against the arm `shut` as the check of revision 1 emulated it
    blind_emulate  the emulation of the arm `blind` (the checker's own arithmetic, not the runner), first validated on 64 updates of E51's
                   arm `keyed` against the saved weights of row 64. Environment variables: E55R3_UPDATES (4096 if absent), E55R3_RUNS (for
                   example "unrelated:B,conflict:S"; all four if absent), E55R3_DTYPE (64 if absent: the gradient is computed in double
                   precision; 32: in single precision, as torch does, and about twice as fast), E55R3_ARM ("blind" if absent; "shut"),
                   E55R3_VALIDATE (0 leaves out the comparison with E51's saved weights), E55R3_SAVE (a folder outside the repository in
                   which the emulated weights after the last update are written; nothing is written if absent)
    blind_states   an emulated network of the arm `blind` read from the folder E55R3_SAVE: the states of the cells in reserve on the first
                   family's held-out pairs, position by position, in double and in single precision
"""
import importlib.util, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("check_e55_brief", os.path.join(HERE, "check_e55_brief.py"))
K = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(K)
V = K.V
BOS, EOS, H0, U0, H1, U1 = K.BOS, K.EOS, K.H0, K.U0, K.H1, K.U1
RES, OLD, Z_RES, ADDED, away = V.RES, V.OLD, K.Z_RES, K.ADDED, K.away          # RES: rows g * 256 + i, g = 0, 1, 2, i = 128 .. 255
ROWS55 = tuple(V.ROWS) + (8192, 16384)
NAMES = {"ih": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"), "hh": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"), "lm": ("lm.weight", "lm.bias"), "l2": ("l2.weight", "l2.bias")}
PARTS = {}


def blocks(cells):
    """The brief's `blocks`: (matrix, rows) for the four blocks of a start."""
    return (("ih", cells), ("hh", cells), ("lm", ADDED), ("l2", ADDED))


def block(W, mn, rows):
    wk, bk = NAMES[mn]
    return np.concatenate([W[wk][rows], W[bk][rows][:, None]], 1).astype(np.float64)


def zero_blocks(W, cells):
    W["rnn.weight_hh_l0"][np.ix_(cells, np.arange(U0))] = 0.0; W["rnn.bias_hh_l0"][cells] = 0.0
    W["lm.weight"][np.ix_(ADDED, np.arange(U0))] = 0.0; W["lm.bias"][ADDED] = 0.0


def make_blind(W, dirs, cells, zero=True):
    """The brief's `make_blind`: the whole of each row, with its bias as a last column, away from the directions in use (double precision,
    stored in single precision), then the entries that the projection brings to zero set to exactly zero."""
    for mn, rows in blocks(cells):
        wk, bk = NAMES[mn]; D = away(block(W, mn, rows), dirs[mn])
        W[wk][rows] = D[:, :-1].astype(np.float32); W[bk][rows] = D[:, -1].astype(np.float32)
    if zero:
        zero_blocks(W, cells)


def start55(form, rule):
    """The brief's `load_start55`: the keyed start of E51 with the rows of the reserve (form `blind`) or the gate rows of the reserve (form
    `shut`, with 10 added to the bias of the input map there) and rows 256 to 511 of lm and l2 made blind."""
    Wk = V.start("keyed", rule); dirs, in_use, _, _ = V.saved_directions(rule); W = {k: v.copy() for k, v in Wk.items()}
    if form == "blind":
        make_blind(W, dirs, RES)
    else:
        make_blind(W, dirs, Z_RES); W["rnn.bias_ih_l0"][Z_RES] += np.float32(10.0)
    return W, Wk, dirs, in_use.astype(bool)


# ---------------------------------------------------------------- the gradient in a chosen precision (K.forward and K.loss_and_grad, double there)
def lossgrad(W, x, y, mask, local, dt=np.float64):
    P = {k: np.asarray(v, dt) for k, v in W.items()}; B, L = x.shape; U = U1
    mm = lambda a, Wm: (a.reshape(-1, a.shape[-1]) @ Wm).reshape(a.shape[:-1] + (Wm.shape[1],)); fl = lambda a: a.reshape(-1, a.shape[-1])
    u = P["book.weight"][x]; z1 = np.tanh(mm(u, P["l1.weight"].T) + P["l1.bias"]); gi = mm(z1, P["rnn.weight_ih_l0"].T) + P["rnn.bias_ih_l0"]
    WhT = np.ascontiguousarray(P["rnn.weight_hh_l0"].T); h = np.zeros((B, U), dt); hp_, r_, z_, n_, g_, hs = [], [], [], [], [], []
    for t in range(L):
        gh = h @ WhT + P["rnn.bias_hh_l0"]
        r = V.sigmoid(gi[:, t, :U] + gh[:, :U]); z = V.sigmoid(gi[:, t, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        hp_.append(h); r_.append(r); z_.append(z); n_.append(n); g_.append(gh[:, 2 * U:]); h = (1 - z) * n + z * h; hs.append(h)
    Hs = np.stack(hs, 1); z2 = np.tanh(mm(z1, P["l2.weight"].T) + P["l2.bias"] + mm(Hs, P["lm.weight"].T) + P["lm.bias"])
    logits = mm(z2, P["l3.weight"].T) + P["l3.bias"]
    lg = logits[mask].astype(np.float64); lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    N = int(mask.sum()); loss = float(-np.log(p[np.arange(N), y[mask]]).mean())
    Sg = (lambda Wm: np.sign(Wm) * np.abs(Wm).mean(dtype=dt)) if local else (lambda Wm: Wm)
    d = p.copy(); d[np.arange(N), y[mask]] -= 1; d /= N; dlog = np.zeros(logits.shape, dt); dlog[mask] = d.astype(dt)
    G = {}
    G["l3.weight"] = fl(dlog).T @ fl(z2); G["l3.bias"] = fl(dlog).sum(0)
    da2 = mm(dlog, Sg(P["l3.weight"])) * (1 - z2 ** 2)
    G["l2.weight"] = fl(da2).T @ fl(z1); G["l2.bias"] = fl(da2).sum(0); G["lm.weight"] = fl(da2).T @ fl(Hs); G["lm.bias"] = fl(da2).sum(0)
    dz1 = mm(da2, Sg(P["l2.weight"])); dHs = mm(da2, Sg(P["lm.weight"]))
    Shh = np.ascontiguousarray(Sg(P["rnn.weight_hh_l0"])); Ghh = np.zeros((3 * U, U), dt); Gbhh = np.zeros(3 * U, dt); dgi = np.zeros((B, L, 3 * U), dt); nxt = np.zeros((B, U), dt)
    for t in range(L - 1, -1, -1):
        dh = dHs[:, t] + nxt; z, n, r, hp, ghn = z_[t], n_[t], r_[t], hp_[t], g_[t]
        dan = dh * (1 - z) * (1 - n ** 2); dar = dan * ghn * r * (1 - r); daz = dh * (hp - n) * z * (1 - z)
        dgi[:, t] = np.concatenate([dar, daz, dan], 1); dgh = np.concatenate([dar, daz, dan * r], 1)
        Ghh += dgh.T @ hp; Gbhh += dgh.sum(0); nxt = dh * z + dgh @ Shh
    G["rnn.weight_hh_l0"], G["rnn.bias_hh_l0"] = Ghh, Gbhh
    G["rnn.weight_ih_l0"] = fl(dgi).T @ fl(z1); G["rnn.bias_ih_l0"] = fl(dgi).sum(0)
    da1 = (dz1 + mm(dgi, Sg(P["rnn.weight_ih_l0"]))) * (1 - z1 ** 2)
    G["l1.weight"] = fl(da1).T @ fl(u); G["l1.bias"] = fl(da1).sum(0)
    Gb = np.zeros(P["book.weight"].shape, dt); np.add.at(Gb, x.reshape(-1), fl(mm(da1, Sg(P["l1.weight"])))); G["book.weight"] = Gb
    return loss, G


def emulate55(W, task, rule, arm, dirs, in_use, updates, at=(), measure=None, dt=np.float64):
    """The update of the brief's runner as read from exp51_reserve.step, R51.Rule and the brief's Rule55.
    arm `keyed`: E51's rule (the old rows and rows 0 to 255 of lm and l2: gradient and step projected).
    arm `shut`: the same with the gate rows of the reserve and rows 256 to 511 of lm and l2 added; then `zero_blocks` on the gate rows.
    arm `blind`: first the gradient of the rows kept blind is projected (Rule55.before), then E51's rule acts on its own rows (gradient
    before the step, the step after it, the restoring), then the whole of each row kept blind is projected and the named entries are set to
    zero (`make_blind` in Rule55.after). AdamW: rate 0.001, betas 0.9 and 0.98, eps 1e-8, no decay; its two moments are kept in double
    precision. The weights are stored in single precision after every step."""
    W = {k: v.copy() for k, v in W.items()}; m = {k: np.zeros(v.shape) for k, v in W.items()}; v_ = {k: np.zeros(v.shape) for k, v in W.items()}
    rowsets = K.MATSETS["shut"] if arm == "shut" else K.MATSETS["keyed"]; bt = K.batches(task); local = rule == "S"
    if measure and 0 in at:
        print(measure(0, W, None), flush=True)
    for g in range(1, updates + 1):
        x, y, mask = next(bt); loss, G = lossgrad(W, x, y, mask, local, dt)
        G = {k: a.astype(np.float32) for k, a in G.items()}; G["book.weight"][BOS] = 0          # the row of the start symbol takes no gradient (padding_idx)
        if arm == "blind":
            for mn, rows in blocks(RES):
                wk, bk = NAMES[mn]; Gm = away(np.concatenate([G[wk][rows], G[bk][rows][:, None]], 1).astype(np.float64), dirs[mn])
                G[wk][rows] = Gm[:, :-1].astype(np.float32); G[bk][rows] = Gm[:, -1].astype(np.float32)
        for mn, rows in rowsets.items():
            wk, bk = NAMES[mn]; Gm = np.concatenate([G[wk], G[bk][:, None]], 1).astype(np.float64); Gm[rows] = away(Gm[rows], dirs[mn])
            G[wk] = Gm[:, :-1].astype(np.float32); G[bk] = Gm[:, -1].astype(np.float32)
        snap = {k: W[k].copy() for k in W}
        for k in W:
            gk = G[k].astype(np.float64); m[k] = 0.9 * m[k] + 0.1 * gk; v_[k] = 0.98 * v_[k] + 0.02 * gk * gk
            W[k] = (W[k] - 1e-3 * (m[k] / (1 - 0.9 ** g)) / (np.sqrt(v_[k] / (1 - 0.98 ** g)) + 1e-8)).astype(np.float32)
        for mn, rows in rowsets.items():
            wk, bk = NAMES[mn]; S = np.concatenate([W[wk].astype(np.float64) - snap[wk], (W[bk].astype(np.float64) - snap[bk])[:, None]], 1); Sr = away(S[rows], dirs[mn])
            W[wk][rows] = (snap[wk][rows].astype(np.float64) + Sr[:, :-1]).astype(np.float32); W[bk][rows] = (snap[bk][rows].astype(np.float64) + Sr[:, -1]).astype(np.float32)
        for k in ("l1.weight", "l1.bias", "l3.bias"):
            W[k] = snap[k]
        W["l3.weight"][:, :H0] = snap["l3.weight"][:, :H0]; W["book.weight"][in_use] = snap["book.weight"][in_use]
        if arm == "blind":
            make_blind(W, dirs, RES)
        elif arm == "shut":
            zero_blocks(W, Z_RES)
        if measure and g in at:
            print(measure(g, W, loss), flush=True)
    return W


# ---------------------------------------------------------------- figures on a network
def reserve_states(net, items):
    """Teacher forcing: the largest magnitude of a state of a cell in reserve at each position (items of one length), and over all."""
    prof = None
    for idx, x in K.rows_of(items):
        h = np.abs(net.run(x, keep=True)[1]["h"][:, :, U0:]).max((0, 2))
        prof = h if prof is None or len(h) != len(prof) else np.maximum(prof, h)
    return prof


def remainder(W, dirs, cells):
    return {mn: float(np.abs(block(W, mn, rows) @ dirs[mn]).max()) for mn, rows in blocks(cells)}


def radius(W):
    """At the state zero of the reserve on a byte in use the gates of a cell in reserve stand at one half, and a small state h of the reserve
    is taken to 0.5 (0.5 Wn h) + 0.5 h, with Wn the candidate rows' weights from the cells in reserve: the largest magnitude of an
    eigenvalue of 0.5 I + 0.25 Wn. Above 1 a small state grows from one position to the next."""
    Wn = W["rnn.weight_hh_l0"][2 * U1 + U0:3 * U1, U0:U1].astype(np.float64)
    return float(np.abs(np.linalg.eigvals(0.5 * np.eye(U1 - U0) + 0.25 * Wn)).max())


def first_batch_state(net, task):
    x, y, mask = next(K.batches(task)); h = np.abs(net.run(x, keep=True)[1]["h"][:, :, U0:])
    return float(h.max()), float(h[:, 1].max())


# ================================================================ part `blind_start`
def blind_start():
    print("== blind_start: the start `blind` in numpy (the brief's `make_blind` and `load_start55`), and the figures of the brief's check 1")
    S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]; samp = V.check1_sample(S1)
    for rule in V.RULES:
        W, Wk, dirs, in_use = start55("blind", rule); print(f"--- rule {rule}")
        # do the directions in use contain the constant?
        for mn in ("ih", "hh", "lm", "l2"):
            e = np.zeros(dirs[mn].shape[0]); e[-1] = 1.0
            print(f" directions in use at {mn}: {dirs[mn].shape[1]} in {dirs[mn].shape[0]} dimensions; the part of the constant (the vector that is 1 at the last coordinate and 0 elsewhere) outside them has the 2-norm {np.linalg.norm(away(e[None], dirs[mn])):.2e}")
        Wp = {k: v.copy() for k, v in Wk.items()}; make_blind(Wp, dirs, RES, zero=False)         # the projection alone, before `zero_blocks`
        print(f" after the projection and before `zero_blocks`: largest magnitude of the bias of the rows of the reserve in the recurrent matrix {np.abs(Wp['rnn.bias_hh_l0'][RES]).max():.1e}, of the bias of rows 256 to 511 of lm {np.abs(Wp['lm.bias'][ADDED]).max():.1e}, "
              f"of the weights from the old cells into the rows of the reserve {np.abs(Wp['rnn.weight_hh_l0'][RES][:, :U0]).max():.1e}, into rows 256 to 511 of lm {np.abs(Wp['lm.weight'][ADDED][:, :U0]).max():.1e}; "
              f"of the bias of the rows of the reserve in the input map {np.abs(Wp['rnn.bias_ih_l0'][RES]).max():.3f} and of rows 256 to 511 of l2 {np.abs(Wp['l2.bias'][ADDED]).max():.3f} (these two are not brought to zero)")
        # (a)
        diff = {}
        for k in W:
            rows = np.nonzero((W[k] != Wk[k]).reshape(len(W[k]), -1).any(1))[0]
            if len(rows):
                diff[k] = (len(rows), int(rows.min()), int(rows.max()))
        ok_a = (set(np.nonzero((W["rnn.weight_ih_l0"] != Wk["rnn.weight_ih_l0"]).any(1))[0]) <= set(RES.tolist()) and set(np.nonzero((W["rnn.weight_hh_l0"] != Wk["rnn.weight_hh_l0"]).any(1))[0]) <= set(RES.tolist()))
        print(f" 1(a) tensors that differ from the keyed start (rows that differ, first, last): {diff}; in the two matrices of the recurrent layer every row that differs is a row of the reserve: {ok_a}")
        # (b)
        print(" 1(b) largest magnitude of an entry of D @ Vm (limit 1e-5): " + ", ".join(f"{mn} {v:.1e}" for mn, v in remainder(W, dirs, RES).items())
              + "; in the keyed start: " + ", ".join(f"{mn} {v:.2f}" for mn, v in remainder(Wk, dirs, RES).items()))
        # (c)
        ns, nk = V.Net(W), V.Net(Wk); ns32, nk32 = V.Net(W, np.float32), V.Net(Wk, np.float32)
        for label, its in (("the 1,200 held-out pairs at 12 letters", held), ("check 1's 2,000 items without letters", samp)):
            f, f32 = K.gate_figures(ns, its), K.gate_figures(ns32, its)
            d = np.abs(V.answer_logits(ns, its) - V.answer_logits(nk, its, in_use=in_use)).max(); d32 = np.abs(V.answer_logits(ns32, its) - V.answer_logits(nk32, its, in_use=in_use)).max()
            print(f" 1(c) {label}: largest magnitude of a state of a cell in reserve {f['largest_state']:.2e} (single precision {f32['largest_state']:.2e}; limit 1e-5); largest magnitude of an added readout unit at the answer positions "
                  f"{f['largest_added_unit']:.2e} (single precision {f32['largest_added_unit']:.2e}; limit 1e-5); largest difference of an output at an answer position from the keyed start under the keyed forward {d:.2e} (single precision {d32:.2e}; limit 1e-4); "
                  f"1 - z of a cell in reserve between {1 - f['largest_opening']:.6f} and {f['largest_opening']:.6f}")
        prof = reserve_states(ns, held)
        print(f"      the largest state of a cell in reserve on the held-out pairs, at positions 0, 1, 5, 10, 15, 21: {[float(f'{prof[t]:.2e}') for t in (0, 1, 5, 10, 15, 21)]}; "
              f"largest magnitude of an eigenvalue of the map that takes a small state of the reserve from one position to the next: {radius(W):.3f}")
        for task in V.TASKS:
            a, b = first_batch_state(ns, task)
            print(f" 1(c) first batch of `{task}`: largest magnitude of a state of a cell in reserve {a:.3f} (limit: above 1e-3); at the item's first byte {b:.3f}")
        # (d)
        gs = V.score_first_family(ns, S1, HELD)[1]; gk = V.score_first_family(nk, S1, HELD, in_use=in_use)[1]
        gs32 = V.score_first_family(ns32, S1, HELD)[1]; gk32 = V.score_first_family(nk32, S1, HELD, in_use=in_use)[1]
        print(f" 1(d) generated answers that differ from the keyed start's under the keyed forward: {sum(a != b for n in gs for a, b in zip(gs[n], gk[n]))} of {sum(len(v) for v in gs.values())} "
              f"(in single precision {sum(a != b for n in gs32 for a, b in zip(gs32[n], gk32[n]))}; limit 0)")
        # two starts that are wrong
        Wg = {k: v.copy() for k, v in Wk.items()}; make_blind(Wg, dirs, Z_RES)                   # only the gate rows made blind (the start `shut` without its bias)
        Wi = {k: v.copy() for k, v in Wk.items()}                                               # the rows of the reserve made blind in the input map only
        for mn, rows in (("ih", RES), ("lm", ADDED), ("l2", ADDED)):
            wk, bk = NAMES[mn]; D = away(block(Wi, mn, rows), dirs[mn]); Wi[wk][rows] = D[:, :-1].astype(np.float32); Wi[bk][rows] = D[:, -1].astype(np.float32)
        for label, Wx in (("the keyed start as it is, under the plain forward (no projection)", Wk), ("only the gate rows of the reserve made blind", Wg), ("the rows of the reserve made blind in the input map and not in the recurrent matrix", Wi)):
            nx = V.Net(Wx); f = K.gate_figures(nx, held); d = np.abs(V.answer_logits(nx, held) - V.answer_logits(nk, held, in_use=in_use)).max()
            gx = V.generate(nx, held); gkh = V.generate(nk, held, in_use=in_use)
            print(f"      a wrong start, {label}: on the held-out pairs largest state {f['largest_state']:.2e}, largest added unit {f['largest_added_unit']:.2e}, largest difference of an output {d:.2e}, "
                  f"answers that differ {sum(a != b for a, b in zip(gx, gkh))} of 1,200; 1(b) " + ", ".join(f"{mn} {v:.1e}" for mn, v in remainder(Wx, dirs, RES).items()))


PARTS["blind_start"] = blind_start


# ================================================================ part `blind_check2`
def blind_check2():
    print("== blind_check2: the brief's check 2 after eight emulated updates")
    S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]; samp = V.check1_sample(S1)
    gate = lambda Wx, net, rows, newb: np.concatenate([Wx["rnn.weight_ih_l0"][rows].astype(np.float64), Wx["rnn.bias_ih_l0"][rows][:, None].astype(np.float64)], 1) @ np.concatenate([net.code[newb], np.ones((len(newb), 1))], 1).T
    for rule in V.RULES:
        W0, Wk, dirs, in_use = start55("blind", rule); n0 = V.Net(W0); a0 = {n: V.answer_logits(n0, its) for n, its in (("held", held), ("sample", samp))}
        for task in V.TASKS:
            newb = list(range(128, 226)) if task == "unrelated" else [ord(ch) for ch in "rtxn"]
            W = emulate55(W0, task, rule, "blind", dirs, in_use, 8); net = V.Net(W); net32 = V.Net(W, np.float32)
            old = {mn: float(np.abs((block(W, mn, K.MATSETS["keyed"][mn]) - block(W0, mn, K.MATSETS["keyed"][mn])) @ dirs[mn]).max()) for mn in NAMES}
            changed = all(not np.array_equal(W[NAMES[mn][0]], W0[NAMES[mn][0]]) for mn in NAMES)
            zeros = (not W["rnn.weight_hh_l0"][RES][:, :U0].any() and not W["rnn.bias_hh_l0"][RES].any() and not W["lm.weight"][ADDED][:, :U0].any() and not W["lm.bias"][ADDED].any())
            rem = remainder(W, dirs, RES)
            f = {n: K.gate_figures(net, its) for n, its in (("held", held), ("sample", samp))}; f32 = {n: K.gate_figures(net32, its) for n, its in (("held", held), ("sample", samp))}
            d = {n: float(np.abs(V.answer_logits(net, its) - a0[n]).max()) for n, its in (("held", held), ("sample", samp))}
            mv = gate(W, net, RES, newb) - gate(W0, n0, RES, newb)
            gh, g0 = V.generate(net, held), V.generate(n0, held)
            print(f" rule {rule}, {task}, arm `blind`: (a) old rows, largest entry of (D_after - D_start) @ Vm: " + ", ".join(f"{mn} {v:.1e}" for mn, v in old.items()) + f" (limit 1e-5); every matrix changed: {changed}; the entries of `zero_blocks` exactly zero: {zeros}; "
                  f"(d) rows kept blind, largest entry of D_after @ Vm: " + ", ".join(f"{mn} {v:.1e}" for mn, v in rem.items()) + f" (limit 1e-5); largest state of a cell in reserve on the two panels {max(f[n]['largest_state'] for n in f):.2e} "
                  f"(single precision {max(f32[n]['largest_state'] for n in f32):.2e}; limit 1e-5); largest added unit at the answer positions {max(f[n]['largest_added_unit'] for n in f):.2e}; largest change of an output at an answer position: held-out {d['held']:.2e}, 2,000 items {d['sample']:.2e} (limit 1e-4); "
                  f"change of what a row of the reserve receives from the input map on a byte of the new task: largest magnitude {np.abs(mv).max():.3f}, mean magnitude {np.abs(mv).mean():.4f} (must exceed 0.01); held-out answers that differ from the start's {sum(a != b for a, b in zip(gh, g0))} of 1,200; "
                  f"largest eigenvalue of the map of a small state {radius(W):.3f}")
        # the arm `shut` of revision 3 (the two bias vectors set to zero as well) against the arm `shut` as the check of revision 1 emulated it with its blocks set to zero
        Ws3, _, _, _ = start55("shut", rule); Ws1, _, _, _ = K.shut_start(rule)
        print(f" rule {rule}, the start `shut` of revision 3 against that of revision 1: tensors that differ " + str({k: float(np.abs(Ws3[k].astype(np.float64) - Ws1[k]).max()) for k in Ws3 if not np.array_equal(Ws3[k], Ws1[k])}))
        for task in V.TASKS:
            A = emulate55(Ws3, task, rule, "shut", dirs, in_use, 8); B, _ = K.emulate_updates(Ws1, task, rule, K.MATSETS["shut"], dirs, in_use, 8, zero_blocks=True)
            print(f" rule {rule}, {task}, arm `shut` after eight updates, revision 3 against revision 1 with its two blocks set to zero: largest difference of a weight {max(float(np.abs(A[k].astype(np.float64) - B[k]).max()) for k in A):.1e}")


PARTS["blind_check2"] = blind_check2


# ================================================================ part `blind_emulate`
def blind_emulate():
    print("== blind_emulate: the emulation of the brief's update, in the checker's own arithmetic")
    dt = np.float32 if os.environ.get("E55R3_DTYPE", "64") == "32" else np.float64; arm = os.environ.get("E55R3_ARM", "blind")
    S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]
    runs = [tuple(r.split(":")) for r in os.environ.get("E55R3_RUNS", "unrelated:S,conflict:S,unrelated:B,conflict:B").split(",")]
    for rule in (V.RULES if os.environ.get("E55R3_VALIDATE", "1") == "1" else ()):
        Wk = V.start("keyed", rule); dirs, in_use, _, _ = V.saved_directions(rule); in_use = in_use.astype(bool)
        We = emulate55(Wk, "unrelated", rule, "keyed", dirs, in_use, 64, dt=dt); Ws, _ = V.weights("unrelated", "keyed", rule, 64)
        print(f" validation ({'single' if dt is np.float32 else 'double'} precision), E51 `keyed` on `unrelated`, rule {rule}, 64 updates: largest difference of a weight from the saved weights of row 64 "
              f"{max(float(np.abs(We[k].astype(np.float64) - Ws[k]).max()) for k in We):.1e}; largest change of a weight from the start {max(float(np.abs(Ws[k].astype(np.float64) - Wk[k]).max()) for k in Ws):.1e}", flush=True)
    n_up = int(os.environ.get("E55R3_UPDATES", "4096")); at = [r for r in ROWS55 if r <= n_up]
    for rule in V.RULES:
        for task in V.TASKS:
            if (task, rule) not in runs:
                continue
            W0, Wk, dirs, in_use = start55(arm, rule); pan = K.reserve_panel(task); panels = V.new_panels(task); ref = {}
            cells = RES if arm == "blind" else Z_RES

            def measure(g, W, loss):
                net = V.Net(W); net32 = V.Net(W, np.float32); prof = reserve_states(net, held); prof32 = reserve_states(net32, held)
                res, gens = V.score_first_family(net, S1, HELD); ref.setdefault("gens", gens); ref.setdefault("a0", V.answer_logits(net, held)); g0 = ref["gens"]
                dk0 = sum(a != b for n in gens if n.startswith("k0_") for a, b in zip(gens[n], g0[n])); dk12 = sum(a != b for n in gens if n.startswith("k12_") for a, b in zip(gens[n], g0[n]))
                s3 = V.score_new(net, panels); s3s = V.score_new(net, panels, shut_all=True)
                nt = lambda s: 100 * sum(s[k]["exact"] for k in s if k != "taught") / sum(s[k]["n"] for k in s if k != "taught")
                f = K.gate_figures(net, held); fn = K.gate_figures(net, pan); rem = remainder(W, dirs, cells)
                return (f" row {g:6d}: loss {'-' if loss is None else format(loss, '.3f')}; never-taught {nt(s3):5.1f}, with the cells in reserve held shut {nt(s3s):5.1f}; first family: answers that differ from row 0's {dk12} of 1,200 held-out and {dk0} of 31,684 without letters, "
                        f"reduction score {res['reduction_score']:.2f}, largest change of an output at an answer position on the held-out pairs {np.abs(V.answer_logits(net, held) - ref['a0']).max():.2e}; largest state of a cell in reserve on the held-out pairs {prof.max():.2e} "
                        f"(single precision {prof32.max():.2e}), at positions 1, 11, 21: {prof[1]:.1e}, {prof[11]:.1e}, {prof[21]:.1e}; largest added readout unit at the answer positions {f['largest_added_unit']:.2e}; largest eigenvalue of the map of a small state {radius(W):.3f}; "
                        f"largest entry of D @ Vm at the rows kept blind {max(rem.values()):.1e}; on the new task's never-taught items: largest state of a cell in reserve {fn['largest_state']:.2f}, cells open at some position {int((fn['peak'] > 0.1).sum())}")
            t0 = time.time(); print(f"--- emulation, {task}, rule {rule}, arm `{arm}`, {n_up} updates, the gradient in {'single' if dt is np.float32 else 'double'} precision", flush=True)
            Wend = emulate55(W0, task, rule, arm, dirs, in_use, n_up, at, measure, dt)
            if os.environ.get("E55R3_SAVE"):
                np.savez(os.path.join(os.environ["E55R3_SAVE"], f"emulated_r3_{task}_{rule}_{arm}_{n_up}.npz"), **Wend)
            print(f"    ({time.time() - t0:.0f} s)", flush=True)


PARTS["blind_emulate"] = blind_emulate


# ================================================================ part `blind_states`
def blind_states():
    print("== blind_states: the emulated networks of the arm `blind` after their last update (read from the folder E55R3_SAVE)")
    import glob
    S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]
    held40 = [it for t in V.DEV for it in V.panel(t, V.split(t, .5, V.SEED)[1], V.SEED, (40,), 300)[40]]     # 1,200 held-out pairs at 40 letters
    for f in sorted(glob.glob(os.path.join(os.environ["E55R3_SAVE"], "emulated_r3_*_blind_*.npz"))):
        name = os.path.basename(f)[len("emulated_r3_"):-4]; task, rule = name.split("_")[:2]
        W = dict(np.load(f)); W0, Wk, dirs, in_use = start55("blind", rule); out = [f" {name}:"]
        for label, dt in (("double", np.float64), ("single", np.float32)):
            net, n0 = V.Net(W, dt), V.Net(W0, dt); prof = reserve_states(net, held)
            res, gens = V.score_first_family(net, S1, HELD); res0, g0 = V.score_first_family(n0, S1, HELD)
            dk0 = sum(a != b for n in gens if n.startswith("k0_") for a, b in zip(gens[n], g0[n])); dk12 = sum(a != b for n in gens if n.startswith("k12_") for a, b in zip(gens[n], g0[n]))
            d = float(np.abs(V.answer_logits(net, held) - V.answer_logits(n0, held)).max())
            out.append(f"   {label} precision: largest state of a cell in reserve on the held-out pairs at positions 1, 6, 11, 16, 21: " + ", ".join(f"{prof[t]:.1e}" for t in (1, 6, 11, 16, 21))
                       + f"; answers that differ from the start's: {dk12} of 1,200 held-out and {dk0} of 31,684 without letters; reduction score {res['reduction_score']:.2f} (start {res0['reduction_score']:.2f}); largest change of an output on the held-out pairs {d:.2e}")
            p40 = reserve_states(net, held40); g40, g40s = V.generate(net, held40), V.generate(n0, held40)
            out.append(f"   {label} precision, the same held-out pairs of the first family written with 40 letters ({len(held40[0][0]) + 3} positions; not a panel of the brief): largest state of a cell in reserve at positions 1, 11, 21, 31, 41, {len(p40) - 1}: "
                       + ", ".join(f"{p40[t]:.1e}" for t in (1, 11, 21, 31, 41, len(p40) - 1)) + f"; answers that differ from the start's {sum(x != y for x, y in zip(g40, g40s))} of {len(held40)}; right answers {V.exact(g40, held40)} (start {V.exact(g40s, held40)})")
        # two paths to the first family, taken away one at a time: (1) the part of the change of the old rows that lies along the directions in use
        # (the rounding of the stored weights, which E51's rule does not remove because it projects each step and not the whole change);
        # (2) the states of the cells in reserve (their update gates held at 1 on every item)
        Wf = {k: v.astype(np.float64) for k, v in W.items()}
        rem_old = {}
        for mn, rows in K.MATSETS["keyed"].items():
            wk, bk = NAMES[mn]; D = block(W, mn, rows) - block(W0, mn, rows); rem_old[mn] = float(np.abs(D @ dirs[mn]).max()); Dp = away(D, dirs[mn])
            Wf[wk][rows] = W0[wk][rows].astype(np.float64) + Dp[:, :-1]; Wf[bk][rows] = W0[bk][rows].astype(np.float64) + Dp[:, -1]
        n0 = V.Net(W0); a0 = V.answer_logits(n0, held); g0 = V.score_first_family(n0, S1, HELD)[1]
        for label, net, key in (("as emulated", V.Net(W), {}), ("the change of the old rows along the directions in use taken away", V.Net(Wf), {}), ("the cells in reserve held shut", V.Net(W), dict(shut_all=True)),
                                ("both", V.Net(Wf), dict(shut_all=True))):
            gens = V.score_first_family(net, S1, HELD, **key)[1]
            out.append(f"   double precision, {label}: answers that differ from the start's {sum(x != y for n in gens for x, y in zip(gens[n], g0[n]))} of 32,884; largest change of an output on the held-out pairs {np.abs(V.answer_logits(net, held, **key) - a0).max():.2e}")
        out.append("   old rows, largest entry of (D_end - D_start) @ Vm: " + ", ".join(f"{mn} {v:.1e}" for mn, v in rem_old.items()))
        ev = np.linalg.eigvals(0.5 * np.eye(U1 - U0) + 0.25 * W["rnn.weight_hh_l0"][2 * U1 + U0:3 * U1, U0:U1].astype(np.float64))
        out.append(f"   the map of a small state of the reserve at a byte in use: eigenvalues of magnitude above 1: {int((np.abs(ev) > 1).sum())} of 128; the largest magnitude {np.abs(ev).max():.3f}, to the power 21: {np.abs(ev).max() ** 21:.1e}")
        print("\n".join(out), flush=True)


PARTS["blind_states"] = blind_states


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t0 = time.time(); PARTS[p](); print(f"   [{p}: {time.time() - t0:.0f} s]", flush=True)
