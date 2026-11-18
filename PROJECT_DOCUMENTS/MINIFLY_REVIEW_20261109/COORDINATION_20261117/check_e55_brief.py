#!/usr/bin/env python3
"""Check of the brief of E55, first part (revision 1): the figures of the document CHECK_OF_TASK_E55_BRIEF_20261118.md.

Every file of another branch is read with `git show` into memory; torch is not used and no code of another branch is run. Nothing is
written unless the part `emulate` is given a folder for its emulated weights (E55_CHECK_SAVE). The forward pass, the items, the panels and the reader of weight files are those of the reviewer's
verification of E51's trial (PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e51_trial_verify.py), imported as `V`.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e55_brief.py <part> [...]

Parts:
    record   the figures of the brief's section "What the record says", from the sixteen receipts of E51 on the worker's branch
    open     E51's arm `open`, from its weights of row 4,096: the update gates of the cells in reserve position by position
    start    the start `shut` written in numpy from the saved keyed start and the saved directions: the figures of the brief's check 1
             (b), (c) and (d); what the gates, the reset gates and the candidates of the cells in reserve receive; the part of the code
             of each byte of the new tasks outside the directions in use
    grad     the gradient of the loss at the start `shut` on one batch of each new task, by a backward pass written here (rule B: the
             exact gradient, verified against central finite differences; rule S: the sweep): its size at the gate rows and elsewhere
    check2   the brief's check 2 after eight emulated updates of the arms `shut` and `shut-gates-free`
    residue  rule S after eight emulated updates: the error that the sweep sends back through the entries that the projection brought
             to zero, with these entries as stored and set to exactly zero
    emulate  an emulation of the runner's update (the checker's own arithmetic, not the runner): first 64 updates of E51's arm `keyed`
             against the saved weights of row 64, then the arm `shut`. Environment variables: E55_CHECK_UPDATES (number of updates,
             512 if absent), E55_CHECK_RUNS (for example "unrelated:B,conflict:S"; all four if absent), E55_CHECK_VALIDATE (0 leaves
             out the comparison with E51's saved weights), E55_CHECK_VARIANTS ("brief" if absent; "zeroed" sets the entries that the
             projection brings to zero to exactly zero after every step; "gates-free" is the arm `shut-gates-free`; "scaled"
             multiplies the gradient of the gate rows of the reserve by 1e4), E55_CHECK_SAVE (a folder outside the repository in which
             the emulated weights after the last update are written; nothing is written if absent)
    positions  the emulated run of `conflict` under rule B at row 4,096, read from the folder E55_CHECK_SAVE: the positions at which the
             gates of the cells in reserve open
"""
import importlib.util, json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
_argv, sys.argv = sys.argv, sys.argv[:1]
_spec = importlib.util.spec_from_file_location("e51_trial_verify", os.path.join(os.path.dirname(HERE), "e51_trial_verify.py"))
sys.path.insert(0, os.path.dirname(HERE))
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)
sys.argv = _argv
V.WREF = "0ab99a377aec3808359ae7054ea29cfd88354ab0"   # the head of the branch of E51's worker on 18 November (the verification pins an earlier commit)

SEED, BOS, EOS = V.SEED, V.BOS, V.EOS
H0, U0, H1, U1 = V.H0, V.U0, V.H1, V.U1
ROWS = V.ROWS
Z_RES = np.arange(U1 + U0, 2 * U1)                       # rows 384 to 511: the update gates of the cells in reserve
ADDED = np.arange(H0, H1)                               # rows 256 to 511 of lm and of l2
PARTS = {}


def learned_from(rec):
    c = [r["score3_never_taught"]["pct"] for r in rec["rows"]]
    for i, r in enumerate(rec["rows"]):
        if all(v >= 80 for v in c[i:]):
            return r["updates"]
    return None


# ================================================================ part `record`
def record():
    print("== record: the figures of 'What the record says', from the receipts (sweep; back-propagation)")
    cpu = 0.0
    for task in V.TASKS:
        for arm in V.ARMS:
            out = []
            for rule in V.RULES:
                rec = V.receipt(task, arm, rule); rows = rec["rows"]; assert [r["updates"] for r in rows] == list(ROWS) and rec["complete"]["updates"] == 4096
                cpu += rec["complete"]["cpu_total"]
                red = {r["updates"]: r["score1_first_family_all_pairs_k0"]["reduction_score"] for r in rows}
                sh = {r["updates"]: r["share_of_answers_equal_to_row0"] for r in rows}
                changed_k0 = max(31684 - sum(v for n, v in sh[g]["per_panel"].items() if n.startswith("k0_")) for g in ROWS)
                changed_k12 = max(1200 - sum(v for n, v in sh[g]["per_panel"].items() if n.startswith("k12_")) for g in ROWS)
                first_below5 = next((g for g in ROWS if red[g] < 5), None)
                last = rows[-1]
                shut = last.get("score3_never_taught_reserve_shut", {}).get("pct")
                res = last.get("reserve", {})
                out.append(f"   rule {rule}: reduction score row 0 {red[0]:.2f}, row 16 {red[16]:.2f}, row 64 {red[64]:.2f}, row 4,096 {red[4096]:.2f}; first row below 5: {first_below5}; "
                           f"answers without letters that differ from row 0's at row 4,096: {100 * (1 - sh[4096]['all_pairs_k0']):.2f}% (largest count at any row {changed_k0} of 31,684; held-out {changed_k12} of 1,200); "
                           f"never-taught score at 4,096 {last['score3_never_taught']['pct']:.1f}, learned from row {learned_from(rec)}; with the cells in reserve shut {shut}; "
                           f"cells in reserve open by more than 0.1 (mean of 1 - z over positions and items): row 0 {rows[0].get('reserve', {}).get('reserve_cells_open_by_more_than_0p1')}, row 4,096 {res.get('reserve_cells_open_by_more_than_0p1')}; "
                           f"mean opening {res.get('mean_opening', float('nan')):.5f}, largest opening of a cell {res.get('largest_opening', float('nan')):.4f}; "
                           f"largest state of a cell in reserve on the first family's held-out panel: row 0 {rows[0].get('reserve', {}).get('largest_reserve_state_on_first_family_panel')}, row 4,096 {res.get('largest_reserve_state_on_first_family_panel')}")
            print(f" {task}, {arm}:"); print("\n".join(out))
    print(f" CPU-hours of the sixteen runs together: {cpu / 3600:.3f}")
    # the opening of the gates in `open` at every row
    for task in V.TASKS:
        for rule in V.RULES:
            rec = V.receipt(task, "open", rule)
            print(f" open, {task}, {rule}: cells open by more than 0.1 at the ten rows {[r['reserve']['reserve_cells_open_by_more_than_0p1'] for r in rec['rows']]}; "
                  f"largest mean opening of a cell {[round(r['reserve']['largest_opening'], 4) for r in rec['rows']]}; never-taught score {[round(r['score3_never_taught']['pct'], 1) for r in rec['rows']]}; "
                  f"the same with the cells in reserve shut {[round(r['score3_never_taught_reserve_shut']['pct'], 1) for r in rec['rows']]}")


PARTS["record"] = record


# ================================================================ helpers on the saved weights
def rows_of(items):
    """Teacher forcing, as the runner scores: x = [start] + prompt + answer without its last byte; items grouped by length."""
    for idx in V.groups_of(items):
        yield idx, np.array([[BOS] + items[i][0] + items[i][1][:-1] for i in idx])


def gate_figures(net, items, **key):
    """Over the cells in reserve, under teacher forcing: the mean of 1 - z over positions and items for each cell (the receipts' figure);
    for each cell the largest over positions of the mean over items of 1 - z (items of one length); the largest 1 - z anywhere; the largest
    magnitude of a state; the largest magnitude of an added readout unit at the answer positions."""
    tot, n, big_open, big_h, big_z2 = np.zeros(U1 - U0), 0, 0.0, 0.0, 0.0
    by_pos = np.zeros(U1 - U0); peak, items_n = np.zeros(U1 - U0), 0
    for idx, x in rows_of(items):
        k = net.run(x, keep=True, **key)[1]; op = 1 - k["z"][:, :, U0:]
        tot += op.sum((0, 1)); n += op.shape[0] * op.shape[1]
        by_pos = np.maximum(by_pos, op.mean(0).max(0)); big_open = max(big_open, float(op.max())); peak += op.max(1).sum(0); items_n += op.shape[0]
        big_h = max(big_h, float(np.abs(k["h"][:, :, U0:]).max())); big_z2 = max(big_z2, float(np.abs(k["z2"][:, :, H0:]).max()))
    return dict(mean=tot / n, by_pos=by_pos, peak=peak / items_n, largest_opening=big_open, largest_state=big_h, largest_added_unit=big_z2)


def reserve_panel(task):
    p = V.new_panels(task)
    return [it for n in (("add97", "sub97") if task == "conflict" else ("held",)) for it in p[n]]


# ================================================================ part `open`
def open_():
    print("== open: E51's arm `open` at row 4,096, from the saved weights; the update gates of the cells in reserve position by position")
    HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]
    for task in V.TASKS:
        pan = reserve_panel(task)
        for rule in V.RULES:
            W, _ = V.weights(task, "open", rule, 4096); net = V.Net(W)
            rec = V.row(V.receipt(task, "open", rule), 4096)
            for label, its in ((f"the never-taught items of `{task}` ({len(pan)})", pan), ("the first family's 1,200 held-out pairs at 12 letters", held)):
                f = gate_figures(net, its)
                print(f" {task}, rule {rule}, on {label}: cells with mean of 1 - z over positions and items above 0.1: {int((f['mean'] > 0.1).sum())}"
                      + (f" (receipt {rec['reserve']['reserve_cells_open_by_more_than_0p1']})" if its is pan else "")
                      + f"; cells whose mean over items of 1 - z exceeds 0.1 at some position: {int((f['by_pos'] > 0.1).sum())}, exceeds 0.5 at some position: {int((f['by_pos'] > 0.5).sum())}; cells whose mean over items of the largest 1 - z over the positions of the item exceeds 0.5: {int((f['peak'] > 0.5).sum())}, exceeds 0.1: {int((f['peak'] > 0.1).sum())}; "
                      f"largest 1 - z of a cell at a position of an item {f['largest_opening']:.4f}; largest magnitude of a state of a cell in reserve {f['largest_state']:.4f}")
            # which positions: the mean over items and cells of 1 - z at each position of the panel's items (all of one length)
            for idx, x in rows_of(pan):
                op = 1 - net.run(x, keep=True)[1]["z"][:, :, U0:]
                print(f"   positions of the new task's items (length {x.shape[1]}): number of cells whose mean over items of 1 - z exceeds 0.5, position by position: {[int((op[:, t].mean(0) > 0.5).sum()) for t in range(x.shape[1])]}")
            s_open = V.score_new(net, V.new_panels(task)); s_shut = V.score_new(net, V.new_panels(task), shut_all=True)
            print(f"   never-taught score with the plain forward {V.fmt_new(s_open)}; with the cells in reserve held shut on every item {V.fmt_new(s_shut)}")


PARTS["open"] = open_


# ================================================================ the start `shut`, written in numpy from the brief's text
BLOCKS = (("ih", "rnn.weight_ih_l0", "rnn.bias_ih_l0", Z_RES), ("hh", "rnn.weight_hh_l0", "rnn.bias_hh_l0", Z_RES),
          ("lm", "lm.weight", "lm.bias", ADDED), ("l2", "l2.weight", "l2.bias", ADDED))


def away(D, Vm):
    return D - (D @ Vm) @ Vm.T


def shut_start(rule, exact_zero=False):
    """The brief's `shut_start`: each of the four blocks, with its bias appended as a last column, is replaced by its projection away from the
    saved directions (double precision, stored in single precision); then 10 is added to the bias of the input map at the gate rows of the
    reserve (in single precision, as `+=` on a float32 tensor does)."""
    Wk = V.start("keyed", rule); dirs, in_use, _, _ = V.saved_directions(rule)
    W = {k: v.copy() for k, v in Wk.items()}
    for m, wk, bk, rows in BLOCKS:
        D = away(np.concatenate([W[wk][rows], W[bk][rows][:, None]], 1).astype(np.float64), dirs[m])
        W[wk][rows] = D[:, :-1].astype(np.float32); W[bk][rows] = D[:, -1].astype(np.float32)
    W["rnn.bias_ih_l0"][Z_RES] += np.float32(10.0)
    return W, Wk, dirs, in_use.astype(bool)


def forward(W, x):
    """The plain forward in double precision with everything kept (the brief's network; no added line). x: rows of equal length."""
    P = {k: np.asarray(v, np.float64) for k, v in W.items()}; B, L = x.shape; U = P["rnn.weight_hh_l0"].shape[1]
    mm = lambda a, Wm: (a.reshape(-1, a.shape[-1]) @ Wm).reshape(a.shape[:-1] + (Wm.shape[1],))      # one product for all rows and positions
    u = P["book.weight"][x]; z1 = np.tanh(mm(u, P["l1.weight"].T) + P["l1.bias"])
    gi = mm(z1, P["rnn.weight_ih_l0"].T) + P["rnn.bias_ih_l0"]
    h = np.zeros((B, U)); c = dict(P=P, x=x, u=u, z1=z1, hp=[], r=[], z=[], n=[], ghn=[], h=[])
    for t in range(L):
        gh = h @ P["rnn.weight_hh_l0"].T + P["rnn.bias_hh_l0"]
        r = V.sigmoid(gi[:, t, :U] + gh[:, :U]); z = V.sigmoid(gi[:, t, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        c["hp"].append(h); c["r"].append(r); c["z"].append(z); c["n"].append(n); c["ghn"].append(gh[:, 2 * U:])
        h = (1 - z) * n + z * h; c["h"].append(h)
    Hs = np.stack(c["h"], 1)
    z2 = np.tanh(mm(z1, P["l2.weight"].T) + P["l2.bias"] + mm(Hs, P["lm.weight"].T) + P["lm.bias"])
    c.update(Hs=Hs, z2=z2, logits=mm(z2, P["l3.weight"].T) + P["l3.bias"], az=gi[:, :, U:2 * U], mm=mm)
    return c


def start():
    print("== start: the start `shut` in numpy (the brief's `shut_start`), and the figures of the brief's check 1")
    S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]; samp = V.check1_sample(S1)
    every = [it for n in S1 for it in S1[n][0]]
    for rule in V.RULES:
        W, Wk, dirs, in_use = shut_start(rule)
        print(f"--- rule {rule}: bytes in use {int(in_use.sum())}; directions {[dirs[m].shape for m in ('ih', 'hh', 'lm', 'l2')]}")
        # (a) what differs from the keyed start
        diff = [k for k in W if not np.array_equal(W[k], Wk[k])]
        print(f" 1(a) tensors that differ from the keyed start: {diff}")
        # (b) the projection
        for m, wk, bk, rows in BLOCKS:
            D = np.concatenate([W[wk][rows], W[bk][rows][:, None]], 1).astype(np.float64)
            raw = np.abs(D @ dirs[m]).max()
            if m == "ih":
                D[:, -1] -= 10.0
            D0 = np.concatenate([Wk[wk][rows], Wk[bk][rows][:, None]], 1).astype(np.float64)
            print(f" 1(b) block {m}: largest magnitude of an entry of D @ Vm: {np.abs(D @ dirs[m]).max():.2e} (limit 1e-5)" + (f"; without the subtraction of 10: {raw:.3f}" if m == "ih" else "")
                  + f"; before the projection (the keyed start): {np.abs(D0 @ dirs[m]).max():.3f}; 2-norm of the block before {np.linalg.norm(D0):.2f} and after {np.linalg.norm(D):.2f}")
        # the entries that the projection brings to zero: are they exactly zero?
        for wk, rows, what in (("rnn.weight_hh_l0", Z_RES, "recurrent matrix, gate rows of the reserve, columns of the old cells"), ("lm.weight", ADDED, "lm, rows 256 to 511, columns of the old cells")):
            blk = W[wk][rows][:, :U0]
            print(f"      {what}: {blk.size} entries, of which exactly zero {int((blk == 0).sum())}, positive {int((blk > 0).sum())}, negative {int((blk < 0).sum())}; largest magnitude {np.abs(blk).max():.1e}")
        print(f"      the saved directions at the recurrent matrix: largest magnitude of an entry on the coordinates of the cells in reserve {np.abs(dirs['hh'][U0:U1]).max():.1e}; entries there that are exactly zero: {int((dirs['hh'][U0:U1] == 0).sum())} of {dirs['hh'][U0:U1].size}")
        # (c) the plain forward on the two panels
        ns, nk = V.Net(W), V.Net(Wk); ns32, nk32 = V.Net(W, np.float32), V.Net(Wk, np.float32)
        for label, its in (("the 1,200 held-out pairs at 12 letters", held), ("check 1's 2,000 items without letters", samp)):
            f = gate_figures(ns, its); f32 = gate_figures(ns32, its)
            d = np.abs(V.answer_logits(ns, its) - V.answer_logits(nk, its, in_use=in_use)).max(); d32 = np.abs(V.answer_logits(ns32, its) - V.answer_logits(nk32, its, in_use=in_use)).max()
            print(f" 1(c) {label}: largest 1 - z of a cell in reserve {f['largest_opening']:.3e} (single precision {f32['largest_opening']:.3e}; limit 1e-4); largest magnitude of a state of a cell in reserve "
                  f"{f['largest_state']:.2e} (limit 0.01); largest magnitude of an added readout unit at the answer positions {f['largest_added_unit']:.2e} (limit 0.01); "
                  f"largest difference of an output at an answer position from the keyed start under the keyed forward {d:.2e} (single precision {d32:.2e}; limit 0.02)")
        # the same check on a start in which the projection was not made (10 added to the bias only): can 1(c) fail?
        Wn = {k: v.copy() for k, v in Wk.items()}; Wn["rnn.bias_ih_l0"][Z_RES] += np.float32(10.0); nn_ = V.Net(Wn)
        for label, its in (("the 1,200 held-out pairs at 12 letters", held),):
            f = gate_figures(nn_, its); d = np.abs(V.answer_logits(nn_, its) - V.answer_logits(nk, its, in_use=in_use)).max()
            print(f"      a start with 10 added to the bias and no projection, {label}: largest 1 - z {f['largest_opening']:.3e}; largest state {f['largest_state']:.2e}; largest added unit {f['largest_added_unit']:.2e}; largest difference of an output {d:.2e}")
        # (d) the generated answers
        gs = V.score_first_family(ns, S1, HELD)[1]; gk = V.score_first_family(nk, S1, HELD, in_use=in_use)[1]
        print(f" 1(d) generated answers of the first family that differ from the keyed start's under the keyed forward: {sum(a != b for n in gs for a, b in zip(gs[n], gk[n]))} of {sum(len(v) for v in gs.values())} (limit 5); "
              f"by panel {({n: sum(a != b for a, b in zip(gs[n], gk[n])) for n in gs})}")
        gn = V.score_first_family(nn_, S1, HELD)[1]
        print(f"      the start without the projection: {sum(a != b for n in gn for a, b in zip(gn[n], gk[n]))} of 32,884 differ")
        # what the reset and candidate rows of the cells in reserve receive on first-family items; the same on the new tasks' items
        for label, its in (("the first family's held-out pairs at 12 letters", held), ("the never-taught items of `unrelated`", reserve_panel("unrelated")), ("the never-taught items of `conflict`", reserve_panel("conflict"))):
            n_, r_, op, hres, az = [], [], [], [], []
            for idx, x in rows_of(its):
                c = forward(W, x)
                n_.append(np.stack(c["n"], 1)[:, :, U0:]); r_.append(np.stack(c["r"], 1)[:, :, U0:]); op.append(1 - np.stack(c["z"], 1)[:, :, U0:]); hres.append(c["Hs"][:, :, U0:])
                az.append(c["az"][:, :, U0:])
                chk = np.abs(c["logits"][:, -3:] - ns.run(x)[0]).max(); assert chk < 1e-9, chk          # this forward against the verification script's
            n_, r_, op, hres, az = (np.concatenate([a.reshape(-1, U1 - U0) for a in v]) for v in (n_, r_, op, hres, az))
            print(f" on {label}: candidate of a cell in reserve: mean magnitude {np.abs(n_).mean():.3f}, largest {np.abs(n_).max():.3f}; reset gate: {r_.min():.3f} to {r_.max():.3f}; "
                  f"1 - z: mean {op.mean():.3e}, largest {op.max():.3e}; what the gate receives from the input map: {az.min():.4f} to {az.max():.4f}; largest state {np.abs(hres).max():.2e}")
        # the room that the rule leaves at the gate rows: the part of the code of each byte of the new tasks outside the directions in use
        code1 = np.concatenate([ns.code, np.ones((258, 1))], 1)
        for label, bytes_ in (("the four code bytes of `conflict` (r, t, x, n)", [ord(c) for c in "rtxn"]), ("the 98 bytes of `unrelated`", list(range(128, 226)))):
            c = code1[bytes_]; res = away(c, dirs["ih"]); l2n, l1n = np.linalg.norm(res, axis=1), np.abs(res).sum(1)
            res2 = away(c, dirs["l2"])
            # the least number of updates in which steps of at most 0.001 a weight move the gate's summed input on this byte by 8 (from 10 to 2, where 1 - z is 0.12)
            print(f" input map, {label}: 2-norm of the code with its constant {np.median(np.linalg.norm(c, axis=1)):.2f} (median); of its part outside the 46 directions in use: "
                  f"{l2n.min():.2f} to {l2n.max():.2f} (median {np.median(l2n):.2f}); sum of magnitudes of that part {l1n.min():.1f} to {l1n.max():.1f} (median {np.median(l1n):.1f}); "
                  f"updates needed to move a gate's summed input by 8 with steps of at most 0.001 a weight: at least {8 / (1e-3 * l1n.max()):.0f} (the byte with the largest part) to {8 / (1e-3 * l1n.min()):.0f}; "
                  f"what the projected gate rows give on these bytes beside the 10: {np.abs((W['rnn.weight_ih_l0'][Z_RES].astype(np.float64) @ ns.code[bytes_].T).T + W['rnn.bias_ih_l0'][Z_RES] - 10).max():.3f} at most "
                  f"(in the keyed start, without the projection: {np.abs((Wk['rnn.weight_ih_l0'][Z_RES].astype(np.float64) @ nk.code[bytes_].T).T + Wk['rnn.bias_ih_l0'][Z_RES]).max():.3f}); "
                  f"l2: part of the code outside the 11 directions in use, 2-norm {np.linalg.norm(res2, axis=1).min():.2f} to {np.linalg.norm(res2, axis=1).max():.2f}")
        # the weights of the gate rows from the cells in reserve: what they would have to be to hold a gate open at a position that presents a byte in use
        g = W["rnn.weight_hh_l0"][Z_RES][:, U0:].astype(np.float64)
        print(f" recurrent matrix, gate rows of the reserve, columns of the cells in reserve: sum of magnitudes of a row {np.abs(g).sum(1).min():.2f} to {np.abs(g).sum(1).max():.2f} (a gate is held at 1 - z = 0.12 "
              f"at a byte in use only if these weights times the states of the cells in reserve sum to -8); with steps of at most 0.001 a weight and states of magnitude 1 in all 128 cells this needs at least {8 / (1e-3 * 128):.0f} updates, "
              f"with states of magnitude 1 in 16 cells at least {8 / (1e-3 * 16):.0f}")


PARTS["start"] = start


# ================================================================ the gradient, by a backward pass written here
def loss_and_grad(W, x, y, mask, local, want_grad=True):
    """The loss of the runner's `step` (mean cross-entropy over the answer positions, credit through the whole item) and its gradient.
    local=False: the exact gradient (rule B). local=True: the sweep (rule S): through each linear map the error goes back by the signs
    of the map's weights times the mean magnitude of those weights (exp3_state.SignLinear), everything else as in rule B."""
    c = forward(W, x); P = c["P"]; B, L = x.shape; U = P["rnn.weight_hh_l0"].shape[1]
    lg = c["logits"][mask]; lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    N = int(mask.sum()); loss = float(-np.log(p[np.arange(N), y[mask]]).mean())
    if not want_grad:
        return loss, None, c
    Sg = (lambda Wm: np.sign(Wm) * np.abs(Wm).mean()) if local else (lambda Wm: Wm)
    d = p.copy(); d[np.arange(N), y[mask]] -= 1; d /= N
    dlog = np.zeros_like(c["logits"]); dlog[mask] = d
    G = {}; z1, z2, Hs = c["z1"], c["z2"], c["Hs"]; fl = lambda a: a.reshape(-1, a.shape[-1]); mm = c["mm"]
    G["l3.weight"] = fl(dlog).T @ fl(z2); G["l3.bias"] = fl(dlog).sum(0)
    da2 = mm(dlog, Sg(P["l3.weight"])) * (1 - z2 ** 2)
    G["l2.weight"] = fl(da2).T @ fl(z1); G["l2.bias"] = fl(da2).sum(0); G["lm.weight"] = fl(da2).T @ fl(Hs); G["lm.bias"] = fl(da2).sum(0)
    dz1 = mm(da2, Sg(P["l2.weight"])); dHs = mm(da2, Sg(P["lm.weight"]))
    Shh = Sg(P["rnn.weight_hh_l0"]); Ghh = np.zeros_like(P["rnn.weight_hh_l0"]); Gbhh = np.zeros(3 * U); dgi = np.zeros((B, L, 3 * U)); nxt = np.zeros((B, U))
    daz_all = np.zeros((B, L, U)); dh_all = np.zeros((B, L, U))
    for t in range(L - 1, -1, -1):
        dh = dHs[:, t] + nxt; z, n, r, hp, ghn = c["z"][t], c["n"][t], c["r"][t], c["hp"][t], c["ghn"][t]
        dan = dh * (1 - z) * (1 - n ** 2); dar = dan * ghn * r * (1 - r); daz = dh * (hp - n) * z * (1 - z)
        dgi[:, t] = np.concatenate([dar, daz, dan], 1); dgh = np.concatenate([dar, daz, dan * r], 1)
        Ghh += dgh.T @ hp; Gbhh += dgh.sum(0); nxt = dh * z + dgh @ Shh
        daz_all[:, t] = daz; dh_all[:, t] = dh
    G["rnn.weight_hh_l0"], G["rnn.bias_hh_l0"] = Ghh, Gbhh
    G["rnn.weight_ih_l0"] = fl(dgi).T @ fl(z1); G["rnn.bias_ih_l0"] = fl(dgi).sum(0)
    da1 = (dz1 + mm(dgi, Sg(P["rnn.weight_ih_l0"]))) * (1 - z1 ** 2)
    G["l1.weight"] = fl(da1).T @ fl(c["u"]); G["l1.bias"] = fl(da1).sum(0)
    Gb = np.zeros_like(P["book.weight"]); np.add.at(Gb, x.reshape(-1), fl(mm(da1, Sg(P["l1.weight"])))); G["book.weight"] = Gb
    c.update(daz=daz_all, dh=dh_all)
    return loss, G, c


def stream_of(task):
    return V.u_stream() if task == "unrelated" else V.conflict_stream()


def batches(task):
    st = stream_of(task)
    while True:
        yield V.batch_tensor([next(st) for _ in range(64)])


MATSETS = {"keyed": {"ih": V.OLD, "hh": V.OLD, "lm": np.arange(H0), "l2": np.arange(H0)},
           "shut": {"ih": np.concatenate([V.OLD, Z_RES]), "hh": np.concatenate([V.OLD, Z_RES]), "lm": np.arange(H1), "l2": np.arange(H1)},
           "shut-gates-free": {"ih": V.OLD, "hh": V.OLD, "lm": np.arange(H1), "l2": np.arange(H1)}}


def grad():
    print("== grad: the gradient at the start `shut` on the first batch of 64 items of each new task's stream")
    for rule in V.RULES:
        W, Wk, dirs, in_use = shut_start(rule); local = rule == "S"
        for task in V.TASKS:
            x, y, mask = next(batches(task)); loss, G, c = loss_and_grad(W, x, y, mask, local)
            if rule == "B" and task == "unrelated":                      # the backward pass against finite differences (double precision)
                rng = np.random.RandomState(0); worst = 0.0
                for k, pick in (("rnn.weight_ih_l0", Z_RES), ("rnn.weight_ih_l0", V.RES[-128:]), ("rnn.weight_hh_l0", V.OLD[:128]), ("lm.weight", ADDED), ("l2.weight", np.arange(H0)), ("l3.weight", np.arange(128, 192)), ("book.weight", np.arange(128, 226)), ("rnn.bias_ih_l0", Z_RES)):
                    for _ in range(3):
                        i = int(rng.choice(pick)); idx = (i,) if W[k].ndim == 1 else (i, int(rng.randint(W[k].shape[1])))
                        e = 1e-4 * max(1.0, abs(float(W[k][idx]))); Wp = {n: v.astype(np.float64) for n, v in W.items()}; Wm_ = {n: v.astype(np.float64) for n, v in W.items()}
                        Wp[k][idx] += e; Wm_[k][idx] -= e
                        fd = (loss_and_grad(Wp, x, y, mask, False, False)[0] - loss_and_grad(Wm_, x, y, mask, False, False)[0]) / (2 * e)
                        worst = max(worst, abs(fd - G[k][idx]) / max(abs(fd), abs(G[k][idx]), 1e-12))
                print(f" the backward pass against central finite differences of the loss, 24 entries of eight tensors (the gate rows of the reserve among them): largest relative difference {worst:.1e}")
            print(f"--- rule {rule}, {task}: loss {loss:.3f}; items of {x.shape[1]} positions")
            q = lambda a: f"median {np.median(np.abs(a)):.1e}, largest {np.abs(a).max():.1e}, share above 1e-8 {100 * (np.abs(a) > 1e-8).mean():.0f}%, median of |g| / (|g| + 1e-8) {np.median(np.abs(a) / (np.abs(a) + 1e-8)):.2f}"
            for m, wk, bk, _ in BLOCKS[:2]:
                Gm = np.concatenate([G[wk], G[bk][:, None]], 1)
                for label, rows, proj in (("gate rows of the reserve, as computed", Z_RES, False), ("gate rows of the reserve, projected away from the directions in use (the rule's first addition)", Z_RES, True),
                                          ("candidate rows of the cells in reserve", np.arange(2 * U1 + U0, 3 * U1), False), ("reset rows of the cells in reserve", np.arange(U0, U1), False),
                                          ("old rows, projected", V.OLD, True)):
                    g = away(Gm[rows], dirs[m]) if proj else Gm[rows]
                    print(f"   {wk} {label}: magnitude of an entry of the gradient: {q(g)}")
            for k, sl, label in (("lm.weight", (slice(None), slice(U0, U1)), "lm, columns of the cells in reserve"), ("rnn.weight_hh_l0", (V.OLD, slice(U0, U1)), "recurrent matrix, old rows, columns of the cells in reserve"),
                                 ("l3.weight", (slice(None), slice(H0, H1)), "l3, columns 256 to 511"), ("book.weight", (slice(128, 226) if task == "unrelated" else [ord(ch) for ch in "rtxn"], slice(None)), "byte table, rows of the new task's bytes")):
                print(f"   {label}: {q(G[k][sl])}")
            # the error at the gate itself, and the first step of the optimizer at the gate rows of the input map, as a change of the gate's summed input on the new task's bytes
            print(f"   error arriving at the state of a cell in reserve (dL/dh): median magnitude {np.median(np.abs(c['dh'][:, :, U0:])):.1e}; at its gate's summed input: {np.median(np.abs(c['daz'][:, :, U0:])):.1e} (the same for an old cell: {np.median(np.abs(c['dh'][:, :, :U0])):.1e} and {np.median(np.abs(c['daz'][:, :, :U0])):.1e})")
            Gm = away(np.concatenate([G["rnn.weight_ih_l0"], G["rnn.bias_ih_l0"][:, None]], 1)[Z_RES], dirs["ih"])
            stepm = away(-1e-3 * Gm / (np.abs(Gm) + 1e-8), dirs["ih"])                                 # the first step of AdamW (m = g, v = g * g after the correction), then the rule's second addition
            net = V.Net(W); newb = list(range(128, 226)) if task == "unrelated" else [ord(ch) for ch in "rtxn"]
            sh = stepm @ np.concatenate([net.code[newb], np.ones((len(newb), 1))], 1).T               # cells x bytes: the change of the summed input of the gate
            l1n = np.abs(away(np.concatenate([net.code[newb], np.ones((len(newb), 1))], 1), dirs["ih"])).sum(1)
            print(f"   first step of the optimizer at the gate rows of the input map, after the rule: change of a gate's summed input on a byte of the new task: mean {sh.mean():+.1e}, mean magnitude {np.abs(sh).mean():.1e}, "
                  f"largest magnitude {np.abs(sh).max():.1e} (the bound for steps of 0.001 a weight is {1e-3 * l1n.max():.1e}); on the bytes in use: {np.abs(stepm @ np.concatenate([net.code[in_use], np.ones((int(in_use.sum()), 1))], 1).T).max():.1e}")


PARTS["grad"] = grad


# ================================================================ an emulation of the first updates (the checker's own arithmetic; not the runner)
MATNAMES = {"ih": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"), "hh": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"), "lm": ("lm.weight", "lm.bias"), "l2": ("l2.weight", "l2.bias")}


def emulate_updates(W, task, rule, rowsets, dirs, in_use, updates, at=(), measure=None, scale_gate_rows=1.0, zero_blocks=False):
    """The runner's update as read from exp51_reserve.step and Rule (and the brief's Rule55): the gradient (rule B exact, rule S the sweep),
    the projection of the gradient of the chosen rows, AdamW (rate 0.001, betas 0.9 and 0.98, eps 1e-8, no decay), the projection of the
    step, the restoring of l1, of the old columns and the bias of l3 and of the rows of the byte table of the bytes in use. The weights are
    stored in single precision after every step; the arithmetic between is double precision (torch's is single: the two differ in rounding).
    scale_gate_rows multiplies the gradient of the gate rows of the reserve before the optimizer (1.0: the brief). zero_blocks sets the
    entries that the projection brings to zero (weights from old cells into the gates of the reserve and into rows 256 to 511 of lm) to
    exactly zero after each step."""
    W = {k: v.copy() for k, v in W.items()}; m = {k: np.zeros(v.shape) for k, v in W.items()}; v_ = {k: np.zeros(v.shape) for k, v in W.items()}
    bt = batches(task); local = rule == "S"; out = []
    if measure and 0 in at:
        out.append(measure(0, W, None)); print(out[-1], flush=True)
    for g in range(1, updates + 1):
        x, y, mask = next(bt); loss, G, _ = loss_and_grad(W, x, y, mask, local)
        G = {k: a.astype(np.float32) for k, a in G.items()}; G["book.weight"][BOS] = 0          # the byte table's row of the start symbol takes no gradient (padding_idx)
        for mn, rows in rowsets.items():
            wk, bk = MATNAMES[mn]; Gm = np.concatenate([G[wk], G[bk][:, None]], 1).astype(np.float64); Gm[rows] = away(Gm[rows], dirs[mn])
            if mn in ("ih", "hh") and scale_gate_rows != 1.0:
                Gm[Z_RES] *= scale_gate_rows
            G[wk] = Gm[:, :-1].astype(np.float32); G[bk] = Gm[:, -1].astype(np.float32)
        snap = {k: W[k].copy() for k in W}
        for k in W:
            gk = G[k].astype(np.float64); m[k] = 0.9 * m[k] + 0.1 * gk; v_[k] = 0.98 * v_[k] + 0.02 * gk * gk
            W[k] = (W[k] - 1e-3 * (m[k] / (1 - 0.9 ** g)) / (np.sqrt(v_[k] / (1 - 0.98 ** g)) + 1e-8)).astype(np.float32)
        for mn, rows in rowsets.items():
            wk, bk = MATNAMES[mn]; S = np.concatenate([W[wk].astype(np.float64) - snap[wk], (W[bk].astype(np.float64) - snap[bk])[:, None]], 1); Sr = away(S[rows], dirs[mn])
            W[wk][rows] = (snap[wk][rows].astype(np.float64) + Sr[:, :-1]).astype(np.float32); W[bk][rows] = (snap[bk][rows].astype(np.float64) + Sr[:, -1]).astype(np.float32)
        if rowsets:
            for k in ("l1.weight", "l1.bias", "l3.bias"):
                W[k] = snap[k]
            W["l3.weight"][:, :H0] = snap["l3.weight"][:, :H0]; W["book.weight"][in_use] = snap["book.weight"][in_use]
        if zero_blocks:
            W["rnn.weight_hh_l0"][np.ix_(Z_RES, np.arange(U0))] = 0.0; W["lm.weight"][np.ix_(ADDED, np.arange(U0))] = 0.0
        if measure and g in at:
            out.append(measure(g, W, loss)); print(out[-1], flush=True)
    return W, out


def emulate():
    print("== emulate: the first updates, in the checker's own arithmetic")
    S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]
    runs = [tuple(r.split(":")) for r in os.environ.get("E55_CHECK_RUNS", "unrelated:S,conflict:S,unrelated:B,conflict:B").split(",")]
    # 1. is the emulation the runner's update? E51's arm `keyed`, 64 updates on `unrelated`, against the saved weights of row 64
    for rule in (V.RULES if os.environ.get("E55_CHECK_VALIDATE", "1") == "1" else ()):
        Wk = V.start("keyed", rule); dirs, in_use, _, _ = V.saved_directions(rule); in_use = in_use.astype(bool)
        We, _ = emulate_updates(Wk, "unrelated", rule, MATSETS["keyed"], dirs, in_use, 64)
        Ws, _ = V.weights("unrelated", "keyed", rule, 64)
        print(f" validation, E51 `keyed` on `unrelated`, rule {rule}, 64 updates: largest difference of a weight from the saved weights of row 64: "
              + "; ".join(f"{k} {np.abs(We[k].astype(np.float64) - Ws[k]).max():.1e} (change from the start {np.abs(Ws[k].astype(np.float64) - Wk[k]).max():.1e})" for k in sorted(We)))
    # 2. the arm `shut`
    n_up = int(os.environ.get("E55_CHECK_UPDATES", "512")); at = [r for r in ROWS if r <= n_up]
    variants = os.environ.get("E55_CHECK_VARIANTS", "brief").split(",")
    for rule in V.RULES:
        W0, Wk, dirs, in_use = shut_start(rule); gk = None
        for task in V.TASKS:
            if (task, rule) not in runs:
                continue
            pan = reserve_panel(task); panels = V.new_panels(task)
            for variant in variants:
                def measure(g, W, loss):
                    net = V.Net(W); f = gate_figures(net, pan); ff = gate_figures(net, held)
                    peak = np.zeros(U1 - U0); n = 0
                    for idx, x in rows_of(pan):
                        op = 1 - net.run(x, keep=True)[1]["z"][:, :, U0:]; peak += op.max(1).sum(0); n += len(idx)
                    s3 = V.score_new(net, panels); s3s = V.score_new(net, panels, shut_all=True)
                    nt = lambda s: 100 * sum(s[k]["exact"] for k in s if k != "taught") / sum(s[k]["n"] for k in s if k != "taught")
                    gh = V.generate(net, held)
                    nonlocal_ref.setdefault("g0", gh); every = ""
                    if g == 0 or g >= 256:                       # all 32,884 answers: the reduction score and the answers that differ from row 0's
                        res, gens = V.score_first_family(net, S1, HELD); nonlocal_ref.setdefault("gens0", gens); g0 = nonlocal_ref["gens0"]
                        every = (f"; reduction score {res['reduction_score']:.2f}; of the 31,684 answers without letters {sum(a != b for n in gens if n.startswith('k0_') for a, b in zip(gens[n], g0[n]))} differ from row 0's")
                    return (f" row {g:5d}: loss {'-' if loss is None else format(loss, '.3f')}; never-taught {nt(s3):5.1f}, with the cells in reserve shut {nt(s3s):5.1f}; on the new task's items: cells with mean opening above 0.1: "
                            f"{int((f['mean'] > 0.1).sum())}, cells whose mean over items of the largest 1 - z of an item exceeds 0.5: {int((peak / n > 0.5).sum())} (exceeds 0.1: {int((peak / n > 0.1).sum())}), largest 1 - z {f['largest_opening']:.2e}, largest state {f['largest_state']:.2e}; "
                            f"on the first family's held-out items: largest 1 - z {ff['largest_opening']:.2e}, largest state {ff['largest_state']:.2e}, largest added unit {ff['largest_added_unit']:.2e}, "
                            f"answers that differ from row 0's {sum(a != b for a, b in zip(gh, nonlocal_ref['g0']))} of 1,200" + every)
                nonlocal_ref = {}
                arm = "shut-gates-free" if variant == "gates-free" else "shut"
                t0 = time.time()
                print(f"--- emulation, {task}, rule {rule}, arm `{arm}`" + ("" if variant in ("brief", "gates-free") else f", variant {variant}") + f", {n_up} updates", flush=True)
                Wend, out = emulate_updates(W0, task, rule, MATSETS[arm], dirs, in_use, n_up, at, measure, scale_gate_rows=1e4 if variant == "scaled" else 1.0, zero_blocks=variant == "zeroed")
                if os.environ.get("E55_CHECK_SAVE"):             # the emulated weights after the last update, for further figures (a scratch folder)
                    np.savez(os.path.join(os.environ["E55_CHECK_SAVE"], f"emulated_{task}_{rule}_{arm}_{variant}_{n_up}.npz"), **Wend)
                print(f"    ({time.time() - t0:.0f} s)", flush=True)


PARTS["emulate"] = emulate


def check2():
    print("== check2: the brief's check 2 in the emulation: eight updates from the start `shut`")
    S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]; samp = V.check1_sample(S1)
    for rule in V.RULES:
        W0, Wk, dirs, in_use = shut_start(rule); n0 = V.Net(W0); a0 = {n: V.answer_logits(n0, its) for n, its in (("held", held), ("sample", samp))}
        for task in V.TASKS:
            newb = list(range(128, 226)) if task == "unrelated" else [ord(ch) for ch in "rtxn"]
            for arm in ("shut", "shut-gates-free"):
                W, _ = emulate_updates(W0, task, rule, MATSETS[arm], dirs, in_use, 8); net = V.Net(W)
                dv = {}
                for mn, (wk, bk) in MATNAMES.items():
                    Dd = np.concatenate([W[wk].astype(np.float64) - W0[wk], (W[bk].astype(np.float64) - W0[bk])[:, None]], 1)
                    dv[mn] = (np.abs(Dd[MATSETS[arm][mn]] @ dirs[mn]).max(), np.abs(Dd[Z_RES] @ dirs[mn]).max() if mn in ("ih", "hh") else None, bool(np.abs(Dd).max() > 0))
                f = {n: gate_figures(net, its) for n, its in (("held", held), ("sample", samp))}
                d = {n: np.abs(V.answer_logits(net, its) - a0[n]).max() for n, its in (("held", held), ("sample", samp))}
                code1 = np.concatenate([net.code[newb], np.ones((len(newb), 1))], 1); code0 = np.concatenate([n0.code[newb], np.ones((len(newb), 1))], 1)
                gate = lambda Wx, cd: np.concatenate([Wx["rnn.weight_ih_l0"][Z_RES].astype(np.float64), Wx["rnn.bias_ih_l0"][Z_RES][:, None].astype(np.float64)], 1) @ cd.T
                mv = gate(W, code1) - gate(W0, code0)
                print(f" rule {rule}, {task}, arm `{arm}`: (a) largest entry of (D_after - D_start) @ Vm at the projected rows: " + ", ".join(f"{mn} {dv[mn][0]:.1e}" for mn in dv) + f" (limit 1e-5); every matrix changed: {all(dv[mn][2] for mn in dv)}; "
                      f"(b) largest 1 - z of a cell in reserve on the two panels {max(f[n]['largest_opening'] for n in f):.3e} (limit 1e-4); largest change of an output at an answer position: held-out {d['held']:.2e}, 2,000 items {d['sample']:.2e} (limit 1e-3); "
                      f"largest state of a cell in reserve {max(f[n]['largest_state'] for n in f):.2e}; largest added unit at the answer positions {max(f[n]['largest_added_unit'] for n in f):.2e}; "
                      f"(c) largest entry of (D_after - D_start) @ Vm at the gate rows of the reserve: input map {dv['ih'][1]:.1e}, recurrent matrix {dv['hh'][1]:.1e} (must exceed 1e-4 in `shut-gates-free`); "
                      f"change of a gate's summed input on a byte of the new task after the eight updates: mean {mv.mean():+.4f}, most negative {mv.min():+.4f}, most positive {mv.max():+.4f}")


PARTS["check2"] = check2


def residue():
    print("== residue: rule S after eight updates of the arm `shut`: the error that the sweep sends back through the entries that the projection brought to zero")
    for task in V.TASKS:
        W0, Wk, dirs, in_use = shut_start("S"); W, _ = emulate_updates(W0, task, "S", MATSETS["shut"], dirs, in_use, 8)
        bt = batches(task)
        for _ in range(9):
            x, y, mask = next(bt)
        Wz = {k: v.copy() for k, v in W.items()}; Wz["rnn.weight_hh_l0"][np.ix_(Z_RES, np.arange(U0))] = 0.0; Wz["lm.weight"][np.ix_(ADDED, np.arange(U0))] = 0.0
        l1_, G1, c1 = loss_and_grad(W, x, y, mask, True); l0_, G0, c0 = loss_and_grad(Wz, x, y, mask, True)
        blk = W["lm.weight"][ADDED][:, :U0]
        old = lambda G: away(np.concatenate([G["rnn.weight_ih_l0"], G["rnn.bias_ih_l0"][:, None]], 1)[V.OLD], dirs["ih"])
        e1, e0 = c1["dh"][:, :, :U0], c0["dh"][:, :, :U0]
        print(f" {task}: lm, rows 256 to 511, columns of the old cells after the eight updates: largest magnitude {np.abs(blk).max():.1e}, non-zero {int((blk != 0).sum())} of {blk.size}; loss with these entries as stored {l1_:.6f} and set to zero {l0_:.6f}; "
              f"error arriving at the states of the old cells: 2-norm {np.linalg.norm(e0):.3e} with the entries at zero, 2-norm of the difference made by the signs of the remainders {np.linalg.norm(e1 - e0):.3e} "
              f"({100 * np.linalg.norm(e1 - e0) / np.linalg.norm(e0):.0f}% of it); the gradient of the old rows of the input map after the rule's projection: 2-norm {np.linalg.norm(old(G0)):.3e}, difference {np.linalg.norm(old(G1) - old(G0)):.3e} "
              f"({100 * np.linalg.norm(old(G1) - old(G0)) / np.linalg.norm(old(G0)):.0f}%), cosine between the two {float((old(G1) * old(G0)).sum() / (np.linalg.norm(old(G1)) * np.linalg.norm(old(G0)))):.3f}")


PARTS["residue"] = residue

def positions():
    print("== positions: the emulated run of `conflict` under rule B at row 4,096 (weights written by the part `emulate` with E55_CHECK_SAVE): where the gates open")
    W = dict(np.load(os.path.join(os.environ["E55_CHECK_SAVE"], "emulated_conflict_B_shut_brief_4096.npz"))); net = V.Net(W); pan = reserve_panel("conflict")
    for idx, x in rows_of(pan):
        op = 1 - net.run(x, keep=True)[1]["z"][:, :, U0:]; cells = np.nonzero(op.max(1).mean(0) > 0.1)[0]
        print(f" cells for which the mean over the {len(idx)} items of the largest 1 - z over the positions exceeds 0.1: {len(cells)}")
        print(f" mean over the items and these cells of 1 - z, position by position: {np.round(op[:, :, cells].mean((0, 2)), 3).tolist()}")
        print(f" pairs of an item and a cell in reserve with 1 - z above 0.5, position by position: {(op > 0.5).sum((0, 2)).tolist()}")
        print(f" the bytes of the first item: {[chr(b) if b < 128 else int(b) for b in x[0]]}")


PARTS["positions"] = positions


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t0 = time.time(); PARTS[p](); print(f"   [{p}: {time.time() - t0:.0f} s]", flush=True)
