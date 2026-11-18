#!/usr/bin/env python3
"""Check of the brief of E55, first part (revision 1): the figures of the document CHECK_OF_TASK_E55_BRIEF_20261118.md.

Read-only: every file of another branch is read with `git show` into memory; nothing is written, nothing is trained, torch is not used and
no code of another branch is run. The forward pass, the items, the panels and the reader of weight files are those of the reviewer's
verification of E51's trial (PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e51_trial_verify.py), imported as `V`.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e55_brief.py <part> [...]

Parts:
    record   the figures of the brief's section "What the record says", from the sixteen receipts of E51 on the worker's branch
    open     E51's arm `open`, from its weights of row 4,096: the update gates of the cells in reserve position by position
    start    the start `shut` written in numpy from the saved keyed start and the saved directions: the figures of the brief's check 1
             (b), (c) and (d); what the reset and candidate rows of the cells in reserve receive; the room that the rule leaves
    grad     the exact gradient of the loss at the start `shut` on one batch of each new task, by a hand-written backward pass in
             numpy (double precision, verified against finite differences): the size of the gradient at the gate rows and elsewhere
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
V.WREF = "origin/claude/e51-reserve-017dSEpW"          # the worker's branch as it stands (the verification pins an earlier commit)

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
                nk0 = sum(r_["n"] if "n" in r_ else 0 for r_ in [])  # not used
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
    by_pos = np.zeros(U1 - U0)
    for idx, x in rows_of(items):
        k = net.run(x, keep=True, **key)[1]; op = 1 - k["z"][:, :, U0:]
        tot += op.sum((0, 1)); n += op.shape[0] * op.shape[1]
        by_pos = np.maximum(by_pos, op.mean(0).max(0)); big_open = max(big_open, float(op.max()))
        big_h = max(big_h, float(np.abs(k["h"][:, :, U0:]).max())); big_z2 = max(big_z2, float(np.abs(k["z2"][:, :, H0:]).max()))
    return dict(mean=tot / n, by_pos=by_pos, largest_opening=big_open, largest_state=big_h, largest_added_unit=big_z2)


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
                      + f"; cells whose mean over items of 1 - z exceeds 0.1 at some position: {int((f['by_pos'] > 0.1).sum())}, exceeds 0.5 at some position: {int((f['by_pos'] > 0.5).sum())}; "
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
    u = P["book.weight"][x]; z1 = np.tanh(u @ P["l1.weight"].T + P["l1.bias"])
    gi = z1 @ P["rnn.weight_ih_l0"].T + P["rnn.bias_ih_l0"]
    h = np.zeros((B, U)); c = dict(P=P, x=x, u=u, z1=z1, hp=[], r=[], z=[], n=[], ghn=[], h=[])
    for t in range(L):
        gh = h @ P["rnn.weight_hh_l0"].T + P["rnn.bias_hh_l0"]
        r = V.sigmoid(gi[:, t, :U] + gh[:, :U]); z = V.sigmoid(gi[:, t, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        c["hp"].append(h); c["r"].append(r); c["z"].append(z); c["n"].append(n); c["ghn"].append(gh[:, 2 * U:])
        h = (1 - z) * n + z * h; c["h"].append(h)
    Hs = np.stack(c["h"], 1)
    z2 = np.tanh(z1 @ P["l2.weight"].T + P["l2.bias"] + Hs @ P["lm.weight"].T + P["lm.bias"])
    c.update(Hs=Hs, z2=z2, logits=z2 @ P["l3.weight"].T + P["l3.bias"], az=gi[:, :, U:2 * U])
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

if __name__ == "__main__":
    for p in sys.argv[1:]:
        t0 = time.time(); PARTS[p](); print(f"   [{p}: {time.time() - t0:.0f} s]", flush=True)
