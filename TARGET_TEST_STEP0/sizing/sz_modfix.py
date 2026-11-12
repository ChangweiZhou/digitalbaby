"""Sizing sweep, stage 1: modulus fixtures and the configuration-reach admission (brief rev. 3, "What is varied").

For addition (family 1) and subtraction (family 2) at each of 97, 31 and 11:
  * positive: a pair whose sum or difference wraps round the modulus is answered correctly by the generator (the reveal it writes), the
    oracle, the prefix-only solver (once from episodes, through elimination, once from disclosure lines), the checker's wiring (the oracle
    and solver it builds from a bundle) and the sweep's direct-use responder (sz_score.direct_use_answer);
  * negative: the same component deliberately left at another modulus gets the wraparound case wrong and admission refuses it, and a
    component handed a configuration other than the cell's is refused by its configuration hash before anything is computed.
The expected answers are computed here with plain arithmetic from the fixture's own values; nothing is read from the components under test.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_check as CK      # noqa: E402
import sz_config as CF     # noqa: E402
import sz_oracle as OR     # noqa: E402
import sz_score as SC      # noqa: E402
import sz_solver as SV     # noqa: E402
import sz_stream as G      # noqa: E402

OTHER = {97: 31, 31: 11, 11: 97}          # where a component is "left" in the negative fixture


def tiny_world(M, fam):
    """Contexts A, B; cues 0 (a or m) and 1 (b or n). (B, cue 1) wraps: v(B) + u(1) = M + 2 for addition, v(B) - u(1) = -3 for subtraction."""
    if fam == 0:
        v = {0: 1, 1: M - 3}; u = {0: 1, 1: 5}
        expect = (M - 3 + 5) - M                       # 2: the sum passes M
    else:
        v = {0: 4, 1: 2}; u = {0: 3, 1: 5}
        expect = (2 - 5) + M                           # M - 3: the difference passes 0
    # elimination (B, 1) = (B, 0) + (A, 1) - (A, 0): the unreduced sum is M + 2 (addition) or 2M - 3 (subtraction), so the elimination
    # itself passes M and a solver at another modulus cannot land on the answer by accident
    return v, u, expect


def full_world(cfg, fam, v, u):
    NCTX, NCUE = cfg["contexts"], cfg["cues_per_family"]
    V = [[0] * NCTX for _ in range(2)]; U = [[0] * NCUE for _ in range(2)]
    for x, a in v.items():
        V[fam][x] = a
    for c, a in u.items():
        U[fam][c] = a
    return dict(v=V, u=U, u_new={}, changed=[])


def outcome(M, fam, a, b):
    return (a + b) % M if fam == 0 else (a - b) % M


def episodes_bytes(cfg, fam, v, u, M):
    """(A, cue 0), (B, cue 0), (A, cue 1): enough to fix (B, cue 1) by elimination. The reveals are written with plain arithmetic here."""
    out = b""
    for x, c in ((0, 0), (1, 0), (0, 1)):
        out += bytes([65 + x]) + b"!#" + bytes([G.cue_byte(cfg, fam, c)]) + b"=" + b"%02d" % outcome(M, fam, v[x], u[c]) + b"\n"
    return out


def lines_bytes(cfg, fam, v, u):
    return bytes([65 + 1]) + b":%02d " % v[1] + bytes([G.cue_byte(cfg, fam, 1)]) + b":%02d" % u[1] + b"\n"


def component_answers(cell_cfg, comp_cfg, fam):
    """Each component answers the wraparound pair (B, cue 1) of family `fam`; comp_cfg is the configuration the components are given."""
    M = cell_cfg["modulus"]
    v, u, expect = tiny_world(M, fam)
    W = full_world(comp_cfg, fam, v, u)
    a = {}
    a["generator_true_outcome"] = G.true_outcome(comp_cfg, W, fam, 1, 1, 0)
    e = dict(x=1, f=fam, c=1, fill=b"!#", o=G.true_outcome(comp_cfg, W, fam, 1, 1, 0))
    a["generator_reveal_bytes"] = int(G.episode_bytes(comp_cfg, e)[-3:-1])
    a["oracle"] = OR.Oracle(comp_cfg, W["v"], W["u"], {}, 10 ** 9).answer(fam, 1, 1, 0)
    s = SV.PrefixSolver(comp_cfg, None); s.feed(episodes_bytes(cell_cfg, fam, v, u, M)); a["solver_from_episodes"] = s.query(fam, 1, 1)
    s = SV.PrefixSolver(comp_cfg, None); s.feed(lines_bytes(cell_cfg, fam, v, u)); a["solver_from_lines"] = s.query(fam, 1, 1)
    s = SV.PrefixSolver(comp_cfg, None); s.feed(episodes_bytes(cell_cfg, fam, v, u, M))
    a["solver_prompt"] = s.answer_prompt(bytes([66]) + b"!#" + bytes([G.cue_byte(cell_cfg, fam, 1)]) + b"=")
    # the checker's wiring: the oracle and the solver it builds from a bundle carry the bundle's configuration
    Bfake = dict(cfg=comp_cfg, world=W, meta=dict(change_at=0), offsets=dict(fx=dict(starts=[0], end=len(episodes_bytes(cell_cfg, fam, v, u, M)))),
                 streams=dict(fx=episodes_bytes(cell_cfg, fam, v, u, M)),
                 probes=[dict(at=1, kind="x")])
    a["checker_oracle"] = CK.oracle_for(Bfake).answer(fam, 1, 1, 0)
    got = []
    CK.walk(Bfake, "fx", lambda s_, p, e_: got.append(s_.query(fam, 1, 1)), probes=[dict(at=1)])
    a["checker_solver"] = got[0]
    a["scorer_direct_use"] = SC.direct_use_answer(comp_cfg, fam, v[1], u[1])
    return a, expect


def admit(cell_cfg, comp_cfgs=None):
    """Admission of one configuration: every component must carry the cell's configuration (hash) and answer every wraparound case right.
    comp_cfgs: {component: configuration it is actually handed} (default: the cell's own, for every component)."""
    want = CF.chash(cell_cfg); comp_cfgs = comp_cfgs or {}
    names = ("generator", "oracle", "solver", "checker", "scorer")
    hashes = {n: CF.chash(comp_cfgs.get(n, cell_cfg)) for n in names}
    # what each component itself reports having received
    v, u, _ = tiny_world(cell_cfg["modulus"], 0)
    reported = dict(oracle=OR.Oracle(comp_cfgs.get("oracle", cell_cfg), [[0]], [[0]], {}, 0).cfg_hash,
                    solver=SV.PrefixSolver(comp_cfgs.get("solver", cell_cfg)).cfg_hash,
                    generator=G.cfg_hash(comp_cfgs.get("generator", cell_cfg)))
    reach = {n: hashes[n] == want for n in names}
    reach.update({f"{n}_reports": reported[n] == want for n in reported})
    cases, wrong = [], []
    for fam in (0, 1):
        res = {}
        for n in names:
            a, expect = component_answers(cell_cfg, comp_cfgs.get(n, cell_cfg), fam)
            res.update({k: val for k, val in a.items() if k.startswith(n)})
        for k, val in res.items():
            if val != expect:
                wrong.append(dict(family=fam + 1, component=k, answered=val, expected=expect))
        cases.append(dict(family=fam + 1, operation="addition" if fam == 0 else "subtraction", expected=expect, answers=res))
    ok = all(reach.values()) and not wrong
    return dict(admitted=ok, modulus=cell_cfg["modulus"], cfg_hash=want, reach=reach, cases=cases, wrong=wrong)


def run():
    out = {}
    for M in CF.MODULI:
        cell = CF.make(M, 4)
        pos = admit(cell)
        neg = {}
        for comp in ("generator", "oracle", "solver", "checker", "scorer"):
            left = CF.make(OTHER[M], 4)
            r = admit(cell, {comp: left})
            neg[comp] = dict(left_at=OTHER[M], admitted=r["admitted"], refused_by_hash=not all(r["reach"].values()),
                             wrong_answers=[w for w in r["wrong"]])
        # the same, but with the configuration hash forged to the cell's: behaviour alone must still catch it
        beh = {}
        for comp in ("generator", "oracle", "solver", "checker", "scorer"):
            caught = []
            for fam in (0, 1):
                a, expect = component_answers(cell, CF.make(OTHER[M], 4), fam)
                for k, val in a.items():
                    if k.startswith(comp):
                        caught.append(val != expect)
            beh[comp] = dict(cases=len(caught), caught=sum(caught))
        out[str(M)] = dict(positive=pos, negative=neg, negative_by_behaviour_alone=beh,
                           passed=pos["admitted"] and all(not n["admitted"] for n in neg.values())
                           and all(b["caught"] == b["cases"] and b["cases"] > 0 for b in beh.values()))
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({k: (v["passed"] if isinstance(v, dict) else v) for k, v in r.items()}))
