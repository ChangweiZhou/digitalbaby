"""Sizing sweep, stage 1: the cost profile and the forecast for the 84 fits (brief rev. 3, "Running it").

World seed 104 only (in no roster). Nothing run here is a fit; every learner is discarded.
  * generation of the longest configuration (modulus 97, 256 presentations), all six of step 0's streams, and step 0's learner-free
    checks on it (other seed for check 8's fixture: 105, used for nothing else); the same checks at the other eight settings, for the
    admission preview and for their cost;
  * four paths at 256 presentations, the first 500 updates of each: arm A ordinary, arm B ordinary, arm B with lines (wider), arm B with
    matched extra bytes; then, on each learner as it stands, one full panel (twins and taught panel) and one first-encounter probe are
    answered and timed, and the learner-free references of the full stream are computed and timed;
  * sizes of every stream at every setting (bytes, target positions, updates from the actual schedule with its probe cuts).

    python3 sz_profile.py gen      generation and checks (writes runs/profile_gen.json)
    python3 sz_profile.py path P   one path, P in A_ordinary, B_ordinary, B_wider, B_extra (writes runs/profile_path_P.json)
    python3 sz_profile.py sizes    sizes of every setting (writes runs/profile_sizes.json)
"""
import json
import sys
import time
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_check as CK      # noqa: E402
import sz_config as CF     # noqa: E402
import sz_learner as L     # noqa: E402
import sz_panels as P      # noqa: E402
import sz_score as SC      # noqa: E402
import sz_stream as G      # noqa: E402

W = CF.PROFILE_WORLD
OTHER = 105
RUNS = HERE / "runs"
PATHS = dict(A_ordinary=("A", "ordinary"), B_ordinary=("B", "ordinary"), B_wider=("B", "wider"), B_extra=("B", "extra"))
N_UPDATES = 500


def cpu():
    return time.process_time()


def gen():
    out = {}
    for M, Pn in sorted(CF.settings(), key=lambda s: s[1]):
        cfg = CF.make(M, Pn)
        t0 = cpu(); B = G.build(cfg, W); t_gen = cpu() - t0
        t0 = cpu(); checks = CK.run_checks(B, OTHER); t_chk = cpu() - t0
        fails = [k for k, v in checks.items() if not v["passed"]]
        out[f"{M}/{Pn}"] = dict(modulus=M, presentations=Pn, cfg_hash=CF.chash(cfg), episodes=len(B["episodes"]),
                                bytes={k: len(v) for k, v in B["streams"].items()}, hashes=G.hashes(B)["streams"],
                                cpu_generation=round(t_gen, 2), cpu_checks=round(t_chk, 2), checks_failed=fails,
                                checks={k: dict(passed=v["passed"], condition_held=v["condition_held"], fixture_failed_as_required=v["fixture_failed_as_required"])
                                        for k, v in checks.items()},
                                twin_redraws=B["split"]["twin_redraws"], split_redraws=B["split"]["split_redraws"],
                                revision_placement_attempts=B["meta"]["placement_attempts"])
        print(M, Pn, "gen", round(t_gen, 1), "checks", round(t_chk, 1), "failed", fails, flush=True)
    return out


def path(name):
    T.set_num_threads(1)
    arm, stream = PATHS[name]
    cfg = CF.make(97, 256)
    t0 = cpu(); B = G.build(cfg, W, streams_wanted=(stream,)); t_gen = cpu() - t0
    taught, _ = P.load(W, 97)
    t0 = cpu(); ev, cuts, end = SC.plan(B, stream, arm, taught); t_plan = cpu() - t0
    if arm == "B":
        lr = L.StreamLearner(W + 1000, B["streams"][stream], cuts)
    else:
        lr = L.EpisodeLearner(W + 1000, [G.episode_bytes(cfg, e) for e in B["episodes"]], cuts)
    stop = lr.grid[N_UPDATES - 1]
    nxt = [e for e in ev if e["pos"] <= stop]
    for e in nxt:
        lr.advance_to(e["pos"]); lr.respond(e["prompts"])
    lr.advance_to(stop)
    assert lr.updates == N_UPDATES
    t_train = lr.cpu_train
    panel = next(e for e in ev if e["type"] == "panel" and e["name"] == "A")
    t0 = cpu(); lr.respond(panel["prompts"]); t_panel = cpu() - t0
    twin_only = next(e for e in ev if e["type"] == "panel" and e["name"] == "I")
    t0 = cpu(); lr.respond(twin_only["prompts"]); t_panel_I = cpu() - t0
    fe = next(e for e in ev if e["type"] == "first")
    t0 = cpu()
    for _ in range(20):
        lr.respond(fe["prompts"])
    t_first = (cpu() - t0) / 20
    t0 = cpu(); refs = SC.references(B, stream, arm, ev, grid=lr.grid if arm == "B" else None); t_refs = cpu() - t0
    t0 = cpu(); SC.readings(B, ev, [[b"00\n"] * len(e["prompts"]) for e in ev], refs, stream == "wider"); t_read = cpu() - t0
    n_first = sum(1 for e in ev if e["type"] == "first")
    tok = sum(n for _, n, _ in lr.loss_log)
    return dict(path=name, arm=arm, stream=stream, world=W, model_seed=W + 1000, configuration=dict(modulus=97, presentations=256),
                bytes=len(B["streams"][stream]), episodes=len(B["episodes"]), updates=lr.updates, targets_in_profiled_updates=tok,
                total_updates_of_the_fit=len(lr.grid), total_targets_of_the_fit=(len(B["streams"][stream]) if arm == "B" else
                                                                                  sum(len(G.episode_bytes(cfg, e)) for e in B["episodes"])),
                probe_cuts=len(lr.cuts), cpu_generation_one_stream=round(t_gen, 2), cpu_plan=round(t_plan, 2),
                cpu_train=round(t_train, 2), cpu_per_update=t_train / lr.updates, cpu_per_target=t_train / tok,
                cpu_panel_A_twins_and_taught=round(t_panel, 3), cpu_panel_I_twins=round(t_panel_I, 3), cpu_first_encounter_probe=t_first,
                n_first_encounter_probes=n_first, cpu_references_full_stream=round(t_refs, 2), cpu_readings=round(t_read, 2),
                loss_first_10=[round(x[2], 3) for x in lr.loss_log[:10]], loss_last_10=[round(x[2], 3) for x in lr.loss_log[-10:]],
                torch=T.__version__, threads=T.get_num_threads())


def sizes():
    out = {}
    for M, Pn in CF.settings():
        cfg = CF.make(M, Pn)
        B = G.build(cfg, W, streams_wanted=("ordinary", "wider", "extra"))
        taught, _ = P.load(W, M)
        row = {}
        for name, (arm, stream) in PATHS.items():
            ev, cuts, end = SC.plan(B, stream, arm, taught)
            grid = sorted(set(range(L.CHUNK if arm == "B" else L.BATCH, end, L.CHUNK if arm == "B" else L.BATCH)) | {c for c in cuts if 0 < c < end} | {end})
            tgt = len(B["streams"][stream]) if arm == "B" else sum(len(G.episode_bytes(cfg, e)) for e in B["episodes"])
            row[name] = dict(bytes=len(B["streams"][stream]), targets=tgt, updates=len(grid), probe_cuts=len(grid) - len(set(range(L.CHUNK if arm == "B" else L.BATCH, end, L.CHUNK if arm == "B" else L.BATCH)) | {end}),
                             first_encounter_probes=sum(1 for e in ev if e["type"] == "first"),
                             outcome_targets=2 * len(B["episodes"]))
        out[f"{M}/{Pn}"] = dict(modulus=M, presentations=Pn, episodes=len(B["episodes"]), paths=row)
    return out


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    what = sys.argv[1]
    t0 = cpu()
    if what == "gen":
        r = gen(); f = RUNS / "profile_gen.json"
    elif what == "path":
        r = path(sys.argv[2]); f = RUNS / f"profile_path_{sys.argv[2]}.json"
    else:
        r = sizes(); f = RUNS / "profile_sizes.json"
    r = dict(r, cpu_seconds_this_process=round(cpu() - t0, 1)) if isinstance(r, dict) else r
    f.write_text(json.dumps(r, indent=1))
    print("wrote", f, round(cpu() - t0, 1), "CPU-s")
