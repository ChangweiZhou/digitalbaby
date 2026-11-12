"""Sizing sweep: one fit, or the learner-free admission of one setting and world (brief rev. 3). For stage 2, on the coordinator's release.

    python3 sz_fit.py checks <modulus> <presentations> <world>              step 0's checks at that configuration and world, plus the modulus
                                                                            admission; writes runs/checks_<M>_<P>_<world>.json
    python3 sz_fit.py fit <arm> <stream> <modulus> <presentations> <world>  one fit; refuses unless its setting and world were admitted by
                                                                            `checks` (same configuration hash, same stream hash); writes
                                                                            runs/fit_<arm>_<stream>_<M>_<P>_<world>.json, refuses to overwrite

arm A runs on the ordinary stream only; arm B on ordinary, wider (the aligned lines) or extra (matched extra bytes, 97/4 only).
Every fit records the configuration it ran, its hash, the stream hash, the taught panel's hash, the cut positions, bytes, target positions,
updates, and CPU for generation, training and scoring separately (checks are recorded with the setting and world).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_check as CK      # noqa: E402
import sz_config as CF     # noqa: E402
import sz_learner as L     # noqa: E402
import sz_modfix as MF     # noqa: E402
import sz_panels as P      # noqa: E402
import sz_score as SC      # noqa: E402
import sz_stream as G      # noqa: E402

RUNS = HERE / "runs"
ALLOWED_WORLDS = set(CF.ROSTER_WORLDS) | {CF.PROFILE_WORLD}


def commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()


def checks_path(M, Pn, w):
    return RUNS / f"checks_{M}_{Pn}_{w}.json"


def do_checks(M, Pn, w):
    assert w in ALLOWED_WORLDS, "world seeds 1, 2, 3 are never touched as worlds"
    cfg = CF.make(M, Pn)
    t0 = time.process_time(); B = G.build(cfg, w); t_gen = time.process_time() - t0
    roster = list(CF.ROSTER_WORLDS)
    other = roster[(roster.index(w) + 1) % 3] if w in roster else 105
    t0 = time.process_time(); ch = CK.run_checks(B, other); t_chk = time.process_time() - t0
    adm = MF.admit(cfg)
    ok = all(v["passed"] for v in ch.values()) and adm["admitted"]
    out = dict(modulus=M, presentations=Pn, world=w, configuration=cfg, cfg_hash=CF.chash(cfg), stream_hashes=G.hashes(B)["streams"],
               checks={k: dict(passed=v["passed"], condition_held=v["condition_held"], fixture_failed_as_required=v["fixture_failed_as_required"]) for k, v in ch.items()},
               checks_detail=ch, modulus_admission=adm, admitted=ok, cpu_generation=round(t_gen, 2), cpu_checks=round(t_chk, 2), source_commit=commit())
    checks_path(M, Pn, w).write_text(json.dumps(out, indent=1, default=str))
    return out


def do_fit(arm, stream, M, Pn, w):
    T.set_num_threads(1)
    assert w in ALLOWED_WORLDS
    assert (arm, stream) in (("A", "ordinary"), ("B", "ordinary"), ("B", "wider"), ("B", "extra"))
    assert stream != "extra" or (M, Pn) == (97, 4), "the matched extra-bytes stream is trained on at the proposed size only"
    out_path = RUNS / f"fit_{arm}_{stream}_{M}_{Pn}_{w}.json"
    if out_path.exists():
        raise SystemExit(f"refused: {out_path.name} exists")
    cfg = CF.make(M, Pn)
    adm = json.loads(checks_path(M, Pn, w).read_text())
    if not adm["admitted"] or adm["cfg_hash"] != CF.chash(cfg):
        raise SystemExit("refused: this setting and world are not admitted (or were admitted under another configuration)")
    t0 = time.process_time(); B = G.build(cfg, w, streams_wanted=(stream,)); t_gen = time.process_time() - t0
    if G.sha(B["streams"][stream]) != adm["stream_hashes"][stream]:
        raise SystemExit("refused: the stream differs from the admitted one")
    taught, taught_sha = P.load(w, M)
    ev, cuts, end = SC.plan(B, stream, arm, taught)
    seed = w + 1000
    if arm == "B":
        data = B["streams"][stream]; om = T.zeros(len(data), dtype=T.bool)
        q = T.tensor(B["offsets"][stream]["eqs"]); om[q + 1] = True; om[q + 2] = True       # the two answer digits of every episode
        lr = L.StreamLearner(seed, data, cuts, outcome_mask=om)
    else:
        lr = L.EpisodeLearner(seed, [G.episode_bytes(cfg, e) for e in B["episodes"]], cuts)
    responses = SC.run(lr, ev, end)
    t0 = time.process_time()
    refs = SC.references(B, stream, arm, ev, grid=lr.grid if arm == "B" else None)
    R = SC.readings(B, ev, responses, refs, stream == "wider")
    cls, nums = SC.classify(R)
    t_ref = time.process_time() - t0
    tenths = [dict(tenth=t + 1, outcome_loss=float(lr.tenths[t, 1, 0] / max(1, lr.tenths[t, 1, 1])), outcome_targets=int(lr.tenths[t, 1, 1]),
                   other_loss=float(lr.tenths[t, 0, 0] / max(1, lr.tenths[t, 0, 1])), other_targets=int(lr.tenths[t, 0, 1])) for t in range(10)]
    targets = lr.targets
    out = dict(brief="OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md rev. 3", arm=arm, stream=stream, modulus=M, presentations=Pn, world=w,
               configuration=cfg, cfg_hash=CF.chash(cfg), stream_sha256=G.sha(B["streams"][stream]), taught_panel_sha256=taught_sha,
               model_seed=seed, optimizer=L.opt_groups(lr.m), batch=("64 episodes in stream order" if arm == "A" else "one stream, chunks of 256 target positions"),
               loss="mean over the targets of a batch or chunk; every next byte; no answer mask", one_update_per_batch_or_chunk=True,
               bytes=len(B["streams"][stream]), target_positions=targets, updates=lr.updates, cut_positions=lr.cuts, episodes=len(B["episodes"]),
               cpu=dict(generation=round(t_gen, 2), training=round(lr.cpu_train, 2), scoring_responses=round(lr.cpu_respond, 2),
                        scoring_references_and_readings=round(t_ref, 2), checks_recorded_with_setting=adm["cpu_checks"]),
               responses_written=lr.responses, cls=cls, class_numbers=nums, loss_tenths=tenths,
               loss_log_every_50=[x for i, x in enumerate(lr.loss_log) if i % 50 == 0 or i == len(lr.loss_log) - 1],
               readings=R, source_commit=commit(), torch=T.__version__,
               note="A sizing fit: no prediction is judged and nothing is concluded about any learner or rule.")
    out_path.write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    if sys.argv[1] == "checks":
        r = do_checks(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])); print("admitted" if r["admitted"] else "NOT ADMITTED")
    elif sys.argv[1] == "fit":
        r = do_fit(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        print(r["cls"], json.dumps(r["class_numbers"]), json.dumps(r["cpu"]))
    else:
        print(__doc__)
