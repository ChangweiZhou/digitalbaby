"""Source-locked three-family paired science run; no FE1 arm or trajectory."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

from audit_calibration import audit_calibration
from audit_fixture import audit as audit_fixture
from audit_round import audit_life, audit_r_family, audit_t_pair
from common_platform import run_fourstore_life
from fixture import make_world
from technical_arm import ARMS, HERE, OUT as TECH, atomic_json, construct
from t2_graph import build_pair, qualify_technical_roster
from common_platform import fresh_native

RESULTS = HERE / "results"
LOCK_PATH = RESULTS / "SOURCE_LOCK.json"
FAILURE = RESULTS / "FAILURE.json"
WRITER_LOCK = RESULTS / ".single_writer.lock"
GRAPH_MANIFEST = RESULTS / "GRAPH_MANIFEST.json"
SCIENCE_WORLDS = tuple(range(171001, 171033))
TECH_WORLD = 170000
TECHNICAL_AUDITOR_V1_SHA256 = "f51d34aec10f94ec69b426f50bc8ffdd190039b98d3b17f530354b74200a085c"
RESOURCE_POLICY = {"science_worlds": len(SCIENCE_WORLDS),
                   "atomic_unit": "one complete world-arm life",
                   "max_total_receipt_bytes": 2_000_000_000,
                   "max_total_wall_s": 86400,
                   "max_rss_raw": 1_500_000_000,
                   "rss_unit": "macOS ru_maxrss bytes"}
SOURCES = (
    "DESIGN.md", "r2_spec.md", "z1_spec.md", "t2_spec.md",
    "fixture.py", "audit_fixture.py", "common_platform.py", "r2_model.py",
    "z1_model.py", "t2_graph.py", "t2_model.py", "calibrate_r2.py",
    "audit_calibration.py", "technical_arm.py", "audit_round.py",
    "qualify_t2.py", "run_science.py", "analyze_round.py",
)
PARENTS = (
    "BYTE_CORE_V9/brain_byte.py", "BYTE_CORE_1_20260922/bytecore.py",
    "FULL151_VISIBLE_CONTEXT_BRIDGE_20260927/content_model.py",
    "FULL151_INPUT_STEP6_FADING_PAIR_20260927/model.py",
    "V88/minifly_rce_v87_followup_runner.py",
    "V51_Mac_M3/elm_v51_core_scope.py",
    "minifly/V82E/src/model_evo.py",
    "minifly/V82E/src/common_evo.py", "minifly/V82E/config.json",
)


@contextmanager
def single_writer():
    """Hold a process-wide advisory lock across preflight or science."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    fd = os.open(WRITER_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another preflight/science writer is active") from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


_PARENT_PATHS: tuple[Path, ...] | None = None


def _loaded_parent_paths() -> tuple[Path, ...]:
    """Freeze the actual transitive local Python import closure, including V88's dynamic modules."""
    global _PARENT_PATHS
    if _PARENT_PATHS is None:
        # These imports instantiate no scientific life. They close the modules
        # loaded lazily by the Z and T constructors before the source lock.
        import z1_model  # noqa: F401
        import t2_model  # noqa: F401
        fresh_native()
        project = HERE.parent.resolve()
        paths = set()
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            path = Path(filename).resolve()
            if (path.suffix == ".py" and path.is_relative_to(project) and
                    not path.is_relative_to(HERE) and
                    not any(part.startswith(".venv") for part in path.parts)):
                paths.add(path)
                # Ancestor model modules load their own iteration config at
                # import time; it is executable scientific state too.
                config = path.parent.parent / "config.json"
                if config.is_file():
                    paths.add(config.resolve())
        paths.update((HERE.parent / name).resolve() for name in PARENTS)
        _PARENT_PATHS = tuple(sorted(paths))
    return _PARENT_PATHS


def _environment() -> dict:
    import importlib.metadata
    return {"python": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "numba")}}


@lru_cache(maxsize=1)
def _native_object_hashes() -> dict:
    base = fresh_native()
    import brain_byte as bb
    observed = bb.bc.object_hashes(base.fly)
    if observed != bb.bc.OBJECT_HASHES:
        raise RuntimeError("native Full151 B/Q object digest drift")
    return observed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planned_lock() -> dict:
    paths = [HERE / name for name in SOURCES]
    parent_paths = _loaded_parent_paths()
    paths += parent_paths
    paths += [TECH / "R0_CALIBRATION.json", TECH / "T2_GRAPH_ROSTER.json"]
    paths += [TECH / f"{arm}_{TECH_WORLD}.json" for arm in ARMS]
    paths.append(GRAPH_MANIFEST)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise RuntimeError(f"source/technical inputs missing: {missing}")
    doc = {"schema": "MINIFLY-THREE-MECHANISM-SOURCE-LOCK-v1",
           "arms": list(ARMS), "technical_world": TECH_WORLD,
           "science_worlds": list(SCIENCE_WORLDS),
           "inherited_python_closure": [str(path) for path in parent_paths if path.suffix == ".py"],
           "environment": _environment(),
           "native_full151_objects": _native_object_hashes(),
           "source_sha256": {str(path.resolve()): _sha(path) for path in paths}}
    doc["digest"] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
    return doc


def technical_preflight() -> dict:
    calibration = json.loads((TECH / "R0_CALIBRATION.json").read_text())
    cal_audit = audit_calibration(calibration)
    graph = json.loads((TECH / "T2_GRAPH_ROSTER.json").read_text())
    if graph.get("schema") != "MINIFLY-T2-TECHNICAL-ROSTER-v1":
        raise AssertionError("wrong T2 graph roster")
    graph_audit = qualify_technical_roster(graph["graphs"], tuple(graph["worlds"]))
    if graph_audit != graph["qualification"]:
        raise AssertionError("T2 graph qualification changed")
    technical_graph = next((row for row in graph["graphs"]
                            if row.get("world") == TECH_WORLD), None)
    if technical_graph is None:
        raise AssertionError("reserved technical graph receipt missing")
    arm_audits = {}
    receipts = {}
    seconds = 0.0
    sizes = 0
    peak_rss = 0
    recorded_auditors = set()
    current_auditor = _sha(HERE / "audit_round.py")
    for arm in ARMS:
        path = TECH / f"{arm}_{TECH_WORLD}.json"
        receipt = json.loads(path.read_text())
        if receipt.get("arm") != arm or receipt.get("technical") is not True:
            raise AssertionError("technical arm identity")
        kwargs = {"expected_world": TECH_WORLD}
        if arm == "Rrand":
            kwargs["r2_receipt"] = receipts["R2"]
        if arm in ("T0_2", "T2"):
            kwargs["graph_expected"] = technical_graph
        arm_audits[arm] = audit_life(receipt, arm, **kwargs)
        receipts[arm] = receipt
        for name, digest in receipt["source_sha256"].items():
            if name == "audit_round.py":
                if digest not in (TECHNICAL_AUDITOR_V1_SHA256, current_auditor):
                    raise AssertionError(f"technical {arm} unknown historical auditor hash")
                recorded_auditors.add(digest)
            elif _sha(HERE / name) != digest:
                raise AssertionError(f"technical {arm} source drift: {name}")
        seconds += receipt["elapsed_wall_s"]
        sizes += path.stat().st_size
        peak_rss = max(peak_rss, receipt["max_rss_raw"])
    family_audits = {"R": audit_r_family(receipts["R0"], receipts["R2"],
                                         receipts["Rrand"], expected_world=TECH_WORLD),
                     "T": audit_t_pair(receipts["T0_2"], receipts["T2"],
                                       expected_world=TECH_WORLD,
                                       graph_expected=technical_graph)}
    return {"calibration": cal_audit, "graph": graph_audit, "arms": arm_audits,
            "family_audits": family_audits,
            "auditor_upgrade": {"technical_receipt_auditor_sha256": sorted(recorded_auditors),
                                "current_strict_auditor_sha256": current_auditor,
                                "all_technical_receipts_freshly_reaudited": True},
            "technical_sum_wall_s": seconds, "technical_sum_bytes": sizes,
            "technical_peak_rss_raw": peak_rss,
            "estimated_32world_wall_h_sequential": len(SCIENCE_WORLDS) * seconds / 3600,
            "estimated_science_receipt_bytes": len(SCIENCE_WORLDS) * sizes}


def preflight():
    with single_writer():
        if LOCK_PATH.exists() or FAILURE.exists():
            raise RuntimeError("source lock or terminal failure already exists")
        if any((RESULTS / "worlds").glob("*/*.json")):
            raise RuntimeError("science receipts already exist before source lock")
        if any("FE1" in arm for arm in ARMS):
            raise AssertionError("FE1 is excluded from the science roster")
        outcome = technical_preflight()
        if (outcome["estimated_32world_wall_h_sequential"] * 3600 >
                RESOURCE_POLICY["max_total_wall_s"] or
                outcome["estimated_science_receipt_bytes"] >
                RESOURCE_POLICY["max_total_receipt_bytes"] or
                outcome["technical_peak_rss_raw"] > RESOURCE_POLICY["max_rss_raw"]):
            raise RuntimeError("technical projection exceeds frozen resource ceiling")
        base = fresh_native()
        graph_receipts = {}
        for world in SCIENCE_WORLDS:
            audit_fixture(make_world(world))
            # The constructor accepts source B, the fixed receptor map, and a
            # reserved world ID only. Its result is sealed before science.
            graph_receipts[str(world)] = build_pair(base.fly.m.B,
                                                  base.fly.m.pn_type_index,
                                                  world).receipt
        atomic_json(GRAPH_MANIFEST,
                    {"schema": "MINIFLY-THREE-MECHANISM-GRAPH-MANIFEST-v1",
                     "worlds": list(SCIENCE_WORLDS), "graphs": graph_receipts})
        lock = planned_lock()
        lock["technical_audit"] = outcome
        lock["resource_policy"] = dict(RESOURCE_POLICY)
        # The digest covers the complete resource and qualification decision.
        lock["digest"] = hashlib.sha256(json.dumps({k: v for k, v in lock.items() if k != "digest"},
                                                sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode()).hexdigest()
        atomic_json(LOCK_PATH, lock)
        print(json.dumps({"preflight": "PASS", "worlds": len(SCIENCE_WORLDS),
                          "arms": len(ARMS), "estimated_sequential_hours":
                          outcome["estimated_32world_wall_h_sequential"]}), flush=True)


def require_lock() -> dict:
    lock = json.loads(LOCK_PATH.read_text())
    computed = planned_lock()
    if (lock.get("schema") != computed["schema"] or lock.get("arms") != list(ARMS) or
            lock.get("science_worlds") != list(SCIENCE_WORLDS) or
            lock.get("source_sha256") != computed["source_sha256"] or
            lock.get("inherited_python_closure") != computed["inherited_python_closure"] or
            lock.get("environment") != computed["environment"] or
            lock.get("native_full151_objects") != computed["native_full151_objects"] or
            lock.get("resource_policy") != RESOURCE_POLICY):
        raise RuntimeError("frozen source/roster drift")
    digest = hashlib.sha256(json.dumps({k: v for k, v in lock.items() if k != "digest"},
                                       sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    if lock.get("digest") != digest:
        raise RuntimeError("source lock digest mismatch")
    return lock


def _graph_manifest(lock: dict) -> dict:
    doc = json.loads(GRAPH_MANIFEST.read_text())
    if (doc.get("schema") != "MINIFLY-THREE-MECHANISM-GRAPH-MANIFEST-v1" or
            doc.get("worlds") != list(SCIENCE_WORLDS) or
            set(doc.get("graphs", {})) != {str(w) for w in SCIENCE_WORLDS} or
            _sha(GRAPH_MANIFEST) != lock["source_sha256"][str(GRAPH_MANIFEST.resolve())]):
        raise AssertionError("sealed science graph manifest mismatch")
    for world in SCIENCE_WORLDS:
        if doc["graphs"][str(world)].get("world") != world:
            raise AssertionError("graph manifest world identity")
    return doc["graphs"]


def _receipt_resources(lock: dict) -> dict:
    """Count every atomic arm receipt already committed, including on resume."""
    out = {"committed_arms": 0, "receipt_bytes": 0, "arm_wall_s": 0.0,
           "max_rss_raw": 0}
    for path in sorted((RESULTS / "worlds").glob("*/*.json")):
        try:
            world = int(path.parent.name)
        except ValueError as exc:
            raise AssertionError(f"unexpected receipt directory: {path}") from exc
        arm = path.stem
        if world not in SCIENCE_WORLDS or arm not in ARMS:
            raise AssertionError(f"unplanned scientific receipt: {path}")
        receipt = json.loads(path.read_text())
        if (receipt.get("world") != world or receipt.get("arm") != arm or
                receipt.get("source_lock_digest") != lock["digest"] or
                receipt.get("audit", {}).get("pass") is not True):
            raise AssertionError(f"invalid committed receipt header: {path}")
        wall = receipt.get("elapsed_wall_s")
        rss = receipt.get("max_rss_raw")
        if (not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0 or
                not isinstance(rss, (int, float)) or not math.isfinite(rss) or rss <= 0):
            raise AssertionError(f"invalid committed receipt resource: {path}")
        out["committed_arms"] += 1
        out["receipt_bytes"] += path.stat().st_size
        out["arm_wall_s"] += float(wall)
        out["max_rss_raw"] = max(out["max_rss_raw"], int(rss))
    return out


def _check_resource(totals: dict, policy: dict):
    if (totals["receipt_bytes"] > policy["max_total_receipt_bytes"] or
            totals["arm_wall_s"] > policy["max_total_wall_s"] or
            totals["max_rss_raw"] > policy["max_rss_raw"]):
        raise RuntimeError(f"prespecified cumulative resource ceiling reached: {totals}")


@contextmanager
def _arm_wall_limit(remaining_seconds: float):
    if remaining_seconds <= 0:
        raise RuntimeError("prespecified cumulative arm wall-time ceiling reached")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("scientific runner cannot share an active wall timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError("prespecified cumulative arm wall-time ceiling reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _audit_saved(receipt: dict, arm: str, world: int, graphs: dict,
                 *, r2_receipt: dict | None = None) -> dict:
    if receipt.get("world") != world or receipt.get("arm") != arm:
        raise AssertionError(f"saved receipt identity mismatch: {world}/{arm}")
    kwargs = {"expected_world": world}
    if arm == "Rrand":
        if r2_receipt is None:
            raise AssertionError("Rrand requires its same-world R2 receipt")
        kwargs["r2_receipt"] = r2_receipt
    if arm in ("T0_2", "T2"):
        kwargs["graph_expected"] = graphs[str(world)]
    return audit_life(receipt, arm, **kwargs)


def _audit_r_world(world: int) -> dict:
    folder = RESULTS / "worlds" / str(world)
    rows = {arm: json.loads((folder / f"{arm}.json").read_text())
            for arm in ("R0", "R2", "Rrand")}
    return audit_r_family(rows["R0"], rows["R2"], rows["Rrand"],
                          expected_world=world)


def _audit_t_world(world: int, graph_expected: dict) -> dict:
    folder = RESULTS / "worlds" / str(world)
    t0 = json.loads((folder / "T0_2.json").read_text())
    t2 = json.loads((folder / "T2.json").read_text())
    return audit_t_pair(t0, t2, expected_world=world,
                        graph_expected=graph_expected)


def science():
    with single_writer():
        if FAILURE.exists():
            raise RuntimeError("prior scientific failure; no automatic restart")
        if (RESULTS / "RUN_COMPLETE.json").exists():
            print(json.dumps({"science": "ALREADY_COMPLETE"}), flush=True)
            return
        lock = None
        world = None
        arm = None
        begun = time.monotonic()
        independently_verified = 0
        try:
            lock = require_lock()
            graphs = _graph_manifest(lock)
            totals = _receipt_resources(lock)
            _check_resource(totals, lock["resource_policy"])
            for world in SCIENCE_WORLDS:
                audit_fixture(make_world(world))
                pair = None
                for arm in ARMS:
                    path = RESULTS / "worlds" / str(world) / f"{arm}.json"
                    r2 = None
                    if arm == "Rrand":
                        r2_path = RESULTS / "worlds" / str(world) / "R2.json"
                        r2 = json.loads(r2_path.read_text())
                        _audit_saved(r2, "R2", world, graphs)
                    if path.exists():
                        saved = json.loads(path.read_text())
                        if saved.get("source_lock_digest") != lock["digest"]:
                            raise RuntimeError(f"incompatible completed arm {world}/{arm}")
                        _audit_saved(saved, arm, world, graphs, r2_receipt=r2)
                        independently_verified += 1
                        if arm == "Rrand":
                            _audit_r_world(world)
                        if arm == "T2":
                            _audit_t_world(world, graphs[str(world)])
                        continue
                    require_lock()
                    remaining = lock["resource_policy"]["max_total_wall_s"] - totals["arm_wall_s"]
                    started = time.monotonic()
                    with _arm_wall_limit(remaining):
                        if arm in ("T0_2", "T2") and pair is None:
                            base = fresh_native()
                            pair = build_pair(base.fly.m.B, base.fly.m.pn_type_index, world)
                            if pair.receipt != graphs[str(world)]:
                                raise AssertionError(f"sealed T graph drift: {world}")
                        actor, graph_receipt = construct(arm, world, pair_cache=pair,
                                                         r2_receipt=r2)
                        receipt = run_fourstore_life(world, actor)
                        receipt.update(arm=arm, source_lock_digest=lock["digest"],
                                       graph_receipt=graph_receipt,
                                       max_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                        receipt["audit"] = _audit_saved(receipt, arm, world, graphs,
                                                         r2_receipt=r2)
                    receipt["elapsed_wall_s"] = time.monotonic() - started
                    receipt["max_rss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    prospective = {"committed_arms": totals["committed_arms"] + 1,
                                   "receipt_bytes": totals["receipt_bytes"] + len(json.dumps(
                                       receipt, sort_keys=True, allow_nan=False).encode()),
                                   "arm_wall_s": totals["arm_wall_s"] + receipt["elapsed_wall_s"],
                                   "max_rss_raw": max(totals["max_rss_raw"],
                                                      receipt["max_rss_raw"])}
                    _check_resource(prospective, lock["resource_policy"])
                    atomic_json(path, receipt)
                    totals = prospective
                    independently_verified += 1
                    if arm == "Rrand":
                        _audit_r_world(world)
                    if arm == "T2":
                        _audit_t_world(world, graphs[str(world)])
                    print(json.dumps({"world": world, "arm": arm,
                                      "committed_arms": totals["committed_arms"],
                                      "of": len(SCIENCE_WORLDS) * len(ARMS),
                                      "elapsed_wall_s": receipt["elapsed_wall_s"]}), flush=True)
            if totals["committed_arms"] != len(SCIENCE_WORLDS) * len(ARMS):
                raise AssertionError("science arm roster incomplete at final commit")
            require_lock()
            atomic_json(RESULTS / "RUN_COMPLETE.json",
                        {"source_lock_digest": lock["digest"],
                         "worlds": len(SCIENCE_WORLDS), "arms_per_world": len(ARMS),
                         "committed_arms": totals["committed_arms"],
                         "cumulative_arm_wall_s": totals["arm_wall_s"],
                         "receipt_bytes": totals["receipt_bytes"],
                         "max_rss_raw": totals["max_rss_raw"],
                         "runner_wall_s": time.monotonic() - begun})
            print(json.dumps({"science": "COMPLETE",
                              "committed_arms": totals["committed_arms"]}), flush=True)
        except BaseException as exc:
            try:
                counts = _receipt_resources(lock) if lock is not None else None
            except Exception as count_exc:
                counts = {"resource_count_error": repr(count_exc),
                          "receipt_files_present": len(list((RESULTS / "worlds").glob("*/*.json")))}
            atomic_json(FAILURE, {"world": world, "arm": arm,
                                  "error": repr(exc), "traceback": traceback.format_exc(),
                                  "committed": counts,
                                  "independently_verified_in_this_attempt": independently_verified,
                                  "of": len(SCIENCE_WORLDS) * len(ARMS),
                                  "source_lock_digest": None if lock is None else lock["digest"]})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "science"))
    {"preflight": preflight, "science": science}[parser.parse_args().phase]()
