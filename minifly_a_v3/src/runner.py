"""Build one Package A V3-CLAUDE arm from separate canonical births and run one full technical life.

Receipts are gzip JSON, written once (os.link onto the final name; never overwritten).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import stores  # noqa: E402
import systems  # noqa: E402
from common_platform import run_fourstore_life  # noqa: E402
from fixture import make_world  # noqa: E402

ARMS = ("R0", "R1", "R1_rand", "R0_signed", "R3", "R3_randtarget",
        "Z0_resource", "Z2", "Z2_rand", "P0", "P1", "P2", "P4")
TECH_WORLD = 190000
RESULTS = paths.ROOT / "results"
SCIENCE_WORLDS = tuple(range(190001, 190065))
TOP_FILES = ("SPEC_LOCK.md", "ARM_ROSTER.json", "results/calibration/R_CALIBRATION.json")


def source_hashes() -> dict:
    """Executed closure: all src/ and tests/ modules, the spec, roster, calibration and the scaffold MANIFEST
    (every package file is checked against that MANIFEST before each life)."""
    import lock
    out = lock.code_files()
    out["package/MANIFEST.json"] = hashlib.sha256((paths.PKG / "MANIFEST.json").read_bytes()).hexdigest()
    for f in TOP_FILES:
        out[f] = hashlib.sha256((paths.ROOT / f).read_bytes()).hexdigest()
    return out


def receipt_path(kind: str, arm: str, world: int) -> Path:
    return RESULTS / kind / arm / f"{world}.json.gz"


def load_receipt(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def write_once(path: Path, doc: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = gzip.compress(json.dumps(doc, sort_keys=True, allow_nan=False, separators=(",", ":")).encode(), 6)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    try:
        os.link(tmp, path)          # fails if a receipt already exists
    finally:
        tmp.unlink()
    return len(raw)


def r_calibration() -> dict:
    return json.loads((RESULTS / "calibration" / "R_CALIBRATION.json").read_text())


def construct(arm: str, world: int, *, kind: str = "technical"):
    births = []
    if arm in systems.R_ARMS:
        cal = r_calibration()
        banks = {}
        for bank, k in (("shared", "native"), ("private", "content")):
            banks[bank] = []
            for j in range(4):
                model, rc = stores.birth(k)
                banks[bank].append(model)
                births.append({"bank": bank, "store": j, **rc})
        gates = None
        if arm == "R1_rand":
            r1path = receipt_path(kind, "R1", world)
            r1 = load_receipt(r1path)
            doc = make_world(world)
            gates = {b: systems.r1_random_schedule(doc, b, rows) for b, rows in r1["mechanism_events"].items()}
        system = systems.RSystem(banks["shared"], banks["private"], arm,
                                 scales=(cal["scales"]["shared"], cal["scales"]["private"]),
                                 theta=cal["theta"], rand_gates=gates)
        params = {"scales": cal["scales"], "theta": cal["theta"], "window": systems.R_WINDOW,
                  "paired_R1_receipt_sha256": (hashlib.sha256(r1path.read_bytes()).hexdigest()
                                               if arm == "R1_rand" else None),
                  "calibration_sha256": hashlib.sha256(
                      (RESULTS / "calibration" / "R_CALIBRATION.json").read_bytes()).hexdigest()}
    elif arm in stores.Z_ARMS:
        models = []
        for j in range(4):
            model, rc = stores.birth_z(arm, world, j)
            models.append(model)
            births.append({"bank": "native", "store": j, **rc})
        table, zsha = None, None
        if arm == "Z2_rand":
            zpath = receipt_path(kind, "Z2", world)
            z2 = load_receipt(zpath)
            zsha = hashlib.sha256(zpath.read_bytes()).hexdigest()
            table = {(b, row[1], row[2]): {name: v[2] for name, v in row[7].items()}
                     for b, rows in z2["mechanism_events"].items() for row in rows if row[0] == "Z"}
        system = systems.LedgerFourStore(models, z_ref_table=table)
        params = {"tau_L": stores.Z_TAU_L, "kappa": stores.Z_KAPPA, "writable_coordinates": int(len(models[0].z_indices)),
                  "paired_Z2_receipt_sha256": zsha}
    elif arm in stores.P_ARMS:
        models = []
        for j in range(4):
            model, rc = stores.birth_p(arm)
            models.append(model)
            births.append({"bank": "native", "store": j, **rc})
        system = systems.LedgerFourStore(models)
        params = {"eps": stores.P_EPS, "tau": stores.P_TAU, "floor": stores.P_FLOOR,
                  "support_sha256": list(models[0].p_support[:2])}
    else:
        raise ValueError(arm)
    if len({b["fly_id"] for b in births}) != len(births):
        raise AssertionError("stores share a newborn fly object")
    for b in births:
        b.pop("fly_id")
    return system, births, params


def run(arm: str, world: int = TECH_WORLD, *, kind: str = "technical") -> dict:
    import portable_birth as pb
    import lock
    if kind == "science":
        if world not in SCIENCE_WORLDS:
            raise ValueError("science receipts only for worlds 190001-190064")
        lk = lock.verify_lock()
    elif kind.startswith("technical"):
        if world != TECH_WORLD:
            raise ValueError("technical receipts only for world 190000")
        lk = None
        lock.package_files()            # every scaffold file must match its MANIFEST
    else:
        raise ValueError(kind)
    dest = receipt_path(kind, arm, world)
    if dest.exists():
        raise FileExistsError(f"receipt exists: {dest}")
    t0 = time.monotonic()
    system, births, params = construct(arm, world, kind=kind)
    t1 = time.monotonic()
    doc = run_fourstore_life(world, system, technical=kind.startswith("technical"))
    t2 = time.monotonic()
    doc["schema"] = "MINIFLY-A3-CLAUDE-RECEIPT-v1"
    doc["package"] = "MINIFLY_A_V3_CLAUDE (new implementation; not Muse V2)"
    doc["arm"] = arm
    doc["births"] = births
    doc["params"] = params
    doc["canonical_B_sha256"] = pb.EXPECTED_B
    doc["source_sha256"] = source_hashes()
    doc["lock_digest"] = None if lk is None else lk["lock_digest"]
    doc["kind"] = kind
    doc["resources"] = {"construct_s": t1 - t0, "life_s": t2 - t1,
                        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    size = write_once(dest, doc)
    return {"arm": arm, "world": world, "life_s": round(t2 - t1, 1), "receipt_bytes": size,
            "peak_rss_bytes": doc["resources"]["peak_rss_bytes"]}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], kind=sys.argv[2] if len(sys.argv) > 2 else "technical_final")), flush=True)
