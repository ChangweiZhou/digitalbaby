"""Check campaign receipts and summarize native endpoints without training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "campaign"


def read(path):
    return json.loads(path.read_text())


def summary(panel):
    rows = panel["outputs"]
    assert len(rows) == panel["n"]
    metrics = {"n": len(rows), "exact": sum(x["output"] == x["target"] for x in rows),
               "first_byte": sum(x["output"][0] == x["target"][0] for x in rows),
               "eos": sum(x["output"][-1] == 256 for x in rows)}
    assert metrics["exact"] == panel["exact"] and metrics["eos"] == panel["own_eos"]
    return metrics


def quartet(a, b):
    assert len(a) == len(b) == 64
    assert [r["prefix"] for r in a] == [r["prefix"] for r in b]
    wins = [all(row["output"] == row["target"] for row in (*a[i:i+2], *b[i:i+2]))
            for i in range(0, 64, 2)]
    return {"total": sum(wins), "families": [sum(wins[:16]), sum(wins[16:])]}


def retention(before, after):
    assert [(x["prefix"], x["target"]) for x in before] == [(x["prefix"], x["target"]) for x in after]
    indices = [i for i,x in enumerate(before) if x["output"] == x["target"]]
    return [sum(after[i]["output"] == after[i]["target"] for i in indices), len(indices)]


def revision(old, edited, noedit):
    assert [x["prefix"] for x in old] == [x["prefix"] for x in edited] == [x["prefix"] for x in noedit]
    assert [x["target"] for x in old] == [x["target"] for x in noedit]
    changes = [i for i,(a,b) in enumerate(zip(old,edited)) if a["target"] != b["target"]]
    previously = [i for i in changes if old[i]["output"] == old[i]["target"]]
    still_valid = [i for i,(a,b) in enumerate(zip(old,edited))
                   if a["target"] == b["target"] and a["output"] == a["target"]]
    return {"changed_targets":len(changes),
            "edit_exact_new_on_previously_correct": [sum(edited[i]["output"] == edited[i]["target"] for i in previously),
                                                     len(previously)],
            "noedit_exact_new_on_previously_correct": [sum(noedit[i]["output"] == edited[i]["target"] for i in previously),
                                                       len(previously)],
            "edit_obsolete_exact_on_changed": [sum(edited[i]["output"] == old[i]["target"] for i in changes),len(changes)],
            "edit_still_valid_preserved": [sum(edited[i]["output"] == old[i]["target"] for i in still_valid),len(still_valid)]}


def run():
    lock = read(RUN / "LOCK.json")
    blocks = {}
    for block in lock["blocks"]:
        block_results = {}
        for arm in lock["arms"]:
            root = RUN / f"{block}-{arm}"
            status = read(root / "STATUS.json") if (root / "STATUS.json").is_file() else {}
            result = {"status":status, "stages":{}}
            raw = {}
            for stage, clock, expected in (("cold",0,0),("A_10k",10000,20000),("A_100k",100000,200000),
                                            ("B",110000,220000),("A_prime",120000,240000),
                                            ("noedit",120000,260000)):
                file = root / f"{stage}.json"
                if not file.exists():
                    break
                data = read(file)
                assert (data["clock"],data["updates_this_worker"]) == (clock,expected)
                panels = data["panels"]
                raw[stage] = panels
                laws = [int(stage == "A_prime"), int(stage != "A_prime")]
                if stage == "cold":
                    assert data["state_hashes"][0] == data["state_hashes"][1]
                    for x,y in zip(panels[0]["held_A_law0"]["outputs"],panels[1]["held_A_law1"]["outputs"]):
                        assert x["output"] == y["output"]
                joined = quartet(*(panels[i][f"held_A_law{laws[i]}"]["outputs"] for i in range(2)))
                if arm == "bag" or stage == "cold":
                    assert joined["total"] == 0
                result["stages"][stage] = {"A_quartets":joined,
                    "by_history":[{name:summary(rows) for name,rows in p.items()} for p in panels]}
            if "noedit" in raw:
                result["lifecycle"] = [{
                    "B_retained_after_edit":retention(raw["B"][i]["held_B"]["outputs"],raw["A_prime"][i]["held_B"]["outputs"]),
                    "B_retained_after_noedit":retention(raw["B"][i]["held_B"]["outputs"],raw["noedit"][i]["held_B"]["outputs"]),
                    "A_revision_relative_to_pre_edit":revision(raw["B"][i][f"held_A_law{i}"]["outputs"],
                                                            raw["A_prime"][i][f"held_A_law{1-i}"]["outputs"],
                                                            raw["noedit"][i][f"held_A_law{i}"]["outputs"])}
                    for i in range(2)]
            block_results[arm] = result
        blocks[block] = block_results

    updates = sum(r["status"].get("updates_this_worker",0) for b in blocks.values() for r in b.values())
    complete = all(r["status"].get("stage") == "COMPLETE" for b in blocks.values() for r in b.values())
    decision = "IN_PROGRESS"
    targets = {}
    if all("A_100k" in r["stages"] for b in blocks.values() for r in b.values()):
        for name,b in blocks.items():
            minimum = max(lock["panel"]["required_ordered_joint_at_least"],
                          b["stock"]["stages"]["A_100k"]["A_quartets"]["total"] + 4)
            ordered = b["ordered"]["stages"]["A_100k"]["A_quartets"]["total"]
            targets[name] = {"ordered":ordered,"required":minimum,"passed":ordered>=minimum}
        decision = "NATIVE_SCREEN_PASSED_SHAM_REQUIRED" if all(t["passed"] for t in targets.values()) else "NATIVE_SCREEN_FAILED"
    if complete:
        assert updates == lock["main_events"]
    return {"status":"COMPLETE" if complete else "IN_PROGRESS","updates":updates,
            "predicted_symbols":updates*lock["targets_per_event"],"native_screen":decision,
            "threshold_by_block":targets,"panel":lock["panel"],"blocks":blocks}


if __name__ == "__main__":
    report = run()
    out = RUN / "SUMMARY.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2) + "\n")
    tmp.replace(out)
    print({k:report[k] for k in ("status","updates","predicted_symbols","native_screen","threshold_by_block")})
    for block, arms in report["blocks"].items():
        for arm,r in arms.items():
            print(block, arm, r["status"].get("stage"),
                  r["status"].get("updates_this_worker"),
                  {k:v["A_quartets"]["total"] for k,v in r["stages"].items()})
