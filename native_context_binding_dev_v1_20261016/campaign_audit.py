"""Independent receipt checks. Reads existing outcomes; never executes training."""
import math
import hashlib
import campaign_common as c


def score_check(value, truth):
    rows = value["rows"]
    if {r["prompt"] for r in rows} != set(truth) or len(rows) != len(truth):
        raise AssertionError("Incomplete or duplicated score set")
    correct = []
    for row in rows:
        if row["answer"] != truth[row["prompt"]]:
            raise AssertionError("Wrong evaluator answer")
        if row["status"] not in ("EOS", "TRUNCATED"):
            raise AssertionError("Invalid generation status")
        emitted = bytes.fromhex(row["emitted"])
        if len(emitted) > 16:
            raise AssertionError("Output cap exceeded")
        got = row["status"] == "EOS" and row["emitted"] == row["answer"]
        if row["correct"] is not got:
            raise AssertionError("Own-output grading mismatch")
        correct.append(got)
    if value["accuracy"] != sum(correct)/len(correct):
        raise AssertionError("Score aggregation mismatch")


def audit(receipt, check_source=True):
    import bind_evaluator as evaluator
    if receipt["schema"] != "NATIVE_BIND_SCIENCE_V1" or receipt["stage"] not in ("DEV", "CONFIRM_A", "CONFIRM_B"):
        raise AssertionError("Wrong scientific envelope")
    stage = receipt["stage"]
    selected = None if stage == "DEV" else c.read(c.RESULTS / "DEV/SELECTION_LOCK.json")["selected"]
    manifest = c.manifest(stage, selected)
    if receipt["confirmation_admission_sha256"]!=manifest["confirmation_admission_sha256"]:
        raise AssertionError("Changed confirmation admission lock")
    if receipt["world"] not in manifest["worlds"] or receipt["arm"] not in manifest["roster"]:
        raise AssertionError("Out-of-manifest receipt")
    if (receipt["old_n"], receipt["new_n"]) != (manifest["old_n"], manifest["new_n"]):
        raise AssertionError("Changed scientific dose")
    if receipt["records_executed"] != 2*(receipt["old_n"]+receipt["new_n"]):
        raise AssertionError("Wrong full-job count")
    if check_source and receipt["source_identity"] != c.identity():
        raise AssertionError("Changed operative source")
    groups = evaluator.split(receipt["world"])
    truths = {name: {f"{a}+{b}=".encode().hex(): str(a+b).encode().hex() for a,b in pairs}
              for name,pairs in groups.items()}
    if any(set(truths[a]) & set(truths[b]) for a,b in (("old","new"),("old","held"),("new","held"))):
        raise AssertionError("Exposure leakage")
    if set(receipt["histories"]) != {"W", "N_OLD"}:
        raise AssertionError("Missing prior-teaching control")
    learns = receipt["arm"] != "NATIVE" and "FROZEN" not in receipt["arm"]
    readings = {"MEM", "LEGACY"} if receipt["arm"] == "NATIVE" else {"MEM"}
    positions = [0,16,64,256,512,768,1024] if stage == "DEV" else [0,16,64,256,512,1024,1536,2048]
    for history,h in receipt["histories"].items():
        cursors = {"OLD":0, "NEW":0}
        for row in h["windows"]:
            phase = row["phase"]
            allowed = history == "W" or phase == "NEW"
            if row["start"] != cursors[phase] or row["n"] != 4 or row["history"] != history or row["allowed"] is not allowed:
                raise AssertionError("Forbidden branch or duplicated/skipped records")
            cursors[phase] += 4
            if row["actor_steps"] != int(learns and allowed):
                raise AssertionError("Incorrect theta/optimizer update permission")
            if not allowed and (row["alpha_l1"] != 0 or not row["theta_unchanged"] or not row["native_passive_only"]):
                raise AssertionError("N_OLD wrote native facts or theta")
        if cursors != {"OLD":receipt["old_n"], "NEW":receipt["new_n"]}:
            raise AssertionError("Incomplete continuous history")
        resources = h["resource"]
        records = evaluator.stream(receipt["world"],"old",receipt["old_n"])+evaluator.stream(receipt["world"],"new",receipt["new_n"])
        if receipt["data_stream_sha256"]!=hashlib.sha256(repr(records).encode()).hexdigest():
            raise AssertionError("Wrong ordered training stream")
        for row in h["windows"]:
            if row["theta_unchanged"] is not (row["theta_before"]==row["theta_after"]):
                raise AssertionError("False theta/optimizer unchanged certificate")
        if resources["records"] != len(records) or resources["EOS_targets"] != len(records):
            raise AssertionError("Production resource count mismatch")
        if resources["actor_steps"]!=sum(row["actor_steps"] for row in h["windows"]):
            raise AssertionError("Unlogged optimizer update")
        if resources["input_bytes"] != sum(len(p) for p,a in records) or resources["answer_bytes"] != sum(len(a) for p,a in records):
            raise AssertionError("Byte exposure count mismatch")
        if resources["native_store_events"] != sum((2+len(p)+len(a)+1)*32 for p,a in records):
            raise AssertionError("Native event accounting mismatch")
        if any(not math.isfinite(v) or v < 0 for k,v in resources.items() if isinstance(v,(int,float))):
            raise AssertionError("Nonfinite or negative resource charge")
        samples = [p for p in h["points"] if p["tag"] == "SAMPLE"]
        if [p["records"] for p in samples] != positions:
            raise AssertionError("Changed sample checkpoints")
        previous = {"MEM":-1., "LEGACY":-1.}
        for p in h["points"]:
            if set(p["readings"]) != readings:
                raise AssertionError("Missing baseline readout")
            for reading,value in p["readings"].items():
                score_check(value, truths["held"])
                cpu = p["CPU_"+reading]
                if not math.isfinite(cpu) or cpu < previous[reading]:
                    raise AssertionError("Nonmonotonic training CPU")
                previous[reading] = cpu
        if [(b["phase"],b["tag"]) for b in h["boundaries"]] != [("BIRTH","BIRTH"),("OLD","BEFORE_REST"),("OLD","AFTER_REST"),("NEW","BEFORE_REST"),("NEW","AFTER_REST")]:
            raise AssertionError("Missing acquisition/retention boundary")
        for b in h["boundaries"]:
            if set(b["readings"]) != readings:
                raise AssertionError("Missing boundary readout")
            for reading,sets in b["readings"].items():
                for group in ("old","new"):
                    score_check(sets[group], truths[group])
        births = h["birth_receipts"]
        if len(births) != 32 or len({b["fly_id"] for b in births}) != 32:
            raise AssertionError("Stores did not have independent canonical births")
        if any(b["canonical_B_sha256"]!="32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964" for b in births):
            raise AssertionError("Wrong canonical native anatomy")
    return dict(accepted=True, records=receipt["records_executed"], stage=stage,
                world=receipt["world"], arm=receipt["arm"], science=True,
                exposure_sets_checked=True, all_history_permissions_checked=True,
                complete_byte_and_EOS_grading=True, source_checked=check_source)
