"""Compact Markdown tables from completed ordered-front-end receipts."""
from campaign import RUN, read


def cell(num, den):
    return f"{num}/{den}"


def pool(history, match):
    entries = [v for h in history for name,v in h.items() if name.startswith(match)]
    return "/".join(map(str, (sum(x["exact"] for x in entries), sum(x["n"] for x in entries))))


def total_frac(lives, key):
    return cell(*(sum(d[key][i] for d in lives) for i in range(2)))


def primary(run):
    print("| Birth | Encoder | Taught A exact | Held A first byte | Held A exact | Held A EOS | Held quartets at 10k → 100k |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for block, arms in run["blocks"].items():
        for arm,r in arms.items():
            early = r["stages"]["A_10k"]["A_quartets"]["total"]
            final = r["stages"]["A_100k"]
            h = final["by_history"]
            held = [x[f"held_A_law{i}"] for i,x in enumerate(h)]
            print("|", " | ".join([block.rsplit("-",1)[-1],arm,pool(h,"taught_A"),
                    cell(sum(x["first_byte"] for x in held),128), pool(h,"held_A"),
                    cell(sum(x["eos"] for x in held),128), f"{early}/32 → {final['A_quartets']['total']}/32"]), "|")


def lifecycle(run):
    print("| Birth | Encoder | B preserved after edit | B preserved after no edit | Previously correct flipped A corrected | Same corrections in no edit |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for block, arms in run["blocks"].items():
        for arm, r in arms.items():
            lives = r["lifecycle"]
            change = [x["A_revision_relative_to_pre_edit"] for x in lives]
            print("|", " | ".join([block.rsplit("-",1)[-1],arm,
                    total_frac(lives,"B_retained_after_edit"),total_frac(lives,"B_retained_after_noedit"),
                    total_frac(change,"edit_exact_new_on_previously_correct"),
                    total_frac(change,"noedit_exact_new_on_previously_correct")]), "|")


def vocabulary(vocab):
    print("| Birth | Encoder | Taught topology, familiar actors | Taught topology, fresh actors | Held topology, familiar actors | Held topology, fresh actors | Held quartets, familiar actors |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for block, arms in vocab["by_block"].items():
        for arm, r in arms.items():
            keys = ("taught_topology_familiar_actor_bytes","taught_topology_fresh_actor_bytes",
                    "held_topology_familiar_actor_bytes","held_topology_fresh_actor_bytes")
            counts = [pool(r["metrics"], k) for k in keys]
            joint = r["joint_by_held_panel"]["held_topology_familiar_actor_bytes"]["total"]
            print("|"," | ".join([block.rsplit("-",1)[-1],arm,*counts,cell(joint,32)]),"|")


def all_mask(run):
    print("| Birth | Encoder | Family 1 balanced exact | Family 2 balanced exact |")
    print("| --- | --- | ---: | ---: |")
    for block, arms in run["blocks"].items():
        for arm,r in arms.items():
            family = [sum(law["metrics"]["balanced_exact"] for law in r[str(i)])/2 for i in range(2)]
            print("|", " | ".join([block.rsplit("-",1)[-1],arm,*(f"{x:.1%}" for x in family)]), "|")


if __name__ == "__main__":
    run = read(RUN/"SUMMARY.json")
    assert run["status"] == "COMPLETE"
    primary(run)
    print()
    lifecycle(run)
    if (RUN/"VOCABULARY_DIAGNOSTIC.json").exists():
        print()
        vocabulary(read(RUN/"VOCABULARY_DIAGNOSTIC.json"))
    if (RUN/"ALL_MASK_MODEL_SCORES.json").exists():
        print()
        all_mask(read(RUN/"ALL_MASK_MODEL_SCORES.json"))
