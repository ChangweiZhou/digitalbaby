"""Build CATALOG.md and CATALOG.csv from entries.psv (single source of truth)."""
import csv, collections, pathlib
here = pathlib.Path(__file__).parent
FAMILIES = [
 ("CELL", "Cellular and synaptic plasticity"),
 ("SYS", "Systems memory, forgetting and cognitive maps"),
 ("SLP", "Sleep and offline processing"),
 ("RL", "Reward, prediction error and credit assignment"),
 ("CTRL", "Working memory, control and learning to learn"),
 ("CONC", "Concepts, abstraction and reasoning"),
 ("PERC", "Perceptual and cortical learning"),
 ("MOT", "Motor and cerebellar learning"),
 ("LANG", "Language, social and developmental learning"),
 ("NUM", "Number and arithmetic learning"),
 ("EDU", "Cognitive-psychology and educational learning effects"),
 ("MOD", "Neuromodulation and learning state"),
]
rows = []
for line in (here / "entries.psv").read_text().splitlines():
    f = line.split("|")
    assert len(f) == 9, line
    rows.append(dict(zip(["fam","mechanism","variant","sources","grade","contested","src","register","note"], f)))
counter = collections.Counter(); ids = []
for r in rows:
    counter[r["fam"]] += 1
    r["id"] = "%s-%02d" % (r["fam"], counter[r["fam"]])
with open(here / "CATALOG.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id","family","mechanism","variant","sources","evidence_grade","contested","source_status","in_minifly_register","note"])
    names = dict(FAMILIES)
    for r in rows:
        w.writerow([r["id"], names[r["fam"]], r["mechanism"], r["variant"], r["sources"], r["grade"],
                    "Y" if r["contested"] == "Y" else "", r["src"], r["register"], r["note"]])
grades = collections.Counter(r["grade"] for r in rows)
contested = sum(r["contested"] == "Y" for r in rows)
searched = sum(r["src"] == "S" for r in rows)
out = []
out.append("# Human learning mechanisms catalog (2026-10-23)\n")
out.append("""A catalog of mechanisms by which humans (and, where noted, animals whose circuits underlie human learning) learn, in the same spirit as MiniFly's 254-version mechanism register. Each row gives the **mechanism**, its **variants**, **sources**, and an **evidence grade**. It is a literature map for choosing mechanisms to test in MiniFly, not a systematic review.

**Status: compiled from web searches run on 2026-10-23.** Nothing here has been implemented or tested in MiniFly.
""")
out.append("## How to read the evidence grade\n")
out.append("""| Grade | Meaning |
|---|---|
| A | Robust, replicated human evidence from interventions (drugs, lesions, stimulation) or from large meta-analyses of behavioural effects |
| B | Good human evidence that is mostly correlational (imaging, behaviour, small causal studies), or strong animal causal evidence with human corroboration |
| C | Mainly animal, in-vitro or ex-vivo human tissue; human evidence indirect, single-lab or absent |
| D | Theory or computational model with limited direct empirical test |
| E | Well tested and largely not supported, or failed to replicate |

* **Contested = Y** means there is a substantive, documented disagreement (failed replications, rival accounts, or method disputes), independent of the grade.
* **Source status:** `S` = at least one source for the row was returned by a search run for this catalog; `M` = row rests on background knowledge and was not re-checked. Items marked "(not re-checked)" inside Sources were named from memory.
* **In MiniFly register** names the family in our 254-version register (families: importance/availability, bounded allocation, replay, episodic retrieval, consolidation, metaplasticity cascade, surprise, fast weights, silent memory, context gating, recycling, forgetting, uncertainty, subspace, sparse geometry) that appears to cover the same idea. It was judged from family names only, not from reading each of the 254 versions.
""")
out.append("## Summary\n")
out.append("* **%d mechanism rows** across %d families (%d with at least one source retrieved this session; %d from background knowledge only).\n" % (len(rows), len(FAMILIES), searched, len(rows)-searched))
out.append("* Evidence grades: " + ", ".join("%s=%d" % (g, grades.get(g,0)) for g in "ABCDE") + ". Contested rows: %d.\n" % contested)
out.append("""* Grades come from my reading of search snippets and abstracts plus background knowledge. Effect sizes quoted in notes are from those summaries; several are from secondary sources and are labelled as such.
* Grade A rows are mostly behavioural or educational effects, or effects with human pharmacology; most cellular and systems mechanisms are grade B or C because human causal evidence is hard to obtain.
""")
out.append("## Limitations\n")
out.append("""* Searches returned snippets and abstracts; full texts were rarely read. Details can be wrong where a source was secondary.
* Coverage is broad, not exhaustive: roughly one to three searches per mechanism family member. Some well-known mechanisms are included from memory (source status `M`) and should be verified before citing.
* Evidence grades are judgements. Where the literature is contested, the grade reflects the evidence most directly bearing on humans, and the Contested flag carries the disagreement.
* Human evidence is rarely about the exact learning rule MiniFly would implement. A high grade means the phenomenon is well established, not that a particular algorithmic variant is.
""")
for code, name in FAMILIES:
    out.append("\n## %s: %s\n" % (code, name))
    out.append("| ID | Mechanism | Variants | Sources | Grade | Contested | Reg. | Notes |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["fam"] == code:
            reg = r["register"] if r["register"] != "-" else ""
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], r["mechanism"], r["variant"], r["sources"], r["grade"],
                "Y" if r["contested"] == "Y" else "", reg, r["note"]))
(here / "CATALOG.md").write_text("\n".join(out) + "\n")
print(len(rows), dict(grades), "contested", contested, "searched", searched)
