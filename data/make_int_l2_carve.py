"""int_l2_2k: L2-hybrid-scale (2,000q) carve of int_mix6k — cliff 400 (InT rows,
hash-order prefix of the mix's cliff stratum, disjoint from the 300 L5 holdout
by construction) + frontier 150 + solved 1,450 (hash-ordered carves of the
mix6k strata kept verbatim in int_mix6k)."""
import json, collections
from pathlib import Path
from expert_iter.utils import stable_hash
from expert_iter.registry import ADAPTERS
import expert_iter.data  # registry

MIX=Path("data/mixes/int_mix6k_qwen3-4b-2507.jsonl")
OUT=Path("data/mixes"); NAME="int_l2_2k_qwen3-4b-2507"
rows=[json.loads(l) for l in open(MIX)]
int_cliff=[r for r in rows if r["qid"].startswith("int-")]            # already in stable_hash('int-mix') order
frontier=[r for r in rows if r["meta"]["passrate_class"]=="frontier"]
solved=[r for r in rows if r["meta"]["passrate_class"]=="solved"]
frontier.sort(key=lambda r: stable_hash("int-l2", r["question"]))
solved.sort(key=lambda r: stable_hash("int-l2", r["question"]))
carve=int_cliff[:400]+frontier[:150]+solved[:1450]
assert len(carve)==2000 and len({r["qid"] for r in carve})==2000
hold={json.loads(l)["qid"] for l in open("data/mixes/int_mix6k_qwen3-4b-2507_cliff_holdout.jsonl")}
assert not hold & {r["qid"] for r in carve}, "overlaps L5 holdout!"
with open(OUT/f"{NAME}.jsonl","w") as f:
    for r in carve: f.write(json.dumps(r,ensure_ascii=False)+"\n")
man={"name":NAME,"total":2000,"cliff":400,"frontier":150,"solved":1450,
     "subset_of":str(MIX),"cliff_rule":"first 400 of the mix's InT cliff stratum in its stable_hash('int-mix',problem) order (disjoint from the 300-row L5 cliff holdout by construction)",
     "frontier_solved_rule":"stable_hash('int-l2', question) order over int_mix6k's frontier/solved strata (verbatim mix6k rows)",
     "mirrors":"L2 hybrid openr1_hybrid_c400_n2000 (cliff 400 / frontier 150 / solved 1450)",
     "gold":"meta.gold_solution present on all rows"}
json.dump(man, open(OUT/f"{NAME}_manifest.json","w"), indent=2, ensure_ascii=False)
print(json.dumps(man,indent=2,ensure_ascii=False))
recs=ADAPTERS["local_jsonl"]().load({"path":str(OUT/f"{NAME}.jsonl")})
gs=sum(1 for r in recs if r.meta.get("gold_solution"))
src=collections.Counter(r.meta.get("row_source","?") for r in recs if r.qid.startswith("int-"))
print(f"\nadapter: {len(recs)} records | gold_solution {gs} | cliff sources {dict(src.most_common())}")
