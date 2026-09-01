"""int_mix6k: mix6k's solved+frontier strata kept verbatim; the cliff stratum
(2,730) replaced by CMU-AIRe/InT-hard-set rows + a disjoint holdout (<=540),
both drawn from ONE stable_hash-ordered split (exchangeable)."""
import json, re, collections
import datasets as hfd
from pathlib import Path
import expert_iter.verifier  # registry population
from expert_iter.registry import VERIFIERS, ADAPTERS
from expert_iter.records import QuestionRecord
from expert_iter.utils import stable_hash

OUT=Path("data/mixes"); NAME="int_mix6k_qwen3-4b-2507"
CLIFF_QUOTA=2730; HOLDOUT_MAX=540
drops=collections.OrderedDict()

rows=list(hfd.load_dataset("CMU-AIRe/InT-hard-set")["train"]); n0=len(rows)
rows=[ (i,r) for i,r in enumerate(rows) if str(r["answer"]).strip() not in ("","None") ]
drops["empty_answer"]=n0-len(rows)
PROOF=re.compile(r"\b(prove|show that|justify your|demonstrate that)\b", re.I)
k=len(rows); rows=[(i,r) for i,r in rows if not PROOF.search(r["problem"])]; drops["proof_worded"]=k-len(rows)
# self-grade (parallel)
V=VERIFIERS["math_strict"]()
items=[(QuestionRecord(qid=str(i),question="",final_answer=str(r["answer"]).strip(),domain="math"),
        "\\boxed{"+str(r["answer"]).strip()+"}") for i,r in rows]
verd=V.verify_batch(items)
k=len(rows); rows=[(i,r) for (i,r),v in zip(rows,verd) if v.correct]; drops["gold_not_self_gradable"]=k-len(rows)
k=len(rows); rows=[(i,r) for i,r in rows if r["reference_solution"] not in (None,"None","")]; drops["no_reference_solution"]=k-len(rows)
# intra-set dedup (normalized prefix)
def normp(t): return re.sub(r"[^a-z0-9]","",t.lower())[:120]
seen=set(); uniq=[]
for i,r in rows:
    p=normp(r["problem"])
    if p in seen: continue
    seen.add(p); uniq.append((i,r))
drops["intra_dedup"]=len(rows)-len(uniq); rows=uniq
# rename-robust dedup vs ALL benchmarks + mix6k + mix8k cliff holdout
def shingles(t,k=8):
    w=re.sub(r"[^a-z0-9 ]"," ",t.lower()).split()
    return {" ".join(w[i:i+k]) for i in range(max(0,len(w)-k+1))}
import expert_iter.data as data
from expert_iter.config import Config
guard=[]  # (label, question)
cfg=Config.load("configs/bench_eval_hard.yaml")
for b in cfg.eval.benchmarks:
    args=b.adapter_args if b.adapter else {**data.BENCHMARK_PRESETS[b.name],"bench_name":b.name}
    guard+= [("bench:"+b.name,r.question) for r in ADAPTERS[b.adapter or "hf_benchmark"]().load(args)]
BD=Path("runs/bench/Qwen_Qwen3-4B-Instruct-2507_bench_eval_20260901_110721/benchmarks")
for s in ("aime24","aime25","aime26","math500_hard"):
    guard+=[("bench:"+s,json.loads(l)["question"]) for l in open(BD/f"{s}.jsonl")]
mix6k=[json.loads(l) for l in open("data/mixes/openr1_mix6k_qwen3-4b-2507.jsonl")]
guard+=[("mix6k",r["question"]) for r in mix6k]
guard+=[("openr1_holdout",json.loads(l)["question"]) for l in open("data/mixes/openr1_mix8k_qwen3-4b-2507_cliff_holdout.jsonl")]
gidx=collections.defaultdict(set)
for gi,(lab,q) in enumerate(guard):
    for s in shingles(q): gidx[s].add(gi)
def dup_of(p):
    cand=collections.Counter()
    for s in shingles(p):
        g=gidx.get(s)
        if g and len(g)<=3:
            for gi in g: cand[gi]+=1
    if not cand: return None
    gi,shared=cand.most_common(1)[0]
    return guard[gi][0] if shared>=5 else None
k=len(rows); kept=[]; dup_by=collections.Counter()
for i,r in rows:
    d=dup_of(r["problem"])
    if d: dup_by[d]+=1
    else: kept.append((i,r))
rows=kept; drops["dup_vs_bench_or_mix6k"]=k-len(rows)
# hash-ordered split
rows.sort(key=lambda ir: stable_hash("int-mix", ir[1]["problem"]))
def mk(i,r):
    return {"qid":"int-"+stable_hash("int-hard",r["problem"]),
            "question":r["problem"].strip(),"final_answer":str(r["answer"]).strip(),"domain":"math",
            "meta":{"hf_name":"CMU-AIRe/InT-hard-set","split":"train","row_idx":i,
                    "row_source":r["source"],"gold_solution":r["reference_solution"],
                    "gold_solution_source":r["reference_solution_source"],
                    "int_hard_filter":"zero reward @64-128 rollouts on Qwen3-4B-Instruct-2507 (InT paper's sampling)",
                    "passrate_class":"cliff"}}
print("drops:", json.dumps(drops), "| usable InT cliff pool:", len(rows))
HOLDOUT_N=300  # InT-only holdout, exchangeable (same hash-ordered split)
int_cliff=[mk(i,r) for i,r in rows[:len(rows)-HOLDOUT_N]]
hold=[mk(i,r) for i,r in rows[len(rows)-HOLDOUT_N:]]
sf=[r for r in mix6k if r["meta"]["passrate_class"] in ("solved","frontier")]
# top up the cliff stratum to CLIFF_QUOTA with mix6k's own OpenR1 cliffs
# (hash-ordered; cannot overlap InT rows because InT rows duplicating mix6k
# were already dropped in dup_vs_bench_or_mix6k)
need=CLIFF_QUOTA-len(int_cliff)
o_cliffs=[r for r in mix6k if r["meta"]["passrate_class"]=="cliff"]
o_cliffs.sort(key=lambda r: stable_hash("int-mix-topup", r["question"]))
topup=o_cliffs[:max(0,need)]
cliff=int_cliff+topup
mix=sf+cliff
qids=[r["qid"] for r in mix+hold]; assert len(qids)==len(set(qids))
with open(OUT/f"{NAME}.jsonl","w") as f:
    for r in mix: f.write(json.dumps(r,ensure_ascii=False)+"\n")
with open(OUT/f"{NAME}_cliff_holdout.jsonl","w") as f:
    for r in hold: f.write(json.dumps(r,ensure_ascii=False)+"\n")
man={"name":NAME,"total":len(mix),"solved_frontier_from":"openr1_mix6k_qwen3-4b-2507.jsonl (verbatim, passrate_class in {solved,frontier})",
     "n_solved_frontier":len(sf),"n_cliff":len(cliff),"n_cliff_int":len(int_cliff),"n_cliff_openr1_topup":len(topup),"n_cliff_holdout":len(hold),
     "cliff_source":"CMU-AIRe/InT-hard-set[train] (+ openr1 mix6k cliff top-up, hash-ordered, to keep the 2,730 quota)","cliff_pool_after_filters":len(rows),
     "drops":drops,"dup_hit_by":dict(dup_by),
     "order":"stable_hash('int-mix', problem) — hash-deterministic, holdout drawn from the SAME ordered split (exchangeable)",
     "filters":"empty-answer, proof-worded regex, math_strict gold self-grade, reference_solution required (mix AND holdout), intra-set prefix dedup, rename-robust shingle dedup (>=5 rare 8-word shingles) vs 10 benchmarks + mix6k + openr1 mix8k cliff holdout",
     "gold_solution":"meta.gold_solution = InT reference_solution (human/gemini)"}
with open(OUT/f"{NAME}_manifest.json","w") as f: json.dump(man,f,indent=2,ensure_ascii=False)
print(json.dumps(man,indent=2,ensure_ascii=False))
# validation via the adapter
recs=ADAPTERS["local_jsonl"]().load({"path":str(OUT/f"{NAME}.jsonl")})
gs=sum(1 for r in recs if r.meta.get("gold_solution"))
print(f"\nadapter load: {len(recs)} records, gold_solution {gs} ({gs/len(recs)*100:.0f}%)")
h=ADAPTERS["local_jsonl"]().load({"path":str(OUT/f"{NAME}_cliff_holdout.jsonl")})
print(f"holdout: {len(h)} records; overlap with mix qids: {len(set(r.qid for r in h)&set(r.qid for r in recs))}")
srcs=collections.Counter(r.meta.get("row_source","?") for r in recs if str(r.qid).startswith("int-"))
print("cliff row_source:", dict(srcs.most_common()))
