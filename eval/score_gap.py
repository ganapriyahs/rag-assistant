import csv

from app.rag_chain import retrieve


def norm(s: str) -> str:
    return "".join(s.lower().split())


with open("eval/questions.csv", newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["answerable"].strip().lower() == "yes"]

for r in rows:
    keys = [norm(k) for k in r["key_phrase"].split("|")]
    chunks = retrieve(r["question"])
    top = chunks[0]["rerank_score"]
    print(r["question"])
    for c in chunks:
        hit = any(k in norm(c["text"]) for k in keys)
        print(f"   {c['rerank_score']:6.2f}   gap {top - c['rerank_score']:5.2f}   {'HIT' if hit else ''}")