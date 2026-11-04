import csv
import json
import re
from datetime import datetime

from app.ingest import CHUNK_SIZE, CHUNK_OVERLAP
from app.rag_chain import retrieve, TOP_K


def norm(s: str) -> str:
    """Remove all whitespace and lowercase, so PDF spacing quirks don't matter."""
    return re.sub(r"\s+", "", s).lower()


def load_questions(path="eval/questions.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate():
    results = []
    for row in load_questions():
        chunks = retrieve(row["question"], top_k=TOP_K)
        answerable = row["answerable"] == "yes"

        rank = None  # position (1..k) of the first chunk containing the key phrase
        if answerable:
            keys = [norm(k) for k in row["key_phrase"].split("|")]
            for i, c in enumerate(chunks, start=1):
                text = norm(c["text"])
                if any(k in text for k in keys):
                    rank = i
                    break

        results.append({
            "question": row["question"],
            "answerable": answerable,
            "rank": rank,
            "top_chunk_preview": chunks[0]["text"][:150].replace("\n", " "),
        })
    return results


def report(results):
    answerable = [r for r in results if r["answerable"]]
    hit3 = sum(1 for r in answerable if r["rank"] is not None)
    hit1 = sum(1 for r in answerable if r["rank"] == 1)
    for r in answerable:
        status = f"HIT@{r['rank']}" if r["rank"] else "MISS"
        print(f"{status:7} {r['question']}")
        if not r["rank"]:
            print(f"        top chunk: {r['top_chunk_preview']}")
    print(f"\nhit@1: {hit1}/{len(answerable)}   hit@{TOP_K}: {hit3}/{len(answerable)}")


if __name__ == "__main__":
    results = evaluate()
    report(results)

    out = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "config": {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "top_k": TOP_K},
        "results": results,
    }
    with open("eval/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)