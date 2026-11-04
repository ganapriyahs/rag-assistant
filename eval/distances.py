import csv

from app.rag_chain import get_collection

collection = get_collection()
with open("eval/questions.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

scores = []
for row in rows:
    res = collection.query(query_texts=[row["question"]], n_results=3, include=["distances"])
    scores.append((res["distances"][0][0], row["answerable"], row["question"]))

for dist, answerable, question in sorted(scores):
    print(f"{dist:.3f}  [{answerable:3}]  {question}")