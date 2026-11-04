import csv

from app.rag_chain import generate_answer

with open("eval/questions.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

unanswerable = [r for r in rows if r["answerable"] == "no"]
controls = [r for r in rows if r["answerable"] == "yes"][:3]

for row in unanswerable + controls:
    result = generate_answer(row["question"])
    print("Q:", row["question"], f"[answerable: {row['answerable']}]")
    print("A:", result["answer"])
    print("sources:", result["sources"], "\n")