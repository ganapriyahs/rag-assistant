import sys

from app.rag_chain import generate_answer

question = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

for i in range(runs):
    result = generate_answer(question)
    print(f"--- run {i + 1} | sources: {result['sources']}")
    print(result["answer"][:300])