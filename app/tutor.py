import json
import os
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv

from app.outline import build_outline, section_text

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
BOOK_PATH = Path(__file__).parent.parent / "data" / "sample_docs" / "inference_engineering.txt"
NUM_CTX = 8192  # Ollama's default window is small; a lesson plus the conversation needs more

SYSTEM = (
    "You are a patient tutor teaching one lesson from a textbook. "
    "Use ONLY the lesson text you are given. If the text doesn't cover something, say so."
)

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["quote", "explanation"],
            },
        }
    },
    "required": ["points"],
}
# The property order matters: the model fills the fields in order, so it quotes the
# evidence first and then writes a question that the quote answers.
QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "question": {"type": "string"},
        "answer_key": {"type": "string"},
    },
    "required": ["evidence", "question", "answer_key"],
}
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partly_correct", "not_yet"]},
        "feedback": {"type": "string"},
    },
    "required": ["verdict", "feedback"],
}
VERDICT_LABELS = {"correct": "Correct", "partly_correct": "Partly correct", "not_yet": "Not yet"}

client = ollama.Client(host=OLLAMA_HOST)


def chat(messages: list[dict], schema: dict | None = None, temperature: float | None = None) -> str:
    options = {"num_ctx": NUM_CTX}
    if temperature is not None:
        options["temperature"] = temperature
    response = client.chat(model=OLLAMA_MODEL, messages=messages, options=options, format=schema)
    return response["message"]["content"]


def squash(s: str) -> str:
    """Lowercase, straighten quotes and drop all whitespace, so a quote matches across line breaks."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return "".join(s.lower().split())


def explain(lesson: str, tries: int = 3) -> str:
    """Teach the lesson as short points, each backed by a sentence quoted from the book.

    Points whose quote isn't really in the lesson are dropped.
    """
    prompt = (
        f"LESSON TEXT:\n{lesson}\n\n"
        "Teach this lesson to a complete beginner in 3 to 5 points. For each point, copy ONE sentence "
        "from the lesson exactly as 'quote', then write 'explanation': that sentence restated in simple "
        "words, using only what the sentence says and adding no new facts."
    )
    good = []
    for _ in range(tries):
        raw = chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            schema=EXPLAIN_SCHEMA,
            temperature=0.3,
        )
        try:
            points = json.loads(raw)["points"]
        except (json.JSONDecodeError, KeyError):
            continue
        good = [p for p in points if p.get("quote") and squash(p["quote"]) in squash(lesson)]
        if len(good) >= 2:
            break
    if not good:
        raise RuntimeError("Could not produce an explanation backed by quotes from the lesson.")
    return "\n\n".join(
        f"{i}. {p['explanation']}\n   Book: \"{p['quote']}\"" for i, p in enumerate(good, start=1)
    )


def make_question(lesson: str, tries: int = 3) -> dict:
    """Ask for a question built on a sentence quoted from the lesson, and check the quote is real."""
    prompt = (
        f"LESSON TEXT:\n{lesson}\n\n"
        "Pick ONE important sentence from the lesson and copy it exactly as 'evidence'. "
        "Then write ONE question that can be answered using only that sentence, and an 'answer_key': "
        "ONE short sentence stating the single main idea a correct answer must contain. "
        "The question must not mention 'the lesson' or 'the text'."
    )
    for _ in range(tries):
        raw = chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            schema=QUESTION_SCHEMA,
            temperature=0.3,
        )
        try:
            q = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if q.get("evidence") and squash(q["evidence"]) in squash(lesson):
            return q
    raise RuntimeError("Could not produce a question whose evidence quotes the lesson.")


def grade(question: str, answer_key: str, answer: str) -> dict:
    prompt = (
        f"QUESTION: {question}\n"
        f"ANSWER KEY: {answer_key}\n"
        f"STUDENT ANSWER: {answer}\n\n"
        "Use 'correct' if the student's answer expresses the main idea of the answer key, even in "
        "different words or with less detail. Use 'partly_correct' only if it lacks the main idea but "
        "contains related true content. Use 'not_yet' if it is wrong. An empty, off-topic, "
        "'I don't know' or placeholder answer is always 'not_yet'. "
        "In 'feedback', explain in 1-2 sentences what the answer key says."
    )
    raw = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        schema=GRADE_SCHEMA,
        temperature=0,
    )
    return json.loads(raw)


def main():
    number = sys.argv[1]
    text = BOOK_PATH.read_text(encoding="utf-8")
    entries = build_outline(text)
    entry = next(e for e in entries if e["number"] == number)
    lesson = section_text(text, entries, number).replace("\f", "\n")

    print(f"\n=== {entry['label']} (PDF p. {entry['pdf_page']}) ===\n")
    print(explain(lesson))

    q = make_question(lesson)
    print(f"\nQuestion: {q['question']}")

    if len(sys.argv) > 2:
        answer = sys.argv[2]
        print(f"\nYour answer: {answer}")
    else:
        answer = input("\nYour answer: ")

    result = grade(q["question"], q["answer_key"], answer)
    print(f"\n{VERDICT_LABELS[result['verdict']]}. {result['feedback']}")
    print(f"\nAnswer key: {q['answer_key']}")
    print(f"From the book (PDF p. {entry['pdf_page']}): \"{q['evidence']}\"")


if __name__ == "__main__":
    main()