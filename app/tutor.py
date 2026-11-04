import json
import os
import random
import re
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
BLANK = "_____"

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
CLOZE_SCHEMA = {
    "type": "object",
    "properties": {"blank": {"type": "string"}},
    "required": ["blank"],
}

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


def exact_quote(lesson: str, quote: str) -> str:
    """Return the quote as it is written in the lesson, with line breaks turned into spaces.

    The model's copy can lose spaces at line breaks ("actualprobability"); this finds the
    real text again. Returns the model's copy unchanged if it can't be located.
    """
    squashed, positions = [], []
    for i, ch in enumerate(lesson):
        for c in squash(ch):  # squash() drops whitespace, so spaces and newlines add nothing
            squashed.append(c)
            positions.append(i)
    target = squash(quote)
    at = "".join(squashed).find(target)
    if not target or at == -1:
        return quote
    start, end = positions[at], positions[at + len(target) - 1]
    return " ".join(lesson[start : end + 1].split())


def tokens(s: str) -> list[str]:
    """Lowercase words with simple plural endings removed, e.g. 'Models!' -> 'model'."""
    words = re.findall(r"[a-z0-9]+", s.lower().replace("’", "'"))
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]


def explain(lesson: str, tries: int = 3) -> tuple[str, list[str]]:
    """Teach the lesson as short points, each backed by a sentence quoted from the book.

    Points whose quote isn't really in the lesson are dropped. Returns the text to show
    and the list of exact quotes it used.
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
    for p in good:
        p["quote"] = exact_quote(lesson, p["quote"])
    text = "\n\n".join(
        f"{i}. {p['explanation']}\n   Book: \"{p['quote']}\"" for i, p in enumerate(good, start=1)
    )
    return text, [p["quote"] for p in good]


def make_cloze(quotes: list[str], title: str, tries: int = 2) -> dict:
    """Build a fill-in-the-blank question from one of the sentences the explanation just taught.

    The model only chooses which term to blank out. Our code checks that the term appears in
    the sentence exactly once (or the answer would leak) and isn't just the lesson's own title.
    """
    for quote in random.sample(quotes, len(quotes)):
        prompt = (
            f"SENTENCE: {quote}\n\n"
            "Choose the single most important technical term in this sentence (1 to 3 words), "
            "copied exactly as it appears. It must not be a common everyday word."
        )
        for _ in range(tries):
            raw = chat(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                schema=CLOZE_SCHEMA,
                temperature=0.5,
            )
            try:
                blank = json.loads(raw).get("blank", "").strip()
            except json.JSONDecodeError:
                continue
            pattern = re.compile(re.escape(blank), re.IGNORECASE)
            if len(blank) < 3 or len(pattern.findall(quote)) != 1:
                continue
            if all(t in tokens(title) for t in tokens(blank)):  # would just give away the lesson title
                continue
            return {"question": pattern.sub(BLANK, quote), "answer": blank, "evidence": quote}
    raise RuntimeError("Could not build a fill-in-the-blank question from the lesson.")


def check_cloze(answer: str, blank: str) -> bool:
    """Correct if every word of the expected term is in the answer, with at most 3 extra words."""
    want, got = tokens(blank), tokens(answer)
    return bool(want) and all(w in got for w in want) and len(got) <= len(want) + 3


def main():
    number = sys.argv[1]
    text = BOOK_PATH.read_text(encoding="utf-8")
    entries = build_outline(text)
    entry = next(e for e in entries if e["number"] == number)
    lesson = section_text(text, entries, number).replace("\f", "\n")

    print(f"\n=== {entry['label']} (PDF p. {entry['pdf_page']}) ===\n")
    explanation, quotes = explain(lesson)
    print(explanation)

    c = make_cloze(quotes, entry["title"])
    print(f"\nFill in the blank:\n  {c['question']}")

    if len(sys.argv) > 2:
        answer = sys.argv[2]
        print(f"\nYour answer: {answer}")
    else:
        answer = input("\nYour answer: ")

    if check_cloze(answer, c["answer"]):
        print("\nCorrect.")
    else:
        print(f"\nNot yet. The missing term was: {c['answer']}")
    print(f"From the book (PDF p. {entry['pdf_page']}): \"{c['evidence']}\"")


if __name__ == "__main__":
    main()