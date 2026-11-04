import json
from datetime import date
from pathlib import Path

PROGRESS_PATH = Path(__file__).parent.parent / "data" / "progress.json"
MAX_REVIEW_ATTEMPTS = 3  # stop re-offering a missed lesson after this many tries


def load_progress(path: Path = PROGRESS_PATH) -> dict:
    """Read the learner's progress file; a missing file means a fresh start."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(progress: dict, path: Path = PROGRESS_PATH) -> None:
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def record_result(progress: dict, number: str, correct: bool, today: str | None = None) -> None:
    """Update the progress dict in place after one quiz question."""
    item = progress.setdefault(number, {"attempts": 0, "correct": 0})
    item["attempts"] += 1
    item["correct"] += int(correct)
    item["last_correct"] = correct
    item["last_date"] = today or date.today().isoformat()


def lessons(entries: list[dict]) -> list[dict]:
    """The teachable lessons in book order: outline leaves, excluding back matter."""
    result = []
    for i, e in enumerate(entries):
        is_leaf = i + 1 == len(entries) or entries[i + 1]["level"] <= e["level"]
        if is_leaf and not e.get("backmatter"):
            result.append(e)
    return result


def choose_next(entries: list[dict], progress: dict) -> dict | None:
    """Pick the next lesson: a recently missed one first, otherwise the first unseen one.

    Returns None when every lesson has been done.
    """
    todo = lessons(entries)
    for e in todo:
        item = progress.get(e["number"])
        if item and item["last_correct"] is False and item["attempts"] < MAX_REVIEW_ATTEMPTS:
            return e
    for e in todo:
        if e["number"] not in progress:
            return e
    return None