from pathlib import Path

from app.outline import build_outline, section_text

text = Path("data/sample_docs/inference_engineering.txt").read_text(encoding="utf-8")
entries = build_outline(text)

# a leaf has nothing nested under it: the next entry is not deeper
leaves = [
    e for i, e in enumerate(entries)
    if i + 1 == len(entries) or entries[i + 1]["level"] <= e["level"]
]
sizes = sorted((len(section_text(text, entries, e["number"])), e["number"], e["title"]) for e in leaves)
lengths = [s[0] for s in sizes]

print(f"{len(leaves)} leaves, median {lengths[len(lengths) // 2]} chars, max {lengths[-1]}")
print("over 8000 chars:", [s for s in sizes if s[0] > 8000])

last = next(e for e in entries if e["number"] == "7.6")
print("END OF 7.6:", repr(section_text(text, entries, "7.6")[-500:]))