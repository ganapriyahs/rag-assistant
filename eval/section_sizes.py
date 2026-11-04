from pathlib import Path

from app.outline import build_outline, section_text

text = Path("data/sample_docs/inference_engineering.txt").read_text(encoding="utf-8")
entries = build_outline(text)

sizes = []
for level in (1, 2):
    sizes = sorted(
        (len(section_text(text, entries, e["number"])), e["number"], e["title"])
        for e in entries
        if e["level"] == level
    )
    lengths = [s[0] for s in sizes]
    print(f"level {level}: {len(sizes)} sections, median {lengths[len(lengths) // 2]} chars, max {lengths[-1]}")
    print("  smallest:", sizes[:2])
    print("  largest: ", sizes[-3:])