import re

from app.ingest import CHAPTER_HEADER, build_page_index, locate

# A table-of-contents line: a title, dotted leaders (".....", ". ....."), then a page number.
LEADER = re.compile(r"^(?P<title>.*?[^\s.])\s*(?:\.\s*){2,}(?P<page>\d+)\s*$")
BARE_NUMBER = re.compile(r"^\d+\.\d+$")  # a section number sitting alone on its line
NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")  # "1.2.1 Title"
CHAPTER = re.compile(r"^Chapter (\d+):\s*(.+)$")
APPENDIX = re.compile(r"^Appendix ([A-Z]):\s*(.+)$")

# Verified against the PDF: every chapter and appendix divider page ("CHAPTER N",
# "APPENDIX A") and every numbered heading sits exactly 2 pages after the page number
# the table of contents lists.
TOC_TO_PDF_PAGE = 2


def parse_toc(text: str) -> list[dict]:
    """Parse the table of contents (everything before the first real chapter header) into an outline.

    Each entry has: number, title, level, toc_page and label. Back matter (appendices,
    their unnumbered sub-entries, acknowledgements) is marked backmatter=True.
    """
    lines = text.splitlines()
    end = next((i for i, line in enumerate(lines) if CHAPTER_HEADER.match(line.strip())), len(lines))

    entries = []
    pending = None  # a bare section number waiting for its title on a later line
    for line in lines[:end]:
        s = re.sub(r"[\x00-\x1f\x7f]", " ", line).strip()  # blank out control characters like \x07
        if not s:
            continue
        if BARE_NUMBER.match(s):
            pending = s
            continue

        m = LEADER.match(s)
        if not m:
            pending = None
            continue

        title, page = m.group("title"), int(m.group("page"))
        mc = CHAPTER.match(title)
        ma = APPENDIX.match(title)
        mn = NUMBERED.match(title)

        if mc:
            entries.append(
                {
                    "number": mc.group(1),
                    "title": mc.group(2).strip(),
                    "level": 0,
                    "toc_page": page,
                    "label": f"Chapter {mc.group(1)}: {mc.group(2).strip()}",
                }
            )
        elif ma:
            entries.append(
                {
                    "number": ma.group(1),
                    "title": ma.group(2).strip(),
                    "level": 0,
                    "toc_page": page,
                    "label": f"Appendix {ma.group(1)}: {ma.group(2).strip()}",
                    "backmatter": True,
                }
            )
        elif mn or pending:
            number, name = (mn.group(1), mn.group(2)) if mn else (pending, title)
            entries.append(
                {
                    "number": number,
                    "title": name.strip(),
                    "level": number.count("."),
                    "toc_page": page,
                    "label": f"{number} {name.strip()}",
                }
            )
        elif title.lower().startswith("acknowledg"):
            entries.append(
                {
                    "number": "Ack",
                    "title": title,
                    "level": 0,
                    "toc_page": page,
                    "label": title,
                    "backmatter": True,
                }
            )
        elif entries and entries[-1].get("backmatter") and entries[-1]["number"] != "Ack":
            # an unnumbered sub-entry under the last appendix, e.g. "Architecture" under Appendix B
            parent = entries[-1]["number"].split(".")[0]
            count = sum(1 for e in entries if e["number"].startswith(parent + "."))
            entries.append(
                {
                    "number": f"{parent}.{count + 1}",
                    "title": title,
                    "level": 1,
                    "toc_page": page,
                    "label": title,
                    "backmatter": True,
                }
            )
        pending = None
    return entries


def locate_entries(text: str, entries: list[dict]) -> list[dict]:
    """Add 'pdf_page', 'start_offset' and 'approx' to each outline entry.

    Chapters, appendices and back matter use the table-of-contents page plus a verified
    constant, because the extracted text has no divider lines to search for; they start
    at their page's first character. Numbered sections are searched for in the body,
    taking the earliest match of either heading layout. If a heading can't be found, it
    falls back to the start of its TOC page + 2 and is marked approx=True.
    """
    body = re.search(r"^Chapter \d+: [^.\n]+$", text, re.M)
    start = body.start() if body else 0  # skip the table of contents itself
    index = build_page_index(text)

    for e in entries:
        e["approx"] = False
        if e["level"] == 0 or e.get("backmatter"):
            e["pdf_page"] = e["toc_page"] + TOC_TO_PDF_PAGE
            e["start_offset"] = index[e["pdf_page"] - 1][0]
            continue

        title = r"\s+".join(re.escape(word) for word in e["title"].split())
        number = re.escape(e["number"])
        patterns = [
            # Two-line form: the number alone on its line. A running chapter header
            # can sit between the number and the title.
            rf"(?:^|\f){number}[ \t]*\n\s*(?:Chapter\s+\d+:[^\n]*\s+)?{title}",
            # One-line form, with body text on the very next line. The running page
            # header looks similar but is followed by a blank line and a page number.
            rf"(?:^|\f){number}[ \t]+{title}[ \t]*\n(?=\S)",
        ]
        matches = [re.compile(p, re.M).search(text, start) for p in patterns]
        m = min((x for x in matches if x), key=lambda x: x.start(), default=None)

        if m:
            # m.end() - 1 is a character inside the heading, so it can't be a form feed
            e["pdf_page"] = locate(index, m.end() - 1)[0]
            e["start_offset"] = m.start() + (1 if text[m.start()] == "\f" else 0)
        else:
            e["pdf_page"] = e["toc_page"] + TOC_TO_PDF_PAGE
            e["start_offset"] = index[e["pdf_page"] - 1][0]
            e["approx"] = True
    return entries


def build_outline(text: str) -> list[dict]:
    """Parse the table of contents, locate every heading, and add each entry's end_offset."""
    entries = locate_entries(text, parse_toc(text))
    for i, e in enumerate(entries):
        # the section ends where the next heading at the same or a higher level starts
        following = next(
            (n for n in entries[i + 1 :] if n["level"] <= e["level"] and n["start_offset"] is not None),
            None,
        )
        e["end_offset"] = following["start_offset"] if following else len(text)
    return entries


def section_text(text: str, entries: list[dict], number: str) -> str:
    """Return the text of one section, from its heading to the start of the next heading."""
    entry = next(e for e in entries if e["number"] == number)
    if entry["start_offset"] is None:
        raise ValueError(f"Heading {number} was not found in the text")
    return text[entry["start_offset"] : entry["end_offset"]]