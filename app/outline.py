import re

from app.ingest import CHAPTER_HEADER, build_page_index, locate

# A table-of-contents line: a title, dotted leaders (".....", ". ....."), then a page number.
LEADER = re.compile(r"^(?P<title>.*?[^\s.])\s*(?:\.\s*){2,}(?P<page>\d+)\s*$")
BARE_NUMBER = re.compile(r"^\d+\.\d+$")  # a section number sitting alone on its line
NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")  # "1.2.1 Title"
CHAPTER = re.compile(r"^Chapter (\d+):\s*(.+)$")
TOC_TO_PDF_PAGE = 2

def parse_toc(text: str) -> list[dict]:
    """Parse the table of contents (everything before the first real chapter header) into an outline."""
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
        if mc:
            entries.append({"number": mc.group(1), "title": mc.group(2).strip(), "level": 0, "toc_page": page})
        else:
            mn = NUMBERED.match(title)
            if mn:
                number, name = mn.group(1), mn.group(2)
            elif pending:
                number, name = pending, title
            else:
                pending = None
                continue
            entries.append(
                {"number": number, "title": name.strip(), "level": number.count("."), "toc_page": page}
            )
        pending = None
    return entries

def locate_entries(text: str, entries: list[dict]) -> list[dict]:
    """Add 'pdf_page' to each outline entry: the PDF page where its heading appears.

    Chapters use the table-of-contents page plus a verified constant, because the
    extracted text has no divider lines to search for. Sections are searched for
    in the body. The value is None if a heading can't be found.
    """
    body = re.search(r"^Chapter \d+: [^.\n]+$", text, re.M)
    start = body.start() if body else 0  # skip the table of contents itself
    index = build_page_index(text)

    for e in entries:
        if e["level"] == 0:
            e["pdf_page"] = e["toc_page"] + TOC_TO_PDF_PAGE
            continue
        title = r"\s+".join(re.escape(word) for word in e["title"].split())
        # a running page header ("Chapter 7: Production") can sit between number and title
        prefix = rf"{re.escape(e['number'])}\s+(?:Chapter\s+\d+:[^\n]*\s+)?"
        m = re.compile(rf"(?:^|\f){prefix}{title}", re.M).search(text, start)
        # m.end() - 1 is a character inside the title text, so it can't be the form feed itself
        e["pdf_page"] = locate(index, m.end() - 1)[0] if m else None
    return entries