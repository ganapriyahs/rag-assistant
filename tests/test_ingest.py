from app.ingest import chunk_text


def test_chunk_text_splits_long_text():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_chunk_text_respects_overlap():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    # consecutive chunks should share the overlapping tail/head
    assert chunks[0][-50:] == text[450:500]


def test_chunk_text_handles_short_text():
    text = "short text"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_chunk_text_strips_whitespace_only_chunks():
    text = "real content" + " " * 600  # trailing chunk would be all whitespace
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert all(c.strip() for c in chunks)