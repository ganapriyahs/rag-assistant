from app.progress import choose_next, lessons, record_result


def make_entries():
    return [
        {"number": "1", "level": 0},
        {"number": "1.1", "level": 1},
        {"number": "1.2", "level": 1},
        {"number": "1.2.1", "level": 2},
        {"number": "1.2.2", "level": 2},
        {"number": "2", "level": 0},
        {"number": "A", "level": 0, "backmatter": True},
    ]


def test_lessons_are_outline_leaves_without_back_matter():
    assert [e["number"] for e in lessons(make_entries())] == ["1.1", "1.2.1", "1.2.2", "2"]


def test_starts_at_the_first_lesson():
    assert choose_next(make_entries(), {})["number"] == "1.1"


def test_moves_on_after_a_correct_answer():
    progress = {}
    record_result(progress, "1.1", True, today="2026-11-04")
    assert choose_next(make_entries(), progress)["number"] == "1.2.1"


def test_reviews_a_missed_lesson_before_new_ones():
    progress = {}
    record_result(progress, "1.1", True)
    record_result(progress, "1.2.1", False)
    assert choose_next(make_entries(), progress)["number"] == "1.2.1"


def test_gives_up_on_a_lesson_after_three_misses():
    progress = {}
    record_result(progress, "1.1", True)
    for _ in range(3):
        record_result(progress, "1.2.1", False)
    assert choose_next(make_entries(), progress)["number"] == "1.2.2"


def test_returns_none_when_everything_is_done():
    progress = {}
    for number in ["1.1", "1.2.1", "1.2.2", "2"]:
        record_result(progress, number, True)
    assert choose_next(make_entries(), progress) is None


def test_record_result_counts_attempts():
    progress = {}
    record_result(progress, "1.1", True)
    record_result(progress, "1.1", False)
    assert progress["1.1"]["attempts"] == 2
    assert progress["1.1"]["correct"] == 1
    assert progress["1.1"]["last_correct"] is False