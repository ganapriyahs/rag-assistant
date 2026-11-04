from app.tutor import check_cloze, exact_quote


def test_check_cloze_accepts_the_term_in_different_forms():
    assert check_cloze("distillation", "Distillation")
    assert check_cloze("It's distillation.", "Distillation")
    assert check_cloze("fine tuning", "fine-tuning")
    assert check_cloze("inference time", "inference times")


def test_check_cloze_rejects_wrong_or_empty_answers():
    assert not check_cloze("I don't know", "Distillation")
    assert not check_cloze("", "Distillation")
    assert not check_cloze("tuning", "fine-tuning")  # only half of the term
    assert not check_cloze("a long answer that happens to mention distillation somewhere inside", "distillation")


def test_exact_quote_restores_spaces_lost_at_line_breaks():
    lesson = "shows the student\nmodel the teacher model’s answers"
    quote = "shows the studentmodel the teacher model's answers"
    assert exact_quote(lesson, quote) == "shows the student model the teacher model’s answers"