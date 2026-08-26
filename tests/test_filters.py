import string

from src.modules.filters import (is_hallucination, post_process_translation,
                                 trim_degenerate_tail)

TRANSLATOR = str.maketrans('', '', string.punctuation)


def test_empty_output_filtered():
    assert is_hallucination("", TRANSLATOR, [])
    assert is_hallucination("   ", TRANSLATOR, [])


def test_short_phrases_pass_after_list_removal():
    # Phrase lists were removed by request: short real utterances display now.
    assert not is_hallucination("Thank you for watching!", TRANSLATOR, [])
    assert not is_hallucination("Thanks", TRANSLATOR, [])
    assert not is_hallucination("Okay", TRANSLATOR, [])


def test_punctuation_only_filtered():
    assert is_hallucination(".", TRANSLATOR, [])
    assert is_hallucination("!!", TRANSLATOR, [])
    assert is_hallucination("...", TRANSLATOR, [])


def test_refusal_boilerplate_filtered():
    assert is_hallucination(
        "申し訳ありませんが、この文章は文脈が不明確であるため、正確な翻訳を提供できません。",
        TRANSLATOR, [])
    assert is_hallucination("I cannot provide an accurate translation of this.", TRANSLATOR, [])


def test_leading_fragment_punctuation_stripped():
    assert post_process_translation(", so I'm going to the gym teacher.") == \
        "So I'm going to the gym teacher."


def test_trim_degenerate_tail_salvages_prefix():
    line = ("The previous meeting felt rather drewd, a bit sluggish at the very "
            "same time, I was in the gustle and gawk of a little bit more slurry "
            "than I was in the slugger-thump-thump-thump-thres-thres-thres-thres"
            "-h-h-h-h-h-h-h-h-h-h-h-h-h-h-h-h")
    trimmed = trim_degenerate_tail(line)
    assert trimmed.startswith("The previous meeting felt rather drewd")
    assert "thres-thres" not in trimmed
    assert "h-h-h" not in trimmed


def test_trim_degenerate_tail_keeps_clean_text():
    clean = "Today we're playing Minecraft together with everyone."
    assert trim_degenerate_tail(clean) == clean


def test_trim_degenerate_tail_all_junk_becomes_empty():
    assert trim_degenerate_tail("ununununununununun") == ""
    assert trim_degenerate_tail("thres-thres-thres-thres-thres") == ""


def test_real_speech_not_filtered():
    assert not is_hallucination("Today we're playing Minecraft together", TRANSLATOR, [])
    assert not is_hallucination("That boss fight was really hard", TRANSLATOR, [])


def test_recent_repetition_filtered():
    history = ["Let's go", "Let's go", "Let's go"]
    assert is_hallucination("Let's go", TRANSLATOR, history)


def test_post_process_translation():
    assert post_process_translation("hello   world .") == "Hello world."
    assert post_process_translation("i think im ready") == "I think I'm ready"
    assert post_process_translation("what!!!") == "What!"
