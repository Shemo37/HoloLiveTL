import string

from src.modules.filters import is_hallucination, post_process_translation

TRANSLATOR = str.maketrans('', '', string.punctuation)


def test_known_hallucinations_filtered():
    assert is_hallucination("Thank you for watching!", TRANSLATOR, [])
    assert is_hallucination("Don't forget to subscribe", TRANSLATOR, [])
    assert is_hallucination("", TRANSLATOR, [])
    assert is_hallucination("Thanks", TRANSLATOR, [])


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
