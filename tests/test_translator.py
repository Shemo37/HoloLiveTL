from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.modules.translator import (
    DeepLTranslator,
    FuguMTTranslator,
    TranslatorUnavailable,
    apply_glossary,
    create_translator,
    DEEPL_FREE_ENDPOINT,
    DEEPL_PRO_ENDPOINT,
    MAX_CONSECUTIVE_FAILURES,
)
from src.modules.processor import resolve_pipeline


def ok_response(text="Hello"):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"translations": [{"text": text}]}
    return response


def error_response(status_code):
    response = MagicMock()
    response.status_code = status_code
    return response


def test_free_key_uses_free_endpoint():
    assert DeepLTranslator("abc:fx").endpoint == DEEPL_FREE_ENDPOINT
    assert DeepLTranslator("abc").endpoint == DEEPL_PRO_ENDPOINT


def test_translate_success_and_payload():
    translator = DeepLTranslator("key:fx")
    with patch("src.modules.translator.requests.post", return_value=ok_response("Hello")) as post:
        result = translator.translate("こんにちは", context="前の行")

    assert result == "Hello"
    assert translator.chars_sent == len("こんにちは")
    payload = post.call_args.kwargs["json"]
    assert payload["text"] == ["こんにちは"]
    assert payload["source_lang"] == "JA"
    assert payload["target_lang"] == "EN-US"
    assert payload["context"] == "前の行"


def test_context_omitted_when_none():
    translator = DeepLTranslator("key:fx")
    with patch("src.modules.translator.requests.post", return_value=ok_response()) as post:
        translator.translate("こんにちは")
    assert "context" not in post.call_args.kwargs["json"]


def test_quota_exhausted_raises_unavailable():
    translator = DeepLTranslator("key:fx")
    with patch("src.modules.translator.requests.post", return_value=error_response(456)):
        with pytest.raises(TranslatorUnavailable):
            translator.translate("テスト")


def test_bad_key_raises_unavailable():
    translator = DeepLTranslator("key:fx")
    with patch("src.modules.translator.requests.post", return_value=error_response(403)):
        with pytest.raises(TranslatorUnavailable):
            translator.translate("テスト")


def test_single_transient_failure_returns_none():
    translator = DeepLTranslator("key:fx")
    with patch("src.modules.translator.requests.post",
               side_effect=requests.ConnectionError("down")):
        assert translator.translate("テスト") is None


def test_consecutive_failures_raise_unavailable():
    translator = DeepLTranslator("key:fx")
    with patch("src.modules.translator.requests.post",
               side_effect=requests.ConnectionError("down")):
        with pytest.raises(TranslatorUnavailable):
            for _ in range(MAX_CONSECUTIVE_FAILURES):
                translator.translate("テスト")


def test_success_resets_failure_counter():
    translator = DeepLTranslator("key:fx")
    with patch("src.modules.translator.requests.post",
               side_effect=[requests.ConnectionError("down"), ok_response(),
                            requests.ConnectionError("down")]):
        assert translator.translate("a") is None
        assert translator.translate("b") == "Hello"
        assert translator.translate("c") is None  # counter restarted, no raise
    assert translator.consecutive_failures == 1


def test_apply_glossary():
    glossary = {"White God": "Fubuki", "fox": "Fubuki"}
    assert apply_glossary("The white god laughed", glossary) == "The Fubuki laughed"
    assert apply_glossary("A foxtrot", glossary) == "A foxtrot"  # whole word only
    assert apply_glossary("no matches here", {}) == "no matches here"


def test_create_translator_requires_engine_and_key():
    assert create_translator(SimpleNamespace(translation_engine="whisper",
                                             deepl_api_key="key:fx")) is None
    assert create_translator(SimpleNamespace(translation_engine="deepl",
                                             deepl_api_key=None)) is None
    translator = create_translator(SimpleNamespace(translation_engine="deepl",
                                                   deepl_api_key="key:fx"))
    assert isinstance(translator, DeepLTranslator)


def make_fugumt(pipe):
    translator = FuguMTTranslator(SimpleNamespace(model_cache_dir="/tmp"))
    translator.pipe = pipe
    return translator


def test_fugumt_translate_success():
    pipe = MagicMock(return_value=[{"translation_text": "The cat is cute."}])
    translator = make_fugumt(pipe)
    assert translator.translate("猫はかわいいです。") == "The cat is cute."
    pipe.assert_called_once_with("猫はかわいいです。")


def test_fugumt_context_ignored():
    pipe = MagicMock(return_value=[{"translation_text": "Hello"}])
    translator = make_fugumt(pipe)
    assert translator.translate("こんにちは", context="前の行") == "Hello"
    pipe.assert_called_once_with("こんにちは")  # context not forwarded


def test_fugumt_transient_failure_then_unavailable():
    translator = make_fugumt(MagicMock(side_effect=RuntimeError("boom")))
    assert translator.translate("テスト") is None  # first failure: skip chunk
    with pytest.raises(TranslatorUnavailable):
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            translator.translate("テスト")


def test_create_translator_fugumt(monkeypatch):
    monkeypatch.setattr(FuguMTTranslator, "load", lambda self: None)
    translator = create_translator(SimpleNamespace(translation_engine="fugumt",
                                                   model_cache_dir="/tmp"))
    assert isinstance(translator, FuguMTTranslator)


def test_create_translator_fugumt_load_failure_falls_back(monkeypatch):
    def boom(self):
        raise ImportError("no transformers")
    monkeypatch.setattr(FuguMTTranslator, "load", boom)
    assert create_translator(SimpleNamespace(translation_engine="fugumt",
                                             model_cache_dir="/tmp")) is None


def test_resolve_pipeline():
    config = SimpleNamespace(output_mode="translate", language_code="en")
    translator = object()
    assert resolve_pipeline(config, translator) == ("transcribe", "ja", True)
    assert resolve_pipeline(config, None) == ("translate", "en", False)

    config = SimpleNamespace(output_mode="transcribe", language_code="ja")
    assert resolve_pipeline(config, translator) == ("transcribe", "ja", False)
