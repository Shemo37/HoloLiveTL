import builtins
import sys
from types import SimpleNamespace

import pytest

from src.modules import asr_backend
from src.modules.asr_backend import create_backend, TransformersBackend, FasterWhisperBackend


def make_config(backend="faster-whisper"):
    return SimpleNamespace(
        asr_backend=backend,
        asr_beam_size=5,
        use_dynamic_chunking=True,
        model_cache_dir="/nonexistent",
    )


def test_falls_back_to_transformers_when_faster_whisper_missing(monkeypatch):
    # Simulate faster_whisper not being installed
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(FasterWhisperBackend, "load",
                        lambda self: (_ for _ in ()).throw(ImportError("no faster_whisper")))
    monkeypatch.setattr(TransformersBackend, "load", lambda self: None)

    backend = create_backend(make_config(), "cpu")
    assert isinstance(backend, TransformersBackend)


def test_explicit_transformers_backend_skips_faster_whisper(monkeypatch):
    def boom(self):
        raise AssertionError("faster-whisper should not be loaded")

    monkeypatch.setattr(FasterWhisperBackend, "load", boom)
    monkeypatch.setattr(TransformersBackend, "load", lambda self: None)

    backend = create_backend(make_config("transformers"), "cpu")
    assert isinstance(backend, TransformersBackend)


def test_fallback_posts_gui_status(monkeypatch):
    monkeypatch.setattr(FasterWhisperBackend, "load",
                        lambda self: (_ for _ in ()).throw(RuntimeError("cuDNN missing")))
    monkeypatch.setattr(TransformersBackend, "load", lambda self: None)

    messages = []
    gui_queue = SimpleNamespace(put=messages.append)
    backend = create_backend(make_config(), "cuda:0", gui_queue)

    assert isinstance(backend, TransformersBackend)
    assert any(msg[0] == "status" for msg in messages)


def test_faster_whisper_selected_when_available(monkeypatch):
    monkeypatch.setattr(FasterWhisperBackend, "load", lambda self: None)
    backend = create_backend(make_config(), "cuda:0")
    assert isinstance(backend, FasterWhisperBackend)
    assert backend.device == "cuda"
