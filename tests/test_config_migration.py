import json

from src.modules.config import (Config, CONFIG_VERSION,
                                HOLOLIVE_HOTWORD_PRESETS, merge_hotwords)


def make_config(tmp_path, monkeypatch, contents=None):
    monkeypatch.chdir(tmp_path)
    if contents is not None:
        with open(tmp_path / "translator_config.json", "w") as f:
            json.dump(contents, f)
    return Config()


def test_fresh_config_uses_new_defaults(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch)
    assert config.dynamic_max_chunk_duration == 8.0
    assert config.dynamic_silence_timeout == 0.9
    assert config.asr_backend == "faster-whisper"
    assert config.asr_beam_size == 5
    assert config.min_confidence == 0.30
    assert config.config_version == CONFIG_VERSION


def test_v1_config_with_old_defaults_is_migrated(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch, {
        "dynamic_max_chunk_duration": 15.0,
        "dynamic_silence_timeout": 1.2,
    })
    assert config.dynamic_max_chunk_duration == 8.0
    assert config.dynamic_silence_timeout == 0.9
    assert config.config_version == CONFIG_VERSION


def test_v1_config_with_custom_values_is_preserved(tmp_path, monkeypatch):
    # fbk.json-shaped preset: user-tuned chunking must survive migration
    config = make_config(tmp_path, monkeypatch, {
        "dynamic_max_chunk_duration": 5.0,
        "dynamic_silence_timeout": 1.0,
        "vad_threshold": 0.3,
        "volume_threshold": 0.005,
        "use_speaker_diarization": True,
    })
    assert config.dynamic_max_chunk_duration == 5.0
    assert config.dynamic_silence_timeout == 1.0
    assert config.vad_threshold == 0.3
    assert config.use_speaker_diarization is True
    assert config.config_version == CONFIG_VERSION


def test_v2_config_not_remigrated(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch, {
        "config_version": 2,
        "dynamic_max_chunk_duration": 15.0,  # deliberately set by the user
    })
    assert config.dynamic_max_chunk_duration == 15.0
    assert config.config_version == CONFIG_VERSION


def test_v2_empty_hotwords_migrated_to_preset(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch, {
        "config_version": 2,
        "asr_hotwords": "",
    })
    assert config.asr_hotwords == HOLOLIVE_HOTWORD_PRESETS["JP"]
    assert config.config_version == CONFIG_VERSION


def test_v2_custom_hotwords_preserved(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch, {
        "config_version": 2,
        "asr_hotwords": "Shirakami Fubuki, sukonbu",
    })
    assert config.asr_hotwords == "Shirakami Fubuki, sukonbu"
    assert config.config_version == CONFIG_VERSION


def test_v3_empty_hotwords_not_remigrated(tmp_path, monkeypatch):
    # A v3 user who deliberately cleared the field keeps it empty
    config = make_config(tmp_path, monkeypatch, {
        "config_version": 3,
        "asr_hotwords": "",
    })
    assert config.asr_hotwords == ""


def test_hotword_presets_fit_asr_budget():
    # faster-whisper truncates hotwords at ~223 tokens; ~700 chars of romaji
    # stays comfortably under that, so no preset silently loses its tail
    for branch, preset in HOLOLIVE_HOTWORD_PRESETS.items():
        assert len(preset) < 700, f"{branch} preset too long for ASR hotwords"


def test_merge_hotwords():
    assert merge_hotwords("", "A, B") == "A, B"
    assert merge_hotwords("A, B", "") == "A, B"
    assert merge_hotwords("A, B", "b, C") == "A, B, C"  # case-insensitive dedup
    assert merge_hotwords("A, B,", " C ,, D") == "A, B, C, D"  # stray commas/spaces
    jp = HOLOLIVE_HOTWORD_PRESETS["JP"]
    assert merge_hotwords(jp, jp) == jp  # idempotent


def test_validate_new_keys(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch, {
        "asr_backend": "bogus",
        "asr_beam_size": 99,
        "min_confidence": 7.5,
        "translation_engine": "google",
    })
    assert config.validate() is False
    assert config.asr_backend == "faster-whisper"
    assert config.asr_beam_size == 5
    assert config.min_confidence == 0.30
    assert config.translation_engine == "whisper"


def test_translation_defaults(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch)
    assert config.translation_engine == "whisper"
    assert config.deepl_api_key is None
    assert config.asr_hotwords == HOLOLIVE_HOTWORD_PRESETS["JP"]
    assert config.glossary == {}
