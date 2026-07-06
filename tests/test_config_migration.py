import json

from src.modules.config import Config, CONFIG_VERSION


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


def test_validate_new_keys(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch, {
        "asr_backend": "bogus",
        "asr_beam_size": 99,
        "min_confidence": 7.5,
    })
    assert config.validate() is False
    assert config.asr_backend == "faster-whisper"
    assert config.asr_beam_size == 5
    assert config.min_confidence == 0.30
