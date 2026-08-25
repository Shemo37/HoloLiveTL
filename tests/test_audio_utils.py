import numpy as np

from src.modules.audio_utils import normalize_audio, enhance_audio_quality


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def test_near_silence_is_not_amplified():
    audio = np.random.default_rng(0).normal(0, 0.0001, 16000).astype(np.float32)
    out = normalize_audio(audio.copy())
    assert rms(out) <= rms(audio) * 1.01


def test_quiet_speech_gain_capped_at_5x():
    audio = 0.01 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000)).astype(np.float32)
    out = normalize_audio(audio.copy())
    assert rms(out) <= rms(audio) * 5.0 * 1.01
    assert rms(out) > rms(audio)  # still boosted


def test_normal_speech_reaches_target_rms():
    audio = 0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000)).astype(np.float32)
    out = normalize_audio(audio.copy())
    assert abs(rms(out) - 0.3) < 0.02


def test_loud_audio_peak_capped():
    audio = 1.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000)).astype(np.float32)
    out = normalize_audio(audio.copy())
    assert np.max(np.abs(out)) <= 0.95 + 1e-6


def test_enhance_audio_quality_no_nans():
    for scale in (0.0, 0.0001, 0.05, 1.0, 10.0):
        audio = scale * np.random.default_rng(1).normal(0, 1, 16000).astype(np.float32)
        out = enhance_audio_quality(audio, sample_rate=16000)
        assert np.all(np.isfinite(out))


def test_enhance_audio_quality_removes_dc_offset():
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000)) + 0.5).astype(np.float32)
    out = enhance_audio_quality(audio, sample_rate=16000)
    assert abs(np.mean(out)) < 0.01
