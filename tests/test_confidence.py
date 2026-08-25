import math

from src.modules.asr_backend import (
    ASRSegment,
    confidence_from_segments,
    drop_no_speech_segments,
)


def seg(text="hello", start=0.0, end=1.0, avg_logprob=None, no_speech_prob=None):
    return ASRSegment(text=text, start=start, end=end,
                      avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


def test_confidence_clamped_to_unit_interval():
    high = confidence_from_segments([seg(avg_logprob=0.0, no_speech_prob=0.0)])
    low = confidence_from_segments([seg(avg_logprob=-10.0, no_speech_prob=1.0)])
    assert 0.0 <= low <= high <= 1.0


def test_higher_logprob_means_higher_confidence():
    good = confidence_from_segments([seg(avg_logprob=-0.1, no_speech_prob=0.1)])
    bad = confidence_from_segments([seg(avg_logprob=-1.5, no_speech_prob=0.1)])
    assert good > bad


def test_no_speech_prob_penalizes_confidence():
    clean = confidence_from_segments([seg(avg_logprob=-0.2, no_speech_prob=0.0)])
    noisy = confidence_from_segments([seg(avg_logprob=-0.2, no_speech_prob=0.8)])
    assert clean > noisy


def test_duration_weighting():
    # A long good segment should dominate a short bad one
    segments = [
        seg(start=0.0, end=9.0, avg_logprob=-0.1, no_speech_prob=0.0),
        seg(start=9.0, end=9.5, avg_logprob=-3.0, no_speech_prob=0.0),
    ]
    conf = confidence_from_segments(segments)
    assert conf > 0.8


def test_expected_value_single_segment():
    conf = confidence_from_segments([seg(avg_logprob=-0.5, no_speech_prob=0.2)])
    assert abs(conf - math.exp(-0.5) * 0.8) < 1e-9


def test_heuristic_fallback_without_logprobs():
    # transformers backend provides no probabilities
    assert confidence_from_segments([seg()], "this has several words") == 0.85
    assert confidence_from_segments([seg()], "word") == 0.75
    assert confidence_from_segments([], "") == 0.6


def test_drop_no_speech_segments():
    keep = seg(text="real speech", avg_logprob=-0.3, no_speech_prob=0.1)
    drop = seg(text="garbage", avg_logprob=-1.5, no_speech_prob=0.95)
    borderline = seg(text="quiet but confident", avg_logprob=-0.5, no_speech_prob=0.95)
    unscored = seg(text="transformers path")

    kept = drop_no_speech_segments([keep, drop, borderline, unscored])
    assert keep in kept
    assert borderline in kept  # confident decode survives high no_speech_prob
    assert unscored in kept    # no probabilities -> never dropped here
    assert drop not in kept
