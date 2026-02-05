"""
Translation statistics tracking module
"""
import time
from collections import deque


class TranslatorStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.chunks_processed = 0
        self.translations_made = 0
        self.hallucinations_filtered = 0
        self.total_processing_time = 0
        self.confidence_scores = deque(maxlen=100)  # Keep last 100 scores

    def add_chunk(self, processing_time, had_translation, was_hallucination, confidence=None):
        self.chunks_processed += 1
        self.total_processing_time += processing_time
        if had_translation:
            if was_hallucination:
                self.hallucinations_filtered += 1
            else:
                self.translations_made += 1
                if confidence is not None:
                    self.confidence_scores.append(confidence)

    @property
    def total_chunks(self):
        """Total number of audio chunks processed"""
        return self.chunks_processed

    @property
    def successful_translations(self):
        """Number of successful (non-filtered) translations"""
        return self.translations_made

    @property
    def filtered_count(self):
        """Number of translations filtered as hallucinations"""
        return self.hallucinations_filtered

    @property
    def average_confidence(self):
        """Average confidence score of recent translations"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def session_duration(self):
        """Duration of the current session in seconds"""
        return time.time() - self.start_time

    @property
    def translations_per_minute(self):
        """Rate of successful translations per minute"""
        duration_minutes = self.session_duration / 60
        if duration_minutes < 0.1:  # Less than 6 seconds
            return 0.0
        return self.translations_made / duration_minutes

    @property
    def average_processing_time(self):
        """Average time to process each chunk"""
        if self.chunks_processed == 0:
            return 0.0
        return self.total_processing_time / self.chunks_processed

    def get_summary(self):
        """Get a summary dictionary of all statistics"""
        return {
            "total_chunks": self.total_chunks,
            "successful_translations": self.successful_translations,
            "filtered_count": self.filtered_count,
            "average_confidence": self.average_confidence,
            "session_duration": self.session_duration,
            "translations_per_minute": self.translations_per_minute,
            "average_processing_time": self.average_processing_time
        }
