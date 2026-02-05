"""
Speaker Diarization Module using pyannote-audio
Identifies different speakers in audio streams
"""
import os
import logging
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

# Speaker colors for UI display (can be customized)
SPEAKER_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
]


@dataclass
class SpeakerSegment:
    """Represents a segment of speech from a specific speaker"""
    speaker_id: str
    speaker_label: str
    start_time: float
    end_time: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DiarizationResult:
    """Result of speaker diarization on an audio chunk"""
    segments: List[SpeakerSegment]
    num_speakers: int
    dominant_speaker: Optional[str] = None

    def get_speaker_at_time(self, time: float) -> Optional[str]:
        """Get the speaker ID at a specific time"""
        for seg in self.segments:
            if seg.start_time <= time <= seg.end_time:
                return seg.speaker_id
        return None

    def get_dominant_speaker(self) -> Optional[str]:
        """Get the speaker with the most speaking time"""
        if not self.segments:
            return None

        speaker_times = defaultdict(float)
        for seg in self.segments:
            speaker_times[seg.speaker_id] += seg.duration

        if speaker_times:
            return max(speaker_times, key=speaker_times.get)
        return None


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote-audio
    Requires HuggingFace token with access to pyannote models
    """

    def __init__(self,
                 hf_token: Optional[str] = None,
                 device: Optional[str] = None,
                 min_speakers: int = 1,
                 max_speakers: int = 5):
        """
        Initialize the speaker diarizer

        Args:
            hf_token: HuggingFace token for accessing pyannote models
            device: Device to run on ('cuda' or 'cpu')
            min_speakers: Minimum expected number of speakers
            max_speakers: Maximum expected number of speakers
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        self.pipeline = None
        self.is_loaded = False
        self.speaker_mapping = {}  # Maps raw speaker IDs to friendly names
        self.speaker_counter = 0

        logger.info(f"SpeakerDiarizer initialized (device: {self.device})")

    def load_model(self) -> bool:
        """Load the pyannote diarization pipeline"""
        if self.is_loaded:
            return True

        if not self.hf_token:
            logger.error("HuggingFace token required for pyannote models")
            logger.error("Set HF_TOKEN environment variable or pass hf_token parameter")
            logger.error("Get your token at: https://huggingface.co/settings/tokens")
            logger.error("Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            return False

        try:
            logger.info("Loading pyannote speaker diarization model...")
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )

            # Move to GPU if available
            if self.device == "cuda":
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                logger.info("Diarization model loaded on GPU")
            else:
                logger.info("Diarization model loaded on CPU")

            self.is_loaded = True
            return True

        except ImportError:
            logger.error("pyannote.audio not installed. Run: pip install pyannote.audio")
            return False
        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}")
            logger.error("Make sure you have accepted the license at:")
            logger.error("https://huggingface.co/pyannote/speaker-diarization-3.1")
            return False

    def _get_speaker_label(self, raw_speaker_id: str) -> str:
        """Convert raw speaker ID to friendly label"""
        if raw_speaker_id not in self.speaker_mapping:
            self.speaker_counter += 1
            self.speaker_mapping[raw_speaker_id] = f"Speaker {self.speaker_counter}"
        return self.speaker_mapping[raw_speaker_id]

    def _get_speaker_color(self, speaker_label: str) -> str:
        """Get color for a speaker label"""
        # Extract speaker number and use it to pick a color
        try:
            num = int(speaker_label.split()[-1])
            return SPEAKER_COLORS[(num - 1) % len(SPEAKER_COLORS)]
        except (ValueError, IndexError):
            return SPEAKER_COLORS[0]

    def diarize(self,
                audio: np.ndarray,
                sample_rate: int = 16000) -> Optional[DiarizationResult]:
        """
        Perform speaker diarization on audio

        Args:
            audio: Audio data as numpy array (mono, normalized)
            sample_rate: Sample rate of the audio

        Returns:
            DiarizationResult or None if failed
        """
        if not self.is_loaded:
            if not self.load_model():
                return None

        try:
            # Ensure audio is the right format
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono

            # Normalize if needed
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))

            # Convert to torch tensor with proper shape
            waveform = torch.from_numpy(audio).float().unsqueeze(0)

            # Create input dict for pyannote
            audio_input = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            # Run diarization
            diarization = self.pipeline(
                audio_input,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )

            # Convert to our format
            segments = []
            speakers_found = set()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_label = self._get_speaker_label(speaker)
                speakers_found.add(speaker_label)

                segments.append(SpeakerSegment(
                    speaker_id=speaker,
                    speaker_label=speaker_label,
                    start_time=turn.start,
                    end_time=turn.end
                ))

            result = DiarizationResult(
                segments=segments,
                num_speakers=len(speakers_found)
            )
            result.dominant_speaker = result.get_dominant_speaker()

            logger.debug(f"Diarization found {result.num_speakers} speaker(s)")
            return result

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return None

    def diarize_with_transcript(self,
                                audio: np.ndarray,
                                transcript_segments: List[Dict],
                                sample_rate: int = 16000) -> List[Dict]:
        """
        Align speaker diarization with transcript segments

        Args:
            audio: Audio data
            transcript_segments: List of dicts with 'start', 'end', 'text' keys
            sample_rate: Sample rate

        Returns:
            List of segments with added 'speaker' and 'speaker_color' keys
        """
        diarization = self.diarize(audio, sample_rate)

        if not diarization or not diarization.segments:
            # No diarization, return segments without speaker info
            for seg in transcript_segments:
                seg['speaker'] = None
                seg['speaker_color'] = None
            return transcript_segments

        # Assign speakers to each transcript segment
        for seg in transcript_segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', seg_start + 1)
            seg_mid = (seg_start + seg_end) / 2

            # Find the speaker at the middle of this segment
            speaker = diarization.get_speaker_at_time(seg_mid)

            if speaker:
                speaker_label = self._get_speaker_label(speaker)
                seg['speaker'] = speaker_label
                seg['speaker_color'] = self._get_speaker_color(speaker_label)
            else:
                # Use dominant speaker as fallback
                if diarization.dominant_speaker:
                    seg['speaker'] = self._get_speaker_label(diarization.dominant_speaker)
                    seg['speaker_color'] = self._get_speaker_color(seg['speaker'])
                else:
                    seg['speaker'] = None
                    seg['speaker_color'] = None

        return transcript_segments

    def get_simple_speaker(self,
                           audio: np.ndarray,
                           sample_rate: int = 16000) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the dominant speaker for an audio chunk (simplified interface)

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            Tuple of (speaker_label, speaker_color) or (None, None)
        """
        result = self.diarize(audio, sample_rate)

        if result and result.dominant_speaker:
            label = self._get_speaker_label(result.dominant_speaker)
            color = self._get_speaker_color(label)
            return label, color

        return None, None

    def reset_speakers(self):
        """Reset speaker mapping for a new session"""
        self.speaker_mapping.clear()
        self.speaker_counter = 0
        logger.info("Speaker mapping reset")

    def set_speaker_name(self, speaker_label: str, custom_name: str):
        """Set a custom name for a speaker"""
        # Find the raw ID for this label
        for raw_id, label in self.speaker_mapping.items():
            if label == speaker_label:
                self.speaker_mapping[raw_id] = custom_name
                logger.info(f"Renamed {speaker_label} to {custom_name}")
                return
        logger.warning(f"Speaker {speaker_label} not found")

    def get_speaker_stats(self) -> Dict[str, float]:
        """Get speaking time statistics (must be called after diarize)"""
        # This would need to track cumulative stats across calls
        # For now, return the current mapping
        return {v: 0.0 for v in self.speaker_mapping.values()}


class SimpleSpeakerDetector:
    """
    Lightweight speaker change detection without full diarization
    Uses voice activity and basic audio features
    """

    def __init__(self):
        self.last_features = None
        self.current_speaker = 1
        self.change_threshold = 0.3

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract simple audio features for speaker comparison"""
        # Simple features: energy, zero-crossing rate, basic spectral
        if len(audio) < 160:  # Too short
            return np.zeros(5)

        # Energy
        energy = np.sqrt(np.mean(audio ** 2))

        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2

        # Simple spectral centroid approximation
        fft = np.abs(np.fft.fft(audio))[:len(audio)//2]
        freqs = np.arange(len(fft))
        spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)

        # Spectral spread
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft) / (np.sum(fft) + 1e-10))

        # Peak frequency
        peak_freq = np.argmax(fft)

        return np.array([energy, zcr, spectral_centroid, spectral_spread, peak_freq])

    def detect_speaker_change(self, audio: np.ndarray) -> Tuple[bool, int]:
        """
        Detect if there's a speaker change

        Returns:
            Tuple of (is_change, speaker_id)
        """
        features = self.extract_features(audio)

        if self.last_features is None:
            self.last_features = features
            return False, self.current_speaker

        # Calculate feature distance
        distance = np.linalg.norm(features - self.last_features)
        distance = distance / (np.linalg.norm(self.last_features) + 1e-10)  # Normalize

        is_change = distance > self.change_threshold

        if is_change:
            self.current_speaker = (self.current_speaker % 5) + 1  # Cycle through speakers 1-5

        self.last_features = features * 0.3 + self.last_features * 0.7  # Smooth update

        return is_change, self.current_speaker

    def reset(self):
        """Reset detector state"""
        self.last_features = None
        self.current_speaker = 1
