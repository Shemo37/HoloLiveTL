"""
Configuration module for Live Translator
"""
import json
import os

# Constants
MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0"
SAMPLE_RATE = 16000
CHUNK_DURATION = 5
LANGUAGE_CODE = "en"
VOLUME_THRESHOLD = 0.003
USE_VAD_FILTER = True
VAD_THRESHOLD = 0.25
DEFAULT_BG_COLOR = '#282828'
DEFAULT_FONT_COLOR = '#FFFFFF'
DEFAULT_BG_MODE = 'transparent'
DEFAULT_WINDOW_OPACITY = 0.85

# Speaker diarization defaults
DEFAULT_USE_DIARIZATION = False
DEFAULT_MIN_SPEAKERS = 1
DEFAULT_MAX_SPEAKERS = 5

# Speaker colors for UI
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

# Model cache directory
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "translator_models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


class Config:
    """Configuration class with persistence"""

    def __init__(self):
        self.config_file = "translator_config.json"
        self.load_config()

    def load_config(self):
        default_config = {
            # Audio settings
            "volume_threshold": VOLUME_THRESHOLD,
            "chunk_duration": CHUNK_DURATION,
            "language_code": LANGUAGE_CODE,
            "use_vad_filter": USE_VAD_FILTER,
            "vad_threshold": VAD_THRESHOLD,
            "selected_audio_device": None,

            # Dynamic chunking
            "use_dynamic_chunking": True,
            "dynamic_max_chunk_duration": 15.0,
            "dynamic_silence_timeout": 1.2,
            "dynamic_min_speech_duration": 0.3,

            # Appearance settings
            "window_opacity": DEFAULT_WINDOW_OPACITY,
            "font_size": 24,
            "subtitle_bg_color": DEFAULT_BG_COLOR,
            "subtitle_font_color": DEFAULT_FONT_COLOR,
            "subtitle_bg_mode": DEFAULT_BG_MODE,
            "font_weight": "bold",
            "text_shadow": True,
            "border_width": 2,
            "border_color": "#000000",

            # Translation settings
            "output_mode": "translate",

            # Speaker diarization settings
            "use_speaker_diarization": DEFAULT_USE_DIARIZATION,
            "min_speakers": DEFAULT_MIN_SPEAKERS,
            "max_speakers": DEFAULT_MAX_SPEAKERS,
            "hf_token": None,  # HuggingFace token for pyannote
            "show_speaker_colors": True,
            "speaker_label_format": "bracket",  # 'bracket', 'prefix', 'color_only'

            # Model settings
            "model_cache_dir": MODEL_CACHE_DIR
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")

        self.__dict__.update(default_config)

    def save_config(self):
        """Save current configuration to file"""
        # Exclude certain keys from saving
        exclude_keys = {'config_file'}
        config_data = {k: v for k, v in self.__dict__.items()
                       if k not in exclude_keys and not k.startswith('_')}
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_speaker_color(self, speaker_num: int) -> str:
        """Get color for a speaker number (1-indexed)"""
        return SPEAKER_COLORS[(speaker_num - 1) % len(SPEAKER_COLORS)]

    def validate(self) -> bool:
        """Validate configuration values"""
        valid = True

        # Validate numeric ranges
        if not (0.0 <= self.volume_threshold <= 1.0):
            print(f"Warning: volume_threshold {self.volume_threshold} out of range, resetting to default")
            self.volume_threshold = VOLUME_THRESHOLD
            valid = False

        if not (0.0 <= self.vad_threshold <= 1.0):
            print(f"Warning: vad_threshold {self.vad_threshold} out of range, resetting to default")
            self.vad_threshold = VAD_THRESHOLD
            valid = False

        if not (0.0 <= self.window_opacity <= 1.0):
            print(f"Warning: window_opacity {self.window_opacity} out of range, resetting to default")
            self.window_opacity = DEFAULT_WINDOW_OPACITY
            valid = False

        if not (8 <= self.font_size <= 72):
            print(f"Warning: font_size {self.font_size} out of range, resetting to 24")
            self.font_size = 24
            valid = False

        if not (1 <= self.min_speakers <= 10):
            print(f"Warning: min_speakers {self.min_speakers} out of range, resetting to 1")
            self.min_speakers = 1
            valid = False

        if not (1 <= self.max_speakers <= 10):
            print(f"Warning: max_speakers {self.max_speakers} out of range, resetting to 5")
            self.max_speakers = 5
            valid = False

        if self.min_speakers > self.max_speakers:
            print(f"Warning: min_speakers > max_speakers, resetting both")
            self.min_speakers = 1
            self.max_speakers = 5
            valid = False

        return valid

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        exclude_keys = {'config_file'}
        return {k: v for k, v in self.__dict__.items()
                if k not in exclude_keys and not k.startswith('_')}

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
