"""
Configuration module for Live Translator
"""
import json
import os

# Constants
CONFIG_VERSION = 2
MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0"
# Pre-converted CTranslate2 build of the same weights, used by faster-whisper
FASTER_MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"
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
            # Note: speaker diarization works best with chunks >= 10s; raise
            # dynamic_max_chunk_duration when enabling it.
            "dynamic_max_chunk_duration": 8.0,
            "dynamic_silence_timeout": 0.9,
            "dynamic_min_speech_duration": 0.3,

            # Audio preprocessing (high-pass + peak normalization)
            "enhance_audio": True,

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
            # "translate" (JP->EN), "transcribe" (JP->JP), or "both"
            # (JP transcription + EN translation stacked in the subtitle;
            # runs two decodes per chunk, so roughly 2x ASR cost)
            "output_mode": "translate",
            # Language of the incoming audio, used as the model token in
            # transcribe mode (language_code stays the display language)
            "source_language_code": "ja",

            # Speaker diarization settings
            "use_speaker_diarization": DEFAULT_USE_DIARIZATION,
            "min_speakers": DEFAULT_MIN_SPEAKERS,
            "max_speakers": DEFAULT_MAX_SPEAKERS,
            "hf_token": None,  # HuggingFace token for pyannote
            "show_speaker_colors": True,
            "speaker_label_format": "bracket",  # 'bracket', 'prefix', 'color_only'

            # Text translation stage (optional): "whisper" (built-in),
            # "fugumt" (local MT model, free), or "deepl" (API key required)
            "translation_engine": "whisper",
            "deepl_api_key": None,
            # Comma-separated names/terms fed to the ASR decoder as hotwords
            "asr_hotwords": "",
            # {"wrong": "right"} whole-word fixes applied to English output
            "glossary": {},

            # Model settings
            "model_cache_dir": MODEL_CACHE_DIR,
            # ASR engine: "faster-whisper" (CTranslate2, fast) or "transformers"
            "asr_backend": "faster-whisper",
            # faster-whisper precision: auto/float16/int8_float16/int8
            "compute_type": "auto",
            "asr_beam_size": 5,
            # Subtitles below this confidence are dropped
            "min_confidence": 0.30,
            "config_version": CONFIG_VERSION
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    self._migrate_config(default_config, loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")

        self.__dict__.update(default_config)

    @staticmethod
    def _migrate_config(config, loaded_config):
        """Migrate pre-versioned configs in place. Only values still equal to
        an old default are bumped to the new default; user-tuned values are
        left untouched."""
        # Renamed/renormalized keys migrate regardless of version
        if config.get("asr_backend") == "faster_whisper":
            config["asr_backend"] = "faster-whisper"
        if "beam_size" in loaded_config and "asr_beam_size" not in loaded_config:
            config["asr_beam_size"] = loaded_config["beam_size"]
        config.pop("beam_size", None)

        if loaded_config.get("config_version"):
            return
        old_defaults_to_new = {
            "dynamic_max_chunk_duration": (15.0, 8.0),
            "dynamic_silence_timeout": (1.2, 0.9),
        }
        for key, (old_default, new_default) in old_defaults_to_new.items():
            if config.get(key) == old_default:
                print(f"Config migration: {key} {old_default} -> {new_default}")
                config[key] = new_default
        config["config_version"] = CONFIG_VERSION

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

        if self.output_mode not in ("translate", "transcribe", "both"):
            print(f"Warning: output_mode {self.output_mode!r} invalid, resetting to translate")
            self.output_mode = "translate"
            valid = False

        if self.asr_backend == "faster_whisper":  # legacy spelling
            self.asr_backend = "faster-whisper"
        if self.asr_backend not in ("faster-whisper", "transformers"):
            print(f"Warning: asr_backend {self.asr_backend!r} invalid, resetting to faster-whisper")
            self.asr_backend = "faster-whisper"
            valid = False

        if self.compute_type not in ("auto", "float16", "int8_float16", "int8"):
            print(f"Warning: compute_type {self.compute_type!r} invalid, resetting to auto")
            self.compute_type = "auto"
            valid = False

        if not (1 <= int(self.asr_beam_size) <= 10):
            print(f"Warning: asr_beam_size {self.asr_beam_size} out of range, resetting to 5")
            self.asr_beam_size = 5
            valid = False

        if not (0.0 <= float(self.min_confidence) <= 1.0):
            print(f"Warning: min_confidence {self.min_confidence} out of range, resetting to 0.30")
            self.min_confidence = 0.30
            valid = False

        if self.translation_engine not in ("whisper", "fugumt", "deepl"):
            print(f"Warning: translation_engine {self.translation_engine!r} invalid, resetting to whisper")
            self.translation_engine = "whisper"
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
