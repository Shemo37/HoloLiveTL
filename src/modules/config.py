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

# Model cache directory
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "translator_models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class Config:
    def __init__(self):
        self.config_file = "translator_config.json"
        self.load_config()

    def load_config(self):
        default_config = {
            "volume_threshold": VOLUME_THRESHOLD,
            "chunk_duration": CHUNK_DURATION,
            "language_code": LANGUAGE_CODE,
            "window_opacity": DEFAULT_WINDOW_OPACITY,
            "font_size": 24,
            "use_vad_filter": USE_VAD_FILTER,
            "vad_threshold": VAD_THRESHOLD,
            "subtitle_bg_color": DEFAULT_BG_COLOR,
            "subtitle_font_color": DEFAULT_FONT_COLOR,
            "subtitle_bg_mode": DEFAULT_BG_MODE,
            "font_weight": "bold",
            "text_shadow": True,
            "border_width": 2,
            "border_color": "#000000",
            "output_mode": "translate",
            "selected_audio_device": None,
            "use_dynamic_chunking": True,
            "dynamic_max_chunk_duration": 15.0,
            "dynamic_silence_timeout": 1.2,
            "dynamic_min_speech_duration": 0.3,
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
        config_data = {k: v for k, v in self.__dict__.items() if k != 'config_file'}
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
