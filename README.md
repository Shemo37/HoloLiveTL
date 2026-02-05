# HoloLiveTL - Real-Time Japanese Translation for VTuber Streams

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![AI](https://img.shields.io/badge/AI-Whisper%20%2B%20VAD-orange.svg)

*Real-time Japanese to English subtitle overlay for VTuber livestreams and Japanese content*

</div>

## Demo
![giphy](https://github.com/user-attachments/assets/302705e5-21f9-42e8-9f4e-4234c0cd71ed)

## Overview

HoloLiveTL captures audio from your system and translates Japanese speech into English subtitles in real time. Uses the **kotoba-whisper-bilingual** model for ASR/translation, **Silero VAD** for voice activity detection, and optionally **pyannote** for speaker diarization.

> **Note:** This uses GPU heavily. Make sure your system can handle the extra load.

## Features

- **Real-time translation** — Japanese to English subtitle overlay via kotoba-whisper-bilingual
- **Dynamic chunking** — Speech-aware audio segmentation using VAD, only processes when speech is detected
- **Speaker diarization** — Identifies and color-codes different speakers (requires HuggingFace token + pyannote)
- **Hallucination filtering** — Filters out common model hallucinations and repetitive output automatically
- **Customizable subtitles** — Font size, colors, transparency, text shadow, border, and more
- **Translation history** — Scrollable history panel with per-speaker labels and timestamps
- **Presets** — Save and load configurations (e.g., per-streamer presets)
- **GPU accelerated** — CUDA support with automatic CPU fallback

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (recommended)
- Windows OS

## Installation

1. Clone the repo

```bash
git clone https://github.com/Shemo37/HoloLiveTL.git
cd HoloLiveTL
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. For speaker diarization (optional), get a HuggingFace token and accept the pyannote license:
   - Create a token at https://huggingface.co/settings/tokens
   - Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Set `HF_TOKEN` as an environment variable

## Usage

**Windows (recommended):**
```
Double-click LiveTranslator.bat
```

**Or run directly:**
```bash
python main.py
```

The app will download the required models on first run and cache them locally.

## Configuration

Settings are saved automatically to `translator_config.json`. You can also use presets in the `presets/` folder.

| Setting | Description |
|---|---|
| Audio device | Select which audio input to capture |
| Dynamic chunking | Speech-aware segmentation (on by default) |
| Volume threshold | Minimum RMS level to trigger processing |
| VAD threshold | Voice activity detection sensitivity |
| Output mode | `translate` (JP→EN) or `transcribe` (JP→JP) |
| Speaker diarization | Color-coded speaker labels (requires HF token) |
| Subtitle appearance | Font size, colors, opacity, shadow, border |

## Project Structure

```
HoloLiveTL/
├── main.py                 # Entry point
├── LiveTranslator.bat      # Windows launcher
├── requirements.txt        # Dependencies
├── presets/                # Saved config presets
│   └── fbk.json
└── src/
    ├── gui/
    │   └── main_window.py  # Tkinter GUI + subtitle overlay
    └── modules/
        ├── config.py       # Configuration and defaults
        ├── recorder.py     # Audio capture (fixed + dynamic chunking)
        ├── processor.py    # ASR/translation pipeline
        ├── diarization.py  # Speaker diarization (pyannote)
        ├── filters.py      # Hallucination detection and text cleanup
        ├── audio_utils.py  # Audio device discovery and enhancement
        ├── model_utils.py  # Model download and caching
        └── stats.py        # Runtime statistics
```

## Known Issues

- Some hallucinations may still slip through the filter
- Translation accuracy can vary depending on audio quality

## Star

If you find this project helpful, please consider giving it a star! ⭐

---
