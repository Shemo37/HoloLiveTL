
import numpy as np
import soundcard as sc
from scipy.signal import butter, lfilter

def find_audio_device(selected_device_name=None):
    """Find the best audio capture device"""
    print("🔍 Searching for audio capture devices...")
    all_mics = sc.all_microphones(include_loopback=True)
    if not all_mics:
        print("❌ No audio devices found at all.")
        return None

    if selected_device_name:
        for mic in all_mics:
            if mic.name == selected_device_name:
                print(f"🎚️ Using selected device: '{mic.name}'")
                return mic
        print(f"⚠️ Could not find previously selected device '{selected_device_name}'. Searching for alternatives.")

    preferred_names = ["cable", "stereo mix", "what u hear", "loopback", "virtual"]
    for name in preferred_names:
        for mic in all_mics:
            if name in mic.name.lower():
                print(f"✅ Found preferred capture device: '{mic.name}'")
                return mic
    try:
        default_mic = sc.default_microphone(include_loopback=True)
        print(f"⚠️ Using default system loopback device: '{default_mic.name}'")
        return default_mic
    except Exception:
        print(f"⚠️ No default loopback found. Falling back to first available device: '{all_mics[0].name}'")
        return all_mics[0]

def highpass_filter(data, cutoff=100, fs=16000, order=5):
    """Apply high-pass filter to audio data"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def normalize_audio(audio_data):
    """Peak-normalize audio without boosting the noise floor.

    Whisper's log-mel frontend is level-robust; the old RMS-0.3 target boosted
    quiet/noise-only chunks ~100x, which is a known hallucination trigger, and
    pushed voiced peaks into the soft-clip. Peak normalization only scales
    chunks that would clip or are unusually quiet, and never amplifies more
    than 4x.
    """
    audio_data = audio_data - np.mean(audio_data)

    peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0
    if peak > 0:
        gain = min(0.95 / peak, 4.0)
        if gain < 1.0 or peak < 0.24:  # attenuate clipping, gently lift quiet speech
            audio_data = audio_data * gain

    audio_data = np.where(np.abs(audio_data) > 0.95,
                         np.sign(audio_data) * (0.95 + 0.05 * np.tanh((np.abs(audio_data) - 0.95) * 10)),
                         audio_data)

    return audio_data

def enhance_audio_quality(audio_data, sample_rate=16000):
    """Apply light audio cleanup for speech recognition.

    Kept intentionally minimal: a low-cut filter for rumble plus peak
    normalization. The old sub-threshold downward expander distorted quiet
    speech onsets and is gone.
    """
    audio_data = highpass_filter(audio_data, cutoff=60, fs=sample_rate, order=3)
    audio_data = normalize_audio(audio_data)

    return audio_data
