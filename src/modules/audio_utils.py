
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
    """Normalize toward a target RMS with two safety rails:

    - near-silence (RMS below the noise floor) is never amplified, so room
      noise can't be boosted into something Whisper hallucinates over
    - gain is capped at 5x, and the result is hard-capped at 0.95 peak
    """
    audio_data = audio_data - np.mean(audio_data)

    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms > 0.001:  # leave the noise floor alone
        target_rms = 0.3
        gain = min(target_rms / rms, 5.0)
        audio_data = audio_data * gain

    peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0
    if peak > 0.95:
        audio_data = audio_data * (0.95 / peak)

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
