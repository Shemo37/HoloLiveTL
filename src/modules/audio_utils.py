
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
    """Normalize audio to improve recognition accuracy while preserving loud sounds"""
    audio_data = audio_data - np.mean(audio_data)
    
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms > 0:
        target_rms = 0.3
        audio_data = audio_data * (target_rms / rms)
    
    audio_data = np.where(np.abs(audio_data) > 0.95, 
                         np.sign(audio_data) * (0.95 + 0.05 * np.tanh((np.abs(audio_data) - 0.95) * 10)), 
                         audio_data)
    
    return audio_data

def enhance_audio_quality(audio_data, sample_rate=16000):
    """Apply audio enhancements for better speech recognition including loud sounds"""
    audio_data = highpass_filter(audio_data, cutoff=60, fs=sample_rate, order=3)
    audio_data = normalize_audio(audio_data)
    
    threshold = 0.005
    audio_data = np.where(np.abs(audio_data) < threshold, 
                         audio_data * 0.3, audio_data)
    
    return audio_data
