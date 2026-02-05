
import os
import numpy as np
import torch
import traceback
from queue import Queue
from .audio_utils import find_audio_device
from .config import SAMPLE_RATE

def recorder_thread(stop_event, audio_queue, config, gui_queue, selected_device_name=None):
    if config.use_dynamic_chunking:
        print("🎙️ Recorder thread started (Dynamic Chunking Mode).")
        dynamic_recorder_thread(stop_event, audio_queue, config, gui_queue, selected_device_name)
    else:
        print("🎙️ Recorder thread started (Fixed Chunk Mode).")
        fixed_recorder_thread(stop_event, audio_queue, config, gui_queue, selected_device_name)

def fixed_recorder_thread(stop_event, audio_queue, config, gui_queue, selected_device_name):
    try:
        target_mic = find_audio_device(selected_device_name)
        if target_mic is None:
            raise RuntimeError("No audio devices found. Cannot start recording.")
        with target_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
            while not stop_event.is_set():
                data = mic.record(numframes=int(SAMPLE_RATE * config.chunk_duration))
                if not stop_event.is_set():
                    audio_queue.put(data)
    except Exception as e:
        print(f"🔴 Recorder Thread Error (Fixed): {e}")
        traceback.print_exc()
        gui_queue.put(("error", "Audio device error! Check console."))
    finally:
        print("🎙️ Recorder thread stopped (Fixed).")

def dynamic_recorder_thread(stop_event, audio_queue, config, gui_queue, selected_device_name):
    try:
        target_mic = find_audio_device(selected_device_name)
        if target_mic is None:
            raise RuntimeError("No audio devices found. Cannot start recording.")
            
        print("🎙️ [Dynamic] Loading Silero VAD model for recorder...")
        torch.set_num_threads(1)
        
        # Load VAD from cache
        vad_path = os.path.join(config.model_cache_dir, "vad_model", "silero_vad.jit")
        vad_failed_path = vad_path + ".failed"
        
        if not os.path.exists(vad_path) and not os.path.exists(vad_failed_path):
            print("🎙️ [Dynamic] VAD model not found, downloading...")
            from .model_utils import ensure_model_downloaded
            from .config import MODEL_ID
            ensure_model_downloaded(MODEL_ID, config.model_cache_dir)
        
        if os.path.exists(vad_failed_path):
            print("⚠️ [Dynamic] VAD model download previously failed. Using volume-based detection only.")
            vad_model = None
        elif not os.path.exists(vad_path):
            print("⚠️ [Dynamic] VAD model file not found. Using volume-based detection only.")
            vad_model = None
        else:
            try:
                vad_model = torch.jit.load(vad_path, map_location='cpu')
                print("🎙️ [Dynamic] VAD model loaded.")
            except Exception as e:
                print(f"⚠️ [Dynamic] Failed to load VAD model: {e}. Using volume-based detection only.")
                vad_model = None

        VAD_FRAME_DURATION_MS = 30
        VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

        is_speaking = False
        speech_buffer = []
        silence_frames_after_speech = 0
        
        silence_timeout_frames = int(config.dynamic_silence_timeout * 1000 / VAD_FRAME_DURATION_MS)
        max_chunk_frames = int(config.dynamic_max_chunk_duration * 1000 / VAD_FRAME_DURATION_MS)

        with target_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
            print("🎙️ [Dynamic] Now listening...")
            while not stop_event.is_set():
                frame_data = mic.record(numframes=VAD_FRAME_SIZE)
                
                rms = np.sqrt(np.mean(frame_data ** 2))
                peak_level = np.max(np.abs(frame_data))
                is_loud_sound = peak_level > 0.1
                
                if rms < config.volume_threshold and not is_loud_sound:
                    is_speech = False
                else:
                    if vad_model is not None:
                        # Use VAD model if available
                        audio_tensor = torch.from_numpy(frame_data.flatten()).float()
                        if len(audio_tensor) < 512:
                            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - len(audio_tensor)))
                        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
                        
                        vad_threshold = config.vad_threshold * 0.5 if is_loud_sound else config.vad_threshold
                        is_speech = speech_prob > vad_threshold or is_loud_sound
                        
                        if is_loud_sound:
                            print(f"🔊 Loud sound detected! Peak: {peak_level:.3f}, RMS: {rms:.3f}, VAD: {speech_prob:.3f}")
                    else:
                        # Fallback to volume-based detection only
                        is_speech = rms > config.volume_threshold or is_loud_sound
                        
                        if is_loud_sound:
                            print(f"🔊 Loud sound detected! Peak: {peak_level:.3f}, RMS: {rms:.3f} (VAD disabled)")

                if is_speaking:
                    speech_buffer.append(frame_data)
                    if is_speech:
                        silence_frames_after_speech = 0
                    else:
                        silence_frames_after_speech += 1
                    
                    chunk_ended = (silence_frames_after_speech > silence_timeout_frames) or \
                                  (len(speech_buffer) > max_chunk_frames)

                    if chunk_ended:
                        audio_chunk = np.concatenate(speech_buffer)
                        chunk_duration_s = len(audio_chunk) / SAMPLE_RATE
                        chunk_peak = np.max(np.abs(audio_chunk))
                        is_loud_chunk = chunk_peak > 0.1
                        
                        min_duration = config.dynamic_min_speech_duration * 0.5 if is_loud_chunk else config.dynamic_min_speech_duration
                        min_samples = int(SAMPLE_RATE * 0.5) if is_loud_chunk else int(SAMPLE_RATE * 1.0)
                        
                        if chunk_duration_s > min_duration and len(audio_chunk) >= min_samples:
                            chunk_type = "LOUD" if is_loud_chunk else "speech"
                            print(f"🎤 Detected {chunk_type} chunk of {chunk_duration_s:.2f}s (peak: {chunk_peak:.3f}). Sending for processing.")
                            audio_queue.put(audio_chunk)
                        else:
                            print(f"⏩ Skipped short chunk: {chunk_duration_s:.2f}s (peak: {chunk_peak:.3f})")
                        
                        is_speaking = False
                        speech_buffer = []
                        silence_frames_after_speech = 0

                elif is_speech:
                    is_speaking = True
                    speech_buffer.append(frame_data)
                    silence_frames_after_speech = 0

    except Exception as e:
        print(f"🔴 Recorder Thread Error (Dynamic): {e}")
        traceback.print_exc()
        gui_queue.put(("error", "Audio device error! Check console."))
    finally:
        print("🎙️ Recorder thread stopped (Dynamic).")
