
import os
import time
import numpy as np
from collections import deque
import torch
import traceback
from queue import Queue, Full
from .audio_utils import find_audio_device
from .config import SAMPLE_RATE

def put_latest(audio_queue, chunk):
    """Queue a chunk, dropping the OLDEST queued audio if the queue is full.

    Live subtitles want the newest audio; if the model falls behind, blocking
    here would make subtitles drift permanently behind the stream.
    """
    while True:
        try:
            audio_queue.put_nowait(chunk)
            return
        except Full:
            try:
                audio_queue.get_nowait()
                print("\u26a0\ufe0f Audio queue full - dropped oldest chunk to stay live")
            except Exception:
                pass

LEVEL_INTERVAL_S = 0.1  # throttle for ("audio_level", rms) GUI messages


class _LevelReporter:
    """Throttled audio-level feed for the GUI meter."""

    def __init__(self, gui_queue, interval=LEVEL_INTERVAL_S):
        self.gui_queue = gui_queue
        self.interval = interval
        self._last = 0.0

    def report(self, rms):
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            try:
                self.gui_queue.put_nowait(("audio_level", float(rms)))
            except Exception:
                pass


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
        level = _LevelReporter(gui_queue)
        with target_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
            while not stop_event.is_set():
                data = mic.record(numframes=int(SAMPLE_RATE * config.chunk_duration))
                level.report(np.sqrt(np.mean(data ** 2)))
                if not stop_event.is_set():
                    put_latest(audio_queue, data)
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

        # 32ms = exactly 512 samples at 16 kHz, the frame size Silero VAD
        # expects; the old 30ms frames needed zero-padding that distorted
        # every VAD decision.
        VAD_FRAME_DURATION_MS = 32
        VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

        is_speaking = False
        speech_buffer = []
        silence_frames_after_speech = 0
        # ~300ms of pre-roll so the first mora isn't clipped when VAD fires
        preroll_frames = max(1, int(300 / VAD_FRAME_DURATION_MS))
        preroll = deque(maxlen=preroll_frames)
        
        silence_timeout_frames = int(config.dynamic_silence_timeout * 1000 / VAD_FRAME_DURATION_MS)
        max_chunk_frames = int(config.dynamic_max_chunk_duration * 1000 / VAD_FRAME_DURATION_MS)

        level = _LevelReporter(gui_queue)
        with target_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
            print("🎙️ [Dynamic] Now listening...")
            while not stop_event.is_set():
                frame_data = mic.record(numframes=VAD_FRAME_SIZE)
                
                rms = np.sqrt(np.mean(frame_data ** 2))
                level.report(rms)
                peak_level = np.max(np.abs(frame_data))
                is_loud_sound = peak_level > 0.1
                
                if rms < config.volume_threshold and not is_loud_sound:
                    is_speech = False
                else:
                    if vad_model is not None:
                        # Use VAD model if available (frames are exactly 512 samples)
                        audio_tensor = torch.from_numpy(frame_data.flatten()).float()
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
                            put_latest(audio_queue, audio_chunk)
                        else:
                            print(f"⏩ Skipped short chunk: {chunk_duration_s:.2f}s (peak: {chunk_peak:.3f})")
                        
                        is_speaking = False
                        speech_buffer = []
                        silence_frames_after_speech = 0
                        preroll.clear()

                elif is_speech:
                    is_speaking = True
                    # Seed with buffered pre-roll: VAD fires a frame or two into
                    # the utterance, so the attack lives in these frames.
                    speech_buffer = list(preroll)
                    speech_buffer.append(frame_data)
                    silence_frames_after_speech = 0
                else:
                    preroll.append(frame_data)

    except Exception as e:
        print(f"🔴 Recorder Thread Error (Dynamic): {e}")
        traceback.print_exc()
        gui_queue.put(("error", "Audio device error! Check console."))
    finally:
        print("🎙️ Recorder thread stopped (Dynamic).")
