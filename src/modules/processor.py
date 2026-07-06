"""
Audio processor module with speaker diarization support
"""
import time
import threading
import numpy as np
import torch
import string
import traceback
import logging
from queue import Queue, Full, Empty
from collections import deque
from .audio_utils import enhance_audio_quality
from .asr_backend import create_backend, confidence_from_segments
from .translator import create_translator, apply_glossary, TranslatorUnavailable
from .filters import post_process_translation, is_hallucination
from .config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def resolve_pipeline(config, translator):
    """Decide the ASR task/target and whether a text translator runs after it.

    With a text translator active, ASR transcribes Japanese and the translator
    produces the English; otherwise Whisper's built-in translate task is used.
    """
    if config.output_mode == "translate":
        if translator is not None:
            return "transcribe", "ja", True
        return "translate", "en", False
    return "transcribe", config.language_code, False


def processor_thread(stop_event, audio_queue, config, stats, gui_queue):
    """Main processor thread. Emits subtitles immediately after ASR; speaker
    diarization (if enabled) runs in a separate worker and attaches labels
    afterwards via "speaker_update" messages."""
    print("Processing thread started.")

    diar_queue = None
    diar_thread = None

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")

        # Start async diarization worker (never blocks subtitle output)
        if getattr(config, 'use_speaker_diarization', False):
            from .diarization_worker import diarization_worker
            diar_queue = Queue(maxsize=3)
            diar_thread = threading.Thread(
                target=diarization_worker,
                args=(stop_event, diar_queue, config, gui_queue, device),
                daemon=True,
            )
            diar_thread.start()
        else:
            gui_queue.put(("diarization_status", False))

        backend = create_backend(config, device, gui_queue)

        text_translator = create_translator(config)
        task, target_lang, use_text_translator = resolve_pipeline(config, text_translator)
        print(f"ASR backend: {backend.name}, task: '{task}', target language: '{target_lang}'"
              + (", translation: DeepL" if use_text_translator else ""))

        gui_queue.put(("model_loaded", None))

        translator = str.maketrans('', '', string.punctuation)
        translation_history = []
        min_confidence = getattr(config, 'min_confidence', 0.30)
        glossary = getattr(config, 'glossary', {}) or {}
        source_context = deque(maxlen=2)  # recent JA lines, sent to DeepL as context
        subtitle_id = 0

        while not stop_event.is_set():
            start_time = time.time()
            had_translation, was_hallucination = False, False
            try:
                try:
                    audio_chunk_np = audio_queue.get(timeout=1)
                except Empty:
                    continue

                audio_chunk_np = enhance_audio_quality(audio_chunk_np.flatten(), sample_rate=SAMPLE_RATE)

                if not config.use_dynamic_chunking:
                    # Fixed mode: cheap RMS gate; finer speech gating is done
                    # by the backend's built-in VAD filter.
                    rms = np.sqrt(np.mean(audio_chunk_np ** 2))
                    if rms < config.volume_threshold:
                        stats.add_chunk(time.time() - start_time, False, False)
                        continue

                audio_data = audio_chunk_np.flatten().astype(np.float32)
                min_samples = int(SAMPLE_RATE * 1.0)
                if len(audio_data) < min_samples:
                    print(f"Skipped chunk: too short ({len(audio_data)/SAMPLE_RATE:.2f}s)")
                    stats.add_chunk(time.time() - start_time, False, False)
                    continue

                # Run ASR/translation
                result = backend.transcribe(audio_data, task, target_lang)
                processed_text = result.text.strip()

                confidence_score = confidence_from_segments(result.segments, processed_text)

                had_translation = bool(processed_text)
                is_hallucination_result = False

                if processed_text:
                    if confidence_score < min_confidence:
                        print(f"Filtered low-confidence output: '{processed_text}' "
                              f"(confidence: {confidence_score:.2f} < {min_confidence:.2f})")
                        stats.add_chunk(time.time() - start_time, had_translation, True, confidence_score)
                        continue

                    if use_text_translator:
                        source_text = processed_text
                        try:
                            translated = text_translator.translate(
                                source_text, context=" ".join(source_context) or None)
                        except TranslatorUnavailable as e:
                            # Abandon DeepL for this session; later chunks go
                            # back through Whisper's built-in translation.
                            print(f"Disabling DeepL: {e}")
                            gui_queue.put(("status", "DeepL unavailable - using Whisper translation"))
                            text_translator = None
                            task, target_lang, use_text_translator = resolve_pipeline(config, None)
                            translated = None

                        if use_text_translator:
                            if translated is None:
                                stats.add_chunk(time.time() - start_time, had_translation, True, confidence_score)
                                continue
                            source_context.append(source_text)
                            processed_text = translated.strip()
                        else:
                            # This chunk was transcribed JA with no translation
                            # available; skip it rather than showing raw JA.
                            stats.add_chunk(time.time() - start_time, had_translation, True, confidence_score)
                            continue

                    is_hallucination_result = is_hallucination(processed_text, translator, translation_history)
                    was_hallucination = is_hallucination_result

                    if not is_hallucination_result:
                        processed_text = apply_glossary(post_process_translation(processed_text), glossary)
                        translation_history.append(processed_text)
                        if len(translation_history) > 10:
                            translation_history.pop(0)

                        subtitle_id += 1
                        print(f"Translation: {processed_text} (confidence: {confidence_score:.2f})")

                        # Emit immediately; speaker label (if any) arrives later
                        # via a "speaker_update" message from the diarization worker.
                        gui_queue.put(("subtitle", {
                            "id": subtitle_id,
                            "text": processed_text,
                            "display_text": processed_text,
                            "speaker": None,
                            "speaker_color": None,
                            "confidence": confidence_score
                        }))

                        if diar_queue is not None:
                            try:
                                diar_queue.put_nowait((subtitle_id, audio_data.copy(), time.time()))
                            except Full:
                                logger.debug("Diarization queue full; skipping speaker detection for this chunk")
                    else:
                        print(f"Filtered hallucination: '{processed_text}' (confidence: {confidence_score:.2f})")

                stats.add_chunk(time.time() - start_time, had_translation, was_hallucination, confidence_score)

            except Exception as e:
                print(f"Processor error: {e}")
                traceback.print_exc()
                stats.add_chunk(time.time() - start_time, False, False)

    except Exception as e:
        print(f"Processor Thread Fatal Error: {e}")
        traceback.print_exc()
        gui_queue.put(("error", f"Processing error: {e}"))
    finally:
        print("Processor thread stopped.")
