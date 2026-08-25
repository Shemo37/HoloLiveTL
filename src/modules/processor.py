"""
Audio processor thread.

Pipeline per speech chunk: gates (volume/VAD) -> optional audio enhancement ->
ASR (faster-whisper or transformers via asr_backend) -> optional text
translation stage (DeepL/FuguMT) -> confidence + hallucination filtering ->
subtitle to the GUI. Speaker diarization runs asynchronously in its own
worker thread and attaches labels afterwards via "speaker_update" messages
keyed by subtitle id, so it never adds latency to the subtitle itself.
"""
import os
import time
import threading
import numpy as np
import torch
import string
import traceback
import logging
from collections import deque
from queue import Queue, Full

from .audio_utils import enhance_audio_quality
from .model_utils import ensure_model_downloaded
from .asr_backend import create_backend
from .filters import post_process_translation, is_hallucination, is_low_confidence
from .translator import create_translator, apply_glossary, TranslatorUnavailable
from .diarization_worker import diarization_worker
from .config import SAMPLE_RATE, MODEL_ID

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
    return "transcribe", getattr(config, 'source_language_code', 'ja'), False


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

        # Ensure models are downloaded
        gui_queue.put(("status", "Checking models..."))
        model_dir, vad_dir = ensure_model_downloaded(MODEL_ID, config.model_cache_dir)

        vad_model = None

        if not config.use_dynamic_chunking and config.use_vad_filter:
            try:
                print("Loading Silero VAD model...")
                torch.set_num_threads(1)
                vad_path = os.path.join(vad_dir, "silero_vad.jit")
                vad_model = torch.jit.load(vad_path, map_location='cpu')
                print("VAD model loaded successfully.")
            except Exception as e:
                print(f"Could not load VAD model: {e}. Disabling VAD filter.")
                config.use_vad_filter = False

        # Speaker diarization runs in its own worker thread (pyannote roughly
        # doubles per-chunk latency when run inline, so it never blocks ASR).
        if getattr(config, 'use_speaker_diarization', False):
            diar_queue = Queue(maxsize=4)
            diar_thread = threading.Thread(
                target=diarization_worker,
                args=(stop_event, diar_queue, config, gui_queue, device),
                daemon=True)
            diar_thread.start()
        else:
            gui_queue.put(("diarization_status", False))

        # Optional text translation stage (DeepL / FuguMT)
        gui_queue.put(("status", "Loading translation engine..."))
        text_translator = create_translator(config)
        task, target_lang, use_text_translator = resolve_pipeline(config, text_translator)

        source_lang = getattr(config, 'source_language_code', 'ja')
        # "both" shows the JP transcription above the EN translation.
        dual_output = config.output_mode == "both"

        gui_queue.put(("status", "Loading ASR model..."))
        backend = create_backend(config, device, gui_queue,
                                 model_dir=model_dir, task=task, language=target_lang)
        gui_queue.put(("status", "Warming up..."))
        # Throwaway decode so the first real utterance doesn't pay CUDA init
        backend.warm_up()

        engine_desc = backend.name
        if use_text_translator:
            engine_desc += f" + {text_translator.name}"
        print(f"ASR backend: {backend.name}, task: '{task}', target language: '{target_lang}'"
              + (f", translation: {text_translator.name}" if use_text_translator else ""))
        gui_queue.put(("engine", engine_desc))
        gui_queue.put(("model_loaded", None))

        translator = str.maketrans('', '', string.punctuation)
        translation_history = []
        min_confidence = float(getattr(config, 'min_confidence', 0.30) or 0.0)
        glossary = getattr(config, 'glossary', {}) or {}
        source_context = deque(maxlen=2)  # recent JA lines, sent to DeepL as context
        subtitle_id = 0

        def whisper_translate(audio):
            """Fallback EN decode when the text translator can't deliver."""
            res = backend.transcribe(audio, task="translate", language="en")
            return res.text.strip()

        while not stop_event.is_set():
            start_time = time.time()
            had_translation, was_hallucination = False, False
            try:
                audio_chunk_np = audio_queue.get(timeout=1)
                raw_audio = audio_chunk_np.flatten()

                if not config.use_dynamic_chunking:
                    # Gate on the RAW signal BEFORE enhancement: normalization
                    # would otherwise boost silence past the threshold and make
                    # this check dead code.
                    rms = np.sqrt(np.mean(raw_audio ** 2))
                    if rms < config.volume_threshold:
                        stats.add_chunk(time.time() - start_time, False, False)
                        continue

                    if config.use_vad_filter and vad_model is not None:
                        audio_tensor = torch.from_numpy(raw_audio).float()
                        if len(audio_tensor) < 512:
                            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - len(audio_tensor)))
                        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
                        if speech_prob < config.vad_threshold:
                            stats.add_chunk(time.time() - start_time, False, False)
                            continue

                if getattr(config, 'enhance_audio', True):
                    audio_chunk_np = enhance_audio_quality(raw_audio, sample_rate=SAMPLE_RATE)
                else:
                    audio_chunk_np = raw_audio

                audio_data = audio_chunk_np.flatten().astype(np.float32)
                min_samples = int(SAMPLE_RATE * 1.0)
                if len(audio_data) < min_samples:
                    print(f"Skipped chunk: too short ({len(audio_data)/SAMPLE_RATE:.2f}s)")
                    stats.add_chunk(time.time() - start_time, False, False)
                    continue

                # Run ASR
                result = backend.transcribe(audio_data)
                asr_text = result.text.strip()
                confidence_score = result.confidence
                logger.debug(f"ASR ({result.backend}): {len(result.segments)} segments, "
                             f"confidence {confidence_score:.2f}")

                had_translation = bool(asr_text)
                if not asr_text:
                    stats.add_chunk(time.time() - start_time, False, False, confidence_score)
                    continue

                # Decoder-statistics gates first (real signal, cheap), then
                # the string filters for phrases that score fine on logprob
                low_conf, reason = is_low_confidence(result)
                if not low_conf and confidence_score < min_confidence:
                    low_conf, reason = True, f"confidence {confidence_score:.2f} < {min_confidence:.2f}"
                if low_conf:
                    print(f"Filtered low-confidence output: '{asr_text}' ({reason})")
                    stats.add_chunk(time.time() - start_time, True, True, confidence_score)
                    continue

                # Text translation stage (DeepL / FuguMT): ASR produced JA,
                # the translator produces the EN line.
                jp_text = None
                processed_text = asr_text
                if use_text_translator:
                    jp_text = asr_text
                    english = None
                    try:
                        context = " ".join(source_context) or None
                        english = text_translator.translate(jp_text, context=context)
                    except TranslatorUnavailable as e:
                        print(f"Translation engine disabled for this session: {e}")
                        print("Falling back to Whisper's built-in translation.")
                        gui_queue.put(("status", "Text translator unavailable - using Whisper translation"))
                        text_translator = None
                        use_text_translator = False
                    if english is None:
                        # transient failure (or translator just disabled)
                        english = whisper_translate(audio_data)
                    source_context.append(jp_text)
                    processed_text = english.strip()
                    if not processed_text:
                        stats.add_chunk(time.time() - start_time, True, False, confidence_score)
                        continue

                is_hallucination_result = is_hallucination(processed_text, translator, translation_history)
                was_hallucination = is_hallucination_result

                if not is_hallucination_result:
                    processed_text = apply_glossary(post_process_translation(processed_text), glossary)
                    translation_history.append(processed_text)
                    if len(translation_history) > 10:
                        translation_history.pop(0)

                    subtitle_text = processed_text
                    if dual_output:
                        # JP line: free when the text-translator path already
                        # transcribed JA; otherwise a second decode. Only the
                        # decoder-statistics gate applies (the string filters
                        # are English-oriented).
                        if jp_text is None:
                            try:
                                jp_result = backend.transcribe(
                                    audio_data, task="transcribe", language=source_lang)
                                jp_low, jp_reason = is_low_confidence(jp_result)
                                if not jp_low:
                                    jp_text = jp_result.text.strip()
                                else:
                                    logger.debug(f"Dropped JP line ({jp_reason})")
                            except Exception as e:
                                logger.warning(f"JP transcription decode failed: {e}")
                        if jp_text:
                            subtitle_text = f"{jp_text}\n{processed_text}"

                    subtitle_id += 1
                    print(f"Translation: {processed_text} (confidence: {confidence_score:.2f})")

                    # Emit immediately; the speaker label (if any) arrives
                    # asynchronously later via a "speaker_update" message.
                    gui_queue.put(("subtitle", {
                        "id": subtitle_id,
                        "text": subtitle_text,
                        "display_text": subtitle_text,
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
                if "timeout" not in str(e).lower():
                    print(f"Processor error: {e}")
                    traceback.print_exc()
                stats.add_chunk(time.time() - start_time, False, False)

    except Exception as e:
        print(f"Processor Thread Fatal Error: {e}")
        traceback.print_exc()
        gui_queue.put(("error", f"Processing error: {e}"))
    finally:
        try:
            backend.close()
        except NameError:
            pass
        print("Processor thread stopped.")
