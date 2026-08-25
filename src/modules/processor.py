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
from .audio_utils import enhance_audio_quality
from .model_utils import ensure_model_downloaded
from .asr_backend import load_backend
from .filters import post_process_translation, is_hallucination, is_low_confidence
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

        # Ensure models are downloaded
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

        # Initialize speaker diarization if enabled
        if use_diarization:
            print("Initializing speaker diarization...")
            try:
                from .diarization import SpeakerDiarizer

                hf_token = getattr(config, 'hf_token', None) or os.environ.get('HF_TOKEN')

                if not hf_token:
                    print("WARNING: No HuggingFace token found for speaker diarization")
                    print("Set HF_TOKEN environment variable or configure in settings")
                    print("Get token at: https://huggingface.co/settings/tokens")
                    use_diarization = False
                else:
                    diarizer = SpeakerDiarizer(
                        hf_token=hf_token,
                        device=device.split(':')[0],  # 'cuda' or 'cpu'
                        min_speakers=getattr(config, 'min_speakers', 1),
                        max_speakers=getattr(config, 'max_speakers', 5)
                    )

                    # Pre-load the model
                    if diarizer.load_model():
                        print("Speaker diarization model loaded successfully.")
                    else:
                        print("Failed to load speaker diarization model. Disabling.")
                        use_diarization = False
                        diarizer = None

            except ImportError:
                print("pyannote.audio not installed. Run: pip install pyannote.audio")
                print("Speaker diarization disabled.")
                use_diarization = False
            except Exception as e:
                print(f"Failed to initialize speaker diarization: {e}")
                use_diarization = False

        source_lang = getattr(config, 'source_language_code', 'ja')
        # "both" shows the JP transcription above the EN translation; the
        # primary decode is the translation, the JP line is a second decode.
        dual_output = config.output_mode == "both"
        task = "transcribe" if config.output_mode == "transcribe" else "translate"
        # The language token is the OUTPUT language for this model. Transcribe
        # mode must request Japanese ("ja"), never config.language_code, whose
        # default "en" is the subtitle display language.
        target_lang = "en" if task == "translate" else source_lang
        mode_desc = "dual JP+EN" if dual_output else f"'{task}' targeting '{target_lang}'"
        print(f"Setting model task to: {mode_desc} for Japanese audio.")

        print("Loading ASR model...")
        backend = load_backend(config, device=device, model_dir=model_dir,
                               task=task, language=target_lang)
        # Throwaway decode so the first real utterance doesn't pay CUDA init
        backend.warm_up()
        print("ASR Model loaded successfully.")

        # Notify GUI about diarization status
        if use_diarization:
            print("Speaker diarization: ENABLED")
            gui_queue.put(("diarization_status", True))
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

                # Run ASR/translation
                result = backend.transcribe(audio_data)
                processed_text = result.text.strip()

                confidence_score = result.confidence
                logger.debug(f"ASR ({result.backend}): {len(result.segments)} segments, "
                             f"confidence {confidence_score:.2f}")

                had_translation = bool(processed_text)
                is_hallucination_result = False

                if processed_text:
                    # Decoder-statistics gate first (real signal, cheap), then
                    # the string filters for phrases that score fine on logprob
                    low_conf, reason = is_low_confidence(result)
                    if low_conf:
                        print(f"Filtered low-confidence output: '{processed_text}' ({reason})")
                        stats.add_chunk(time.time() - start_time, True, True, confidence_score)
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

                        subtitle_text = processed_text
                        if dual_output:
                            # Second decode of the same chunk: JP transcription.
                            # Only the decoder-statistics gate applies (the
                            # string filters are English-oriented).
                            try:
                                jp_result = backend.transcribe(
                                    audio_data, task="transcribe", language=source_lang)
                                jp_text = jp_result.text.strip()
                                jp_low, jp_reason = is_low_confidence(jp_result)
                                if jp_text and not jp_low:
                                    subtitle_text = f"{jp_text}\n{processed_text}"
                                elif jp_low:
                                    logger.debug(f"Dropped JP line ({jp_reason})")
                            except Exception as e:
                                logger.warning(f"JP transcription decode failed: {e}")

                        # Format output with speaker label if available
                        if speaker_label:
                            display_text = f"[{speaker_label}] {subtitle_text}"
                            print(f"Translation ({speaker_label}): {processed_text} (confidence: {confidence_score:.2f})")
                        else:
                            display_text = subtitle_text
                            print(f"Translation: {processed_text} (confidence: {confidence_score:.2f})")

                        # Send to GUI with speaker info
                        gui_queue.put(("subtitle", {
                            "text": subtitle_text,
                            "display_text": display_text,
                            "speaker": speaker_label,
                            "speaker_color": speaker_color,
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
        try:
            backend.close()
        except NameError:
            pass
        print("Processor thread stopped.")
