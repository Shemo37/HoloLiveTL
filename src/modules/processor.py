
import time
import numpy as np
import torch
import string
import traceback
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from .audio_utils import enhance_audio_quality
from .model_utils import ensure_model_downloaded, get_kotoba_generate_kwargs, get_kotoba_pipeline_kwargs, optimize_for_vtuber_content
from .filters import post_process_translation, is_hallucination
from .config import SAMPLE_RATE, MODEL_ID

def processor_thread(stop_event, audio_queue, config, stats, gui_queue):
    print("⚙️ Processor thread started.")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device.upper()}")

        # Ensure models are downloaded
        model_dir, vad_dir = ensure_model_downloaded(MODEL_ID, config.model_cache_dir)
        
        vad_model = None
        
        if not config.use_dynamic_chunking and config.use_vad_filter:
            try:
                print("📥 Loading Silero VAD model...")
                torch.set_num_threads(1)
                vad_path = os.path.join(vad_dir, "silero_vad.jit")
                vad_model = torch.jit.load(vad_path, map_location='cpu')
                print("✅ VAD model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Could not load VAD model: {e}. Disabling VAD filter.")
                config.use_vad_filter = False

        print(f"📥 Loading ASR model from cache...")
        
        # Determine the appropriate dtype
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Try to load the model directly first
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            processor = AutoProcessor.from_pretrained(model_dir)
            
            # Create pipeline with loaded model and processor
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=model_dtype,
                device=device,
                **get_kotoba_pipeline_kwargs()
            )
            print("✅ Model loaded successfully from cache.")
        except Exception as e:
            print(f"⚠️ Failed to load model from cache: {e}")
            print("📥 Attempting to download model directly from Hugging Face...")
            
            # Fallback to downloading directly from Hugging Face
            try:
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=MODEL_ID,
                    torch_dtype=model_dtype,
                    device=device,
                    model_kwargs={"attn_implementation": "sdpa"} if torch.cuda.is_available() else {},
                    **get_kotoba_pipeline_kwargs()
                )
                print("✅ Model loaded successfully from Hugging Face.")
            except Exception as e2:
                print(f"🔴 Failed to load model from Hugging Face: {e2}")
                raise Exception(f"Could not load model: {e}, {e2}")

        task = "translate" if config.output_mode == "translate" else "transcribe"
        target_lang = "en" if task == "translate" else config.language_code
        print(f"✅ Setting model task to: '{task}' targeting '{target_lang}' for Japanese audio.")

        generate_kwargs = get_kotoba_generate_kwargs(task, target_lang)
        generate_kwargs = optimize_for_vtuber_content(generate_kwargs)
        print("✅ ASR Model loaded successfully.")

        gui_queue.put(("model_loaded", None))

        translator = str.maketrans('', '', string.punctuation)
        last_valid_translation = ""
        translation_history = []

        while not stop_event.is_set():
            start_time = time.time()
            had_translation, was_hallucination = False, False
            try:
                audio_chunk_np = audio_queue.get(timeout=1)

                audio_chunk_np = enhance_audio_quality(audio_chunk_np.flatten(), sample_rate=SAMPLE_RATE)
                
                if not config.use_dynamic_chunking:
                    rms = np.sqrt(np.mean(audio_chunk_np ** 2))
                    if rms < config.volume_threshold:
                        stats.add_chunk(time.time() - start_time, False, False)
                        continue

                    if config.use_vad_filter and vad_model is not None:
                        audio_tensor = torch.from_numpy(audio_chunk_np.flatten()).float()
                        if len(audio_tensor) < 512:
                            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - len(audio_tensor)))
                        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

                        speech_windows = [prob > config.vad_threshold for prob in [speech_prob]]
                        if sum(speech_windows) < 3:
                            stats.add_chunk(time.time() - start_time, False, False)
                            continue

                audio_data = audio_chunk_np.flatten().astype(np.float32)
                min_samples = int(SAMPLE_RATE * 1.0)
                if len(audio_data) < min_samples:
                    print(f"⏩ Skipped chunk: too short ({len(audio_data)/SAMPLE_RATE:.2f}s)")
                    stats.add_chunk(time.time() - start_time, False, False)
                    continue

                result = pipe({"sampling_rate": SAMPLE_RATE, "raw": audio_data}, 
                            generate_kwargs=generate_kwargs)
                processed_text = result["text"].strip()
                
                confidence_score = 1.0
                if "chunks" in result and result["chunks"]:
                    chunk_confidences = []
                    for chunk in result["chunks"]:
                        if "timestamp" in chunk and chunk["timestamp"]:
                            timestamp = chunk["timestamp"]
                            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                                duration = timestamp[1] - timestamp[0] if timestamp[1] else 1.0
                                chunk_conf = min(0.9, 0.6 + (duration * 0.1))
                                chunk_confidences.append(chunk_conf)
                            else:
                                chunk_confidences.append(0.8)
                        else:
                            chunk_confidences.append(0.7)
                    
                    if chunk_confidences:
                        confidence_score = sum(chunk_confidences) / len(chunk_confidences)
                        print(f"📊 Chunk analysis: {len(chunk_confidences)} chunks, avg confidence: {confidence_score:.2f}")
                else:
                    word_count = len(processed_text.split())
                    if word_count >= 3:
                        confidence_score = 0.85
                    elif word_count >= 1:
                        confidence_score = 0.75
                    else:
                        confidence_score = 0.6

                had_translation = bool(processed_text)
                is_hallucination_result = False
                quality_score = 1.0
                
                if processed_text:
                    is_hallucination_result = is_hallucination(processed_text, translator, translation_history)
                    was_hallucination = is_hallucination_result
                    
                    if not is_hallucination_result:
                        processed_text = post_process_translation(processed_text)
                        translation_history.append(processed_text)
                        if len(translation_history) > 10:
                            translation_history.pop(0)
                        
                        last_valid_translation = processed_text
                        print(f"✅ Translation: {processed_text} (confidence: {confidence_score:.2f})")
                        gui_queue.put(("subtitle", processed_text))
                    else:
                        print(f"🚫 Filtered hallucination: '{processed_text}' (confidence: {confidence_score:.2f})")

                stats.add_chunk(time.time() - start_time, had_translation, was_hallucination)

            except Exception as e:
                if "timeout" not in str(e).lower():
                    print(f"🔴 Processor error: {e}")
                    traceback.print_exc()
                stats.add_chunk(time.time() - start_time, False, False)

    except Exception as e:
        print(f"🔴 Processor Thread Fatal Error: {e}")
        traceback.print_exc()
        gui_queue.put(("error", f"Processing error: {e}"))
    finally:
        print("⚙️ Processor thread stopped.")
