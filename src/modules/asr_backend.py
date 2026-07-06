"""
ASR backend abstraction.

Primary backend is faster-whisper (CTranslate2) running the official
kotoba-whisper-bilingual conversion. If faster-whisper is unavailable or
fails to load (missing package, cuDNN DLLs, download failure), we fall
back to the original HuggingFace Transformers pipeline so the app keeps
working, just slower.
"""
import os
import sys
import math
import logging
import traceback
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import SAMPLE_RATE, MODEL_ID, FASTER_MODEL_ID

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    """One decoded segment. Probability fields are None on the transformers fallback."""
    text: str
    start: float
    end: float
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None


@dataclass
class ASRResult:
    text: str
    segments: List[ASRSegment]


def drop_no_speech_segments(segments):
    """First-line hallucination defense: drop segments the model itself
    flags as probable non-speech decoded with very low confidence."""
    kept = []
    for seg in segments:
        if (seg.no_speech_prob is not None and seg.avg_logprob is not None
                and seg.no_speech_prob > 0.85 and seg.avg_logprob < -1.0):
            logger.debug(f"Dropped no-speech segment: '{seg.text}' "
                         f"(no_speech={seg.no_speech_prob:.2f}, logprob={seg.avg_logprob:.2f})")
            continue
        kept.append(seg)
    return kept


def _heuristic_confidence(text):
    """Word-count heuristic used when the backend provides no probabilities."""
    word_count = len(text.split())
    if word_count >= 3:
        return 0.85
    if word_count >= 1:
        return 0.75
    return 0.6


def confidence_from_segments(segments, text=""):
    """Duration-weighted confidence from real model probabilities:
    exp(avg_logprob) scaled by (1 - no_speech_prob), clamped to [0, 1].
    Falls back to a word-count heuristic when probabilities are absent."""
    scored = [s for s in segments if s.avg_logprob is not None]
    if not scored:
        return _heuristic_confidence(text)

    total_duration = 0.0
    weighted_sum = 0.0
    for seg in scored:
        duration = max(seg.end - seg.start, 0.1)
        seg_conf = math.exp(min(seg.avg_logprob, 0.0)) * (1.0 - (seg.no_speech_prob or 0.0))
        weighted_sum += seg_conf * duration
        total_duration += duration

    return max(0.0, min(1.0, weighted_sum / total_duration))


def _setup_cuda_dlls():
    """Help ctranslate2 find cuBLAS/cuDNN DLLs on Windows.

    torch>=2.4 Windows wheels bundle cuDNN 9 in torch/lib; the pip packages
    nvidia-cublas-cu12 / nvidia-cudnn-cu12 install DLLs under
    site-packages/nvidia/*/bin. Neither location is on the DLL search path
    by default. No-op on other platforms.
    """
    if sys.platform != "win32":
        return
    dll_dirs = []
    try:
        import torch
        dll_dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except ImportError:
        pass
    for path in sys.path:
        nvidia_dir = os.path.join(path, "nvidia")
        if os.path.isdir(nvidia_dir):
            for pkg in ("cublas", "cudnn"):
                bin_dir = os.path.join(nvidia_dir, pkg, "bin")
                if os.path.isdir(bin_dir):
                    dll_dirs.append(bin_dir)
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except (OSError, AttributeError):
                pass


class FasterWhisperBackend:
    """kotoba-whisper-bilingual via faster-whisper / CTranslate2."""

    name = "faster-whisper"

    def __init__(self, config, device):
        self.config = config
        self.device = "cuda" if device.startswith("cuda") else "cpu"
        self.model = None

    def load(self):
        _setup_cuda_dlls()
        from faster_whisper import WhisperModel

        compute_type = "float16" if self.device == "cuda" else "int8"
        download_root = os.path.join(self.config.model_cache_dir, "faster_whisper")
        os.makedirs(download_root, exist_ok=True)

        print(f"Loading faster-whisper model '{FASTER_MODEL_ID}' "
              f"({self.device}, {compute_type})...")
        self.model = WhisperModel(
            FASTER_MODEL_ID,
            device=self.device,
            compute_type=compute_type,
            download_root=download_root,
        )
        print("faster-whisper model loaded.")

    def transcribe(self, audio, task, target_lang):
        # For kotoba-whisper-bilingual, `language` selects the TARGET language
        # ("en" + task="translate" = JA speech -> EN text), same convention as
        # the transformers path.
        use_internal_vad = not self.config.use_dynamic_chunking
        segments_gen, info = self.model.transcribe(
            audio.astype(np.float32),
            language=target_lang,
            task=task,
            beam_size=getattr(self.config, "asr_beam_size", 5),
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            repetition_penalty=1.1,
            vad_filter=use_internal_vad,
            vad_parameters={"min_silence_duration_ms": 500} if use_internal_vad else None,
        )

        segments = [
            ASRSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                compression_ratio=seg.compression_ratio,
            )
            for seg in segments_gen  # inference happens while consuming the generator
        ]
        segments = drop_no_speech_segments(segments)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return ASRResult(text=text, segments=segments)


class TransformersBackend:
    """Original HuggingFace pipeline, kept as an automatic fallback."""

    name = "transformers"

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.pipe = None

    def load(self):
        import torch
        from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
        from .model_utils import ensure_model_downloaded, get_kotoba_pipeline_kwargs

        model_dir, _ = ensure_model_downloaded(MODEL_ID, self.config.model_cache_dir)
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            )
            processor = AutoProcessor.from_pretrained(model_dir)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=model_dtype,
                device=self.device,
                **get_kotoba_pipeline_kwargs()
            )
            print("Transformers model loaded from cache.")
        except Exception as e:
            print(f"Failed to load model from cache: {e}")
            print("Attempting to download model directly from Hugging Face...")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                torch_dtype=model_dtype,
                device=self.device,
                model_kwargs={"attn_implementation": "sdpa"},
                **get_kotoba_pipeline_kwargs()
            )
            print("Transformers model loaded from Hugging Face.")

    def transcribe(self, audio, task, target_lang):
        from .model_utils import get_kotoba_generate_kwargs, optimize_for_vtuber_content

        generate_kwargs = optimize_for_vtuber_content(
            get_kotoba_generate_kwargs(task, target_lang))
        result = self.pipe({"sampling_rate": SAMPLE_RATE, "raw": audio.astype(np.float32)},
                           generate_kwargs=generate_kwargs)

        segments = []
        for chunk in result.get("chunks") or []:
            timestamp = chunk.get("timestamp")
            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start = timestamp[0] or 0.0
                end = timestamp[1] if timestamp[1] is not None else start + 1.0
            else:
                start, end = 0.0, 1.0
            segments.append(ASRSegment(text=chunk.get("text", ""), start=start, end=end))

        return ASRResult(text=result["text"].strip(), segments=segments)


def create_backend(config, device, gui_queue=None):
    """Create the configured ASR backend, falling back to transformers if
    faster-whisper cannot be used."""
    requested = getattr(config, "asr_backend", "faster-whisper")

    if requested != "transformers":
        try:
            backend = FasterWhisperBackend(config, device)
            backend.load()
            return backend
        except Exception as e:
            print(f"faster-whisper backend unavailable ({e}); "
                  f"falling back to transformers pipeline.")
            traceback.print_exc()
            if gui_queue is not None:
                gui_queue.put(("status", "faster-whisper unavailable - using slower fallback"))

    backend = TransformersBackend(config, device)
    backend.load()
    return backend
