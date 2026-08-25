"""
ASR backend abstraction.

One small interface (AsrBackend.transcribe) with two implementations:

  * FasterWhisperBackend  -- CTranslate2 via the faster-whisper package (fast path)
  * TransformersBackend   -- the existing HuggingFace transformers pipeline (fallback)

load_backend() prefers faster-whisper when the package is installed and the
pre-converted CTranslate2 model can be loaded; otherwise it falls back to the
transformers pipeline with a clear console message (the #1 failure mode on
Windows is missing cuDNN 9 DLLs for CTranslate2).

Both backends accept mono float32 PCM at 16 kHz and return an AsrResult.
Heavy imports (torch, transformers, faster_whisper) happen inside the classes
so importing this module stays cheap and the fallback works when faster-whisper
is not installed.
"""
import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import SAMPLE_RATE, MODEL_ID, FASTER_MODEL_ID

logger = logging.getLogger(__name__)


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None


@dataclass
class AsrResult:
    text: str
    confidence: float
    segments: List[AsrSegment] = field(default_factory=list)
    # Real decoder statistics; None on backends that can't provide them.
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    backend: str = ""


def _prepare_audio(audio: np.ndarray) -> np.ndarray:
    """Both backends want a flat float32 array at 16 kHz."""
    return np.asarray(audio, dtype=np.float32).flatten()


def _filter_kwargs(func, kwargs):
    """Drop kwargs the installed version of `func` does not accept.

    faster-whisper adds/renames transcribe() options between releases, so
    introspect the live signature instead of guessing.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        logger.warning("faster-whisper ignores unsupported options: %s", ", ".join(dropped))
    return kept


class AsrBackend:
    """Minimal interface the processor thread depends on."""

    name = "base"

    def transcribe(self, audio: np.ndarray, task=None, language=None) -> AsrResult:
        """audio: mono float32 numpy array sampled at SAMPLE_RATE (16 kHz).

        task/language override the defaults for this one call (used by dual
        JP+EN subtitle mode to run a second decode on the same chunk)."""
        raise NotImplementedError

    def warm_up(self):
        """Run a throwaway decode so the first real utterance is fast."""
        try:
            self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except Exception as e:
            logger.warning("warm-up decode failed (non-fatal): %s", e)

    def close(self):
        pass


class FasterWhisperBackend(AsrBackend):
    """CTranslate2 inference through faster-whisper."""

    name = "faster-whisper"

    def __init__(self, config, task="translate", language="en", device="auto"):
        # ImportError / model-load errors are caught by load_backend -> fallback.
        from faster_whisper import WhisperModel

        device = str(device)
        device_index = 0
        if ":" in device:                     # "cuda:0" -> ("cuda", 0)
            device, idx = device.split(":", 1)
            device_index = int(idx)

        compute_type = getattr(config, 'compute_type', 'auto') or 'auto'
        if compute_type == 'auto':
            compute_type = 'float16' if device == 'cuda' else 'int8'

        self.task = task
        self.language = language
        self.compute_type = compute_type

        self.model = WhisperModel(
            FASTER_MODEL_ID,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            download_root=config.model_cache_dir,
        )

        options = {
            "task": task,
            "language": language,
            "beam_size": int(getattr(config, 'beam_size', 2) or 2),
            # Each queue item is an independent VAD-cut utterance; carrying
            # decoder context across them propagates hallucinations.
            "condition_on_previous_text": False,
            # Greedy first; re-decode hotter only when the output fails the
            # compression-ratio / logprob checks (Whisper's anti-repetition
            # fallback, which the transformers pipeline path lacked).
            "temperature": [0.0, 0.2, 0.4],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            # In faster-whisper, [-1] EXPANDS to the model's non-speech token
            # list (unlike HF transformers, where it means literal index -1).
            "suppress_tokens": [-1],
            "without_timestamps": True,
            "vad_filter": False,   # the recorder already VAD-gates chunks
        }
        self.options = _filter_kwargs(self.model.transcribe, options)
        logger.info("faster-whisper loaded (device=%s, compute_type=%s)", device, compute_type)

    def transcribe(self, audio: np.ndarray, task=None, language=None) -> AsrResult:
        audio = _prepare_audio(audio)

        options = self.options
        if task is not None or language is not None:
            options = dict(options)
            if task is not None and "task" in self.options:
                options["task"] = task
            if language is not None and "language" in self.options:
                options["language"] = language

        # transcribe() returns a lazy generator; decoding happens on iteration.
        seg_iter, _info = self.model.transcribe(audio, **options)

        segments = []
        for seg in seg_iter:
            segments.append(AsrSegment(
                start=float(seg.start or 0.0),
                end=float(seg.end or 0.0),
                text=(seg.text or "").strip(),
                avg_logprob=getattr(seg, 'avg_logprob', None),
                no_speech_prob=getattr(seg, 'no_speech_prob', None),
                compression_ratio=getattr(seg, 'compression_ratio', None),
            ))

        text = " ".join(s.text for s in segments if s.text).strip()
        avg_lp, no_speech, comp_ratio = self._aggregate(segments)
        return AsrResult(
            text=text,
            confidence=self._confidence(avg_lp, no_speech),
            segments=segments,
            avg_logprob=avg_lp,
            no_speech_prob=no_speech,
            compression_ratio=comp_ratio,
            backend=self.name,
        )

    @staticmethod
    def _aggregate(segments):
        """Duration-weighted avg_logprob, max no_speech_prob, max compression_ratio."""
        num = den = 0.0
        no_speech = comp = None
        for s in segments:
            if s.avg_logprob is not None:
                dur = max(0.1, s.end - s.start)
                num += s.avg_logprob * dur
                den += dur
            if s.no_speech_prob is not None:
                no_speech = s.no_speech_prob if no_speech is None else max(no_speech, s.no_speech_prob)
            if s.compression_ratio is not None:
                comp = s.compression_ratio if comp is None else max(comp, s.compression_ratio)
        avg_lp = (num / den) if den else None
        return avg_lp, no_speech, comp

    @staticmethod
    def _confidence(avg_logprob, no_speech_prob):
        """exp(avg token logprob), discounted by speech probability — a real
        model confidence, replacing the old timestamp-length heuristic."""
        if avg_logprob is None:
            return 0.0
        conf = math.exp(max(min(avg_logprob, 0.0), -10.0))
        if no_speech_prob is not None:
            conf *= (1.0 - min(max(no_speech_prob, 0.0), 1.0))
        return min(max(conf, 0.0), 1.0)

    def close(self):
        self.model = None


class TransformersBackend(AsrBackend):
    """The pre-migration path: HF transformers ASR pipeline. Kept as fallback."""

    name = "transformers"

    def __init__(self, model_dir, task="translate", language="en", device="cpu"):
        import torch
        from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
        from .model_utils import get_kotoba_generate_kwargs, get_kotoba_pipeline_kwargs

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            processor = AutoProcessor.from_pretrained(model_dir)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
                **get_kotoba_pipeline_kwargs(),
            )
            print("Model loaded successfully from cache.")
        except Exception as e:
            print(f"Failed to load model from cache: {e}")
            print("Attempting to download model directly from Hugging Face...")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                torch_dtype=torch_dtype,
                device=device,
                model_kwargs=({"attn_implementation": "sdpa"}
                              if torch.cuda.is_available() else {}),
                **get_kotoba_pipeline_kwargs(),
            )
            print("Model loaded successfully from Hugging Face.")

        self._make_generate_kwargs = get_kotoba_generate_kwargs
        self._kwargs_cache = {(task, language): get_kotoba_generate_kwargs(task, language)}

    def transcribe(self, audio: np.ndarray, task=None, language=None) -> AsrResult:
        audio = _prepare_audio(audio)
        key = (task or self.task, language or self.language)
        generate_kwargs = self._kwargs_cache.get(key)
        if generate_kwargs is None:
            generate_kwargs = self._make_generate_kwargs(*key)
            self._kwargs_cache[key] = generate_kwargs
        result = self.pipe({"sampling_rate": SAMPLE_RATE, "raw": audio},
                           generate_kwargs=generate_kwargs)
        text = (result.get("text") or "").strip()

        # No decoder log-probs through the pipeline API: word-count heuristic,
        # flagged as such by avg_logprob=None so filtering no-ops on it.
        words = len(text.split())
        if words >= 3:
            conf = 0.85
        elif words >= 1:
            conf = 0.75
        else:
            conf = 0.0

        duration = len(audio) / float(SAMPLE_RATE)
        return AsrResult(
            text=text,
            confidence=conf,
            segments=[AsrSegment(start=0.0, end=duration, text=text)],
            backend=self.name,
        )

    def close(self):
        self.pipe = None


def load_backend(config, device, model_dir, task, language) -> AsrBackend:
    """
    Build the best available backend.

    config.asr_backend:
        "faster_whisper" (default) -> try faster-whisper, fall back to transformers
        "transformers"             -> the original pipeline path
    """
    preference = getattr(config, 'asr_backend', 'faster_whisper')

    if preference != 'transformers':
        try:
            backend = FasterWhisperBackend(config, task=task, language=language, device=device)
            print(f"ASR engine: faster-whisper ({backend.compute_type})")
            return backend
        except Exception as e:
            logger.warning("faster-whisper unavailable: %s", e)
            print(f"faster-whisper unavailable ({e}).")
            print("Falling back to the transformers engine. If this is unexpected,")
            print("check that 'pip install faster-whisper' succeeded and (on GPU)")
            print("that cuDNN 9 is available.")

    backend = TransformersBackend(model_dir, task=task, language=language, device=device)
    print("ASR engine: transformers")
    return backend
