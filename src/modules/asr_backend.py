"""
ASR backend abstraction.

One small interface (AsrBackend.transcribe) with two implementations:

  * FasterWhisperBackend  -- CTranslate2 via the faster-whisper package (fast path)
  * TransformersBackend   -- the existing HuggingFace transformers pipeline (fallback)

create_backend() prefers faster-whisper when the package is installed and the
pre-converted CTranslate2 model can be loaded; otherwise it falls back to the
transformers pipeline with a clear console message and a GUI status update
(the #1 failure mode on Windows is missing cuDNN 9 DLLs for CTranslate2).

Backends are constructed cheaply and do their heavy lifting in load(), so
selection logic and tests never pay for (or need) torch/ctranslate2 imports.
Both accept mono float32 PCM at 16 kHz and return an ASRResult.
"""
import inspect
import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import SAMPLE_RATE, MODEL_ID, FASTER_MODEL_ID

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    text: str
    start: float = 0.0
    end: float = 0.0
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None


@dataclass
class ASRResult:
    text: str
    confidence: float
    segments: List[ASRSegment] = field(default_factory=list)
    # Real decoder statistics; None on backends that can't provide them.
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    backend: str = ""


# Backwards-compatible aliases (pre-merge naming)
AsrSegment = ASRSegment
AsrResult = ASRResult


def confidence_from_segments(segments, text=None):
    """Duration-weighted confidence from decoder statistics.

    Per segment: exp(avg_logprob) * (1 - no_speech_prob), weighted by segment
    duration so a long clean segment dominates a short garbage one. Segments
    without probabilities (transformers pipeline) fall back to a word-count
    heuristic on `text`.
    """
    scored = [s for s in segments if s.avg_logprob is not None]
    if scored:
        num = den = 0.0
        for s in scored:
            weight = max(float(s.end) - float(s.start), 1e-6)
            conf = math.exp(max(min(s.avg_logprob, 0.0), -10.0))
            if s.no_speech_prob is not None:
                conf *= (1.0 - min(max(s.no_speech_prob, 0.0), 1.0))
            num += conf * weight
            den += weight
        return min(max(num / den, 0.0), 1.0)

    words = len((text or "").split())
    if words >= 3:
        return 0.85
    if words >= 1:
        return 0.75
    return 0.6


def drop_no_speech_segments(segments, no_speech_threshold=0.6, logprob_threshold=-1.0):
    """Filter out segments that are probably silence hallucinations.

    Whisper's own rule: drop only when no_speech_prob is high AND the decode
    is also low-confidence — a confident decode survives a high
    no_speech_prob, and unscored segments are never dropped here.
    """
    kept = []
    for s in segments:
        if (s.no_speech_prob is not None and s.no_speech_prob > no_speech_threshold
                and s.avg_logprob is not None and s.avg_logprob < logprob_threshold):
            logger.debug("Dropped no-speech segment: %r (ns=%.2f, lp=%.2f)",
                         s.text, s.no_speech_prob, s.avg_logprob)
            continue
        kept.append(s)
    return kept


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


def _split_device(device):
    """'cuda:0' -> ('cuda', 0); 'cpu' -> ('cpu', 0)"""
    device = str(device or "auto")
    if ":" in device:
        name, idx = device.split(":", 1)
        try:
            return name, int(idx)
        except ValueError:
            return name, 0
    return device, 0


class AsrBackend:
    """Minimal ASR engine interface: construct cheap, load() heavy."""

    name = "base"

    def __init__(self, config, task="translate", language="en", device="cpu"):
        self.config = config
        self.task = task
        self.language = language
        self.device, self.device_index = _split_device(device)

    def load(self):
        """Load the model. Heavy imports live here, not in __init__."""
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, task=None, language=None) -> ASRResult:
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
        super().__init__(config, task=task, language=language, device=device)
        compute_type = getattr(config, 'compute_type', 'auto') or 'auto'
        if compute_type == 'auto':
            compute_type = 'float16' if self.device == 'cuda' else 'int8'
        self.compute_type = compute_type
        self.model = None
        self.options = {}

    def load(self):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            FASTER_MODEL_ID,
            device=self.device,
            device_index=self.device_index,
            compute_type=self.compute_type,
            download_root=self.config.model_cache_dir,
        )

        options = {
            "task": self.task,
            "language": self.language,
            "beam_size": int(getattr(self.config, 'asr_beam_size', 5) or 5),
            # Each queue item is an independent VAD-cut utterance; carrying
            # decoder context across them propagates hallucinations.
            "condition_on_previous_text": False,
            # Greedy first; re-decode hotter only when the output fails the
            # compression-ratio / logprob checks (Whisper's anti-repetition
            # fallback, which the transformers pipeline path lacked).
            "temperature": [0.0, 0.2, 0.4],
            # Counteract beam search's short-output bias: the EN translate
            # decode otherwise truncates to a fragment of what the JA decode
            # hears. Mild repetition penalty also breaks "もう少し もう少し" loops.
            "length_penalty": 1.5,
            "repetition_penalty": 1.1,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            # In faster-whisper, [-1] EXPANDS to the model's non-speech token
            # list (unlike HF transformers, where it means literal index -1).
            "suppress_tokens": [-1],
            "without_timestamps": True,
            "vad_filter": False,   # the recorder already VAD-gates chunks
        }
        hotwords = (getattr(self.config, 'asr_hotwords', '') or '').strip()
        if hotwords:
            # Hotword prompting collapses the kotoba-whisper-bilingual decoder
            # into single-token garbage ('.', 'ununununun') on every chunk, so
            # it is never forwarded. Use the glossary (text substitution on the
            # finished line) for name correction instead.
            logger.warning("asr_hotwords configured but disabled: hotword prompting "
                           "breaks the %s decoder; use the glossary instead", FASTER_MODEL_ID)
        self.options = _filter_kwargs(self.model.transcribe, options)
        logger.info("faster-whisper loaded (device=%s, compute_type=%s)",
                    self.device, self.compute_type)

    def transcribe(self, audio: np.ndarray, task=None, language=None) -> ASRResult:
        if self.model is None:
            raise RuntimeError("backend not loaded - call load() first")
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

        segments = [
            ASRSegment(
                text=(seg.text or "").strip(),
                start=float(seg.start or 0.0),
                end=float(seg.end or 0.0),
                avg_logprob=getattr(seg, 'avg_logprob', None),
                no_speech_prob=getattr(seg, 'no_speech_prob', None),
                compression_ratio=getattr(seg, 'compression_ratio', None),
            )
            for seg in seg_iter
        ]
        segments = drop_no_speech_segments(segments)

        text = " ".join(s.text for s in segments if s.text).strip()
        avg_lp, no_speech, comp_ratio = self._aggregate(segments)
        return ASRResult(
            text=text,
            confidence=confidence_from_segments(segments, text),
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

    def close(self):
        self.model = None


class TransformersBackend(AsrBackend):
    """The pre-migration path: HF transformers ASR pipeline. Kept as fallback."""

    name = "transformers"

    def __init__(self, config, model_dir=None, task="translate", language="en", device="cpu"):
        super().__init__(config, task=task, language=language, device=device)
        self.model_dir = model_dir or os.path.join(
            getattr(config, 'model_cache_dir', '.'), "whisper_model")
        self.pipe = None
        self._make_generate_kwargs = None
        self._kwargs_cache = {}

    def load(self):
        import torch
        from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
        from .model_utils import get_kotoba_generate_kwargs, get_kotoba_pipeline_kwargs

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        device = f"cuda:{self.device_index}" if use_cuda else "cpu"
        torch_dtype = torch.bfloat16 if use_cuda else torch.float32

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            processor = AutoProcessor.from_pretrained(self.model_dir)
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
                model_kwargs=({"attn_implementation": "sdpa"} if use_cuda else {}),
                **get_kotoba_pipeline_kwargs(),
            )
            print("Model loaded successfully from Hugging Face.")

        self._make_generate_kwargs = get_kotoba_generate_kwargs
        self._kwargs_cache = {(self.task, self.language):
                              get_kotoba_generate_kwargs(self.task, self.language)}

    def transcribe(self, audio: np.ndarray, task=None, language=None) -> ASRResult:
        if self.pipe is None:
            raise RuntimeError("backend not loaded - call load() first")
        audio = _prepare_audio(audio)
        key = (task or self.task, language or self.language)
        generate_kwargs = self._kwargs_cache.get(key)
        if generate_kwargs is None:
            generate_kwargs = self._make_generate_kwargs(*key)
            self._kwargs_cache[key] = generate_kwargs
        result = self.pipe({"sampling_rate": SAMPLE_RATE, "raw": audio},
                           generate_kwargs=generate_kwargs)
        text = (result.get("text") or "").strip()

        duration = len(audio) / float(SAMPLE_RATE)
        segments = [ASRSegment(text=text, start=0.0, end=duration)]
        # No decoder log-probs through the pipeline API; word-count heuristic,
        # flagged as such by avg_logprob=None so stat-based filtering no-ops.
        return ASRResult(
            text=text,
            confidence=confidence_from_segments([], text),
            segments=segments,
            backend=self.name,
        )

    def close(self):
        self.pipe = None


def create_backend(config, device, gui_queue=None, model_dir=None,
                   task="translate", language="en") -> AsrBackend:
    """Build the best available backend and load it.

    config.asr_backend: "faster-whisper" (default; tried first with automatic
    fallback) or "transformers" (skip faster-whisper entirely). A fallback is
    reported on gui_queue as a ("status", message) tuple when one is provided.
    """
    preference = str(getattr(config, 'asr_backend', 'faster-whisper')).replace('_', '-')

    if preference != 'transformers':
        try:
            backend = FasterWhisperBackend(config, task=task, language=language, device=device)
            backend.load()
            print(f"ASR engine: faster-whisper ({backend.compute_type})")
            return backend
        except Exception as e:
            logger.warning("faster-whisper unavailable: %s", e)
            print(f"faster-whisper unavailable ({e}).")
            print("Falling back to the transformers engine. If this is unexpected,")
            print("check that 'pip install faster-whisper' succeeded and (on GPU)")
            print("that cuDNN 9 is available.")
            if gui_queue is not None:
                gui_queue.put(("status", "faster-whisper unavailable - using transformers fallback"))

    backend = TransformersBackend(config, model_dir=model_dir,
                                  task=task, language=language, device=device)
    backend.load()
    print("ASR engine: transformers")
    return backend


def load_backend(config, device, model_dir, task, language) -> AsrBackend:
    """Backwards-compatible wrapper around create_backend()."""
    return create_backend(config, device, model_dir=model_dir, task=task, language=language)
