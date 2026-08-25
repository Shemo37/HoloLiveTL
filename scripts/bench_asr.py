"""
Offline ASR benchmark: compare the transformers and faster-whisper backends
on real audio, with no GUI and no audio device.

Usage:
    python scripts/bench_asr.py --wav sample_ja.wav
    python scripts/bench_asr.py --wav clips_dir/ --repeat 3
    python scripts/bench_asr.py --wav sample_ja.wav --backends faster_whisper
    python scripts/bench_asr.py --wav sample_ja.wav --check-mapping

Per backend/precision it prints: model load time, per-clip latency
(mean/p95), RTF (latency / audio duration), peak VRAM, and the decoded text
so translation quality can be compared side by side.

--check-mapping decodes one clip under both plausible language/task token
mappings for the kotoba bilingual model, settling by output which one
produces English (expected: language="en", task="translate").
"""
import argparse
import os
import statistics
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules.config import Config, SAMPLE_RATE, MODEL_ID  # noqa: E402
from modules.model_utils import ensure_model_downloaded  # noqa: E402
from modules import asr_backend  # noqa: E402


def load_wav(path):
    """Load a WAV file as mono float32 at 16 kHz (rejects other rates)."""
    with wave.open(path, 'rb') as w:
        rate = w.getframerate()
        n = w.getnframes()
        sampwidth = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(n)
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {sampwidth}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, rate // g).astype(np.float32)
        except ImportError:
            raise ValueError(f"{path}: {rate} Hz (need {SAMPLE_RATE} Hz, and scipy is unavailable)")
    return audio


def collect_clips(path):
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.wav')
        )
        if not files:
            sys.exit(f"No .wav files in {path}")
        return files
    return [path]


def peak_vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return None


def reset_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def make_backend(kind, config, device, model_dir):
    if kind == 'faster-whisper':
        backend = asr_backend.FasterWhisperBackend(config, task='translate', language='en', device=device)
    else:
        backend = asr_backend.TransformersBackend(config, model_dir=model_dir,
                                                  task='translate', language='en', device=device)
    backend.load()
    return backend


def bench_backend(kind, config, device, model_dir, clips, repeat):
    label = kind + (f" ({config.compute_type})" if kind == 'faster-whisper' else "")
    print(f"\n=== {label} ===")
    reset_vram()

    t0 = time.perf_counter()
    try:
        backend = make_backend(kind, config, device, model_dir)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return
    load_s = time.perf_counter() - t0
    print(f"  load time: {load_s:.1f}s")

    backend.warm_up()

    latencies = []
    for path in clips:
        audio = load_wav(path)
        duration = len(audio) / SAMPLE_RATE
        texts = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            result = backend.transcribe(audio)
            latencies.append(time.perf_counter() - t0)
            texts.append(result.text)
        lat = latencies[-repeat:]
        mean = statistics.mean(lat)
        print(f"  {os.path.basename(path)} ({duration:.1f}s): "
              f"latency {mean:.2f}s  RTF {mean / duration:.2f}")
        extras = ""
        if result.avg_logprob is not None:
            extras = (f"  [avg_logprob {result.avg_logprob:.2f}, "
                      f"no_speech {result.no_speech_prob:.2f}, "
                      f"comp_ratio {result.compression_ratio:.2f}]")
        print(f"    text: {texts[0]!r}{extras}")

    if latencies:
        mean = statistics.mean(latencies)
        p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        print(f"  overall: mean {mean:.2f}s  p95 {p95:.2f}s over {len(latencies)} runs")
    vram = peak_vram_mb()
    if vram is not None:
        print(f"  peak VRAM: {vram:.0f} MB")
    backend.close()


def check_mapping(config, device, clips):
    """Decode one clip under both language/task mappings and print both outputs."""
    print("\n=== language/task mapping check (kotoba bilingual) ===")
    audio = load_wav(clips[0])
    for language, task in (("en", "translate"), ("ja", "translate")):
        try:
            backend = asr_backend.FasterWhisperBackend(
                config, task=task, language=language, device=device)
            backend.load()
            result = backend.transcribe(audio)
            print(f"  language={language!r} task={task!r} -> {result.text!r}")
            backend.close()
        except Exception as e:
            print(f"  language={language!r} task={task!r} -> FAILED: {e}")
    print("  Expected: language='en' yields English; language='ja' would request JA output.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', required=True, help='WAV file or directory of WAV files (16 kHz mono preferred)')
    ap.add_argument('--repeat', type=int, default=3, help='decodes per clip (default 3)')
    ap.add_argument('--backends', default='transformers,faster-whisper',
                    help='comma list: transformers,faster-whisper')
    ap.add_argument('--compute-types', default='float16,int8_float16',
                    help='faster-whisper precisions to bench on GPU (ignored on CPU)')
    ap.add_argument('--check-mapping', action='store_true',
                    help='also decode one clip under both language token mappings')
    args = ap.parse_args()

    clips = collect_clips(args.wav)
    config = Config()

    try:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    print(f"device: {device}, clips: {len(clips)}, repeat: {args.repeat}")

    model_dir, _ = ensure_model_downloaded(MODEL_ID, config.model_cache_dir)

    for kind in [b.strip() for b in args.backends.split(',') if b.strip()]:
        if kind == 'faster-whisper' and device.startswith('cuda'):
            for ct in [c.strip() for c in args.compute_types.split(',') if c.strip()]:
                config.compute_type = ct
                bench_backend(kind, config, device, model_dir, clips, args.repeat)
        else:
            config.compute_type = 'auto'
            bench_backend(kind, config, device, model_dir, clips, args.repeat)

    if args.check_mapping:
        check_mapping(config, device, clips)


if __name__ == '__main__':
    main()
