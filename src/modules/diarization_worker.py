"""
Async speaker diarization worker.

pyannote diarization is far too slow to run inline before ASR (it roughly
doubles per-chunk latency), so it runs in its own thread: the processor
emits subtitles immediately and this worker attaches speaker labels
afterwards via "speaker_update" GUI messages keyed by subtitle id.
"""
import os
import time
import logging
import traceback
from queue import Empty

from .config import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Items older than this are dropped instead of diarized — the subtitle has
# likely already scrolled away, so the label would be wasted work.
STALE_ITEM_SECONDS = 10.0


def create_diarizer(config, device):
    """Load the pyannote diarizer, or return None (with a reason printed)."""
    hf_token = getattr(config, 'hf_token', None) or os.environ.get('HF_TOKEN')
    if not hf_token:
        print("WARNING: No HuggingFace token found for speaker diarization")
        print("Set HF_TOKEN environment variable or configure in settings")
        print("Get token at: https://huggingface.co/settings/tokens")
        return None

    try:
        from .diarization import SpeakerDiarizer
    except ImportError:
        print("pyannote.audio not installed. Run: pip install pyannote.audio")
        print("Speaker diarization disabled.")
        return None

    try:
        diarizer = SpeakerDiarizer(
            hf_token=hf_token,
            device=device.split(':')[0],  # 'cuda' or 'cpu'
            min_speakers=getattr(config, 'min_speakers', 1),
            max_speakers=getattr(config, 'max_speakers', 5)
        )
        if diarizer.load_model():
            print("Speaker diarization model loaded successfully.")
            return diarizer
        print("Failed to load speaker diarization model. Disabling.")
    except Exception as e:
        print(f"Failed to initialize speaker diarization: {e}")
    return None


def diarization_worker(stop_event, diar_queue, config, gui_queue, device):
    """Consume (subtitle_id, audio, enqueue_time) items and post
    ("speaker_update", {...}) messages back to the GUI."""
    print("Diarization worker started.")
    try:
        diarizer = create_diarizer(config, device)
        gui_queue.put(("diarization_status", diarizer is not None))
        if diarizer is None:
            print("Diarization worker exiting (no diarizer available).")
            return

        while not stop_event.is_set():
            try:
                subtitle_id, audio, enqueue_time = diar_queue.get(timeout=1)
            except Empty:
                continue

            if time.time() - enqueue_time > STALE_ITEM_SECONDS:
                logger.debug(f"Skipping stale diarization item {subtitle_id}")
                continue

            try:
                speaker_label, speaker_color = diarizer.get_simple_speaker(audio, SAMPLE_RATE)
                if speaker_label:
                    gui_queue.put(("speaker_update", {
                        "id": subtitle_id,
                        "speaker": speaker_label,
                        "speaker_color": speaker_color,
                    }))
            except Exception as e:
                logger.warning(f"Diarization error: {e}")

    except Exception as e:
        print(f"Diarization worker fatal error: {e}")
        traceback.print_exc()
        gui_queue.put(("diarization_status", False))
    finally:
        print("Diarization worker stopped.")
