"""
Text translation stage (optional).

With an engine configured, the processor transcribes Japanese and translates
the text here instead of using Whisper's built-in translate task — a dedicated
MT engine reads far better for JA->EN.

Engines:
- DeepL (API key required): highest quality; previous lines are passed as
  (un-billed) context for coherence across subtitle chunks.
- FuguMT (local): free forever, offline, no account; a small JA->EN
  MarianMT model (staka/fugumt-ja-en) run on CPU so it never competes with
  Whisper for VRAM.
"""
import os
import re
import logging

import requests

logger = logging.getLogger(__name__)

DEEPL_FREE_ENDPOINT = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_ENDPOINT = "https://api.deepl.com/v2/translate"

FUGUMT_MODEL_ID = "staka/fugumt-ja-en"

# Give up on DeepL for the rest of the session after this many consecutive
# transient failures; the processor then falls back to Whisper translation.
MAX_CONSECUTIVE_FAILURES = 3


class TranslatorUnavailable(Exception):
    """Raised when the translator should not be used for the rest of the
    session (bad key, quota exhausted, or persistent network failure)."""


class DeepLTranslator:
    """Minimal DeepL REST client for JA->EN subtitle translation."""

    name = "deepl"

    def __init__(self, api_key, timeout=5.0):
        self.api_key = api_key
        # Free-tier keys are suffixed ":fx" and use a separate host
        self.endpoint = DEEPL_FREE_ENDPOINT if api_key.endswith(":fx") else DEEPL_PRO_ENDPOINT
        self.timeout = timeout
        self.chars_sent = 0
        self.consecutive_failures = 0

    def translate(self, text, context=None):
        """Translate Japanese text to English.

        Returns the translation, or None on a transient failure (skip this
        chunk). Raises TranslatorUnavailable when DeepL should be abandoned
        for the session.
        """
        data = {
            "text": [text],
            "source_lang": "JA",
            "target_lang": "EN-US",
        }
        if context:
            data["context"] = context

        try:
            response = requests.post(
                self.endpoint,
                json=data,
                headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            return self._transient_failure(f"network error: {e}")

        if response.status_code == 456:
            raise TranslatorUnavailable("DeepL quota exhausted for this billing period")
        if response.status_code in (401, 403):
            raise TranslatorUnavailable("DeepL API key rejected")
        if response.status_code != 200:
            return self._transient_failure(f"HTTP {response.status_code}")

        try:
            translation = response.json()["translations"][0]["text"]
        except (ValueError, KeyError, IndexError) as e:
            return self._transient_failure(f"unexpected response: {e}")

        self.consecutive_failures = 0
        # The credit is a finite one-time budget on new DeepL accounts, so
        # surface usage in the console at regular milestones.
        prev_chars = self.chars_sent
        self.chars_sent += len(text)
        if self.chars_sent // 10000 > prev_chars // 10000:
            print(f"DeepL usage this session: {self.chars_sent:,} characters")
        return translation

    def _transient_failure(self, reason):
        self.consecutive_failures += 1
        print(f"DeepL translation failed ({reason}) "
              f"[{self.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}]")
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise TranslatorUnavailable(
                f"{MAX_CONSECUTIVE_FAILURES} consecutive DeepL failures, last: {reason}")
        return None


class FuguMTTranslator:
    """Local JA->EN translation via the FuguMT MarianMT model.

    Runs on CPU deliberately: the model is small enough that CPU inference is
    fast (tens of ms per subtitle) and it avoids competing with Whisper and
    pyannote for VRAM.
    """

    name = "fugumt"

    def __init__(self, config):
        self.cache_dir = os.path.join(
            getattr(config, "model_cache_dir", os.path.expanduser("~/.cache")), "fugumt")
        self.pipe = None
        self.consecutive_failures = 0

    def load(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        print(f"Loading FuguMT translation model '{FUGUMT_MODEL_ID}' (CPU)...")
        os.makedirs(self.cache_dir, exist_ok=True)
        tokenizer = MarianTokenizer.from_pretrained(FUGUMT_MODEL_ID, cache_dir=self.cache_dir)
        model = MarianMTModel.from_pretrained(FUGUMT_MODEL_ID, cache_dir=self.cache_dir)
        model.eval()

        def pipe(text):
            # transformers v5 removed the "translation" pipeline task, so
            # drive the model directly; same output shape as the old pipeline.
            batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                tokens = model.generate(**batch, max_new_tokens=256, num_beams=4)
            return [{"translation_text": tokenizer.decode(tokens[0], skip_special_tokens=True)}]

        self.pipe = pipe
        print("FuguMT model loaded.")

    def translate(self, text, context=None):
        """Translate Japanese text to English. `context` is accepted for
        interface parity with DeepL but MarianMT cannot use it."""
        try:
            translation = self.pipe(text)[0]["translation_text"]
        except Exception as e:
            self.consecutive_failures += 1
            print(f"FuguMT translation failed ({e}) "
                  f"[{self.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}]")
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise TranslatorUnavailable(
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive FuguMT failures, last: {e}")
            return None

        self.consecutive_failures = 0
        return translation


def apply_glossary(text, glossary):
    """Case-insensitive whole-word replacement on the English output. Fixes
    recurring mistranslations of names/terms (e.g. "White God" -> "Fubuki")."""
    if not glossary:
        return text
    for wrong, right in glossary.items():
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text, flags=re.IGNORECASE)
    return text


def create_translator(config):
    """Return a translator instance per config, or None to use Whisper's
    built-in translation."""
    engine = getattr(config, "translation_engine", "whisper")

    if engine == "deepl":
        api_key = getattr(config, "deepl_api_key", None)
        if not api_key:
            print("translation_engine is 'deepl' but no deepl_api_key configured; "
                  "using Whisper translation. Get an API key at "
                  "https://www.deepl.com/pro-api (new accounts: one-time 1M character credit)")
            return None
        translator = DeepLTranslator(api_key)
        print(f"Translation engine: DeepL ({'free' if api_key.endswith(':fx') else 'pro'} tier)")
        return translator

    if engine == "fugumt":
        translator = FuguMTTranslator(config)
        try:
            translator.load()
        except Exception as e:
            print(f"FuguMT unavailable ({e}); using Whisper translation.")
            return None
        print("Translation engine: FuguMT (local, CPU)")
        return translator

    return None
