import re

# The phrase-based hallucination lists (exact / whole-line / substring English
# phrase matching) were removed by request: short real utterances ("thank you",
# "okay", "I see") are legitimate subtitles, especially when the EN line is a
# compressed translation. Only structural junk detection remains below.

# Refusal boilerplate the kotoba-whisper-bilingual model leaks from its
# LLM-generated training data. Checked with plain substring matching because
# \b word boundaries don't work inside Japanese text.
BOILERPLATE_FILTER = [
    "正確な翻訳を提供できません", "翻訳を提供できません", "文脈が不明確",
    "翻訳できません", "cannot provide an accurate translation",
    "the context is unclear",
]

QUALITY_INDICATORS = {
    "repetitive_patterns": [r"(.{1,10})\1{3,}", r"(\w+\s+)\1{2,}"],
    "nonsense_patterns": [r"[a-z]{20,}", r"\b\w{1}\s+\w{1}\s+\w{1}\b"],
    "filler_heavy": [r"\b(um|uh|ah|eh|mm)\b.*\b(um|uh|ah|eh|mm)\b.*\b(um|uh|ah|eh|mm)\b"]
}

# Patterns marking the onset of decoder degeneration (repetition loops, letter
# soup). Used to TRIM the junk tail off a line rather than discard the whole
# line: the prefix before the loop started is usually a valid decode.
DEGENERATE_TAIL_PATTERNS = [r"(.{1,10})\1{3,}", r"(\w+\s+)\1{2,}", r"[a-z]{25,}"]


def trim_degenerate_tail(text):
    """Cut a decode at the point where it collapses into repetition junk.

    Returns the clean prefix (possibly '' when the junk starts at the
    beginning). Repetition loops poison only the tail; the text before them
    is worth delivering.
    """
    cut = len(text)
    for pattern in DEGENERATE_TAIL_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            cut = min(cut, m.start())
    if cut >= len(text):
        return text
    return text[:cut].rstrip(" ,;:-–—")

def post_process_translation(text):
    """Clean up and improve translation text"""
    text = ' '.join(text.split())
    # The kotoba translate decode often opens mid-sentence with ", so ..."
    text = re.sub(r'^[\s,;:]+', '', text)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r'\bim\b', "I'm", text)
    text = re.sub(r'\bdont\b', "don't", text)
    text = re.sub(r'\bcant\b', "can't", text)
    text = re.sub(r'\bwont\b', "won't", text)
    
    if len(text.split()) == 1:
        text = text.rstrip('.')
    
    return text.strip()

def is_hallucination(text, translator, translation_history):
    """Check if text is likely a hallucination"""
    if not text or not text.strip():
        return True
    
    text_lower = text.lower().strip()
    text_clean = text.translate(translator).lower().strip()

    # Punctuation-only output ('.', '!!', '...') carries no content; this
    # must hold even when the confidence gates are tuned loose.
    if not text_clean:
        return True

    # Model-leaked refusal boilerplate (JP or EN), plain substring match
    for phrase in BOILERPLATE_FILTER:
        if phrase in text or phrase in text_lower:
            return True

    # Quality checks
    for pattern_list in QUALITY_INDICATORS.values():
        for pattern in pattern_list:
            if re.search(pattern, text_lower):
                return True
    
    # Check for repetition in history
    if len(translation_history) >= 3:
        recent_translations = [t.lower().strip() for t in translation_history[-3:]]
        if text_lower in recent_translations:
            return True
    
    return False


def is_low_confidence(result, no_speech_threshold=0.6, logprob_threshold=-1.0,
                      compression_ratio_threshold=2.4):
    """Model-level hallucination gate using real decoder statistics.

    `result` is an AsrResult. Backends that can't provide a statistic set it
    to None, and that check is skipped - so the transformers fallback path
    passes through untouched. Returns (is_low, reason).
    """
    if result.no_speech_prob is not None and result.no_speech_prob > no_speech_threshold:
        return True, f"no_speech_prob={result.no_speech_prob:.2f}"
    if result.compression_ratio is not None and result.compression_ratio > compression_ratio_threshold:
        return True, f"compression_ratio={result.compression_ratio:.2f} (repetition loop)"
    if result.avg_logprob is not None and result.avg_logprob < logprob_threshold:
        return True, f"avg_logprob={result.avg_logprob:.2f}"
    return False, ""
