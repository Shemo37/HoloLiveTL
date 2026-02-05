import re

# Hallucination filters
SUBSTRING_HALLUCINATION_FILTER = [
    "thank you for watching", "thanks for watching", "don't forget",
    "to subscribe", "subscribe", "bell icon", "see you next time",
    "in the next video", "like and subscribe", "hit the bell",
    "comment below", "let me know", "as a language model", 
    "provide more context", "i'm an ai", "i cannot", "i don't have access",
    "please provide", "more information", "context is needed",
    "in central tokyo, the temperature is likely to rise rapidly from morning",
    "in central tokyo", "the temperature is likely to rise"
]

EXACT_MATCH_HALLUCINATION_FILTER = {
    "i see", "i understand", "i know", "i'm sorry", "thank you", "thanks",
    "you're welcome", "okay", "ok", "all right", "alright", "got it", "right",
    "of course", "excuse me", "please", "the end", "hello", "hi", "hey",
    "um", "uh", "hmm", "well", "so", "like", "you know", "that's right", 
    "such as", "i see, i see", "alright i see", "ah i see", "sound good", 
    "oh i see", "heav-ho", "mm-hmm", "uh-huh", "yeah", "yep", "nope", "nah"
}

PRESERVE_SOUNDS = {
    "ah", "oh", "wow", "no", "yes", "stop", "help", "wait", "go", "come",
    "aah", "ooh", "eeh", "kyaa", "waa", "haa", "yaa", "noo", "ahh", "ohh",
    "nya", "uwu", "owo", "ara", "ehe", "ehehe", "hehe", "hihi", "hoho",
    "yay", "yey", "yup", "nope", "mhm", "mmm", "hmm", "huh", "eh",
    "gg", "nice", "good", "bad", "fail", "win", "lose", "dead", "alive",
    "hai", "iie", "sou", "nani", "mou", "demo", "kedo", "desu", "masu"
}

QUALITY_INDICATORS = {
    "repetitive_patterns": [r"(.{1,10})\1{3,}", r"(\w+\s+)\1{2,}"],
    "nonsense_patterns": [r"[a-z]{15,}", r"\b\w{1}\s+\w{1}\s+\w{1}\b"],
    "filler_heavy": [r"\b(um|uh|ah|eh|mm)\b.*\b(um|uh|ah|eh|mm)\b.*\b(um|uh|ah|eh|mm)\b"]
}

def post_process_translation(text):
    """Clean up and improve translation text"""
    text = ' '.join(text.split())
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
    
    # Check exact matches
    if text_clean in EXACT_MATCH_HALLUCINATION_FILTER:
        if text_clean not in PRESERVE_SOUNDS:
            return True
    
    # Check substring matches
    for phrase in SUBSTRING_HALLUCINATION_FILTER:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text_lower):
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
