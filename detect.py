"""
Language Detection Module
==========================
Fast, accurate language detection for user input.
Pure function, no external state, no translation.
"""

import re
from typing import Dict, Any, Optional
from collections import Counter


# ============================================================================
# CONFIGURATION
# ============================================================================

# Supported languages
SUPPORTED_LANGUAGES = ["en", "ar", "es"]

# Default language when detection is uncertain
DEFAULT_LANGUAGE = "en"

# Confidence threshold for reliable detection
CONFIDENCE_THRESHOLD = 0.6


# ============================================================================
# LANGUAGE CHARACTER PATTERNS
# ============================================================================

# Unicode ranges for different scripts
ARABIC_RANGE = range(0x0600, 0x06FF + 1)  # Arabic script
ARABIC_SUPPLEMENT_RANGE = range(0x0750, 0x077F + 1)
ARABIC_EXTENDED_RANGE = range(0x08A0, 0x08FF + 1)

# Common character sets
ARABIC_CHARS = set(chr(i) for i in ARABIC_RANGE) | \
               set(chr(i) for i in ARABIC_SUPPLEMENT_RANGE) | \
               set(chr(i) for i in ARABIC_EXTENDED_RANGE)

SPANISH_SPECIAL_CHARS = set('áéíóúüñ¿¡ÁÉÍÓÚÜÑ')

# English alphabet (no special characters)
ENGLISH_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')


# ============================================================================
# LANGUAGE-SPECIFIC PATTERNS
# ============================================================================

# Common words by language (most frequent)
COMMON_ENGLISH_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can',
    'could', 'should', 'may', 'might', 'must', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'want', 'like', 'order', 'pizza', 'please'
}

COMMON_SPANISH_WORDS = {
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero',
    'en', 'de', 'a', 'por', 'para', 'con', 'sin', 'es', 'son', 'está',
    'están', 'ser', 'estar', 'haber', 'hacer', 'tener', 'poder', 'querer',
    'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas',
    'mi', 'tu', 'su', 'este', 'ese', 'aquel', 'qué', 'cuál', 'quién',
    'dónde', 'cuándo', 'cómo', 'por qué', 'quiero', 'me', 'te', 'le',
    'nos', 'sí', 'no', 'hola', 'gracias', 'por favor', 'pizza'
}

COMMON_ARABIC_WORDS = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
    'أنا', 'أنت', 'هو', 'هي', 'نحن', 'هم', 'ما', 'من', 'متى', 'أين',
    'كيف', 'لماذا', 'نعم', 'لا', 'شكرا', 'أريد', 'و', 'أو', 'لكن',
    'الذي', 'التي', 'اللذان', 'اللتان', 'الذين', 'اللاتي', 'مرحبا'
}

# Language-specific regex patterns
SPANISH_PATTERNS = [
    r'\b(quiero|necesito|me|te|le|nos|gracias|hola|por favor)\b',
    r'ñ',
    r'¿|¡',
    r'\b(el|la|los|las|un|una)\b',
]

ENGLISH_PATTERNS = [
    r'\b(i|want|need|please|thank|you|hello|the|a|an)\b',
    r"'(s|t|re|ve|ll|d|m)",  # English contractions
]

ARABIC_PATTERNS = [
    r'[\u0600-\u06FF]',  # Arabic script
    r'\b(أريد|شكرا|مرحبا|نعم|لا)\b',
]


# ============================================================================
# CORE DETECTION FUNCTION
# ============================================================================

def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect the language of input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with:
        - language: Detected language code ('en', 'ar', 'es')
        - confidence: Confidence score (0.0 to 1.0)
        
    Example:
        >>> result = detect_language("I want a pizza")
        >>> print(result)  # {'language': 'en', 'confidence': 0.95}
        
        >>> result = detect_language("Quiero una pizza")
        >>> print(result)  # {'language': 'es', 'confidence': 0.92}
        
        >>> result = detect_language("أريد بيتزا")
        >>> print(result)  # {'language': 'ar', 'confidence': 0.98}
    """
    # Validate input
    if not text or not isinstance(text, str):
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0
        }
    
    # Clean and normalize text
    text_clean = text.strip()
    
    if len(text_clean) == 0:
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0
        }
    
    # Calculate scores for each language
    scores = {
        "en": _score_english(text_clean),
        "es": _score_spanish(text_clean),
        "ar": _score_arabic(text_clean)
    }
    
    # Find highest scoring language
    detected_language = max(scores, key=scores.get)
    confidence = scores[detected_language]
    
    # If confidence is too low, default to English
    if confidence < CONFIDENCE_THRESHOLD:
        detected_language = DEFAULT_LANGUAGE
    
    # Normalize confidence to 0-1 range
    confidence = min(1.0, max(0.0, confidence))
    
    return {
        "language": detected_language,
        "confidence": round(confidence, 2)
    }


# ============================================================================
# LANGUAGE-SPECIFIC SCORING
# ============================================================================

def _score_english(text: str) -> float:
    """
    Calculate English language score.
    
    Args:
        text: Text to analyze
        
    Returns:
        Score (0.0 to 1.0+)
    """
    score = 0.0
    text_lower = text.lower()
    words = text_lower.split()
    
    # Character analysis
    char_score = _analyze_characters(text, ENGLISH_CHARS)
    score += char_score * 0.3
    
    # Common word matching
    if words:
        common_word_count = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
        word_ratio = common_word_count / len(words)
        score += word_ratio * 0.5
    
    # Pattern matching
    pattern_score = 0
    for pattern in ENGLISH_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            pattern_score += 0.1
    score += min(pattern_score, 0.2)
    
    # No Spanish/Arabic special characters
    if not any(char in text for char in SPANISH_SPECIAL_CHARS):
        if not any(char in text for char in ARABIC_CHARS):
            score += 0.2
    
    return score


def _score_spanish(text: str) -> float:
    """
    Calculate Spanish language score.
    
    Args:
        text: Text to analyze
        
    Returns:
        Score (0.0 to 1.0+)
    """
    score = 0.0
    text_lower = text.lower()
    words = text_lower.split()
    
    # Spanish special characters (strong indicator)
    special_char_count = sum(1 for char in text if char in SPANISH_SPECIAL_CHARS)
    if special_char_count > 0:
        score += min(special_char_count * 0.15, 0.4)
    
    # Common word matching
    if words:
        common_word_count = sum(1 for word in words if word in COMMON_SPANISH_WORDS)
        word_ratio = common_word_count / len(words)
        score += word_ratio * 0.5
    
    # Pattern matching
    pattern_score = 0
    for pattern in SPANISH_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            pattern_score += 0.1
    score += min(pattern_score, 0.3)
    
    # No Arabic characters
    if not any(char in text for char in ARABIC_CHARS):
        score += 0.1
    
    return score


def _score_arabic(text: str) -> float:
    """
    Calculate Arabic language score.
    
    Args:
        text: Text to analyze
        
    Returns:
        Score (0.0 to 1.0+)
    """
    score = 0.0
    words = text.split()
    
    # Arabic characters (very strong indicator)
    arabic_char_count = sum(1 for char in text if char in ARABIC_CHARS)
    total_chars = len([c for c in text if not c.isspace()])
    
    if total_chars > 0:
        arabic_ratio = arabic_char_count / total_chars
        score += arabic_ratio * 0.7  # Heavy weight for Arabic script
    
    # Common word matching
    if words:
        common_word_count = sum(1 for word in words if word in COMMON_ARABIC_WORDS)
        word_ratio = common_word_count / len(words)
        score += word_ratio * 0.3
    
    # Pattern matching
    for pattern in ARABIC_PATTERNS:
        if re.search(pattern, text):
            score += 0.1
    
    # If any Arabic characters present, boost confidence
    if arabic_char_count > 0:
        score += 0.2
    
    return score


def _analyze_characters(text: str, target_chars: set) -> float:
    """
    Analyze character composition.
    
    Args:
        text: Text to analyze
        target_chars: Set of target characters
        
    Returns:
        Ratio of target characters (0.0 to 1.0)
    """
    # Count alphabetic characters only
    alphabetic_chars = [c for c in text if c.isalpha()]
    
    if not alphabetic_chars:
        return 0.0
    
    # Count how many are in target set
    target_count = sum(1 for c in alphabetic_chars if c in target_chars)
    
    return target_count / len(alphabetic_chars)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_language_supported(language_code: str) -> bool:
    """
    Check if a language is supported.
    
    Args:
        language_code: Language code to check
        
    Returns:
        True if supported
        
    Example:
        >>> is_language_supported("en")  # True
        >>> is_language_supported("fr")  # False
    """
    return language_code in SUPPORTED_LANGUAGES


def get_supported_languages() -> list[str]:
    """
    Get list of supported language codes.
    
    Returns:
        List of language codes
        
    Example:
        >>> langs = get_supported_languages()
        >>> print(langs)  # ['en', 'ar', 'es']
    """
    return SUPPORTED_LANGUAGES.copy()


def detect_with_fallback(
    text: str,
    fallback_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect language with custom fallback.
    
    Args:
        text: Text to analyze
        fallback_language: Language to use if confidence is low (default: 'en')
        
    Returns:
        Detection result with applied fallback
        
    Example:
        >>> result = detect_with_fallback("hmm...", fallback_language="es")
    """
    result = detect_language(text)
    
    # Apply fallback if confidence is too low
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        result["language"] = fallback_language or DEFAULT_LANGUAGE
        result["fallback_applied"] = True
    else:
        result["fallback_applied"] = False
    
    return result


def batch_detect(texts: list[str]) -> list[Dict[str, Any]]:
    """
    Detect language for multiple texts.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        List of detection results
        
    Example:
        >>> texts = ["Hello", "Hola", "مرحبا"]
        >>> results = batch_detect(texts)
    """
    return [detect_language(text) for text in texts]


def get_confidence_level(confidence: float) -> str:
    """
    Convert confidence score to human-readable level.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Confidence level string
        
    Example:
        >>> level = get_confidence_level(0.95)
        >>> print(level)  # "high"
    """
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    elif confidence >= 0.3:
        return "low"
    else:
        return "very_low"


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def analyze_text_details(text: str) -> Dict[str, Any]:
    """
    Detailed analysis of text for debugging.
    
    Args:
        text: Text to analyze
        
    Returns:
        Detailed breakdown of language indicators
        
    Example:
        >>> details = analyze_text_details("Quiero una pizza")
        >>> print(details)
    """
    # Basic detection
    result = detect_language(text)
    
    # Character analysis
    total_chars = len([c for c in text if not c.isspace()])
    arabic_chars = sum(1 for c in text if c in ARABIC_CHARS)
    spanish_chars = sum(1 for c in text if c in SPANISH_SPECIAL_CHARS)
    english_chars = sum(1 for c in text if c in ENGLISH_CHARS)
    
    # Word analysis
    words = text.lower().split()
    english_words = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    spanish_words = sum(1 for w in words if w in COMMON_SPANISH_WORDS)
    arabic_words = sum(1 for w in words if w in COMMON_ARABIC_WORDS)
    
    # Pattern matching
    has_spanish_patterns = any(re.search(p, text, re.IGNORECASE) for p in SPANISH_PATTERNS)
    has_english_patterns = any(re.search(p, text, re.IGNORECASE) for p in ENGLISH_PATTERNS)
    has_arabic_patterns = any(re.search(p, text) for p in ARABIC_PATTERNS)
    
    return {
        "detected_language": result["language"],
        "confidence": result["confidence"],
        "confidence_level": get_confidence_level(result["confidence"]),
        "text_length": len(text),
        "word_count": len(words),
        "character_analysis": {
            "total_chars": total_chars,
            "arabic_chars": arabic_chars,
            "spanish_special_chars": spanish_chars,
            "english_chars": english_chars
        },
        "word_analysis": {
            "english_common_words": english_words,
            "spanish_common_words": spanish_words,
            "arabic_common_words": arabic_words
        },
        "pattern_matching": {
            "spanish_patterns": has_spanish_patterns,
            "english_patterns": has_english_patterns,
            "arabic_patterns": has_arabic_patterns
        },
        "scores": {
            "english": _score_english(text),
            "spanish": _score_spanish(text),
            "arabic": _score_arabic(text)
        }
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the language detection module.
    """
    
    print("Language Detection Module Example")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "I want to order a large pizza please",
        "Quiero ordenar una pizza grande por favor",
        "أريد طلب بيتزا كبيرة من فضلك",
        "Hello, how are you?",
        "Hola, ¿cómo estás?",
        "مرحبا، كيف حالك؟",
        "Can I get two burgers?",
        "¿Puedo obtener dos hamburguesas?",
        "هل يمكنني الحصول على اثنين من البرغر؟",
        "pizza",  # Ambiguous
        "123 456",  # Numbers only
        ""  # Empty
    ]
    
    print("\nDetection Results:")
    print("-" * 50)
    
    for i, text in enumerate(test_cases, 1):
        result = detect_language(text)
        level = get_confidence_level(result["confidence"])
        
        print(f"\n{i}. Text: {text if text else '(empty)'}")
        print(f"   Language: {result['language']}")
        print(f"   Confidence: {result['confidence']:.2f} ({level})")
    
    # Detailed analysis example
    print("\n" + "=" * 50)
    print("Detailed Analysis Example")
    print("=" * 50)
    
    sample_text = "Quiero una pizza grande con extra queso"
    details = analyze_text_details(sample_text)
    
    print(f"\nText: {sample_text}")
    print(f"Detected: {details['detected_language']} ({details['confidence_level']} confidence)")
    print(f"\nWord Analysis:")
    print(f"  Total words: {details['word_count']}")
    print(f"  English words: {details['word_analysis']['english_common_words']}")
    print(f"  Spanish words: {details['word_analysis']['spanish_common_words']}")
    print(f"  Arabic words: {details['word_analysis']['arabic_common_words']}")
    
    print(f"\nLanguage Scores:")
    for lang, score in details['scores'].items():
        print(f"  {lang}: {score:.2f}")
    
    # Batch detection
    print("\n" + "=" * 50)
    print("Batch Detection")
    print("=" * 50)
    
    batch_texts = ["Hello", "Hola", "مرحبا"]
    batch_results = batch_detect(batch_texts)
    
    for text, result in zip(batch_texts, batch_results):
        print(f"{text}: {result['language']} ({result['confidence']:.2f})")
