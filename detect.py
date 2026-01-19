"""
Language Detection Module
==========================
Fast, accurate language detection for user input.
Pure function, no external state, no translation.
"""

from typing import Dict, Any, Optional, List
from langdetect import detect_langs, LangDetectException


# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_LANGUAGES = ["en", "ar", "es"]
DEFAULT_LANGUAGE = "en"
CONFIDENCE_THRESHOLD = 0.6


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
        >>> print(result)
        {'language': 'en', 'confidence': 0.95}
        
        >>> result = detect_language("Quiero una pizza")
        >>> print(result)
        {'language': 'es', 'confidence': 0.92}
        
        >>> result = detect_language("أريد بيتزا")
        >>> print(result)
        {'language': 'ar', 'confidence': 0.98}
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
    
    # Need minimum text for reliable detection
    if len(text_clean) < 3:
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.3
        }
    
    try:
        # Detect using langdetect
        results = detect_langs(text_clean)
        
        if not results:
            return {
                "language": DEFAULT_LANGUAGE,
                "confidence": 0.0
            }
        
        # Get top result
        top_result = results[0]
        detected_code = top_result.lang
        confidence = float(top_result.prob)
        
        # Map to supported languages
        if detected_code in SUPPORTED_LANGUAGES:
            final_language = detected_code
        else:
            # Unsupported language detected, default to English
            final_language = DEFAULT_LANGUAGE
            confidence = max(0.0, confidence - 0.3)
        
        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD and final_language != DEFAULT_LANGUAGE:
            final_language = DEFAULT_LANGUAGE
        
        return {
            "language": final_language,
            "confidence": round(confidence, 2)
        }
        
    except LangDetectException:
        # Detection failed (too short, etc.)
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0
        }
    except Exception:
        # Unexpected error - fail gracefully
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0
        }


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
        >>> is_language_supported("en")
        True
        >>> is_language_supported("fr")
        False
    """
    return language_code in SUPPORTED_LANGUAGES


def get_supported_languages() -> List[str]:
    """
    Get list of supported language codes.
    
    Returns:
        List of language codes
        
    Example:
        >>> langs = get_supported_languages()
        >>> print(langs)
        ['en', 'ar', 'es']
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


def batch_detect(texts: List[str]) -> List[Dict[str, Any]]:
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
        >>> print(level)
        'high'
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
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the language detection module.
    """
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
    
    # Batch detection
    print("\n" + "=" * 50)
    print("Batch Detection")
    print("=" * 50)
    
    batch_texts = ["Hello", "Hola", "مرحبا"]
    batch_results = batch_detect(batch_texts)
    
    for text, result in zip(batch_texts, batch_results):
        print(f"{text}: {result['language']} ({result['confidence']:.2f})")
    
    print("\n" + "=" * 50)
    print("Supported Languages:", ", ".join(get_supported_languages()))
