"""
Language Detection Module (Production)
=======================================
Fast, deterministic language detection with call-level locking.
Only processes final transcripts, never re-detects once locked.
"""

from typing import Dict, Any, Optional, List
from langdetect import detect_langs, LangDetectException
import logging


logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = ["en", "ar", "es"]
DEFAULT_LANGUAGE = "en"
CONFIDENCE_THRESHOLD = 0.7
MIN_TEXT_LENGTH = 5


_call_language_lock: Dict[str, str] = {}


def detect_language(
    text: str,
    call_id: Optional[str] = None,
    is_final: bool = True
) -> Dict[str, Any]:
    """
    Detect language from text with call-level locking.
    
    Args:
        text: Input text to analyze
        call_id: Call identifier for locking (optional)
        is_final: Whether this is a final transcript (default: True)
        
    Returns:
        Dictionary with:
        - language: Detected language code ('en', 'ar', 'es')
        - confidence: Confidence score (0.0 to 1.0)
        - locked: Whether language is locked for this call
        
    Example:
        >>> result = detect_language("I want a pizza", "call_123", is_final=True)
        >>> print(result)
        {'language': 'en', 'confidence': 0.95, 'locked': True}
    """
    # Return locked language if exists
    if call_id and call_id in _call_language_lock:
        locked_lang = _call_language_lock[call_id]
        logger.debug(f"Language locked for {call_id}: {locked_lang}")
        return {
            "language": locked_lang,
            "confidence": 1.0,
            "locked": True
        }
    
    # Only detect on final transcripts
    if not is_final:
        logger.debug("Skipping detection on partial transcript")
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0,
            "locked": False
        }
    
    # Validate input
    if not text or not isinstance(text, str):
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0,
            "locked": False
        }
    
    # Clean text
    text_clean = text.strip()
    
    if len(text_clean) == 0:
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0,
            "locked": False
        }
    
    # Minimum length check
    if len(text_clean) < MIN_TEXT_LENGTH:
        logger.debug(f"Text too short for detection: {len(text_clean)} chars")
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.3,
            "locked": False
        }
    
    try:
        # Detect using langdetect
        results = detect_langs(text_clean)
        
        if not results:
            return {
                "language": DEFAULT_LANGUAGE,
                "confidence": 0.0,
                "locked": False
            }
        
        # Get top result
        top_result = results[0]
        detected_code = top_result.lang
        confidence = float(top_result.prob)
        
        # Map to supported languages
        if detected_code in SUPPORTED_LANGUAGES:
            final_language = detected_code
        else:
            # Unsupported language, use default
            final_language = DEFAULT_LANGUAGE
            confidence = max(0.0, confidence - 0.3)
        
        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Low confidence ({confidence:.2f}), using default language"
            )
            final_language = DEFAULT_LANGUAGE
        
        # Lock language for call if confidence is high enough
        locked = False
        if call_id and confidence >= CONFIDENCE_THRESHOLD:
            _call_language_lock[call_id] = final_language
            locked = True
            logger.info(
                f"Language locked for {call_id}: {final_language} "
                f"(confidence: {confidence:.2f})"
            )
        
        return {
            "language": final_language,
            "confidence": round(confidence, 2),
            "locked": locked
        }
        
    except LangDetectException as e:
        logger.debug(f"LangDetect exception: {str(e)}")
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0,
            "locked": False
        }
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {str(e)}")
        return {
            "language": DEFAULT_LANGUAGE,
            "confidence": 0.0,
            "locked": False
        }


def get_locked_language(call_id: str) -> Optional[str]:
    """
    Get locked language for a call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Locked language code or None
        
    Example:
        >>> lang = get_locked_language("call_123")
        >>> print(lang)  # "en" or None
    """
    return _call_language_lock.get(call_id)


def is_language_locked(call_id: str) -> bool:
    """
    Check if language is locked for a call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        True if locked, False otherwise
        
    Example:
        >>> if is_language_locked("call_123"):
        >>>     print("Language is locked")
    """
    return call_id in _call_language_lock


def unlock_language(call_id: str):
    """
    Unlock language for a call (call cleanup).
    
    Args:
        call_id: Call identifier
        
    Example:
        >>> unlock_language("call_123")
    """
    if call_id in _call_language_lock:
        lang = _call_language_lock.pop(call_id)
        logger.info(f"Language unlocked for {call_id}: {lang}")


def lock_language(call_id: str, language_code: str) -> bool:
    """
    Manually lock language for a call.
    
    Args:
        call_id: Call identifier
        language_code: Language code to lock
        
    Returns:
        True if locked successfully, False if invalid language
        
    Example:
        >>> lock_language("call_123", "es")
        True
    """
    if language_code not in SUPPORTED_LANGUAGES:
        logger.warning(f"Cannot lock unsupported language: {language_code}")
        return False
    
    _call_language_lock[call_id] = language_code
    logger.info(f"Language manually locked for {call_id}: {language_code}")
    return True


def clear_all_locks():
    """
    Clear all language locks (for testing/cleanup).
    
    Example:
        >>> clear_all_locks()
    """
    count = len(_call_language_lock)
    _call_language_lock.clear()
    logger.info(f"Cleared {count} language locks")


def get_locked_calls() -> List[str]:
    """
    Get list of calls with locked languages.
    
    Returns:
        List of call IDs
        
    Example:
        >>> calls = get_locked_calls()
        >>> print(calls)  # ["call_123", "call_456"]
    """
    return list(_call_language_lock.keys())


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


def get_supported_languages() -> List[str]:
    """
    Get list of supported language codes.
    
    Returns:
        List of language codes
        
    Example:
        >>> langs = get_supported_languages()
        >>> print(langs)  # ['en', 'ar', 'es']
    """
    return SUPPORTED_LANGUAGES.copy()


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Language Detection Module (Production)")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("I want to order a large pizza please", "call_001", True),
        ("Quiero ordenar una pizza grande por favor", "call_002", True),
        ("أريد طلب بيتزا كبيرة من فضلك", "call_003", True),
        ("Hello", "call_004", True),
        ("This is partial", "call_001", False),  # Partial, should skip
        ("Another phrase", "call_001", True),  # Locked, should return EN
    ]
    
    print("\nDetection Tests:")
    print("-" * 50)
    
    for text, call_id, is_final in test_cases:
        result = detect_language(text, call_id, is_final)
        level = get_confidence_level(result["confidence"])
        
        print(f"\nText: {text}")
        print(f"Call: {call_id}, Final: {is_final}")
        print(f"Result: {result['language']} (confidence: {result['confidence']:.2f}, {level})")
        print(f"Locked: {result['locked']}")
    
    print("\n" + "=" * 50)
    print("Locked Languages:")
    for call_id in get_locked_calls():
        lang = get_locked_language(call_id)
        print(f"  {call_id}: {lang}")
    
    print("\n" + "=" * 50)
    print("Supported Languages:", ", ".join(get_supported_languages()))
    
    # Cleanup
    clear_all_locks()
    print("\nAll locks cleared")
