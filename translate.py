"""
Translation Layer
=================
Translate text between user language and internal English reasoning language.
Uses Google Cloud Translation API for high-quality translations.
"""

import os
import asyncio
from typing import Optional, Dict, Any
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions


# ============================================================================
# CONFIGURATION
# ============================================================================

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ar": "Arabic", 
    "es": "Spanish"
}

# Internal reasoning language (always English)
INTERNAL_LANGUAGE = "en"

# Translation cache for common phrases (optional optimization)
_translation_cache: Dict[str, str] = {}
ENABLE_CACHE = True
MAX_CACHE_SIZE = 1000


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

def _get_credentials() -> Optional[service_account.Credentials]:
    """
    Get Google Cloud credentials from environment.
    
    Returns:
        Service account credentials or None (uses default)
        
    Environment variables checked:
    - GOOGLE_APPLICATION_CREDENTIALS (path to JSON key file)
    - GOOGLE_CLOUD_CREDENTIALS_JSON (JSON string)
    """
    # Method 1: Path to credentials file
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        return service_account.Credentials.from_service_account_file(creds_path)
    
    # Method 2: JSON string in environment
    creds_json = os.getenv("GOOGLE_CLOUD_CREDENTIALS_JSON")
    if creds_json:
        import json
        creds_dict = json.loads(creds_json)
        return service_account.Credentials.from_service_account_info(creds_dict)
    
    # Method 3: Use default credentials (for GCP environments)
    return None


def _create_client() -> translate.Client:
    """
    Create Google Cloud Translation client.
    
    Returns:
        Initialized Translation Client
        
    Raises:
        RuntimeError: If credentials cannot be loaded
    """
    try:
        credentials = _get_credentials()
        if credentials:
            return translate.Client(credentials=credentials)
        else:
            # Use application default credentials (works in GCP)
            return translate.Client()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Google Cloud Translation client: {str(e)}. "
            f"Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON"
        )


# Global client instance (lazy-loaded)
_client: Optional[translate.Client] = None


def _get_client() -> translate.Client:
    """Get or create the global Translation client."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


# ============================================================================
# CORE TRANSLATION FUNCTIONS
# ============================================================================

async def to_english(text: str, source_language: str) -> str:
    """
    Translate user input to English for internal reasoning.
    
    Args:
        text: Text to translate
        source_language: Source language code ('en', 'ar', 'es')
        
    Returns:
        Translated text in English
        
    Raises:
        ValueError: If text is empty or language unsupported
        RuntimeError: If translation fails
        
    Example:
        >>> english_text = await to_english("Hola, quiero pizza", "es")
        >>> print(english_text)  # "Hello, I want pizza"
    """
    # Validate input
    if not text or text.strip() == "":
        raise ValueError("text cannot be empty")
    
    if source_language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported source language: {source_language}. "
            f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    
    # If already in English, return as-is
    if source_language == INTERNAL_LANGUAGE:
        return text
    
    # Check cache first
    cache_key = f"{source_language}->en:{text}"
    if ENABLE_CACHE and cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    try:
        # Perform translation
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _get_client().translate(
                text,
                source_language=source_language,
                target_language=INTERNAL_LANGUAGE,
                format_="text"
            )
        )
        
        translated_text = result["translatedText"]
        
        # Cache the result
        if ENABLE_CACHE and len(_translation_cache) < MAX_CACHE_SIZE:
            _translation_cache[cache_key] = translated_text
        
        return translated_text
        
    except google_exceptions.BadRequest as e:
        raise ValueError(f"Invalid translation request: {str(e)}")
    except google_exceptions.GoogleAPIError as e:
        raise RuntimeError(f"Google Cloud API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Translation to English failed: {str(e)}")


async def from_english(text: str, target_language: str) -> str:
    """
    Translate internal English response to user's language.
    
    Args:
        text: Text in English to translate
        target_language: Target language code ('en', 'ar', 'es')
        
    Returns:
        Translated text in target language
        
    Raises:
        ValueError: If text is empty or language unsupported
        RuntimeError: If translation fails
        
    Example:
        >>> spanish_text = await from_english("Hello, I want pizza", "es")
        >>> print(spanish_text)  # "Hola, quiero pizza"
    """
    # Validate input
    if not text or text.strip() == "":
        raise ValueError("text cannot be empty")
    
    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported target language: {target_language}. "
            f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    
    # If target is English, return as-is
    if target_language == INTERNAL_LANGUAGE:
        return text
    
    # Check cache first
    cache_key = f"en->{target_language}:{text}"
    if ENABLE_CACHE and cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    try:
        # Perform translation
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _get_client().translate(
                text,
                source_language=INTERNAL_LANGUAGE,
                target_language=target_language,
                format_="text"
            )
        )
        
        translated_text = result["translatedText"]
        
        # Cache the result
        if ENABLE_CACHE and len(_translation_cache) < MAX_CACHE_SIZE:
            _translation_cache[cache_key] = translated_text
        
        return translated_text
        
    except google_exceptions.BadRequest as e:
        raise ValueError(f"Invalid translation request: {str(e)}")
    except google_exceptions.GoogleAPIError as e:
        raise RuntimeError(f"Google Cloud API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Translation from English failed: {str(e)}")


# ============================================================================
# BIDIRECTIONAL TRANSLATION
# ============================================================================

async def translate_bidirectional(
    text: str,
    from_language: str,
    to_language: str
) -> str:
    """
    Translate between any two supported languages.
    Goes through English if needed (pivot translation).
    
    Args:
        text: Text to translate
        from_language: Source language code
        to_language: Target language code
        
    Returns:
        Translated text
        
    Example:
        >>> result = await translate_bidirectional("Hola", "es", "ar")
    """
    # Validate input
    if not text or text.strip() == "":
        raise ValueError("text cannot be empty")
    
    if from_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported source language: {from_language}")
    
    if to_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported target language: {to_language}")
    
    # If same language, return as-is
    if from_language == to_language:
        return text
    
    # If either is English, use direct translation
    if from_language == INTERNAL_LANGUAGE:
        return await from_english(text, to_language)
    
    if to_language == INTERNAL_LANGUAGE:
        return await to_english(text, from_language)
    
    # Otherwise, pivot through English
    english_text = await to_english(text, from_language)
    final_text = await from_english(english_text, to_language)
    
    return final_text


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

async def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect the language of input text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with:
        - language: Detected language code
        - confidence: Confidence score (0.0 to 1.0)
        - is_supported: Whether language is supported
        
    Example:
        >>> result = await detect_language("Hola mundo")
        >>> print(result["language"])  # "es"
    """
    if not text or text.strip() == "":
        raise ValueError("text cannot be empty")
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _get_client().detect_language(text)
        )
        
        language_code = result["language"]
        confidence = result["confidence"]
        
        return {
            "language": language_code,
            "confidence": confidence,
            "is_supported": language_code in SUPPORTED_LANGUAGES
        }
        
    except Exception as e:
        raise RuntimeError(f"Language detection failed: {str(e)}")


# ============================================================================
# BATCH TRANSLATION
# ============================================================================

async def translate_batch_to_english(
    texts: list[str],
    source_language: str
) -> list[str]:
    """
    Translate multiple texts to English in batch.
    More efficient for multiple translations.
    
    Args:
        texts: List of texts to translate
        source_language: Source language code
        
    Returns:
        List of translated texts
        
    Example:
        >>> texts = ["Hola", "Adiós"]
        >>> results = await translate_batch_to_english(texts, "es")
    """
    if not texts:
        return []
    
    # If already English, return as-is
    if source_language == INTERNAL_LANGUAGE:
        return texts
    
    # Translate each (could be optimized with parallel execution)
    translated = []
    for text in texts:
        result = await to_english(text, source_language)
        translated.append(result)
    
    return translated


async def translate_batch_from_english(
    texts: list[str],
    target_language: str
) -> list[str]:
    """
    Translate multiple texts from English in batch.
    
    Args:
        texts: List of English texts to translate
        target_language: Target language code
        
    Returns:
        List of translated texts
        
    Example:
        >>> texts = ["Hello", "Goodbye"]
        >>> results = await translate_batch_from_english(texts, "es")
    """
    if not texts:
        return []
    
    # If target is English, return as-is
    if target_language == INTERNAL_LANGUAGE:
        return texts
    
    # Translate each
    translated = []
    for text in texts:
        result = await from_english(text, target_language)
        translated.append(result)
    
    return translated


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_supported_languages() -> list[str]:
    """
    Get list of supported language codes.
    
    Returns:
        List of language codes
        
    Example:
        >>> langs = get_supported_languages()
        >>> print(langs)  # ['en', 'ar', 'es']
    """
    return list(SUPPORTED_LANGUAGES.keys())


def get_language_name(language_code: str) -> str:
    """
    Get full language name from code.
    
    Args:
        language_code: Language code
        
    Returns:
        Language name
        
    Example:
        >>> name = get_language_name("es")
        >>> print(name)  # "Spanish"
    """
    return SUPPORTED_LANGUAGES.get(language_code, "Unknown")


def is_language_supported(language_code: str) -> bool:
    """
    Check if a language is supported.
    
    Args:
        language_code: Language code to check
        
    Returns:
        True if supported
        
    Example:
        >>> is_language_supported("es")  # True
        >>> is_language_supported("fr")  # False
    """
    return language_code in SUPPORTED_LANGUAGES


def clear_cache() -> int:
    """
    Clear the translation cache.
    
    Returns:
        Number of entries cleared
        
    Example:
        >>> cleared = clear_cache()
        >>> print(f"Cleared {cleared} cached translations")
    """
    global _translation_cache
    count = len(_translation_cache)
    _translation_cache.clear()
    return count


def get_cache_stats() -> Dict[str, Any]:
    """
    Get translation cache statistics.
    
    Returns:
        Dictionary with cache stats
        
    Example:
        >>> stats = get_cache_stats()
        >>> print(f"Cache size: {stats['size']}")
    """
    return {
        "enabled": ENABLE_CACHE,
        "size": len(_translation_cache),
        "max_size": MAX_CACHE_SIZE,
        "utilization": len(_translation_cache) / MAX_CACHE_SIZE if MAX_CACHE_SIZE > 0 else 0
    }


# ============================================================================
# CONVERSATION FLOW HELPER
# ============================================================================

async def translate_conversation_turn(
    user_text: str,
    user_language: str,
    ai_response: str
) -> Dict[str, str]:
    """
    Complete conversation turn translation workflow.
    
    Args:
        user_text: User input in their language
        user_language: User's language code
        ai_response: AI response in English
        
    Returns:
        Dictionary with:
        - user_input_english: User input translated to English
        - ai_response_translated: AI response in user's language
        
    Example:
        >>> result = await translate_conversation_turn(
        >>>     "Hola, quiero pizza",
        >>>     "es",
        >>>     "Great! What size pizza would you like?"
        >>> )
        >>> print(result["user_input_english"])
        >>> print(result["ai_response_translated"])
    """
    # Translate user input to English for AI processing
    user_input_english = await to_english(user_text, user_language)
    
    # Translate AI response to user's language
    ai_response_translated = await from_english(ai_response, user_language)
    
    return {
        "user_input_english": user_input_english,
        "ai_response_translated": ai_response_translated
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the translation layer.
    """
    
    async def example():
        print("Translation Layer Example")
        print("=" * 50)
        
        # Supported languages
        print("\nSupported Languages:")
        for code, name in SUPPORTED_LANGUAGES.items():
            print(f"  {code}: {name}")
        
        # Example translations
        print("\n" + "=" * 50)
        print("Example Translations")
        print("=" * 50)
        
        # Spanish to English
        spanish_text = "Hola, quiero ordenar una pizza grande"
        print(f"\nSpanish: {spanish_text}")
        
        try:
            english = await to_english(spanish_text, "es")
            print(f"English: {english}")
        except Exception as e:
            print(f"Error: {e}")
        
        # English to Arabic
        english_text = "Your order total is $25.50"
        print(f"\nEnglish: {english_text}")
        
        try:
            arabic = await from_english(english_text, "ar")
            print(f"Arabic: {arabic}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Language detection
        print("\n" + "=" * 50)
        print("Language Detection")
        print("=" * 50)
        
        test_texts = [
            "Hello, how are you?",
            "مرحبا، كيف حالك؟",
            "Hola, ¿cómo estás?"
        ]
        
        for text in test_texts:
            try:
                result = await detect_language(text)
                lang_name = get_language_name(result["language"])
                print(f"\nText: {text}")
                print(f"Detected: {lang_name} ({result['language']})")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Supported: {result['is_supported']}")
            except Exception as e:
                print(f"Error: {e}")
        
        # Cache stats
        print("\n" + "=" * 50)
        print("Cache Statistics")
        print("=" * 50)
        stats = get_cache_stats()
        print(f"Enabled: {stats['enabled']}")
        print(f"Size: {stats['size']}/{stats['max_size']}")
        print(f"Utilization: {stats['utilization']:.1%}")
        
        # Conversation flow example
        print("\n" + "=" * 50)
        print("Conversation Flow Example")
        print("=" * 50)
        
        try:
            conversation = await translate_conversation_turn(
                user_text="Quiero dos pizzas grandes",
                user_language="es",
                ai_response="Great! Would you like any drinks with that?"
            )
            
            print(f"\nUser (Spanish): Quiero dos pizzas grandes")
            print(f"→ Translated to English: {conversation['user_input_english']}")
            print(f"\nAI (English): Great! Would you like any drinks with that?")
            print(f"→ Translated to Spanish: {conversation['ai_response_translated']}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Run example
    asyncio.run(example())
