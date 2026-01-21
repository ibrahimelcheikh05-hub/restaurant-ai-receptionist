"""
Translation Module (Production)
================================
Hardened async translation with Google Cloud Translate.
Strict timeouts, failure-safe fallbacks, graceful degradation.
Never blocks, never crashes a call.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions
import json
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = {
    "en": "en",
    "ar": "ar",
    "es": "es"
}

# Timeouts
TRANSLATION_TIMEOUT = 3.0  # seconds
MAX_RETRIES = 2
RETRY_DELAY = 0.5  # seconds

# Limits
MAX_TEXT_LENGTH = 5000
MAX_CONCURRENT_TRANSLATIONS = 20

# Cache
CACHE_ENABLED = True
CACHE_SIZE = 500
CACHE_TTL = 3600  # 1 hour


class TranslationCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, max_size: int = CACHE_SIZE, ttl: int = CACHE_TTL):
        self.cache: Dict[str, tuple[str, datetime]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[str]:
        """Get cached translation if valid."""
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Check TTL
        if datetime.utcnow() - timestamp > timedelta(seconds=self.ttl):
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: str):
        """Cache translation with TTL."""
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.utcnow())
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


_cache = TranslationCache()
_client: Optional[translate.Client] = None
_translation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSLATIONS)


def _get_credentials():
    """Get Google Cloud credentials."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        return service_account.Credentials.from_service_account_file(creds_path)
    
    creds_json = os.getenv("GOOGLE_CLOUD_CREDENTIALS_JSON")
    if creds_json:
        creds_dict = json.loads(creds_json)
        return service_account.Credentials.from_service_account_info(creds_dict)
    
    return None


def _create_client() -> translate.Client:
    """Create Google Cloud Translate client."""
    credentials = _get_credentials()
    if credentials:
        return translate.Client(credentials=credentials)
    return translate.Client()


def _get_client() -> translate.Client:
    """Get or create Translate client."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


def _get_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    """Generate cache key."""
    import hashlib
    content = f"{source_lang}:{target_lang}:{text}"
    return hashlib.md5(content.encode()).hexdigest()


async def to_english(
    text: str,
    source_language: str,
    fallback_to_original: bool = True
) -> str:
    """
    Translate text to English with timeout and fallback.
    
    Args:
        text: Text to translate
        source_language: Source language code ('ar', 'es', etc.)
        fallback_to_original: Return original text on error (default: True)
        
    Returns:
        Translated text (or original if translation fails)
        
    Example:
        >>> english = await to_english("Hola", "es")
        >>> print(english)  # "Hello"
    """
    # Skip if already English
    if source_language == "en":
        return text
    
    # Validate input
    if not text or not isinstance(text, str):
        return text if fallback_to_original else ""
    
    text_clean = text.strip()
    
    if not text_clean:
        return text if fallback_to_original else ""
    
    # Length check
    if len(text_clean) > MAX_TEXT_LENGTH:
        logger.warning(f"Text too long ({len(text_clean)} chars), truncating")
        text_clean = text_clean[:MAX_TEXT_LENGTH]
    
    # Validate source language
    if source_language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported source language: {source_language}")
        return text if fallback_to_original else text_clean
    
    # Check cache
    if CACHE_ENABLED:
        cache_key = _get_cache_key(text_clean, source_language, "en")
        cached = _cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit: {source_language}→en")
            return cached
    
    # Translate with retry and timeout
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with _translation_semaphore:
                result = await asyncio.wait_for(
                    _translate_text(text_clean, source_language, "en"),
                    timeout=TRANSLATION_TIMEOUT
                )
            
            if result:
                # Cache result
                if CACHE_ENABLED:
                    _cache.set(cache_key, result)
                
                logger.debug(
                    f"Translated {source_language}→en: "
                    f"{text_clean[:50]}... → {result[:50]}..."
                )
                return result
            
            # Empty result, use fallback
            logger.warning(f"Empty translation result for: {text_clean[:50]}...")
            return text if fallback_to_original else text_clean
        
        except asyncio.TimeoutError:
            logger.warning(
                f"Translation timeout (attempt {attempt + 1}/{MAX_RETRIES + 1})"
            )
            
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                continue
            
            # Max retries reached
            logger.error(f"Translation failed after {MAX_RETRIES + 1} attempts")
            return text if fallback_to_original else text_clean
        
        except google_exceptions.GoogleAPICallError as e:
            logger.error(f"Google API error (attempt {attempt + 1}): {str(e)}")
            
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                continue
            
            return text if fallback_to_original else text_clean
        
        except Exception as e:
            logger.error(f"Translation error (attempt {attempt + 1}): {str(e)}")
            
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                continue
            
            return text if fallback_to_original else text_clean
    
    # Should never reach here, but safety fallback
    return text if fallback_to_original else text_clean


async def from_english(
    text: str,
    target_language: str,
    fallback_to_original: bool = True
) -> str:
    """
    Translate text from English with timeout and fallback.
    
    Args:
        text: English text to translate
        target_language: Target language code ('ar', 'es', etc.)
        fallback_to_original: Return original text on error (default: True)
        
    Returns:
        Translated text (or original if translation fails)
        
    Example:
        >>> spanish = await from_english("Hello", "es")
        >>> print(spanish)  # "Hola"
    """
    # Skip if target is English
    if target_language == "en":
        return text
    
    # Validate input
    if not text or not isinstance(text, str):
        return text if fallback_to_original else ""
    
    text_clean = text.strip()
    
    if not text_clean:
        return text if fallback_to_original else ""
    
    # Length check
    if len(text_clean) > MAX_TEXT_LENGTH:
        logger.warning(f"Text too long ({len(text_clean)} chars), truncating")
        text_clean = text_clean[:MAX_TEXT_LENGTH]
    
    # Validate target language
    if target_language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported target language: {target_language}")
        return text if fallback_to_original else text_clean
    
    # Check cache
    if CACHE_ENABLED:
        cache_key = _get_cache_key(text_clean, "en", target_language)
        cached = _cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit: en→{target_language}")
            return cached
    
    # Translate with retry and timeout
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with _translation_semaphore:
                result = await asyncio.wait_for(
                    _translate_text(text_clean, "en", target_language),
                    timeout=TRANSLATION_TIMEOUT
                )
            
            if result:
                # Cache result
                if CACHE_ENABLED:
                    _cache.set(cache_key, result)
                
                logger.debug(
                    f"Translated en→{target_language}: "
                    f"{text_clean[:50]}... → {result[:50]}..."
                )
                return result
            
            # Empty result, use fallback
            logger.warning(f"Empty translation result for: {text_clean[:50]}...")
            return text if fallback_to_original else text_clean
        
        except asyncio.TimeoutError:
            logger.warning(
                f"Translation timeout (attempt {attempt + 1}/{MAX_RETRIES + 1})"
            )
            
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                continue
            
            # Max retries reached
            logger.error(f"Translation failed after {MAX_RETRIES + 1} attempts")
            return text if fallback_to_original else text_clean
        
        except google_exceptions.GoogleAPICallError as e:
            logger.error(f"Google API error (attempt {attempt + 1}): {str(e)}")
            
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                continue
            
            return text if fallback_to_original else text_clean
        
        except Exception as e:
            logger.error(f"Translation error (attempt {attempt + 1}): {str(e)}")
            
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                continue
            
            return text if fallback_to_original else text_clean
    
    # Should never reach here, but safety fallback
    return text if fallback_to_original else text_clean


async def _translate_text(
    text: str,
    source_lang: str,
    target_lang: str
) -> Optional[str]:
    """
    Internal translation function.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translated text or None
    """
    if not text:
        return None
    
    try:
        # Run blocking translate call in executor
        loop = asyncio.get_event_loop()
        client = _get_client()
        
        result = await loop.run_in_executor(
            None,
            lambda: client.translate(
                text,
                target_language=target_lang,
                source_language=source_lang
            )
        )
        
        if result and "translatedText" in result:
            return result["translatedText"]
        
        return None
    
    except Exception as e:
        logger.error(f"Translation API error: {str(e)}")
        return None


def is_translation_needed(source_lang: str, target_lang: str) -> bool:
    """
    Check if translation is needed.
    
    Args:
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        True if translation needed
    """
    return source_lang != target_lang


def get_supported_languages() -> list[str]:
    """Get list of supported languages."""
    return list(SUPPORTED_LANGUAGES.keys())


def is_language_supported(language_code: str) -> bool:
    """Check if language is supported."""
    return language_code in SUPPORTED_LANGUAGES


def clear_cache():
    """Clear translation cache."""
    _cache.clear()
    logger.info("Translation cache cleared")


def get_cache_size() -> int:
    """Get current cache size."""
    return _cache.size()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "size": _cache.size(),
        "max_size": _cache.max_size,
        "ttl_seconds": _cache.ttl
    }


async def batch_translate_to_english(
    texts: list[str],
    source_language: str,
    fallback_to_original: bool = True
) -> list[str]:
    """
    Translate multiple texts to English concurrently.
    
    Args:
        texts: List of texts to translate
        source_language: Source language
        fallback_to_original: Return original on error
        
    Returns:
        List of translated texts
    """
    if not texts:
        return []
    
    tasks = [
        to_english(text, source_language, fallback_to_original)
        for text in texts
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions in results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch translation error for text {i}: {str(result)}")
            final_results.append(texts[i] if fallback_to_original else "")
        else:
            final_results.append(result)
    
    return final_results


async def batch_translate_from_english(
    texts: list[str],
    target_language: str,
    fallback_to_original: bool = True
) -> list[str]:
    """
    Translate multiple texts from English concurrently.
    
    Args:
        texts: List of English texts to translate
        target_language: Target language
        fallback_to_original: Return original on error
        
    Returns:
        List of translated texts
    """
    if not texts:
        return []
    
    tasks = [
        from_english(text, target_language, fallback_to_original)
        for text in texts
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions in results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch translation error for text {i}: {str(result)}")
            final_results.append(texts[i] if fallback_to_original else "")
        else:
            final_results.append(result)
    
    return final_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Translation Module (Production)")
        print("=" * 50)
        
        print("\nSupported languages:", ", ".join(get_supported_languages()))
        
        print("\nTest translations:")
        print("-" * 50)
        
        # English to Spanish
        spanish = await from_english("Hello, how are you?", "es")
        print(f"EN→ES: Hello, how are you? → {spanish}")
        
        # Spanish to English
        english = await to_english("Hola, ¿cómo estás?", "es")
        print(f"ES→EN: Hola, ¿cómo estás? → {english}")
        
        # Arabic to English
        english = await to_english("مرحبا، كيف حالك؟", "ar")
        print(f"AR→EN: مرحبا، كيف حالك؟ → {english}")
        
        print("\n" + "=" * 50)
        print("Cache stats:", get_cache_stats())
        
        print("\nBatch translation test:")
        texts = ["Hello", "Goodbye", "Thank you"]
        results = await batch_translate_from_english(texts, "es")
        for eng, spa in zip(texts, results):
            print(f"  {eng} → {spa}")
        
        print("\n" + "=" * 50)
        print("Production translation ready")
    
    asyncio.run(example())
