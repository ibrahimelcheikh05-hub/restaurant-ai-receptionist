"""
Translation Module (Production Hardened)
=========================================
Isolated translation that cannot destabilize calls.

HARDENING UPDATES (v3.0):
✅ All translation is optional (can be disabled per call)
✅ Strict timeouts with guaranteed completion
✅ Bypass logic if translation fails
✅ Caching for repeated phrases
✅ Translation never blocks call loop
✅ NO fatal exceptions raised
✅ Graceful degradation

Version: 3.0.0 (Production Hardened)
Last Updated: 2026-01-22
"""

import os
import asyncio
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
from collections import defaultdict

try:
    from google.cloud import translate_v2 as translate
    from google.api_core.exceptions import GoogleAPIError
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    translate = None

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "en,ar,es").split(",")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

# Timeout configuration (STRICT)
TRANSLATION_TIMEOUT = float(os.getenv("TRANSLATION_TIMEOUT", "2.0"))  # 2s max
CACHE_LOOKUP_TIMEOUT = float(os.getenv("CACHE_LOOKUP_TIMEOUT", "0.1"))  # 100ms max

# Cache configuration
CACHE_TTL_SECONDS = int(os.getenv("TRANSLATION_CACHE_TTL", "3600"))
CACHE_MAX_SIZE = int(os.getenv("TRANSLATION_CACHE_SIZE", "1000"))

# Optional translation (can be disabled globally or per-call)
TRANSLATION_ENABLED = os.getenv("TRANSLATION_ENABLED", "true").lower() == "true"
BYPASS_ON_FAILURE = os.getenv("TRANSLATION_BYPASS_ON_FAILURE", "true").lower() == "true"

# Cost tracking
COST_PER_MILLION_CHARS = 20.0  # $20 per 1M characters


# ============================================================================
# METRICS
# ============================================================================

if METRICS_ENABLED:
    translation_requests_total = Counter(
        'translation_requests_total',
        'Total translation requests',
        ['source_lang', 'target_lang', 'result']
    )
    translation_cache_hits = Counter(
        'translation_cache_hits_total',
        'Translation cache hits'
    )
    translation_cache_misses = Counter(
        'translation_cache_misses_total',
        'Translation cache misses'
    )
    translation_duration = Histogram(
        'translation_duration_seconds',
        'Translation duration'
    )
    translation_timeouts = Counter(
        'translation_timeouts_total',
        'Translation timeout events'
    )
    translation_bypasses = Counter(
        'translation_bypasses_total',
        'Translation bypass events',
        ['reason']
    )
    translation_errors = Counter(
        'translation_errors_total',
        'Translation errors (non-fatal)',
        ['error_type']
    )
    translation_cost = Counter(
        'translation_cost_dollars',
        'Estimated translation cost'
    )


# ============================================================================
# ENUMS
# ============================================================================

class TranslationResult(Enum):
    """Translation operation result."""
    SUCCESS = "success"
    CACHED = "cached"
    BYPASSED = "bypassed"
    TIMEOUT = "timeout"
    ERROR = "error"
    DISABLED = "disabled"


class BypassReason(Enum):
    """Reason for bypassing translation."""
    DISABLED_GLOBALLY = "disabled_globally"
    DISABLED_PER_CALL = "disabled_per_call"
    SAME_LANGUAGE = "same_language"
    EMPTY_TEXT = "empty_text"
    TIMEOUT = "timeout"
    ERROR = "error"
    LIBRARY_UNAVAILABLE = "library_unavailable"


# ============================================================================
# TRANSLATION RESPONSE
# ============================================================================

@dataclass
class TranslationResponse:
    """
    Translation response (never raises exceptions).
    
    Always returns a response - either translated text or original.
    """
    text: str
    result: TranslationResult
    bypass_reason: Optional[BypassReason] = None
    cached: bool = False
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        response = {
            "text": self.text,
            "result": self.result.value,
            "cached": self.cached,
            "duration_ms": round(self.duration_ms, 2)
        }
        
        if self.bypass_reason:
            response["bypass_reason"] = self.bypass_reason.value
        
        return response


# ============================================================================
# PHRASE CACHE
# ============================================================================

class TranslationCacheEntry:
    """Cache entry with TTL."""
    
    def __init__(self, translated_text: str, source_lang: str, target_lang: str):
        self.translated_text = translated_text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.created_at = datetime.utcnow()
        self.access_count = 0
        self.last_accessed = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > CACHE_TTL_SECONDS
    
    def access(self):
        """Record access."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class PhraseCache:
    """
    LRU cache for repeated phrases with timeout protection.
    
    Lookups are guaranteed to complete within CACHE_LOOKUP_TIMEOUT.
    """
    
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self.max_size = max_size
        self.cache: Dict[str, TranslationCacheEntry] = {}
        self.access_order = []
        self._lock = asyncio.Lock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.timeouts = 0
    
    def _generate_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key."""
        content = f"{text}:{source_lang}:{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """
        Get cached translation with timeout.
        
        Guaranteed to return within CACHE_LOOKUP_TIMEOUT or None.
        """
        try:
            # Timeout protection
            result = await asyncio.wait_for(
                self._get_impl(text, source_lang, target_lang),
                timeout=CACHE_LOOKUP_TIMEOUT
            )
            return result
        
        except asyncio.TimeoutError:
            self.timeouts += 1
            logger.warning(
                f"Cache lookup timeout: {len(text)} chars "
                f"(took > {CACHE_LOOKUP_TIMEOUT}s)"
            )
            return None
    
    async def _get_impl(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """Internal get implementation."""
        key = self._generate_key(text, source_lang, target_lang)
        
        async with self._lock:
            entry = self.cache.get(key)
            
            if entry:
                if entry.is_expired():
                    del self.cache[key]
                    self.access_order.remove(key)
                    self.evictions += 1
                    self.misses += 1
                    
                    if METRICS_ENABLED:
                        translation_cache_misses.inc()
                    
                    return None
                
                # Cache hit
                entry.access()
                self.hits += 1
                
                # Update LRU
                self.access_order.remove(key)
                self.access_order.append(key)
                
                if METRICS_ENABLED:
                    translation_cache_hits.inc()
                
                return entry.translated_text
            
            # Cache miss
            self.misses += 1
            if METRICS_ENABLED:
                translation_cache_misses.inc()
            
            return None
    
    async def put(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation: str
    ):
        """Put translation in cache (non-blocking)."""
        try:
            await asyncio.wait_for(
                self._put_impl(text, source_lang, target_lang, translation),
                timeout=CACHE_LOOKUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning("Cache put timeout - skipping")
    
    async def _put_impl(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation: str
    ):
        """Internal put implementation."""
        key = self._generate_key(text, source_lang, target_lang)
        
        async with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Store entry
            entry = TranslationCacheEntry(translation, source_lang, target_lang)
            self.cache[key] = entry
            
            # Update LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.pop(0)
        del self.cache[lru_key]
        self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / max(1, total)
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self.evictions,
            "timeouts": self.timeouts
        }
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


# ============================================================================
# TRANSLATION ENGINE (Isolated)
# ============================================================================

class IsolatedTranslationEngine:
    """
    Isolated translation engine that never blocks call loop.
    
    Guarantees:
    - Always returns within timeout
    - Never raises fatal exceptions
    - Gracefully degrades on failure
    """
    
    def __init__(self):
        self.translation_count = 0
        self.total_characters = 0
        self.timeout_count = 0
        self.error_count = 0
        self.bypass_count = 0
        self.start_time = datetime.utcnow()
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        enabled: bool = True
    ) -> TranslationResponse:
        """
        Translate text with isolation guarantees.
        
        NEVER RAISES EXCEPTIONS - always returns TranslationResponse.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            enabled: Whether translation is enabled for this call
            
        Returns:
            TranslationResponse (never None)
        """
        start_time = time.time()
        
        # Pre-flight checks (bypass conditions)
        bypass = self._check_bypass(text, source_lang, target_lang, enabled)
        if bypass:
            return bypass
        
        # Attempt translation with strict timeout
        try:
            result = await asyncio.wait_for(
                self._do_translate(text, source_lang, target_lang),
                timeout=TRANSLATION_TIMEOUT
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if result:
                self.translation_count += 1
                self.total_characters += len(text)
                
                # Track cost
                if METRICS_ENABLED:
                    cost = (len(text) / 1_000_000) * COST_PER_MILLION_CHARS
                    translation_cost.inc(cost)
                
                return TranslationResponse(
                    text=result,
                    result=TranslationResult.SUCCESS,
                    duration_ms=duration_ms
                )
            else:
                # Translation failed - bypass
                return self._bypass(
                    text,
                    BypassReason.ERROR,
                    duration_ms
                )
        
        except asyncio.TimeoutError:
            self.timeout_count += 1
            duration_ms = (time.time() - start_time) * 1000
            
            logger.warning(
                f"Translation timeout: {len(text)} chars "
                f"({duration_ms:.0f}ms > {TRANSLATION_TIMEOUT*1000}ms)"
            )
            
            if METRICS_ENABLED:
                translation_timeouts.inc()
            
            return self._bypass(
                text,
                BypassReason.TIMEOUT,
                duration_ms
            )
        
        except Exception as e:
            # CRITICAL: Catch ALL exceptions (never propagate)
            self.error_count += 1
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(
                f"Translation error (non-fatal): {type(e).__name__} - {str(e)}"
            )
            
            if METRICS_ENABLED:
                translation_errors.labels(error_type=type(e).__name__).inc()
            
            return self._bypass(
                text,
                BypassReason.ERROR,
                duration_ms
            )
    
    def _check_bypass(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        enabled: bool
    ) -> Optional[TranslationResponse]:
        """
        Check if translation should be bypassed.
        
        Returns TranslationResponse if bypassed, None otherwise.
        """
        # Check if globally disabled
        if not TRANSLATION_ENABLED:
            return self._bypass(text, BypassReason.DISABLED_GLOBALLY)
        
        # Check if disabled for this call
        if not enabled:
            return self._bypass(text, BypassReason.DISABLED_PER_CALL)
        
        # Check empty text
        if not text or not text.strip():
            return self._bypass(text, BypassReason.EMPTY_TEXT)
        
        # Check same language
        if source_lang == target_lang:
            return self._bypass(text, BypassReason.SAME_LANGUAGE)
        
        # Check library availability
        if not GOOGLE_AVAILABLE:
            logger.warning("Translation library not available - bypassing")
            return self._bypass(text, BypassReason.LIBRARY_UNAVAILABLE)
        
        return None  # No bypass
    
    def _bypass(
        self,
        original_text: str,
        reason: BypassReason,
        duration_ms: float = 0.0
    ) -> TranslationResponse:
        """
        Bypass translation and return original text.
        
        This is the SAFE fallback path.
        """
        self.bypass_count += 1
        
        if METRICS_ENABLED:
            translation_bypasses.labels(reason=reason.value).inc()
            translation_requests_total.labels(
                source_lang='unknown',
                target_lang='unknown',
                result='bypassed'
            ).inc()
        
        return TranslationResponse(
            text=original_text,
            result=TranslationResult.BYPASSED,
            bypass_reason=reason,
            duration_ms=duration_ms
        )
    
    async def _do_translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """
        Perform actual translation (isolated).
        
        Returns None on failure (never raises).
        """
        try:
            client = translate.Client()
            
            # Run in executor (sync API)
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                lambda: client.translate(
                    text,
                    source_language=source_lang,
                    target_language=target_lang
                )
            )
            
            translated = result.get("translatedText")
            
            if METRICS_ENABLED:
                translation_requests_total.labels(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    result='success'
                ).inc()
            
            return translated
        
        except GoogleAPIError as e:
            logger.error(f"Google API error (non-fatal): {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Translation error (non-fatal): {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        avg_chars = self.total_characters / max(1, self.translation_count)
        cost = (self.total_characters / 1_000_000) * COST_PER_MILLION_CHARS
        
        return {
            "translation_count": self.translation_count,
            "total_characters": self.total_characters,
            "avg_chars": round(avg_chars, 1),
            "timeouts": self.timeout_count,
            "errors": self.error_count,
            "bypasses": self.bypass_count,
            "estimated_cost_usd": round(cost, 4),
            "uptime_seconds": round(uptime, 2)
        }


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_cache = PhraseCache()
_engine = IsolatedTranslationEngine()


# ============================================================================
# PUBLIC API (Safe, Non-Blocking)
# ============================================================================

async def to_english(
    text: str,
    source_lang: str,
    enabled: bool = True
) -> TranslationResponse:
    """
    Translate text to English (optional, non-blocking).
    
    GUARANTEES:
    - Always returns within timeout
    - Never raises exceptions
    - Returns original text on failure
    
    Args:
        text: Text to translate
        source_lang: Source language code
        enabled: Whether translation is enabled (default True)
        
    Returns:
        TranslationResponse with translated or original text
    """
    start_time = time.time()
    
    # Check if already English
    if source_lang == "en":
        return TranslationResponse(
            text=text,
            result=TranslationResult.BYPASSED,
            bypass_reason=BypassReason.SAME_LANGUAGE
        )
    
    # Check cache first
    cached = await _cache.get(text, source_lang, "en")
    if cached:
        duration_ms = (time.time() - start_time) * 1000
        return TranslationResponse(
            text=cached,
            result=TranslationResult.CACHED,
            cached=True,
            duration_ms=duration_ms
        )
    
    # Translate (isolated, non-blocking)
    response = await _engine.translate(text, source_lang, "en", enabled)
    
    # Cache successful translations
    if response.result == TranslationResult.SUCCESS:
        await _cache.put(text, source_lang, "en", response.text)
    
    return response


async def from_english(
    text: str,
    target_lang: str,
    enabled: bool = True
) -> TranslationResponse:
    """
    Translate text from English (optional, non-blocking).
    
    GUARANTEES:
    - Always returns within timeout
    - Never raises exceptions
    - Returns original text on failure
    
    Args:
        text: English text
        target_lang: Target language code
        enabled: Whether translation is enabled (default True)
        
    Returns:
        TranslationResponse with translated or original text
    """
    start_time = time.time()
    
    # Check if already target language
    if target_lang == "en":
        return TranslationResponse(
            text=text,
            result=TranslationResult.BYPASSED,
            bypass_reason=BypassReason.SAME_LANGUAGE
        )
    
    # Check cache first
    cached = await _cache.get(text, "en", target_lang)
    if cached:
        duration_ms = (time.time() - start_time) * 1000
        return TranslationResponse(
            text=cached,
            result=TranslationResult.CACHED,
            cached=True,
            duration_ms=duration_ms
        )
    
    # Translate (isolated, non-blocking)
    response = await _engine.translate(text, "en", target_lang, enabled)
    
    # Cache successful translations
    if response.result == TranslationResult.SUCCESS:
        await _cache.put(text, "en", target_lang, response.text)
    
    return response


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    enabled: bool = True
) -> TranslationResponse:
    """
    Translate text between any languages (optional, non-blocking).
    
    GUARANTEES:
    - Always returns within timeout
    - Never raises exceptions
    - Returns original text on failure
    
    Args:
        text: Text to translate
        source_lang: Source language
        target_lang: Target language
        enabled: Whether translation is enabled (default True)
        
    Returns:
        TranslationResponse with translated or original text
    """
    start_time = time.time()
    
    # Check if same language
    if source_lang == target_lang:
        return TranslationResponse(
            text=text,
            result=TranslationResult.BYPASSED,
            bypass_reason=BypassReason.SAME_LANGUAGE
        )
    
    # Check cache first
    cached = await _cache.get(text, source_lang, target_lang)
    if cached:
        duration_ms = (time.time() - start_time) * 1000
        return TranslationResponse(
            text=cached,
            result=TranslationResult.CACHED,
            cached=True,
            duration_ms=duration_ms
        )
    
    # Translate (isolated, non-blocking)
    response = await _engine.translate(text, source_lang, target_lang, enabled)
    
    # Cache successful translations
    if response.result == TranslationResult.SUCCESS:
        await _cache.put(text, source_lang, target_lang, response.text)
    
    return response


async def batch_translate(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    enabled: bool = True
) -> List[TranslationResponse]:
    """
    Translate multiple texts concurrently (optional, non-blocking).
    
    GUARANTEES:
    - Each translation isolated
    - Never blocks on failures
    - Returns all results (some may be bypassed)
    
    Args:
        texts: List of texts to translate
        source_lang: Source language
        target_lang: Target language
        enabled: Whether translation is enabled (default True)
        
    Returns:
        List of TranslationResponse
    """
    tasks = [
        translate_text(text, source_lang, target_lang, enabled)
        for text in texts
    ]
    
    return await asyncio.gather(*tasks)


# ============================================================================
# UTILITIES
# ============================================================================

def is_translation_enabled() -> bool:
    """Check if translation is globally enabled."""
    return TRANSLATION_ENABLED


def get_timeout_config() -> Dict[str, float]:
    """Get timeout configuration."""
    return {
        "translation_timeout_seconds": TRANSLATION_TIMEOUT,
        "cache_lookup_timeout_seconds": CACHE_LOOKUP_TIMEOUT
    }


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _cache.get_stats()


def get_translation_stats() -> Dict[str, Any]:
    """Get translation statistics."""
    return _engine.get_stats()


def clear_cache():
    """Clear translation cache."""
    _cache.clear()


def is_supported_language(lang_code: str) -> bool:
    """Check if language is supported."""
    return lang_code in SUPPORTED_LANGUAGES


def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Translation Module (Production Hardened v3.0)")
        print("="*60)
        print(f"Translation Enabled: {TRANSLATION_ENABLED}")
        print(f"Translation Timeout: {TRANSLATION_TIMEOUT}s")
        print(f"Bypass on Failure: {BYPASS_ON_FAILURE}")
        print("="*60)
        
        # Test 1: Translation to English
        print("\n1. Translation to English:")
        response = await to_english("مرحبا بك", "ar", enabled=True)
        print(f"  Text: {response.text}")
        print(f"  Result: {response.result.value}")
        print(f"  Cached: {response.cached}")
        print(f"  Duration: {response.duration_ms:.1f}ms")
        
        # Test 2: Translation from English
        print("\n2. Translation from English:")
        response = await from_english("Hello", "es", enabled=True)
        print(f"  Text: {response.text}")
        print(f"  Result: {response.result.value}")
        
        # Test 3: Disabled translation (bypass)
        print("\n3. Disabled translation:")
        response = await to_english("Hello", "en", enabled=False)
        print(f"  Text: {response.text}")
        print(f"  Result: {response.result.value}")
        print(f"  Bypass Reason: {response.bypass_reason.value if response.bypass_reason else 'N/A'}")
        
        # Test 4: Batch translation
        print("\n4. Batch translation:")
        responses = await batch_translate(
            ["Hello", "Goodbye", "Thank you"],
            "en",
            "es",
            enabled=True
        )
        for i, resp in enumerate(responses):
            print(f"  [{i+1}] {resp.text} (result={resp.result.value})")
        
        # Stats
        print("\n5. Cache statistics:")
        cache_stats = get_cache_stats()
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
        
        print("\n6. Translation statistics:")
        trans_stats = get_translation_stats()
        for key, value in trans_stats.items():
            print(f"  {key}: {value}")
    
    asyncio.run(example())
