"""
Language Detection Module (Production Hardened)
================================================
Hardened language detection for real-time voice calls.

HARDENING UPDATES (v3.0):
✅ Language locking (detect only once per call)
✅ Stored in session (immutable after first detection)
✅ Confidence thresholds enforced
✅ Fallback behavior for low confidence
✅ Prevents mid-call language switching
✅ NO detection on partial transcripts
✅ NO unlock/override mid-call

Version: 3.0.0 (Production Hardened)
Last Updated: 2026-01-22
"""

import os
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

try:
    from langdetect import detect_langs, LangDetectException
    from langdetect import DetectorFactory
    LANGDETECT_AVAILABLE = True
    # Set seed for consistent results
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False

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

# Confidence thresholds
MIN_CONFIDENCE = float(os.getenv("MIN_LANGUAGE_CONFIDENCE", "0.7"))  # 70%
HIGH_CONFIDENCE = float(os.getenv("HIGH_LANGUAGE_CONFIDENCE", "0.9"))  # 90%

# Detection requirements
MIN_TEXT_LENGTH = int(os.getenv("MIN_DETECTION_TEXT_LENGTH", "20"))  # Min chars
REQUIRE_FINAL_TRANSCRIPT = os.getenv("REQUIRE_FINAL_TRANSCRIPT", "true").lower() == "true"


# ============================================================================
# METRICS
# ============================================================================

if METRICS_ENABLED:
    detection_requests_total = Counter(
        'language_detection_requests_total',
        'Total language detection requests',
        ['result']
    )
    detection_by_language = Counter(
        'language_detections_by_language',
        'Language detections by detected language',
        ['language']
    )
    detection_confidence = Histogram(
        'language_detection_confidence',
        'Language detection confidence scores'
    )
    detection_duration = Histogram(
        'language_detection_duration_seconds',
        'Language detection duration'
    )
    detection_locked_calls = Gauge(
        'language_locked_calls',
        'Number of calls with locked languages'
    )
    detection_errors = Counter(
        'language_detection_errors_total',
        'Language detection errors',
        ['error_type']
    )
    detection_fallbacks = Counter(
        'language_detection_fallbacks_total',
        'Fallback to default language',
        ['reason']
    )
    detection_rejected_partial = Counter(
        'language_detection_rejected_partial_total',
        'Rejections due to partial transcript'
    )
    detection_switch_attempts = Counter(
        'language_detection_switch_attempts_total',
        'Blocked mid-call language switch attempts'
    )


# ============================================================================
# ENUMS
# ============================================================================

class DetectionStatus(Enum):
    """Language detection status."""
    SUCCESS = "success"
    FALLBACK = "fallback"
    LOCKED = "locked"
    REJECTED = "rejected"
    ERROR = "error"


class FallbackReason(Enum):
    """Reason for fallback to default language."""
    TEXT_TOO_SHORT = "text_too_short"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    LIBRARY_UNAVAILABLE = "library_unavailable"
    DETECTION_ERROR = "detection_error"
    PARTIAL_TRANSCRIPT = "partial_transcript"


# ============================================================================
# LANGUAGE LOCK (Immutable Session)
# ============================================================================

@dataclass(frozen=True)
class LanguageLock:
    """
    Immutable language lock for a call session.
    
    CRITICAL: frozen=True means language CANNOT be changed after creation.
    This prevents mid-call language switching.
    """
    call_id: str
    language: str
    confidence: float
    locked_at: datetime
    detection_method: str  # "detected" or "fallback"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "locked_at": self.locked_at.isoformat(),
            "detection_method": self.detection_method
        }


# ============================================================================
# DETECTION RESULT
# ============================================================================

@dataclass
class DetectionResult:
    """Language detection result."""
    language: str
    confidence: float
    status: DetectionStatus
    locked: bool
    detection_method: str
    fallback_reason: Optional[FallbackReason] = None
    alternatives: List[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
            "locked": self.locked,
            "detection_method": self.detection_method
        }
        
        if self.fallback_reason:
            result["fallback_reason"] = self.fallback_reason.value
        
        if self.alternatives:
            result["alternatives"] = self.alternatives
        
        return result


# ============================================================================
# DETECTION STATISTICS
# ============================================================================

class DetectionStats:
    """Track detection statistics."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_detections = 0
        self.fallbacks = 0
        self.rejected_partials = 0
        self.switch_attempts_blocked = 0
        self.language_distribution = defaultdict(int)
        self.fallback_reasons = defaultdict(int)
        self.confidence_scores = []
        self.start_time = datetime.utcnow()
    
    def record_detection(
        self,
        status: DetectionStatus,
        language: str,
        confidence: float = 0.0,
        fallback_reason: Optional[FallbackReason] = None
    ):
        """Record detection result."""
        self.total_requests += 1
        
        if status == DetectionStatus.SUCCESS:
            self.successful_detections += 1
            self.language_distribution[language] += 1
            self.confidence_scores.append(confidence)
        
        elif status == DetectionStatus.FALLBACK:
            self.fallbacks += 1
            self.language_distribution[language] += 1
            if fallback_reason:
                self.fallback_reasons[fallback_reason.value] += 1
        
        elif status == DetectionStatus.REJECTED:
            self.rejected_partials += 1
    
    def record_switch_attempt(self):
        """Record blocked switch attempt."""
        self.switch_attempts_blocked += 1
    
    def get_stats(self) -> Dict:
        """Get statistics summary."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        avg_confidence = 0.0
        if self.confidence_scores:
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = self.successful_detections / self.total_requests
        
        return {
            "total_requests": self.total_requests,
            "successful_detections": self.successful_detections,
            "fallbacks": self.fallbacks,
            "rejected_partials": self.rejected_partials,
            "switch_attempts_blocked": self.switch_attempts_blocked,
            "success_rate": round(success_rate, 4),
            "avg_confidence": round(avg_confidence, 4),
            "language_distribution": dict(self.language_distribution),
            "fallback_reasons": dict(self.fallback_reasons),
            "uptime_seconds": round(uptime, 2)
        }


# ============================================================================
# GLOBAL STATE
# ============================================================================

_language_locks: Dict[str, LanguageLock] = {}
_detection_stats = DetectionStats()


# ============================================================================
# LANGUAGE DETECTION (Hardened)
# ============================================================================

def detect_language(
    text: str,
    call_id: str,
    is_final: bool = False
) -> DetectionResult:
    """
    Detect language from text with hardened behavior.
    
    HARDENING RULES:
    1. Only detect on FINAL transcripts (unless configured otherwise)
    2. Detect ONCE per call - language is locked after first detection
    3. NO mid-call language switching
    4. Confidence threshold enforced
    5. Fallback to default on low confidence
    
    Args:
        text: Text to analyze
        call_id: Call identifier
        is_final: Whether this is a final transcript
        
    Returns:
        DetectionResult with language and metadata
    """
    start_time = time.time()
    
    # RULE 1: Check if language is already locked (DETECT ONCE)
    existing_lock = _language_locks.get(call_id)
    if existing_lock:
        logger.debug(
            f"Language already locked for {call_id}: {existing_lock.language} "
            f"(confidence: {existing_lock.confidence:.2f})"
        )
        
        # Track blocked switch attempt
        _detection_stats.record_switch_attempt()
        
        if METRICS_ENABLED:
            detection_switch_attempts.inc()
        
        return DetectionResult(
            language=existing_lock.language,
            confidence=existing_lock.confidence,
            status=DetectionStatus.LOCKED,
            locked=True,
            detection_method=existing_lock.detection_method
        )
    
    # RULE 2: Reject partial transcripts (unless configured to allow)
    if REQUIRE_FINAL_TRANSCRIPT and not is_final:
        logger.debug(
            f"Rejecting partial transcript for {call_id} "
            f"(require_final={REQUIRE_FINAL_TRANSCRIPT})"
        )
        
        _detection_stats.record_detection(
            DetectionStatus.REJECTED,
            DEFAULT_LANGUAGE
        )
        
        if METRICS_ENABLED:
            detection_rejected_partial.inc()
        
        # Return default WITHOUT locking (wait for final)
        return DetectionResult(
            language=DEFAULT_LANGUAGE,
            confidence=0.0,
            status=DetectionStatus.REJECTED,
            locked=False,
            detection_method="none",
            fallback_reason=FallbackReason.PARTIAL_TRANSCRIPT
        )
    
    # RULE 3: Validate text length
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        logger.debug(
            f"Text too short for detection: {len(text)} chars < {MIN_TEXT_LENGTH}"
        )
        
        return _fallback_to_default(
            call_id,
            FallbackReason.TEXT_TOO_SHORT
        )
    
    # RULE 4: Check if library is available
    if not LANGDETECT_AVAILABLE:
        logger.error("langdetect library not available")
        
        return _fallback_to_default(
            call_id,
            FallbackReason.LIBRARY_UNAVAILABLE
        )
    
    # Perform detection
    try:
        detected_langs = detect_langs(text)
        
        if not detected_langs:
            return _fallback_to_default(
                call_id,
                FallbackReason.DETECTION_ERROR
            )
        
        # Get top detection
        top_lang = detected_langs[0]
        language = top_lang.lang
        confidence = top_lang.prob
        
        # RULE 5: Check if detected language is supported
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Unsupported language detected: {language} "
                f"(confidence: {confidence:.2f}), using default"
            )
            
            return _fallback_to_default(
                call_id,
                FallbackReason.UNSUPPORTED_LANGUAGE
            )
        
        # RULE 6: Check confidence threshold
        if confidence < MIN_CONFIDENCE:
            logger.warning(
                f"Low confidence detection: {language} "
                f"(confidence: {confidence:.2f} < {MIN_CONFIDENCE}), using default"
            )
            
            return _fallback_to_default(
                call_id,
                FallbackReason.LOW_CONFIDENCE
            )
        
        # SUCCESS - Lock language permanently
        duration = time.time() - start_time
        
        # Create immutable lock
        lock = LanguageLock(
            call_id=call_id,
            language=language,
            confidence=confidence,
            locked_at=datetime.utcnow(),
            detection_method="detected"
        )
        
        _language_locks[call_id] = lock
        
        # Track metrics
        if METRICS_ENABLED:
            detection_requests_total.labels(result='success').inc()
            detection_by_language.labels(language=language).inc()
            detection_confidence.observe(confidence)
            detection_duration.observe(duration)
            detection_locked_calls.set(len(_language_locks))
        
        _detection_stats.record_detection(
            DetectionStatus.SUCCESS,
            language,
            confidence
        )
        
        logger.info(
            f"Language detected and locked for {call_id}: {language} "
            f"(confidence: {confidence:.2f}, duration: {duration:.3f}s)"
        )
        
        # Build alternatives
        alternatives = [
            {
                "language": lang.lang,
                "confidence": round(lang.prob, 4)
            }
            for lang in detected_langs[:3]
        ]
        
        return DetectionResult(
            language=language,
            confidence=confidence,
            status=DetectionStatus.SUCCESS,
            locked=True,
            detection_method="detected",
            alternatives=alternatives
        )
    
    except LangDetectException as e:
        logger.error(f"Language detection error: {str(e)}")
        
        if METRICS_ENABLED:
            detection_errors.labels(error_type='langdetect_exception').inc()
        
        return _fallback_to_default(
            call_id,
            FallbackReason.DETECTION_ERROR
        )
    
    except Exception as e:
        logger.error(f"Unexpected detection error: {str(e)}")
        
        if METRICS_ENABLED:
            detection_errors.labels(error_type='unknown').inc()
        
        return _fallback_to_default(
            call_id,
            FallbackReason.DETECTION_ERROR
        )


def _fallback_to_default(
    call_id: str,
    reason: FallbackReason
) -> DetectionResult:
    """
    Fallback to default language.
    
    This LOCKS the language to default (immutable).
    
    Args:
        call_id: Call identifier
        reason: Reason for fallback
        
    Returns:
        DetectionResult with default language
    """
    # Create immutable lock with default language
    lock = LanguageLock(
        call_id=call_id,
        language=DEFAULT_LANGUAGE,
        confidence=1.0,
        locked_at=datetime.utcnow(),
        detection_method="fallback"
    )
    
    _language_locks[call_id] = lock
    
    # Track metrics
    if METRICS_ENABLED:
        detection_requests_total.labels(result='fallback').inc()
        detection_by_language.labels(language=DEFAULT_LANGUAGE).inc()
        detection_fallbacks.labels(reason=reason.value).inc()
        detection_locked_calls.set(len(_language_locks))
    
    _detection_stats.record_detection(
        DetectionStatus.FALLBACK,
        DEFAULT_LANGUAGE,
        confidence=1.0,
        fallback_reason=reason
    )
    
    logger.info(
        f"Locked to default language for {call_id}: {DEFAULT_LANGUAGE} "
        f"(reason: {reason.value})"
    )
    
    return DetectionResult(
        language=DEFAULT_LANGUAGE,
        confidence=1.0,
        status=DetectionStatus.FALLBACK,
        locked=True,
        detection_method="fallback",
        fallback_reason=reason
    )


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_locked_language(call_id: str) -> Optional[str]:
    """
    Get locked language for a call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Locked language or None if not yet detected
    """
    lock = _language_locks.get(call_id)
    return lock.language if lock else None


def is_language_locked(call_id: str) -> bool:
    """
    Check if language is locked for a call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        True if language is locked
    """
    return call_id in _language_locks


def get_lock_info(call_id: str) -> Optional[Dict]:
    """
    Get detailed lock information.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Lock information or None
    """
    lock = _language_locks.get(call_id)
    if not lock:
        return None
    
    return lock.to_dict()


def clear_session(call_id: str):
    """
    Clear language lock for a call (call ended).
    
    This is the ONLY way to remove a lock - when call ends.
    NO mid-call unlocking allowed.
    
    Args:
        call_id: Call identifier
    """
    if call_id in _language_locks:
        language = _language_locks[call_id].language
        del _language_locks[call_id]
        
        if METRICS_ENABLED:
            detection_locked_calls.set(len(_language_locks))
        
        logger.info(
            f"Session cleared for {call_id} "
            f"(language was: {language})"
        )


def clear_all_sessions():
    """
    Clear all language locks (for cleanup/testing).
    
    WARNING: This clears ALL sessions.
    """
    count = len(_language_locks)
    _language_locks.clear()
    
    if METRICS_ENABLED:
        detection_locked_calls.set(0)
    
    logger.warning(f"Cleared ALL {count} language locks")


# ============================================================================
# MANUAL LOCKING (Pre-detection)
# ============================================================================

def lock_language(call_id: str, language: str) -> bool:
    """
    Manually lock language for a call (before first detection).
    
    Use case: Restaurant specifies language via API.
    
    Args:
        call_id: Call identifier
        language: Language to lock
        
    Returns:
        True if locked successfully
    """
    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Cannot lock unsupported language: {language}")
        return False
    
    # Check if already locked
    if call_id in _language_locks:
        logger.warning(
            f"Language already locked for {call_id}: "
            f"{_language_locks[call_id].language}"
        )
        return False
    
    # Create immutable lock
    lock = LanguageLock(
        call_id=call_id,
        language=language,
        confidence=1.0,
        locked_at=datetime.utcnow(),
        detection_method="manual"
    )
    
    _language_locks[call_id] = lock
    
    if METRICS_ENABLED:
        detection_locked_calls.set(len(_language_locks))
        detection_by_language.labels(language=language).inc()
    
    logger.info(f"Language manually locked for {call_id}: {language}")
    
    return True


# ============================================================================
# UTILITIES
# ============================================================================

def is_supported_language(language: str) -> bool:
    """Check if language is supported."""
    return language in SUPPORTED_LANGUAGES


def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


def get_default_language() -> str:
    """Get default language."""
    return DEFAULT_LANGUAGE


def get_detection_stats() -> Dict:
    """Get detection statistics."""
    return _detection_stats.get_stats()


def get_active_locks() -> Dict[str, Dict]:
    """Get all active language locks."""
    return {
        call_id: lock.to_dict()
        for call_id, lock in _language_locks.items()
    }


def get_active_lock_count() -> int:
    """Get count of active locks."""
    return len(_language_locks)


# ============================================================================
# CONFIDENCE UTILITIES
# ============================================================================

def get_confidence_threshold() -> Tuple[float, float]:
    """
    Get confidence thresholds.
    
    Returns:
        (min_confidence, high_confidence)
    """
    return MIN_CONFIDENCE, HIGH_CONFIDENCE


def is_high_confidence(confidence: float) -> bool:
    """
    Check if confidence is high.
    
    Args:
        confidence: Confidence score
        
    Returns:
        True if high confidence
    """
    return confidence >= HIGH_CONFIDENCE


def is_acceptable_confidence(confidence: float) -> bool:
    """
    Check if confidence is acceptable.
    
    Args:
        confidence: Confidence score
        
    Returns:
        True if acceptable
    """
    return confidence >= MIN_CONFIDENCE


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Language Detection Module (Production Hardened v3.0)")
    print("="*60)
    print(f"Supported Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"Default Language: {DEFAULT_LANGUAGE}")
    print(f"Min Confidence: {MIN_CONFIDENCE}")
    print(f"Require Final Transcript: {REQUIRE_FINAL_TRANSCRIPT}")
    print("="*60)
    
    # Test detection
    test_texts = {
        "en": "Hello, how can I help you today? I would like to make a reservation.",
        "ar": "مرحبا، كيف يمكنني مساعدتك اليوم؟ أريد أن أحجز طاولة.",
        "es": "Hola, ¿cómo puedo ayudarte hoy? Me gustaría hacer una reserva."
    }
    
    print("\n1. Testing language detection (final transcripts):")
    for expected_lang, text in test_texts.items():
        call_id = f"test_{expected_lang}"
        
        result = detect_language(text, call_id, is_final=True)
        detected = result.language
        confidence = result.confidence
        match = "✓" if detected == expected_lang else "✗"
        
        print(f"\n{match} Text: {text[:50]}...")
        print(f"  Expected: {expected_lang}")
        print(f"  Detected: {detected} (confidence: {confidence:.2f})")
        print(f"  Status: {result.status.value}")
        print(f"  Locked: {result.locked}")
    
    print("\n2. Testing partial transcript rejection:")
    result = detect_language(
        "Hello",
        "test_partial",
        is_final=False  # Partial!
    )
    print(f"  Status: {result.status.value}")
    print(f"  Locked: {result.locked}")
    print(f"  Reason: {result.fallback_reason.value if result.fallback_reason else 'N/A'}")
    
    print("\n3. Testing mid-call switch prevention:")
    # First detection
    result1 = detect_language(
        "Hello, I would like to make a reservation.",
        "test_switch",
        is_final=True
    )
    print(f"  First: {result1.language} (locked={result1.locked})")
    
    # Try to switch (should be blocked)
    result2 = detect_language(
        "Hola, me gustaría hacer una reserva.",  # Spanish
        "test_switch",
        is_final=True
    )
    print(f"  Second: {result2.language} (status={result2.status.value})")
    
    # Stats
    print("\n4. Detection statistics:")
    stats = get_detection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Active locks
    print("\n5. Active locks:")
    locks = get_active_locks()
    for call_id, lock_info in locks.items():
        print(f"  {call_id}: {lock_info['language']} (method={lock_info['detection_method']})")
    
    # Cleanup
    clear_all_sessions()
    print(f"\nAll sessions cleared")
