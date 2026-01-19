"""
Speech-to-Text Layer
====================
Pure audio-to-text conversion using Google Cloud Speech-to-Text.
Provider-swappable, streaming-friendly, production-grade.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions
import io


# ============================================================================
# CONFIGURATION
# ============================================================================

# Supported languages with their Google Cloud codes
SUPPORTED_LANGUAGES = {
    "en": "en-US",      # English (US)
    "ar": "ar-SA",      # Arabic (Saudi Arabia)
    "es": "es-ES",      # Spanish (Spain)
}

# Default audio configuration
DEFAULT_ENCODING = speech.RecognitionConfig.AudioEncoding.LINEAR16
DEFAULT_SAMPLE_RATE = 16000  # Hz
DEFAULT_LANGUAGE = "en-US"

# Audio enhancement features
ENABLE_AUTOMATIC_PUNCTUATION = True
ENABLE_NOISE_REDUCTION = True
USE_ENHANCED_MODELS = True


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


def _create_client() -> speech.SpeechClient:
    """
    Create Google Cloud Speech client.
    
    Returns:
        Initialized SpeechClient
        
    Raises:
        RuntimeError: If credentials cannot be loaded
    """
    try:
        credentials = _get_credentials()
        if credentials:
            return speech.SpeechClient(credentials=credentials)
        else:
            # Use application default credentials (works in GCP)
            return speech.SpeechClient()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Google Cloud Speech client: {str(e)}. "
            f"Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON"
        )


# Global client instance (lazy-loaded)
_client: Optional[speech.SpeechClient] = None


def _get_client() -> speech.SpeechClient:
    """Get or create the global Speech client."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


# ============================================================================
# CORE TRANSCRIPTION FUNCTION
# ============================================================================

async def transcribe_audio(
    audio_bytes: bytes,
    language_hint: Optional[str] = None,
    encoding: Optional[speech.RecognitionConfig.AudioEncoding] = None,
    sample_rate: Optional[int] = None,
    enable_automatic_punctuation: bool = ENABLE_AUTOMATIC_PUNCTUATION,
    enable_noise_reduction: bool = ENABLE_NOISE_REDUCTION,
    use_enhanced: bool = USE_ENHANCED_MODELS
) -> Dict[str, Any]:
    """
    Convert audio bytes to text using Google Cloud Speech-to-Text.
    
    Args:
        audio_bytes: Raw audio data as bytes
        language_hint: Optional language hint ('en', 'ar', 'es')
        encoding: Audio encoding (defaults to LINEAR16)
        sample_rate: Sample rate in Hz (defaults to 16000)
        enable_automatic_punctuation: Add punctuation automatically
        enable_noise_reduction: Enable noise reduction
        use_enhanced: Use enhanced models for better accuracy
        
    Returns:
        Dictionary with:
        - text: Transcribed text
        - language_code: Detected/used language code
        - confidence: Confidence score (0.0 to 1.0)
        - alternatives: List of alternative transcriptions (if any)
        
    Raises:
        ValueError: If audio_bytes is empty or invalid
        RuntimeError: If transcription fails
        
    Example:
        >>> result = await transcribe_audio(audio_data, language_hint="en")
        >>> print(f"Text: {result['text']}")
        >>> print(f"Confidence: {result['confidence']:.2f}")
    """
    # Validate input
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("audio_bytes cannot be empty")
    
    # Determine language code
    language_code = _resolve_language_code(language_hint)
    
    # Build recognition config
    config = _build_recognition_config(
        language_code=language_code,
        encoding=encoding or DEFAULT_ENCODING,
        sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
        enable_automatic_punctuation=enable_automatic_punctuation,
        enable_noise_reduction=enable_noise_reduction,
        use_enhanced=use_enhanced
    )
    
    # Build audio object
    audio = speech.RecognitionAudio(content=audio_bytes)
    
    # Perform transcription
    try:
        # Run blocking operation in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _get_client().recognize(config=config, audio=audio)
        )
        
        # Parse response
        return _parse_response(response, language_code)
        
    except google_exceptions.InvalidArgument as e:
        raise ValueError(f"Invalid audio format or configuration: {str(e)}")
    except google_exceptions.GoogleAPIError as e:
        raise RuntimeError(f"Google Cloud API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


# ============================================================================
# STREAMING TRANSCRIPTION
# ============================================================================

async def transcribe_audio_streaming(
    audio_chunks: List[bytes],
    language_hint: Optional[str] = None,
    encoding: Optional[speech.RecognitionConfig.AudioEncoding] = None,
    sample_rate: Optional[int] = None,
    enable_automatic_punctuation: bool = ENABLE_AUTOMATIC_PUNCTUATION,
    enable_noise_reduction: bool = ENABLE_NOISE_REDUCTION,
    use_enhanced: bool = USE_ENHANCED_MODELS
) -> Dict[str, Any]:
    """
    Transcribe audio from multiple chunks using streaming API.
    More efficient for longer audio or real-time processing.
    
    Args:
        audio_chunks: List of audio byte chunks
        language_hint: Optional language hint ('en', 'ar', 'es')
        encoding: Audio encoding (defaults to LINEAR16)
        sample_rate: Sample rate in Hz (defaults to 16000)
        enable_automatic_punctuation: Add punctuation automatically
        enable_noise_reduction: Enable noise reduction
        use_enhanced: Use enhanced models for better accuracy
        
    Returns:
        Dictionary with transcription results (same format as transcribe_audio)
        
    Raises:
        ValueError: If audio_chunks is empty
        RuntimeError: If transcription fails
        
    Example:
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> result = await transcribe_audio_streaming(chunks, language_hint="en")
    """
    if not audio_chunks or len(audio_chunks) == 0:
        raise ValueError("audio_chunks cannot be empty")
    
    # Determine language code
    language_code = _resolve_language_code(language_hint)
    
    # Build streaming config
    config = _build_recognition_config(
        language_code=language_code,
        encoding=encoding or DEFAULT_ENCODING,
        sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
        enable_automatic_punctuation=enable_automatic_punctuation,
        enable_noise_reduction=enable_noise_reduction,
        use_enhanced=use_enhanced
    )
    
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False  # Only return final results
    )
    
    try:
        # Create request generator
        def request_generator():
            # First request with config
            yield speech.StreamingRecognizeRequest(
                streaming_config=streaming_config
            )
            # Subsequent requests with audio chunks
            for chunk in audio_chunks:
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
        
        # Perform streaming recognition
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            None,
            lambda: _get_client().streaming_recognize(request_generator())
        )
        
        # Parse streaming responses
        return _parse_streaming_response(responses, language_code)
        
    except google_exceptions.InvalidArgument as e:
        raise ValueError(f"Invalid audio format or configuration: {str(e)}")
    except google_exceptions.GoogleAPIError as e:
        raise RuntimeError(f"Google Cloud API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Streaming transcription failed: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _resolve_language_code(language_hint: Optional[str]) -> str:
    """
    Resolve language hint to Google Cloud language code.
    
    Args:
        language_hint: Short language code ('en', 'ar', 'es') or None
        
    Returns:
        Google Cloud language code (e.g., 'en-US')
    """
    if not language_hint:
        return DEFAULT_LANGUAGE
    
    # Normalize hint
    hint = language_hint.lower().strip()
    
    # Check if it's a supported short code
    if hint in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[hint]
    
    # Check if it's already a full code
    if hint in SUPPORTED_LANGUAGES.values():
        return hint
    
    # Default fallback
    return DEFAULT_LANGUAGE


def _build_recognition_config(
    language_code: str,
    encoding: speech.RecognitionConfig.AudioEncoding,
    sample_rate: int,
    enable_automatic_punctuation: bool,
    enable_noise_reduction: bool,
    use_enhanced: bool
) -> speech.RecognitionConfig:
    """
    Build Google Cloud Speech recognition config.
    
    Args:
        language_code: Google Cloud language code
        encoding: Audio encoding format
        sample_rate: Sample rate in Hz
        enable_automatic_punctuation: Enable automatic punctuation
        enable_noise_reduction: Enable noise reduction
        use_enhanced: Use enhanced models
        
    Returns:
        RecognitionConfig object
    """
    # Build alternative language codes for better detection
    alternative_language_codes = [
        code for code in SUPPORTED_LANGUAGES.values()
        if code != language_code
    ]
    
    # Select model
    model = "default"
    if use_enhanced:
        # Enhanced models are more accurate but cost more
        if language_code.startswith("en"):
            model = "phone_call"  # Optimized for phone audio
        else:
            model = "default"
    
    config = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        alternative_language_codes=alternative_language_codes[:2],  # Max 3 total
        enable_automatic_punctuation=enable_automatic_punctuation,
        model=model,
        use_enhanced=use_enhanced,
        # Audio channel config
        audio_channel_count=1,  # Mono
        enable_separate_recognition_per_channel=False,
        # Additional features for noisy environments
        enable_word_time_offsets=False,  # Not needed for basic transcription
        enable_word_confidence=False,     # Not needed for basic transcription
    )
    
    # Add noise reduction metadata hint (unofficial but helpful)
    if enable_noise_reduction:
        config.metadata = speech.RecognitionMetadata(
            interaction_type=speech.RecognitionMetadata.InteractionType.PHONE_CALL,
            recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.PHONE_LINE,
        )
    
    return config


def _parse_response(
    response: speech.RecognizeResponse,
    language_code: str
) -> Dict[str, Any]:
    """
    Parse Google Cloud Speech response into standardized format.
    
    Args:
        response: RecognizeResponse from Google Cloud
        language_code: Language code used for recognition
        
    Returns:
        Standardized result dictionary
    """
    # Default empty result
    result = {
        "text": "",
        "language_code": _normalize_language_code(language_code),
        "confidence": 0.0,
        "alternatives": []
    }
    
    # Extract results
    if not response.results:
        return result
    
    # Get the first result (most confident)
    first_result = response.results[0]
    
    if not first_result.alternatives:
        return result
    
    # Primary alternative (highest confidence)
    primary = first_result.alternatives[0]
    result["text"] = primary.transcript.strip()
    result["confidence"] = primary.confidence if hasattr(primary, "confidence") else 1.0
    
    # Detect actual language if available
    if hasattr(first_result, "language_code"):
        result["language_code"] = _normalize_language_code(first_result.language_code)
    
    # Add alternatives
    for alt in first_result.alternatives[1:]:
        result["alternatives"].append({
            "text": alt.transcript.strip(),
            "confidence": alt.confidence if hasattr(alt, "confidence") else 0.0
        })
    
    return result


def _parse_streaming_response(
    responses,
    language_code: str
) -> Dict[str, Any]:
    """
    Parse streaming recognition responses.
    
    Args:
        responses: Iterator of StreamingRecognizeResponse objects
        language_code: Language code used for recognition
        
    Returns:
        Standardized result dictionary
    """
    # Accumulate final results
    final_text = ""
    max_confidence = 0.0
    detected_language = language_code
    
    for response in responses:
        if not response.results:
            continue
        
        # Only process final results
        for result in response.results:
            if not result.is_final:
                continue
            
            if not result.alternatives:
                continue
            
            # Get best alternative
            alternative = result.alternatives[0]
            final_text += alternative.transcript + " "
            
            # Track max confidence
            if hasattr(alternative, "confidence"):
                max_confidence = max(max_confidence, alternative.confidence)
            
            # Detect language
            if hasattr(result, "language_code"):
                detected_language = result.language_code
    
    return {
        "text": final_text.strip(),
        "language_code": _normalize_language_code(detected_language),
        "confidence": max_confidence if max_confidence > 0 else 1.0,
        "alternatives": []
    }


def _normalize_language_code(language_code: str) -> str:
    """
    Normalize Google Cloud language code to short code.
    
    Args:
        language_code: Full language code (e.g., 'en-US')
        
    Returns:
        Short language code ('en', 'ar', 'es')
    """
    # Map back to short codes
    reverse_map = {v: k for k, v in SUPPORTED_LANGUAGES.items()}
    
    if language_code in reverse_map:
        return reverse_map[language_code]
    
    # Extract language prefix
    prefix = language_code.split("-")[0].lower()
    if prefix in SUPPORTED_LANGUAGES:
        return prefix
    
    # Default to English
    return "en"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_supported_languages() -> List[str]:
    """
    Get list of supported language codes.
    
    Returns:
        List of short language codes
        
    Example:
        >>> langs = get_supported_languages()
        >>> print(langs)  # ['en', 'ar', 'es']
    """
    return list(SUPPORTED_LANGUAGES.keys())


def is_language_supported(language: str) -> bool:
    """
    Check if a language is supported.
    
    Args:
        language: Language code to check
        
    Returns:
        True if supported, False otherwise
        
    Example:
        >>> is_language_supported("en")  # True
        >>> is_language_supported("fr")  # False
    """
    return language.lower() in SUPPORTED_LANGUAGES


async def validate_audio(
    audio_bytes: bytes,
    min_size: int = 1024,
    max_size: int = 10 * 1024 * 1024  # 10 MB
) -> bool:
    """
    Validate audio bytes before transcription.
    
    Args:
        audio_bytes: Audio data to validate
        min_size: Minimum size in bytes
        max_size: Maximum size in bytes
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> await validate_audio(audio_data)
    """
    if not audio_bytes:
        raise ValueError("Audio bytes cannot be None or empty")
    
    size = len(audio_bytes)
    
    if size < min_size:
        raise ValueError(f"Audio too small: {size} bytes (minimum: {min_size})")
    
    if size > max_size:
        raise ValueError(f"Audio too large: {size} bytes (maximum: {max_size})")
    
    return True


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the STT module.
    """
    
    async def example():
        # Example: Read audio file and transcribe
        # audio_file = "sample.wav"
        # with open(audio_file, "rb") as f:
        #     audio_data = f.read()
        
        # For demonstration, using dummy data
        print("STT Module Example")
        print("=" * 50)
        
        # Simulate audio bytes (in production, this comes from microphone/file)
        # audio_data = b"..." # Real audio bytes
        
        print("\nSupported languages:", get_supported_languages())
        print("English supported:", is_language_supported("en"))
        print("French supported:", is_language_supported("fr"))
        
        print("\nConfiguration:")
        print(f"- Default language: {DEFAULT_LANGUAGE}")
        print(f"- Default sample rate: {DEFAULT_SAMPLE_RATE} Hz")
        print(f"- Automatic punctuation: {ENABLE_AUTOMATIC_PUNCTUATION}")
        print(f"- Noise reduction: {ENABLE_NOISE_REDUCTION}")
        print(f"- Enhanced models: {USE_ENHANCED_MODELS}")
        
        # In production:
        # result = await transcribe_audio(audio_data, language_hint="en")
        # print(f"\nTranscription: {result['text']}")
        # print(f"Language: {result['language_code']}")
        # print(f"Confidence: {result['confidence']:.2%}")
    
    # Run example
    asyncio.run(example())
