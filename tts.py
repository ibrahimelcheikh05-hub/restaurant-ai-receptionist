"""
Text-to-Speech Layer
====================
Pure text-to-audio conversion using Google Cloud Text-to-Speech.
Streaming output with cancellation support for barge-in scenarios.
"""

import os
import asyncio
from typing import Optional, AsyncGenerator, Dict, Any
from google.cloud import texttospeech_v1 as texttospeech
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions
import base64


# ============================================================================
# VOICE LIBRARY
# ============================================================================

VOICE_LIBRARY = {
    "en": {
        "language_code": "en-US",
        "name": "en-US-Neural2-F",  # Female voice, natural
        "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE
    },
    "ar": {
        "language_code": "ar-XA",
        "name": "ar-XA-Standard-A",  # Female voice
        "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE
    },
    "es": {
        "language_code": "es-ES",
        "name": "es-ES-Neural2-A",  # Female voice, natural
        "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE
    }
}

# Audio configuration
AUDIO_ENCODING = texttospeech.AudioEncoding.LINEAR16
SAMPLE_RATE = 16000  # Hz (phone quality)
SPEAKING_RATE = 1.0  # Normal speed
PITCH = 0.0  # Normal pitch

# Streaming configuration
CHUNK_SIZE = 4096  # Bytes per chunk for streaming
DEFAULT_LANGUAGE = "en"


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


def _create_client() -> texttospeech.TextToSpeechClient:
    """
    Create Google Cloud TTS client.
    
    Returns:
        Initialized TextToSpeechClient
        
    Raises:
        RuntimeError: If credentials cannot be loaded
    """
    try:
        credentials = _get_credentials()
        if credentials:
            return texttospeech.TextToSpeechClient(credentials=credentials)
        else:
            # Use application default credentials (works in GCP)
            return texttospeech.TextToSpeechClient()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Google Cloud TTS client: {str(e)}. "
            f"Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON"
        )


# Global client instance (lazy-loaded)
_client: Optional[texttospeech.TextToSpeechClient] = None


def _get_client() -> texttospeech.TextToSpeechClient:
    """Get or create the global TTS client."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


# ============================================================================
# CANCELLATION TOKEN
# ============================================================================

class CancellationToken:
    """
    Token for cancelling TTS streaming.
    Supports barge-in scenarios where user interrupts.
    """
    
    def __init__(self):
        self._cancelled = False
        self._lock = asyncio.Lock()
    
    async def cancel(self):
        """Mark this token as cancelled."""
        async with self._lock:
            self._cancelled = True
    
    async def is_cancelled(self) -> bool:
        """Check if cancelled."""
        async with self._lock:
            return self._cancelled
    
    def reset(self):
        """Reset cancellation state."""
        self._cancelled = False


# ============================================================================
# CORE TTS FUNCTION
# ============================================================================

async def stream_tts(
    text: str,
    language_code: str,
    websocket: Any,
    cancellation_token: Optional[CancellationToken] = None,
    speaking_rate: float = SPEAKING_RATE,
    pitch: float = PITCH,
    volume_gain_db: float = 0.0
) -> bool:
    """
    Stream text-to-speech audio to a WebSocket.
    
    Args:
        text: Text to convert to speech
        language_code: Language code ('en', 'ar', 'es')
        websocket: WebSocket connection to stream audio to
        cancellation_token: Optional token for mid-stream cancellation
        speaking_rate: Speech rate (0.25 to 4.0, default 1.0)
        pitch: Voice pitch (-20.0 to 20.0, default 0.0)
        volume_gain_db: Volume adjustment in dB (-96.0 to 16.0, default 0.0)
        
    Returns:
        True if completed successfully, False if cancelled
        
    Raises:
        ValueError: If text is empty or language unsupported
        RuntimeError: If TTS generation fails
        
    Example:
        >>> token = CancellationToken()
        >>> success = await stream_tts("Hello!", "en", ws, token)
        >>> if success:
        >>>     print("Speech completed")
    """
    # Validate input
    if not text or text.strip() == "":
        raise ValueError("Text cannot be empty")
    
    if language_code not in VOICE_LIBRARY:
        raise ValueError(
            f"Unsupported language: {language_code}. "
            f"Supported: {list(VOICE_LIBRARY.keys())}"
        )
    
    # Get voice configuration
    voice_config = VOICE_LIBRARY[language_code]
    
    try:
        # Generate audio
        audio_content = await _synthesize_speech(
            text=text,
            voice_config=voice_config,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume_gain_db=volume_gain_db
        )
        
        # Stream audio in chunks
        return await _stream_audio_chunks(
            audio_content=audio_content,
            websocket=websocket,
            cancellation_token=cancellation_token
        )
        
    except google_exceptions.InvalidArgument as e:
        raise ValueError(f"Invalid TTS parameters: {str(e)}")
    except google_exceptions.GoogleAPIError as e:
        raise RuntimeError(f"Google Cloud API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"TTS streaming failed: {str(e)}")


# ============================================================================
# AUDIO GENERATION
# ============================================================================

async def _synthesize_speech(
    text: str,
    voice_config: Dict[str, Any],
    speaking_rate: float,
    pitch: float,
    volume_gain_db: float
) -> bytes:
    """
    Synthesize speech from text using Google Cloud TTS.
    
    Args:
        text: Text to synthesize
        voice_config: Voice configuration from VOICE_LIBRARY
        speaking_rate: Speech rate
        pitch: Voice pitch
        volume_gain_db: Volume adjustment
        
    Returns:
        Audio content as bytes
    """
    # Build synthesis input
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_config["language_code"],
        name=voice_config["name"],
        ssml_gender=voice_config["ssml_gender"]
    )
    
    # Build audio config
    audio_config = texttospeech.AudioConfig(
        audio_encoding=AUDIO_ENCODING,
        sample_rate_hertz=SAMPLE_RATE,
        speaking_rate=speaking_rate,
        pitch=pitch,
        volume_gain_db=volume_gain_db,
        # Additional quality settings
        effects_profile_id=["telephony-class-application"],  # Optimized for phone
    )
    
    # Perform synthesis (blocking call, run in executor)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _get_client().synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
    )
    
    return response.audio_content


# ============================================================================
# AUDIO STREAMING
# ============================================================================

async def _stream_audio_chunks(
    audio_content: bytes,
    websocket: Any,
    cancellation_token: Optional[CancellationToken] = None,
    chunk_size: int = CHUNK_SIZE
) -> bool:
    """
    Stream audio content in chunks to WebSocket.
    
    Args:
        audio_content: Audio bytes to stream
        websocket: WebSocket connection
        cancellation_token: Optional cancellation token
        chunk_size: Size of each chunk in bytes
        
    Returns:
        True if completed, False if cancelled
    """
    total_size = len(audio_content)
    offset = 0
    
    while offset < total_size:
        # Check for cancellation
        if cancellation_token and await cancellation_token.is_cancelled():
            return False
        
        # Get next chunk
        chunk = audio_content[offset:offset + chunk_size]
        
        # Send chunk to WebSocket
        try:
            await websocket.send_bytes(chunk)
        except Exception as e:
            # WebSocket error - stop streaming
            raise RuntimeError(f"WebSocket send failed: {str(e)}")
        
        offset += chunk_size
        
        # Small delay to prevent overwhelming the connection
        # Adjust based on your needs
        await asyncio.sleep(0.01)
    
    return True


# ============================================================================
# GENERATOR-BASED STREAMING (ALTERNATIVE)
# ============================================================================

async def generate_tts_chunks(
    text: str,
    language_code: str,
    speaking_rate: float = SPEAKING_RATE,
    pitch: float = PITCH,
    volume_gain_db: float = 0.0,
    chunk_size: int = CHUNK_SIZE
) -> AsyncGenerator[bytes, None]:
    """
    Generate TTS audio chunks as an async generator.
    Alternative to stream_tts for custom streaming logic.
    
    Args:
        text: Text to convert to speech
        language_code: Language code ('en', 'ar', 'es')
        speaking_rate: Speech rate
        pitch: Voice pitch
        volume_gain_db: Volume adjustment
        chunk_size: Size of each chunk
        
    Yields:
        Audio chunks as bytes
        
    Raises:
        ValueError: If text is empty or language unsupported
        RuntimeError: If TTS generation fails
        
    Example:
        >>> async for chunk in generate_tts_chunks("Hello!", "en"):
        >>>     await websocket.send_bytes(chunk)
    """
    # Validate input
    if not text or text.strip() == "":
        raise ValueError("Text cannot be empty")
    
    if language_code not in VOICE_LIBRARY:
        raise ValueError(
            f"Unsupported language: {language_code}. "
            f"Supported: {list(VOICE_LIBRARY.keys())}"
        )
    
    # Get voice configuration
    voice_config = VOICE_LIBRARY[language_code]
    
    # Generate audio
    audio_content = await _synthesize_speech(
        text=text,
        voice_config=voice_config,
        speaking_rate=speaking_rate,
        pitch=pitch,
        volume_gain_db=volume_gain_db
    )
    
    # Yield chunks
    offset = 0
    total_size = len(audio_content)
    
    while offset < total_size:
        chunk = audio_content[offset:offset + chunk_size]
        yield chunk
        offset += chunk_size


# ============================================================================
# SSML SUPPORT
# ============================================================================

async def stream_tts_ssml(
    ssml: str,
    language_code: str,
    websocket: Any,
    cancellation_token: Optional[CancellationToken] = None,
    speaking_rate: float = SPEAKING_RATE,
    pitch: float = PITCH,
    volume_gain_db: float = 0.0
) -> bool:
    """
    Stream TTS from SSML (Speech Synthesis Markup Language).
    Allows for advanced control: pauses, emphasis, prosody, etc.
    
    Args:
        ssml: SSML-formatted text
        language_code: Language code ('en', 'ar', 'es')
        websocket: WebSocket connection
        cancellation_token: Optional cancellation token
        speaking_rate: Speech rate
        pitch: Voice pitch
        volume_gain_db: Volume adjustment
        
    Returns:
        True if completed, False if cancelled
        
    Example:
        >>> ssml = '''<speak>
        >>>     Hello! <break time="500ms"/>
        >>>     Your order total is <emphasis>$25.99</emphasis>.
        >>> </speak>'''
        >>> await stream_tts_ssml(ssml, "en", ws)
    """
    if not ssml or ssml.strip() == "":
        raise ValueError("SSML cannot be empty")
    
    if language_code not in VOICE_LIBRARY:
        raise ValueError(f"Unsupported language: {language_code}")
    
    voice_config = VOICE_LIBRARY[language_code]
    
    # Build synthesis input with SSML
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    
    # Build voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_config["language_code"],
        name=voice_config["name"],
        ssml_gender=voice_config["ssml_gender"]
    )
    
    # Build audio config
    audio_config = texttospeech.AudioConfig(
        audio_encoding=AUDIO_ENCODING,
        sample_rate_hertz=SAMPLE_RATE,
        speaking_rate=speaking_rate,
        pitch=pitch,
        volume_gain_db=volume_gain_db,
        effects_profile_id=["telephony-class-application"],
    )
    
    # Synthesize
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _get_client().synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
    )
    
    # Stream
    return await _stream_audio_chunks(
        audio_content=response.audio_content,
        websocket=websocket,
        cancellation_token=cancellation_token
    )


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
    return list(VOICE_LIBRARY.keys())


def get_voice_info(language_code: str) -> Dict[str, Any]:
    """
    Get voice configuration for a language.
    
    Args:
        language_code: Language code
        
    Returns:
        Voice configuration dictionary
        
    Raises:
        ValueError: If language not supported
        
    Example:
        >>> info = get_voice_info("en")
        >>> print(info["name"])  # en-US-Neural2-F
    """
    if language_code not in VOICE_LIBRARY:
        raise ValueError(f"Unsupported language: {language_code}")
    return VOICE_LIBRARY[language_code].copy()


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
    return language_code in VOICE_LIBRARY


async def synthesize_to_file(
    text: str,
    language_code: str,
    output_path: str,
    speaking_rate: float = SPEAKING_RATE,
    pitch: float = PITCH,
    volume_gain_db: float = 0.0
) -> str:
    """
    Synthesize speech and save to file (for testing/caching).
    
    Args:
        text: Text to synthesize
        language_code: Language code
        output_path: Path to save audio file
        speaking_rate: Speech rate
        pitch: Voice pitch
        volume_gain_db: Volume adjustment
        
    Returns:
        Path to saved file
        
    Example:
        >>> path = await synthesize_to_file("Hello!", "en", "hello.wav")
    """
    if language_code not in VOICE_LIBRARY:
        raise ValueError(f"Unsupported language: {language_code}")
    
    voice_config = VOICE_LIBRARY[language_code]
    
    audio_content = await _synthesize_speech(
        text=text,
        voice_config=voice_config,
        speaking_rate=speaking_rate,
        pitch=pitch,
        volume_gain_db=volume_gain_db
    )
    
    # Save to file
    with open(output_path, "wb") as f:
        f.write(audio_content)
    
    return output_path


def create_ssml(
    text: str,
    pauses: Optional[list[tuple[int, str]]] = None,
    emphasis: Optional[list[tuple[int, int]]] = None,
    rate: str = "medium"
) -> str:
    """
    Helper to create SSML from plain text.
    
    Args:
        text: Base text
        pauses: List of (char_position, duration) tuples (e.g., "500ms")
        emphasis: List of (start_pos, end_pos) tuples for emphasis
        rate: Speaking rate ("x-slow", "slow", "medium", "fast", "x-fast")
        
    Returns:
        SSML string
        
    Example:
        >>> ssml = create_ssml(
        >>>     "Hello. How are you?",
        >>>     pauses=[(5, "500ms")],
        >>>     emphasis=[(11, 19)]
        >>> )
    """
    # Start with prosody for rate
    ssml = f'<speak><prosody rate="{rate}">'
    
    # TODO: Implement insertion of pauses and emphasis
    # For now, just wrap the text
    ssml += text
    
    ssml += '</prosody></speak>'
    
    return ssml


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the TTS module.
    """
    
    async def example():
        print("TTS Module Example")
        print("=" * 50)
        
        print("\nSupported languages:", get_supported_languages())
        
        for lang in get_supported_languages():
            info = get_voice_info(lang)
            print(f"\n{lang.upper()}:")
            print(f"  Voice: {info['name']}")
            print(f"  Language: {info['language_code']}")
            print(f"  Gender: {info['ssml_gender']}")
        
        print("\nConfiguration:")
        print(f"- Sample rate: {SAMPLE_RATE} Hz")
        print(f"- Encoding: {AUDIO_ENCODING}")
        print(f"- Chunk size: {CHUNK_SIZE} bytes")
        print(f"- Default speaking rate: {SPEAKING_RATE}")
        
        # Example with mock WebSocket
        class MockWebSocket:
            async def send_bytes(self, data: bytes):
                print(f"  Sent {len(data)} bytes")
        
        ws = MockWebSocket()
        token = CancellationToken()
        
        print("\n\nExample TTS streaming:")
        print("-" * 50)
        
        # Simulate streaming
        # In production, this would stream to a real WebSocket
        # success = await stream_tts(
        #     "Hello! Welcome to our restaurant.",
        #     "en",
        #     ws,
        #     token
        # )
        
        print("\nTo use in production:")
        print("  success = await stream_tts(text, 'en', websocket, token)")
        print("  if not success:")
        print("      print('User interrupted (barge-in)')")
    
    # Run example
    asyncio.run(example())
