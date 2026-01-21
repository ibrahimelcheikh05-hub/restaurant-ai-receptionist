"""
Text-to-Speech Engine (Production Streaming)
=============================================
Hardened real-time streaming TTS with Google Cloud.
Chunked synthesis, immediate barge-in, voice pre-warming, per-call control.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from google.cloud import texttospeech
from google.oauth2 import service_account
import json
import hashlib


logger = logging.getLogger(__name__)


VOICE_LIBRARY = {
    "en": {
        "language_code": "en-US",
        "name": "en-US-Neural2-F",
        "gender": texttospeech.SsmlVoiceGender.FEMALE
    },
    "ar": {
        "language_code": "ar-XA",
        "name": "ar-XA-Standard-A",
        "gender": texttospeech.SsmlVoiceGender.FEMALE
    },
    "es": {
        "language_code": "es-ES",
        "name": "es-ES-Neural2-A",
        "gender": texttospeech.SsmlVoiceGender.FEMALE
    }
}

# Audio configuration
SAMPLE_RATE = 16000
AUDIO_ENCODING = texttospeech.AudioEncoding.LINEAR16
CHUNK_SIZE = 4096  # 4KB chunks for low latency

# Limits
MAX_TEXT_LENGTH = 5000
MAX_CONCURRENT_SYNTHESIS = 10
CACHE_SIZE = 100

# Pre-warming
PREWARM_PHRASES = {
    "en": "Hello, how can I help you?",
    "ar": "مرحبا، كيف يمكنني مساعدتك؟",
    "es": "Hola, ¿cómo puedo ayudarte?"
}


class CancellationToken:
    """Token for cancelling TTS operations."""
    
    def __init__(self):
        self._cancelled = False
        self._event = asyncio.Event()
    
    async def cancel(self):
        """Cancel the operation."""
        if not self._cancelled:
            self._cancelled = True
            self._event.set()
    
    def is_cancelled(self) -> bool:
        """Check if cancelled."""
        return self._cancelled
    
    async def wait_cancelled(self, timeout: float = 0.01) -> bool:
        """Wait for cancellation with timeout."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class SynthesisController:
    """Per-call synthesis controller with cancellation and stats."""
    
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.is_active = True
        self.synthesis_count = 0
        self.total_chars = 0
        self.total_chunks = 0
        self.cancel_count = 0
        self.start_time = datetime.utcnow()
        self.last_synthesis_time: Optional[datetime] = None
    
    def increment_synthesis(self, text_length: int, chunks: int):
        """Record synthesis stats."""
        self.synthesis_count += 1
        self.total_chars += text_length
        self.total_chunks += chunks
        self.last_synthesis_time = datetime.utcnow()
    
    def increment_cancel(self):
        """Record cancellation."""
        self.cancel_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "call_id": self.call_id,
            "synthesis_count": self.synthesis_count,
            "total_chars": self.total_chars,
            "total_chunks": self.total_chunks,
            "cancel_count": self.cancel_count,
            "uptime_seconds": uptime,
            "avg_chars_per_synthesis": self.total_chars / max(1, self.synthesis_count)
        }


_active_controllers: Dict[str, SynthesisController] = {}
_synthesis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYNTHESIS)
_audio_cache: Dict[str, bytes] = {}
_client: Optional[texttospeech.TextToSpeechAsyncClient] = None
_prewarmed_voices: set = set()


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


def _create_client() -> texttospeech.TextToSpeechAsyncClient:
    """Create Google Cloud TTS client."""
    credentials = _get_credentials()
    if credentials:
        return texttospeech.TextToSpeechAsyncClient(credentials=credentials)
    return texttospeech.TextToSpeechAsyncClient()


def _get_client() -> texttospeech.TextToSpeechAsyncClient:
    """Get or create TTS client."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


def _get_cache_key(text: str, language_code: str) -> str:
    """Generate cache key for audio."""
    content = f"{text}:{language_code}"
    return hashlib.md5(content.encode()).hexdigest()


async def _prewarm_voice(language_code: str):
    """Pre-warm voice model with sample phrase."""
    if language_code in _prewarmed_voices:
        return
    
    try:
        phrase = PREWARM_PHRASES.get(language_code, PREWARM_PHRASES["en"])
        
        # Synthesize but don't cache
        await _synthesize_audio(phrase, language_code, use_cache=False)
        
        _prewarmed_voices.add(language_code)
        logger.info(f"Pre-warmed voice for language: {language_code}")
        
    except Exception as e:
        logger.error(f"Error pre-warming voice for {language_code}: {str(e)}")


async def _synthesize_audio(
    text: str,
    language_code: str,
    use_cache: bool = True
) -> bytes:
    """
    Synthesize audio with caching.
    
    Args:
        text: Text to synthesize
        language_code: Language code
        use_cache: Whether to use cache
        
    Returns:
        Audio bytes
    """
    # Check cache
    if use_cache:
        cache_key = _get_cache_key(text, language_code)
        if cache_key in _audio_cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return _audio_cache[cache_key]
    
    # Get voice config
    voice_config = VOICE_LIBRARY.get(language_code, VOICE_LIBRARY["en"])
    
    client = _get_client()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_config["language_code"],
        name=voice_config["name"],
        ssml_gender=voice_config["gender"]
    )
    
    # Get audio config from env or use defaults
    speaking_rate = float(os.getenv("SPEAKING_RATE", "1.0"))
    pitch = float(os.getenv("PITCH", "0.0"))
    volume_gain_db = float(os.getenv("VOLUME_GAIN_DB", "0.0"))
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=AUDIO_ENCODING,
        sample_rate_hertz=SAMPLE_RATE,
        speaking_rate=speaking_rate,
        pitch=pitch,
        volume_gain_db=volume_gain_db
    )
    
    # Synthesize with semaphore control
    async with _synthesis_semaphore:
        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
    
    audio_content = response.audio_content
    
    # Cache if enabled
    if use_cache and len(_audio_cache) < CACHE_SIZE:
        cache_key = _get_cache_key(text, language_code)
        _audio_cache[cache_key] = audio_content
    
    return audio_content


async def stream_tts(
    text: str,
    language_code: str,
    websocket,
    call_id: str,
    cancel_event: Optional[CancellationToken] = None
) -> bool:
    """
    Stream synthesized audio to websocket with barge-in support.
    
    Args:
        text: Text to synthesize
        language_code: Language code ('en', 'ar', 'es')
        websocket: WebSocket connection to stream audio
        call_id: Call identifier
        cancel_event: Cancellation token for barge-in
        
    Returns:
        True if completed without cancellation, False if cancelled
    """
    if not text or not text.strip():
        return True
    
    # Validate text length
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning(f"Text too long ({len(text)} chars), truncating")
        text = text[:MAX_TEXT_LENGTH]
    
    # Get or create controller
    controller = _active_controllers.get(call_id)
    if not controller:
        controller = SynthesisController(call_id)
        _active_controllers[call_id] = controller
    
    if cancel_event is None:
        cancel_event = CancellationToken()
    
    try:
        # Pre-warm voice if needed
        await _prewarm_voice(language_code)
        
        # Check cancellation before synthesis
        if cancel_event.is_cancelled():
            controller.increment_cancel()
            return False
        
        # Synthesize audio
        start_time = datetime.utcnow()
        audio_content = await _synthesize_audio(text, language_code)
        synthesis_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"Synthesized {len(text)} chars in {synthesis_time:.2f}s "
            f"for {call_id}"
        )
        
        if not audio_content:
            logger.warning(f"Empty audio generated for call {call_id}")
            return True
        
        # Stream in chunks with cancellation checks
        total_chunks = (len(audio_content) + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunks_sent = 0
        
        for i in range(0, len(audio_content), CHUNK_SIZE):
            # Check cancellation before each chunk
            if cancel_event.is_cancelled():
                logger.info(
                    f"TTS cancelled for {call_id} at chunk "
                    f"{chunks_sent}/{total_chunks}"
                )
                controller.increment_cancel()
                return False
            
            chunk = audio_content[i:i + CHUNK_SIZE]
            
            try:
                await websocket.send_bytes(chunk)
                chunks_sent += 1
                
                # Yield control to event loop
                await asyncio.sleep(0)
                
                # Quick cancellation check after send
                if await cancel_event.wait_cancelled():
                    logger.info(f"TTS cancelled during send for {call_id}")
                    controller.increment_cancel()
                    return False
                
            except Exception as e:
                logger.error(f"Failed to send audio chunk for {call_id}: {str(e)}")
                return False
        
        # Record stats
        controller.increment_synthesis(len(text), total_chunks)
        
        logger.info(
            f"TTS completed for {call_id}: {len(text)} chars, "
            f"{total_chunks} chunks sent"
        )
        return True
        
    except asyncio.CancelledError:
        logger.info(f"TTS task cancelled for {call_id}")
        controller.increment_cancel()
        return False
    
    except Exception as e:
        logger.error(f"TTS error for {call_id}: {str(e)}", exc_info=True)
        return False


async def synthesize_speech_streaming(
    text: str,
    language_code: str,
    cancel_event: Optional[CancellationToken] = None
) -> AsyncGenerator[bytes, None]:
    """
    Generate audio chunks for streaming without websocket.
    
    Args:
        text: Text to synthesize
        language_code: Language code
        cancel_event: Cancellation token
        
    Yields:
        Audio chunk bytes
    """
    if not text or not text.strip():
        return
    
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    if cancel_event is None:
        cancel_event = CancellationToken()
    
    try:
        # Pre-warm
        await _prewarm_voice(language_code)
        
        # Synthesize
        audio_content = await _synthesize_audio(text, language_code)
        
        # Stream chunks
        for i in range(0, len(audio_content), CHUNK_SIZE):
            if cancel_event.is_cancelled():
                break
            
            chunk = audio_content[i:i + CHUNK_SIZE]
            yield chunk
            
            await asyncio.sleep(0)
    
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.error(f"Streaming synthesis error: {str(e)}")
        return


def create_controller(call_id: str) -> SynthesisController:
    """Create synthesis controller for call."""
    if call_id in _active_controllers:
        return _active_controllers[call_id]
    
    controller = SynthesisController(call_id)
    _active_controllers[call_id] = controller
    
    logger.info(f"Created synthesis controller for {call_id}")
    return controller


def destroy_controller(call_id: str):
    """Destroy synthesis controller and cleanup."""
    controller = _active_controllers.pop(call_id, None)
    
    if controller:
        stats = controller.get_stats()
        logger.info(f"Destroyed controller for {call_id}: {stats}")


def get_controller_stats(call_id: str) -> Optional[Dict[str, Any]]:
    """Get statistics for call's synthesis controller."""
    controller = _active_controllers.get(call_id)
    return controller.get_stats() if controller else None


def get_active_controllers() -> list[str]:
    """Get list of active controller call IDs."""
    return list(_active_controllers.keys())


def get_voice_for_language(language_code: str) -> Dict[str, Any]:
    """Get voice configuration for language."""
    return VOICE_LIBRARY.get(language_code, VOICE_LIBRARY["en"])


def get_supported_languages() -> list[str]:
    """Get list of supported languages."""
    return list(VOICE_LIBRARY.keys())


def update_audio_config(
    speaking_rate: Optional[float] = None,
    pitch: Optional[float] = None,
    volume_gain_db: Optional[float] = None
):
    """
    Update global audio configuration.
    
    Note: This affects new syntheses only, not active ones.
    """
    if speaking_rate is not None:
        if 0.25 <= speaking_rate <= 4.0:
            os.environ["SPEAKING_RATE"] = str(speaking_rate)
        else:
            logger.warning(f"Invalid speaking_rate: {speaking_rate}, must be 0.25-4.0")
    
    if pitch is not None:
        if -20.0 <= pitch <= 20.0:
            os.environ["PITCH"] = str(pitch)
        else:
            logger.warning(f"Invalid pitch: {pitch}, must be -20.0 to 20.0")
    
    if volume_gain_db is not None:
        if -96.0 <= volume_gain_db <= 16.0:
            os.environ["VOLUME_GAIN_DB"] = str(volume_gain_db)
        else:
            logger.warning(f"Invalid volume_gain_db: {volume_gain_db}")


def clear_cache():
    """Clear audio cache."""
    global _audio_cache
    _audio_cache.clear()
    logger.info("Audio cache cleared")


def get_cache_size() -> int:
    """Get current cache size."""
    return len(_audio_cache)


async def prewarm_all_voices():
    """Pre-warm all voice models."""
    tasks = []
    for lang in VOICE_LIBRARY.keys():
        tasks.append(_prewarm_voice(lang))
    
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"Pre-warmed {len(_prewarmed_voices)} voices")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Streaming TTS Engine (Production)")
        print("=" * 50)
        
        print("\nSupported languages:")
        for lang in get_supported_languages():
            voice = get_voice_for_language(lang)
            print(f"  {lang}: {voice['name']}")
        
        print("\nPre-warming voices...")
        await prewarm_all_voices()
        print(f"Pre-warmed voices: {len(_prewarmed_voices)}")
        
        print("\nController management:")
        call_id = "test_call_123"
        controller = create_controller(call_id)
        print(f"  Created controller: {call_id}")
        
        stats = get_controller_stats(call_id)
        if stats:
            print(f"  Stats: {stats}")
        
        destroy_controller(call_id)
        print(f"  Destroyed controller: {call_id}")
        
        print("\n" + "=" * 50)
        print("Production streaming TTS ready")
    
    asyncio.run(example())
