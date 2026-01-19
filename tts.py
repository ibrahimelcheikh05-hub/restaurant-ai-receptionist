"""
Text-to-Speech Layer (Streaming)
=================================
Real-time streaming text-to-speech using Google Cloud Text-to-Speech.
Supports barge-in, chunked audio output, and low latency.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from google.cloud import texttospeech
from google.oauth2 import service_account
import json


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

AUDIO_CONFIG = {
    "audio_encoding": texttospeech.AudioEncoding.LINEAR16,
    "sample_rate_hertz": 16000,
    "speaking_rate": 1.0,
    "pitch": 0.0,
    "volume_gain_db": 0.0
}

CHUNK_SIZE = 4096


class CancellationToken:
    """Token for cancelling TTS mid-stream."""
    
    def __init__(self):
        self._cancelled = False
        self._event = asyncio.Event()
    
    async def cancel(self):
        """Cancel the TTS operation."""
        self._cancelled = True
        self._event.set()
    
    def is_cancelled(self) -> bool:
        """Check if cancelled."""
        return self._cancelled
    
    async def wait_cancelled(self, timeout: float = 0.01):
        """Wait for cancellation with timeout."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


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


_client: Optional[texttospeech.TextToSpeechAsyncClient] = None


def _get_client() -> texttospeech.TextToSpeechAsyncClient:
    """Get or create TTS client."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


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
        call_id: Call identifier for logging
        cancel_event: Cancellation token for barge-in
        
    Returns:
        True if completed without cancellation, False if cancelled
    """
    if not text or not text.strip():
        return True
    
    if cancel_event is None:
        cancel_event = CancellationToken()
    
    try:
        voice_config = VOICE_LIBRARY.get(language_code, VOICE_LIBRARY["en"])
        
        client = _get_client()
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config["language_code"],
            name=voice_config["name"],
            ssml_gender=voice_config["gender"]
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=AUDIO_CONFIG["audio_encoding"],
            sample_rate_hertz=AUDIO_CONFIG["sample_rate_hertz"],
            speaking_rate=AUDIO_CONFIG["speaking_rate"],
            pitch=AUDIO_CONFIG["pitch"],
            volume_gain_db=AUDIO_CONFIG["volume_gain_db"]
        )
        
        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        audio_content = response.audio_content
        
        if not audio_content:
            logger.warning(f"Empty audio generated for call {call_id}")
            return True
        
        total_chunks = (len(audio_content) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for i in range(0, len(audio_content), CHUNK_SIZE):
            if cancel_event.is_cancelled():
                logger.info(f"TTS cancelled for call {call_id} at chunk {i // CHUNK_SIZE}/{total_chunks}")
                return False
            
            chunk = audio_content[i:i + CHUNK_SIZE]
            
            try:
                await websocket.send_bytes(chunk)
            except Exception as e:
                logger.error(f"Failed to send audio chunk for call {call_id}: {str(e)}")
                return False
            
            if await cancel_event.wait_cancelled():
                logger.info(f"TTS cancelled during send for call {call_id}")
                return False
        
        logger.info(f"TTS completed for call {call_id}: {len(text)} chars, {total_chunks} chunks")
        return True
        
    except asyncio.CancelledError:
        logger.info(f"TTS task cancelled for call {call_id}")
        return False
    except Exception as e:
        logger.error(f"TTS error for call {call_id}: {str(e)}", exc_info=True)
        return False


async def synthesize_speech_streaming(
    text: str,
    language_code: str,
    cancel_event: Optional[CancellationToken] = None
) -> AsyncGenerator[bytes, None]:
    """
    Generate audio chunks for streaming.
    
    Args:
        text: Text to synthesize
        language_code: Language code
        cancel_event: Cancellation token
        
    Yields:
        Audio chunk bytes
    """
    if not text or not text.strip():
        return
    
    if cancel_event is None:
        cancel_event = CancellationToken()
    
    try:
        voice_config = VOICE_LIBRARY.get(language_code, VOICE_LIBRARY["en"])
        
        client = _get_client()
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config["language_code"],
            name=voice_config["name"],
            ssml_gender=voice_config["gender"]
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=AUDIO_CONFIG["audio_encoding"],
            sample_rate_hertz=AUDIO_CONFIG["sample_rate_hertz"],
            speaking_rate=AUDIO_CONFIG["speaking_rate"],
            pitch=AUDIO_CONFIG["pitch"],
            volume_gain_db=AUDIO_CONFIG["volume_gain_db"]
        )
        
        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        audio_content = response.audio_content
        
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
    """Update global audio configuration."""
    if speaking_rate is not None:
        AUDIO_CONFIG["speaking_rate"] = speaking_rate
    if pitch is not None:
        AUDIO_CONFIG["pitch"] = pitch
    if volume_gain_db is not None:
        AUDIO_CONFIG["volume_gain_db"] = volume_gain_db


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Streaming TTS Example")
        print("=" * 50)
        
        print("\nSupported languages:")
        for lang in get_supported_languages():
            voice = get_voice_for_language(lang)
            print(f"  {lang}: {voice['name']}")
        
        print("\n" + "=" * 50)
        print("Streaming TTS ready for production use")
    
    asyncio.run(example())
