"""
Speech-to-Text Layer (Streaming)
=================================
Real-time streaming speech recognition using Google Cloud Speech-to-Text.
Supports multiple concurrent calls with low latency.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from google.cloud import speech
from google.cloud.speech import StreamingRecognitionConfig, RecognitionConfig
from google.oauth2 import service_account
import json


logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = {
    "en": "en-US",
    "ar": "ar-SA",
    "es": "es-ES"
}

SAMPLE_RATE = 16000
ENCODING = speech.RecognitionConfig.AudioEncoding.LINEAR16


class StreamSession:
    """Manages a single streaming recognition session."""
    
    def __init__(self, call_id: str, language_code: str):
        self.call_id = call_id
        self.language_code = language_code
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self.is_active = True
        self.stream_task: Optional[asyncio.Task] = None
        self.last_activity = datetime.utcnow()
        
    async def close(self):
        """Close the stream session."""
        self.is_active = False
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass


_active_streams: Dict[str, StreamSession] = {}
_cleanup_task: Optional[asyncio.Task] = None


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


def _create_client() -> speech.SpeechAsyncClient:
    """Create Google Cloud Speech client."""
    credentials = _get_credentials()
    if credentials:
        return speech.SpeechAsyncClient(credentials=credentials)
    return speech.SpeechAsyncClient()


_client: Optional[speech.SpeechAsyncClient] = None


def _get_client() -> speech.SpeechAsyncClient:
    """Get or create Speech client."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


async def start_stream(
    call_id: str,
    language_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Start a streaming recognition session.
    
    Args:
        call_id: Unique call identifier
        language_hint: Language code hint ('en', 'ar', 'es')
        
    Returns:
        Status dictionary
    """
    if call_id in _active_streams:
        logger.warning(f"Stream already exists for call {call_id}")
        return {"status": "already_active", "call_id": call_id}
    
    language_code = SUPPORTED_LANGUAGES.get(
        language_hint or "en",
        "en-US"
    )
    
    session = StreamSession(call_id, language_code)
    _active_streams[call_id] = session
    
    session.stream_task = asyncio.create_task(
        _run_stream(session)
    )
    
    logger.info(f"Started streaming STT for call {call_id}, language: {language_code}")
    
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_stale_streams())
    
    return {
        "status": "started",
        "call_id": call_id,
        "language_code": language_code
    }


async def feed_audio(call_id: str, audio_chunk: bytes) -> bool:
    """
    Feed audio chunk to streaming recognizer.
    
    Args:
        call_id: Call identifier
        audio_chunk: Audio data bytes
        
    Returns:
        True if successful, False if stream not active
    """
    session = _active_streams.get(call_id)
    
    if not session or not session.is_active:
        logger.warning(f"No active stream for call {call_id}")
        return False
    
    if len(audio_chunk) == 0:
        return True
    
    session.last_activity = datetime.utcnow()
    await session.audio_queue.put(audio_chunk)
    
    return True


async def get_transcript(call_id: str) -> Optional[Dict[str, Any]]:
    """
    Get next available transcript (partial or final).
    
    Args:
        call_id: Call identifier
        
    Returns:
        Transcript dictionary or None if no transcript available
    """
    session = _active_streams.get(call_id)
    
    if not session:
        return None
    
    try:
        transcript = await asyncio.wait_for(
            session.transcript_queue.get(),
            timeout=0.1
        )
        return transcript
    except asyncio.TimeoutError:
        return None


async def stop_stream(call_id: str) -> Dict[str, Any]:
    """
    Stop streaming recognition session.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Status dictionary
    """
    session = _active_streams.get(call_id)
    
    if not session:
        return {"status": "not_found", "call_id": call_id}
    
    await session.close()
    del _active_streams[call_id]
    
    logger.info(f"Stopped streaming STT for call {call_id}")
    
    return {"status": "stopped", "call_id": call_id}


async def _run_stream(session: StreamSession):
    """
    Run the streaming recognition loop.
    
    Args:
        session: Stream session instance
    """
    try:
        client = _get_client()
        
        config = RecognitionConfig(
            encoding=ENCODING,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=session.language_code,
            enable_automatic_punctuation=True,
            model="telephony",
            use_enhanced=True
        )
        
        streaming_config = StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False
        )
        
        audio_generator = _audio_generator(session)
        
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            async for chunk in audio_generator
        )
        
        responses = await client.streaming_recognize(
            config=streaming_config,
            requests=requests
        )
        
        async for response in responses:
            if not session.is_active:
                break
            
            await _process_response(session, response)
            
    except asyncio.CancelledError:
        logger.info(f"Stream cancelled for call {session.call_id}")
        raise
    except Exception as e:
        logger.error(f"Stream error for call {session.call_id}: {str(e)}", exc_info=True)
        if session.is_active:
            await session.transcript_queue.put({
                "text": "",
                "is_final": True,
                "confidence": 0.0,
                "language_code": session.language_code.split("-")[0],
                "error": str(e)
            })


async def _audio_generator(session: StreamSession) -> AsyncGenerator[bytes, None]:
    """
    Generate audio chunks for streaming.
    
    Args:
        session: Stream session instance
        
    Yields:
        Audio chunk bytes
    """
    while session.is_active:
        try:
            chunk = await asyncio.wait_for(
                session.audio_queue.get(),
                timeout=5.0
            )
            yield chunk
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Audio generator error: {str(e)}")
            break


async def _process_response(session: StreamSession, response):
    """
    Process streaming recognition response.
    
    Args:
        session: Stream session instance
        response: Google Cloud response
    """
    if not response.results:
        return
    
    result = response.results[0]
    
    if not result.alternatives:
        return
    
    alternative = result.alternatives[0]
    
    transcript_data = {
        "text": alternative.transcript.strip(),
        "is_final": result.is_final,
        "confidence": alternative.confidence if result.is_final else 0.0,
        "language_code": session.language_code.split("-")[0]
    }
    
    if transcript_data["text"]:
        await session.transcript_queue.put(transcript_data)
        
        if result.is_final:
            logger.info(
                f"Final transcript for {session.call_id}: "
                f"{transcript_data['text']} "
                f"(confidence: {transcript_data['confidence']:.2f})"
            )
        else:
            logger.debug(
                f"Partial transcript for {session.call_id}: "
                f"{transcript_data['text']}"
            )


async def _cleanup_stale_streams():
    """Background task to cleanup inactive streams."""
    while True:
        try:
            await asyncio.sleep(60)
            
            now = datetime.utcnow()
            stale_calls = []
            
            for call_id, session in _active_streams.items():
                idle_seconds = (now - session.last_activity).total_seconds()
                
                if idle_seconds > 300:
                    stale_calls.append(call_id)
            
            for call_id in stale_calls:
                logger.warning(f"Cleaning up stale stream: {call_id}")
                await stop_stream(call_id)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {str(e)}")


def get_active_streams() -> list[str]:
    """Get list of active stream call IDs."""
    return list(_active_streams.keys())


def is_stream_active(call_id: str) -> bool:
    """Check if stream is active for call."""
    session = _active_streams.get(call_id)
    return session is not None and session.is_active


async def transcribe_audio(
    audio_bytes: bytes,
    language_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Legacy non-streaming transcription (for compatibility).
    
    Args:
        audio_bytes: Complete audio data
        language_hint: Language code hint
        
    Returns:
        Transcription result
    """
    if not audio_bytes:
        return {
            "text": "",
            "language_code": language_hint or "en",
            "confidence": 0.0
        }
    
    language_code = SUPPORTED_LANGUAGES.get(
        language_hint or "en",
        "en-US"
    )
    
    try:
        client = _get_client()
        
        config = RecognitionConfig(
            encoding=ENCODING,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="telephony",
            use_enhanced=True
        )
        
        audio = speech.RecognitionAudio(content=audio_bytes)
        
        response = await client.recognize(config=config, audio=audio)
        
        if not response.results:
            return {
                "text": "",
                "language_code": language_hint or "en",
                "confidence": 0.0
            }
        
        result = response.results[0]
        alternative = result.alternatives[0]
        
        return {
            "text": alternative.transcript.strip(),
            "language_code": language_hint or "en",
            "confidence": alternative.confidence
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return {
            "text": "",
            "language_code": language_hint or "en",
            "confidence": 0.0,
            "error": str(e)
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Streaming STT Example")
        print("=" * 50)
        
        call_id = "test_call_123"
        
        print(f"\n1. Starting stream for {call_id}")
        result = await start_stream(call_id, "en")
        print(f"   Result: {result}")
        
        print(f"\n2. Stream active: {is_stream_active(call_id)}")
        
        print(f"\n3. Active streams: {get_active_streams()}")
        
        await asyncio.sleep(1)
        
        print(f"\n4. Stopping stream")
        result = await stop_stream(call_id)
        print(f"   Result: {result}")
        
        print("\n" + "=" * 50)
        print("Streaming STT ready for production use")
    
    asyncio.run(example())
