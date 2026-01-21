"""
Speech-to-Text Engine (Production Streaming)
=============================================
Hardened real-time streaming speech recognition with Google Cloud.
Per-call isolation, auto-reconnect, silence endpointing, garbage rejection.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from google.cloud import speech
from google.cloud.speech import StreamingRecognitionConfig, RecognitionConfig
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions
import json


logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = {
    "en": "en-US",
    "ar": "ar-SA",
    "es": "es-ES"
}

SAMPLE_RATE = 16000
ENCODING = speech.RecognitionConfig.AudioEncoding.LINEAR16

# Quality thresholds
MIN_CONFIDENCE = 0.6
MIN_TRANSCRIPT_LENGTH = 3
MAX_TRANSCRIPT_LENGTH = 500

# Timeouts
SILENCE_TIMEOUT = 3.0  # seconds
STREAM_TIMEOUT = 305  # seconds (Google limit is 305s)
RECONNECT_DELAY = 1.0  # seconds

# Audio limits
MAX_AUDIO_QUEUE_SIZE = 100


class StreamState:
    """Stream session state."""
    
    def __init__(self, call_id: str, language_code: str):
        self.call_id = call_id
        self.language_code = language_code
        self.is_active = True
        self.last_audio_time: Optional[datetime] = None
        self.last_transcript_time: Optional[datetime] = None
        self.stream_start_time = datetime.utcnow()
        self.reconnect_count = 0
        self.total_transcripts = 0
        self.error_count = 0


class StreamSession:
    """Manages a single streaming recognition session with fault tolerance."""
    
    def __init__(self, call_id: str, language_code: str):
        self.call_id = call_id
        self.language_code = language_code
        
        # Queues
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE_SIZE)
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        
        # State
        self.state = StreamState(call_id, language_code)
        self.is_active = True
        
        # Tasks
        self.stream_task: Optional[asyncio.Task] = None
        self.watchdog_task: Optional[asyncio.Task] = None
        
        # Locks
        self.shutdown_lock = asyncio.Lock()
        
        logger.info(f"StreamSession created: {call_id}, language: {language_code}")
    
    async def start(self):
        """Start streaming tasks."""
        self.stream_task = asyncio.create_task(self._stream_loop())
        self.watchdog_task = asyncio.create_task(self._watchdog_loop())
    
    async def close(self):
        """Close stream session with cleanup."""
        async with self.shutdown_lock:
            if not self.is_active:
                return
            
            logger.info(f"Closing stream session: {self.call_id}")
            
            self.is_active = False
            self.state.is_active = False
            
            # Cancel tasks
            if self.stream_task and not self.stream_task.done():
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass
            
            if self.watchdog_task and not self.watchdog_task.done():
                self.watchdog_task.cancel()
                try:
                    await self.watchdog_task
                except asyncio.CancelledError:
                    pass
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            logger.info(
                f"Stream closed: {self.call_id}, "
                f"transcripts: {self.state.total_transcripts}, "
                f"reconnects: {self.state.reconnect_count}"
            )
    
    async def _stream_loop(self):
        """Main streaming recognition loop with auto-reconnect."""
        while self.is_active:
            try:
                await self._run_stream()
                
                if not self.is_active:
                    break
                
                # Stream ended, reconnect if still active
                logger.warning(f"Stream ended for {self.call_id}, reconnecting...")
                self.state.reconnect_count += 1
                
                await asyncio.sleep(RECONNECT_DELAY)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error for {self.call_id}: {str(e)}")
                self.state.error_count += 1
                
                if self.state.error_count >= 3:
                    logger.error(f"Too many errors for {self.call_id}, stopping")
                    break
                
                await asyncio.sleep(RECONNECT_DELAY)
    
    async def _run_stream(self):
        """Run a single streaming recognition session."""
        try:
            client = _get_client()
            
            config = RecognitionConfig(
                encoding=ENCODING,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
                model="telephony",
                use_enhanced=True,
                enable_word_time_offsets=False,
                enable_word_confidence=False,
                max_alternatives=1
            )
            
            streaming_config = StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False
            )
            
            audio_generator = self._audio_generator()
            
            requests = (
                speech.StreamingRecognizeRequest(audio_content=chunk)
                async for chunk in audio_generator
            )
            
            responses = await client.streaming_recognize(
                config=streaming_config,
                requests=requests
            )
            
            async for response in responses:
                if not self.is_active:
                    break
                
                await self._process_response(response)
            
        except google_exceptions.OutOfRange:
            logger.info(f"Stream time limit reached for {self.call_id}")
        except google_exceptions.GoogleAPICallError as e:
            logger.error(f"Google API error for {self.call_id}: {str(e)}")
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stream processing error for {self.call_id}: {str(e)}")
            raise
    
    async def _audio_generator(self) -> AsyncGenerator[bytes, None]:
        """Generate audio chunks for streaming with timeout."""
        while self.is_active:
            try:
                chunk = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=SILENCE_TIMEOUT
                )
                
                if chunk is None:  # Sentinel for shutdown
                    break
                
                yield chunk
                
            except asyncio.TimeoutError:
                # Silence timeout - send empty chunk to keep stream alive
                logger.debug(f"Silence timeout for {self.call_id}")
                continue
            except Exception as e:
                logger.error(f"Audio generator error: {str(e)}")
                break
    
    async def _process_response(self, response):
        """Process streaming recognition response with validation."""
        if not response.results:
            return
        
        result = response.results[0]
        
        if not result.alternatives:
            return
        
        alternative = result.alternatives[0]
        transcript = alternative.transcript.strip()
        
        # Garbage rejection
        if not transcript:
            return
        
        if len(transcript) < MIN_TRANSCRIPT_LENGTH:
            logger.debug(f"Transcript too short, ignoring: '{transcript}'")
            return
        
        if len(transcript) > MAX_TRANSCRIPT_LENGTH:
            logger.warning(f"Transcript too long, truncating: {len(transcript)} chars")
            transcript = transcript[:MAX_TRANSCRIPT_LENGTH]
        
        # Confidence check for final results
        confidence = alternative.confidence if result.is_final else 0.0
        
        if result.is_final and confidence < MIN_CONFIDENCE:
            logger.warning(
                f"Low confidence transcript rejected: '{transcript}' "
                f"(confidence: {confidence:.2f})"
            )
            return
        
        # Create transcript data
        transcript_data = {
            "text": transcript,
            "is_final": result.is_final,
            "confidence": confidence,
            "language_code": self.language_code.split("-")[0],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Queue transcript
        try:
            self.transcript_queue.put_nowait(transcript_data)
            self.state.last_transcript_time = datetime.utcnow()
            
            if result.is_final:
                self.state.total_transcripts += 1
                logger.info(
                    f"Final transcript for {self.call_id}: '{transcript}' "
                    f"(confidence: {confidence:.2f})"
                )
            else:
                logger.debug(f"Partial transcript for {self.call_id}: '{transcript}'")
        
        except asyncio.QueueFull:
            logger.warning(f"Transcript queue full for {self.call_id}")
    
    async def _watchdog_loop(self):
        """Monitor stream health and timeouts."""
        try:
            while self.is_active:
                await asyncio.sleep(1.0)
                
                now = datetime.utcnow()
                
                # Check stream age
                stream_age = (now - self.state.stream_start_time).total_seconds()
                if stream_age > STREAM_TIMEOUT:
                    logger.warning(f"Stream timeout for {self.call_id}")
                    # Let it reconnect naturally
                    continue
                
                # Check hard silence timeout
                if self.state.last_audio_time:
                    silence_duration = (now - self.state.last_audio_time).total_seconds()
                    
                    if silence_duration > 60:  # 60 seconds hard limit
                        logger.warning(f"Hard silence timeout for {self.call_id}")
                        await self.close()
                        break
        
        except asyncio.CancelledError:
            pass


_active_streams: Dict[str, StreamSession] = {}
_cleanup_task: Optional[asyncio.Task] = None
_client: Optional[speech.SpeechAsyncClient] = None


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
    
    try:
        session = StreamSession(call_id, language_code)
        _active_streams[call_id] = session
        
        await session.start()
        
        logger.info(f"Started streaming STT for call {call_id}, language: {language_code}")
        
        # Start cleanup task if not running
        global _cleanup_task
        if _cleanup_task is None or _cleanup_task.done():
            _cleanup_task = asyncio.create_task(_cleanup_stale_streams())
        
        return {
            "status": "started",
            "call_id": call_id,
            "language_code": language_code
        }
    
    except Exception as e:
        logger.error(f"Failed to start stream for {call_id}: {str(e)}")
        _active_streams.pop(call_id, None)
        return {
            "status": "error",
            "error": str(e),
            "call_id": call_id
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
    
    if not audio_chunk or len(audio_chunk) == 0:
        return True
    
    try:
        # Non-blocking put with timeout
        session.audio_queue.put_nowait(audio_chunk)
        session.state.last_audio_time = datetime.utcnow()
        return True
    
    except asyncio.QueueFull:
        logger.warning(f"Audio queue full for {call_id}, dropping chunk")
        return False
    except Exception as e:
        logger.error(f"Error feeding audio for {call_id}: {str(e)}")
        return False


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
            timeout=0.05
        )
        return transcript
    
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        logger.error(f"Error getting transcript for {call_id}: {str(e)}")
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
    
    try:
        # Signal shutdown
        try:
            session.audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        
        await session.close()
        
        _active_streams.pop(call_id, None)
        
        logger.info(f"Stopped streaming STT for call {call_id}")
        
        return {
            "status": "stopped",
            "call_id": call_id,
            "total_transcripts": session.state.total_transcripts,
            "reconnects": session.state.reconnect_count
        }
    
    except Exception as e:
        logger.error(f"Error stopping stream for {call_id}: {str(e)}")
        _active_streams.pop(call_id, None)
        return {
            "status": "error",
            "error": str(e),
            "call_id": call_id
        }


async def _cleanup_stale_streams():
    """Background cleanup of stale streams."""
    while True:
        try:
            await asyncio.sleep(30)
            
            now = datetime.utcnow()
            stale = []
            
            for call_id, session in _active_streams.items():
                if not session.is_active:
                    stale.append(call_id)
                    continue
                
                # Check for extended inactivity
                if session.state.last_audio_time:
                    idle = (now - session.state.last_audio_time).total_seconds()
                    if idle > 300:  # 5 minutes
                        logger.warning(f"Stream inactive for {call_id}")
                        stale.append(call_id)
            
            for call_id in stale:
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


def get_stream_stats(call_id: str) -> Optional[Dict[str, Any]]:
    """Get statistics for a stream."""
    session = _active_streams.get(call_id)
    
    if not session:
        return None
    
    return {
        "call_id": call_id,
        "is_active": session.is_active,
        "language_code": session.language_code,
        "total_transcripts": session.state.total_transcripts,
        "reconnect_count": session.state.reconnect_count,
        "error_count": session.state.error_count,
        "uptime_seconds": (datetime.utcnow() - session.state.stream_start_time).total_seconds()
    }


async def transcribe_audio(
    audio_bytes: bytes,
    language_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Legacy batch transcription (for compatibility).
    
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
        logger.error(f"Batch transcription error: {str(e)}")
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
        print("Streaming STT Engine (Production)")
        print("=" * 50)
        
        call_id = "test_call_123"
        
        print(f"\n1. Starting stream: {call_id}")
        result = await start_stream(call_id, "en")
        print(f"   Status: {result['status']}")
        
        print(f"\n2. Stream active: {is_stream_active(call_id)}")
        
        print(f"\n3. Stream stats:")
        stats = get_stream_stats(call_id)
        if stats:
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        await asyncio.sleep(1)
        
        print(f"\n4. Stopping stream")
        result = await stop_stream(call_id)
        print(f"   Status: {result['status']}")
        
        print("\n" + "=" * 50)
        print("Production streaming STT ready")
    
    asyncio.run(example())
