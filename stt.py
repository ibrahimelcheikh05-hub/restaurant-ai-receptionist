"""
Speech-to-Text Module (Production Hardened - Deepgram)
=======================================================
Production-grade streaming speech recognition with comprehensive safety.

HARDENING UPDATES (v4.0):
✅ Stream supervisor with hung stream detection
✅ Confidence filtering and noise suppression
✅ Turn-boundary detection with endpoint events
✅ Hard timeouts for silence and stalled streams
✅ Cancellation support via CancelToken
✅ Length threshold filtering
✅ PARTIAL vs FINAL transcript classification
✅ Structured transcript output (no raw strings)

Version: 4.0.0 (Production Hardened)
Last Updated: 2026-01-22
Cost: $0.0043/min (82% cheaper than Google)
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from enum import Enum
import time
import json

try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
    )
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    logging.warning("Deepgram SDK not installed. Install: pip install deepgram-sdk")

# Latency tracking
try:
    from latency_tracker import get_tracker, LatencyType
    LATENCY_TRACKING_ENABLED = True
except ImportError:
    LATENCY_TRACKING_ENABLED = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# Configuration
STT_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
STT_ENCODING = os.getenv("AUDIO_ENCODING", "LINEAR16")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-US")
STT_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2")
MAX_RECONNECT_ATTEMPTS = int(os.getenv("STT_MAX_RECONNECT", "3"))
RECONNECT_DELAY = int(os.getenv("STT_RECONNECT_DELAY", "2"))
STREAM_TIMEOUT = int(os.getenv("STT_STREAM_TIMEOUT", "300"))  # 5 minutes

# NEW: Hardening configuration
MIN_TRANSCRIPT_LENGTH = int(os.getenv("STT_MIN_LENGTH", "2"))  # Min chars
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("STT_MIN_CONFIDENCE", "0.6"))  # 60%
SILENCE_TIMEOUT = int(os.getenv("STT_SILENCE_TIMEOUT", "30"))  # 30s no audio
STALL_TIMEOUT = int(os.getenv("STT_STALL_TIMEOUT", "15"))  # 15s no transcripts
SUPERVISOR_CHECK_INTERVAL = int(os.getenv("STT_SUPERVISOR_INTERVAL", "5"))  # 5s


# Prometheus Metrics
if METRICS_ENABLED:
    stt_requests_total = Counter(
        'stt_requests_total',
        'Total STT transcription requests',
        ['language', 'status']
    )
    stt_latency_seconds = Histogram(
        'stt_latency_seconds',
        'STT transcription latency',
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    )
    stt_confidence_score = Histogram(
        'stt_confidence_score',
        'STT confidence scores',
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    )
    stt_reconnections_total = Counter(
        'stt_reconnections_total',
        'STT stream reconnection attempts',
        ['result']
    )
    stt_audio_chunks_processed = Counter(
        'stt_audio_chunks_processed',
        'Audio chunks processed'
    )
    stt_stream_duration_seconds = Histogram(
        'stt_stream_duration_seconds',
        'STT stream duration'
    )
    stt_active_streams = Gauge(
        'stt_active_streams',
        'Currently active STT streams'
    )
    stt_words_per_minute = Histogram(
        'stt_words_per_minute',
        'Words transcribed per minute',
        buckets=[50, 100, 150, 200, 250, 300]
    )
    stt_cost_dollars = Counter(
        'stt_cost_dollars_total',
        'Total STT cost in dollars'
    )
    # NEW METRICS (Hardening)
    stt_transcripts_filtered = Counter(
        'stt_transcripts_filtered_total',
        'Transcripts filtered out',
        ['reason']
    )
    stt_hung_streams_killed = Counter(
        'stt_hung_streams_killed_total',
        'Hung streams killed by supervisor'
    )
    stt_silence_timeouts = Counter(
        'stt_silence_timeouts_total',
        'Silence timeout events'
    )
    stt_stall_timeouts = Counter(
        'stt_stall_timeouts_total',
        'Stall timeout events'
    )
    stt_endpoint_detected = Counter(
        'stt_endpoint_detected_total',
        'Turn endpoint detections'
    )


# ============================================================================
# TRANSCRIPT DATA STRUCTURES
# ============================================================================

@dataclass
class TranscriptMetadata:
    """
    Structured transcript output (NO raw strings).
    
    All transcript emissions MUST use this format.
    """
    text: str
    confidence: float
    is_final: bool
    endpoint_detected: bool  # NEW: Turn boundary detection
    timestamp: str
    language: str
    duration: float = 0.0
    word_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "is_final": self.is_final,
            "endpoint_detected": self.endpoint_detected,
            "timestamp": self.timestamp,
            "language": self.language,
            "duration": self.duration,
            "word_count": self.word_count
        }


class CancelToken:
    """Cancellation token for async operations."""
    
    def __init__(self):
        self._cancelled = False
        self._event = asyncio.Event()
    
    async def cancel(self):
        """Cancel the operation."""
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


# ============================================================================
# STREAM SUPERVISOR
# ============================================================================

class StreamSupervisor:
    """
    Monitors STT stream health and kills hung streams.
    
    Tracks:
    - Last audio timestamp
    - Last transcript timestamp
    - Silence duration
    - Stall duration
    
    Actions:
    - Kill hung streams
    - Restart broken connections
    - Emit timeout events
    """
    
    def __init__(self, stream: 'DeepgramSTTStream'):
        self.stream = stream
        self._supervisor_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        logger.debug(f"StreamSupervisor created for {stream.session_id}")
    
    async def start(self):
        """Start supervisor monitoring."""
        self._is_running = True
        self._supervisor_task = asyncio.create_task(self._run_supervisor())
        logger.info(f"Supervisor started for {self.stream.session_id}")
    
    async def stop(self):
        """Stop supervisor monitoring."""
        self._is_running = False
        
        if self._supervisor_task and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            try:
                await asyncio.wait_for(self._supervisor_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        logger.info(f"Supervisor stopped for {self.stream.session_id}")
    
    async def _run_supervisor(self):
        """
        Supervisor monitoring loop.
        
        Checks every SUPERVISOR_CHECK_INTERVAL seconds for:
        - Silence timeout (no audio)
        - Stall timeout (no transcripts)
        - Hung stream (both timeouts)
        """
        try:
            while self._is_running and self.stream.is_active:
                await asyncio.sleep(SUPERVISOR_CHECK_INTERVAL)
                
                # Check cancellation
                if self.stream.cancel_token and self.stream.cancel_token.is_cancelled():
                    logger.info(f"Supervisor cancelled for {self.stream.session_id}")
                    await self._kill_stream("cancelled")
                    break
                
                now = datetime.utcnow()
                
                # Check silence timeout (no audio received)
                if self.stream.last_audio_time:
                    silence_duration = (now - self.stream.last_audio_time).total_seconds()
                    
                    if silence_duration > SILENCE_TIMEOUT:
                        logger.warning(
                            f"SILENCE TIMEOUT: {self.stream.session_id} - "
                            f"{silence_duration:.1f}s since last audio"
                        )
                        
                        if METRICS_ENABLED:
                            stt_silence_timeouts.inc()
                        
                        # Emit silence event (but don't kill yet - might be intentional)
                        if self.stream.on_timeout:
                            await self.stream.on_timeout("silence", silence_duration)
                
                # Check stall timeout (no transcripts received)
                if self.stream.last_transcript_time:
                    stall_duration = (now - self.stream.last_transcript_time).total_seconds()
                    
                    if stall_duration > STALL_TIMEOUT:
                        logger.warning(
                            f"STALL TIMEOUT: {self.stream.session_id} - "
                            f"{stall_duration:.1f}s since last transcript"
                        )
                        
                        if METRICS_ENABLED:
                            stt_stall_timeouts.inc()
                        
                        # Check if stream is hung (both silence AND stall)
                        silence_duration = 0
                        if self.stream.last_audio_time:
                            silence_duration = (now - self.stream.last_audio_time).total_seconds()
                        
                        # If BOTH silence and stall → stream is hung
                        if silence_duration > SILENCE_TIMEOUT:
                            logger.error(
                                f"HUNG STREAM DETECTED: {self.stream.session_id} - "
                                f"silence={silence_duration:.1f}s stall={stall_duration:.1f}s"
                            )
                            
                            await self._kill_stream("hung")
                            break
                        else:
                            # Just stalled transcripts (audio still coming) → try restart
                            logger.warning(f"Attempting stream restart: {self.stream.session_id}")
                            await self.stream.reconnect()
        
        except asyncio.CancelledError:
            logger.debug(f"Supervisor cancelled: {self.stream.session_id}")
        except Exception as e:
            logger.error(
                f"Supervisor crashed: {self.stream.session_id} - {str(e)}",
                exc_info=True
            )
    
    async def _kill_stream(self, reason: str):
        """
        Kill hung stream.
        
        Args:
            reason: Reason for killing (hung, cancelled, etc.)
        """
        logger.error(f"KILLING STREAM: {self.stream.session_id} reason={reason}")
        
        if METRICS_ENABLED:
            stt_hung_streams_killed.inc()
        
        # Force stop
        await self.stream.stop()
        
        # Emit timeout event
        if self.stream.on_timeout:
            await self.stream.on_timeout(reason, 0)


# ============================================================================
# CONFIDENCE FILTER
# ============================================================================

class ConfidenceFilter:
    """
    Filter transcripts based on confidence and length.
    
    Rejects:
    - Low confidence transcripts
    - Too-short transcripts (noise bursts)
    - Empty transcripts
    
    Classifies:
    - PARTIAL vs FINAL
    """
    
    @staticmethod
    def should_accept(
        text: str,
        confidence: float,
        is_final: bool
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if transcript should be accepted.
        
        Args:
            text: Transcript text
            confidence: Confidence score (0-1)
            is_final: Whether this is a final transcript
            
        Returns:
            (should_accept, rejection_reason)
        """
        # Empty text
        if not text or text.strip() == "":
            return False, "empty"
        
        # Length threshold (reject noise bursts)
        if len(text.strip()) < MIN_TRANSCRIPT_LENGTH:
            return False, "too_short"
        
        # Confidence threshold (only for FINAL transcripts)
        # Allow low-confidence partials (they're just hints)
        if is_final and confidence < MIN_CONFIDENCE_THRESHOLD:
            return False, "low_confidence"
        
        return True, None
    
    @staticmethod
    def classify_transcript(
        text: str,
        confidence: float,
        is_final: bool,
        speech_final: bool = False
    ) -> tuple[bool, bool]:
        """
        Classify transcript and detect turn boundaries.
        
        Args:
            text: Transcript text
            confidence: Confidence score
            is_final: Whether Deepgram marked as final
            speech_final: Whether Deepgram detected speech endpoint
            
        Returns:
            (is_final_classification, endpoint_detected)
        """
        # FINAL: High confidence + marked final
        is_final_result = is_final and confidence >= MIN_CONFIDENCE_THRESHOLD
        
        # ENDPOINT: Speech final event OR certain punctuation
        endpoint_detected = speech_final
        
        # Additional endpoint detection: Strong punctuation + final
        if is_final and text.strip().endswith(('.', '!', '?')):
            endpoint_detected = True
        
        return is_final_result, endpoint_detected


# ============================================================================
# DEEPGRAM STT STREAM (HARDENED)
# ============================================================================

class DeepgramSTTStream:
    """
    Deepgram streaming STT session with production hardening.
    
    NEW FEATURES (v4.0):
    - Stream supervisor monitoring
    - Confidence filtering
    - Turn-boundary detection
    - Cancellation support
    - Structured output (no raw strings)
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 16000,
        session_id: Optional[str] = None,
        cancel_token: Optional[CancelToken] = None,  # NEW
        on_transcript: Optional[Callable] = None,    # NEW
        on_timeout: Optional[Callable] = None         # NEW
    ):
        self.language = language
        self.sample_rate = sample_rate
        self.session_id = session_id or f"stt_{int(time.time())}"
        self.cancel_token = cancel_token  # NEW
        self.on_transcript = on_transcript  # NEW: Callback for transcripts
        self.on_timeout = on_timeout  # NEW: Callback for timeouts
        
        # Connection state
        self.is_active = False
        self.reconnect_count = 0
        self.start_time = None
        self.last_activity = None
        
        # NEW: Timestamp tracking for supervisor
        self.last_audio_time: Optional[datetime] = None
        self.last_transcript_time: Optional[datetime] = None
        
        # Transcription tracking
        self.transcripts = deque(maxlen=100)
        self.partial_transcripts = deque(maxlen=10)
        self.total_words = 0
        self.total_duration = 0.0
        
        # Audio quality metrics
        self.audio_chunks_processed = 0
        self.total_audio_bytes = 0
        self.silence_chunks = 0
        
        # Deepgram client
        self.deepgram_client = None
        self.connection = None
        
        # Stream health
        self.consecutive_errors = 0
        
        # NEW: Stream supervisor
        self.supervisor = StreamSupervisor(self)
        
        # NEW: Confidence filter
        self.filter = ConfidenceFilter()
        
        logger.info(
            f"DeepgramSTTStream initialized: {self.session_id} | "
            f"Language: {language} | Model: {STT_MODEL}"
        )
    
    async def start(self) -> bool:
        """Start Deepgram streaming connection with supervisor."""
        if not DEEPGRAM_AVAILABLE:
            logger.error("Deepgram SDK not available")
            return False
        
        try:
            # Get API key
            api_key = os.getenv("DEEPGRAM_API_KEY")
            if not api_key:
                logger.error("DEEPGRAM_API_KEY not set")
                return False
            
            # Initialize Deepgram client
            config = DeepgramClientOptions(
                options={"keepalive": "true"}
            )
            self.deepgram_client = DeepgramClient(api_key, config)
            
            # Create connection
            self.connection = self.deepgram_client.listen.live.v("1")
            
            # Set up event handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            self.connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            
            # Configure streaming options
            options = LiveOptions(
                model=STT_MODEL,
                language=self.language,
                encoding=STT_ENCODING.lower(),
                sample_rate=self.sample_rate,
                punctuate=True,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
                smart_format=True,
                endpointing=300,  # NEW: Endpoint detection
            )
            
            # Start connection
            if not await self.connection.start(options):
                logger.error("Failed to start Deepgram connection")
                return False
            
            self.is_active = True
            self.start_time = datetime.utcnow()
            self.last_activity = datetime.utcnow()
            self.last_audio_time = datetime.utcnow()  # NEW
            
            if METRICS_ENABLED:
                stt_active_streams.inc()
            
            # NEW: Start supervisor
            await self.supervisor.start()
            
            logger.info(f"Deepgram stream started: {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Deepgram stream: {str(e)}", exc_info=True)
            if METRICS_ENABLED:
                stt_requests_total.labels(language=self.language, status='error').inc()
            return False
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """
        Send audio chunk to Deepgram with cancellation check.
        
        Returns:
            False if cancelled or failed
        """
        # NEW: Check cancellation
        if self.cancel_token and self.cancel_token.is_cancelled():
            logger.info(f"Audio send cancelled: {self.session_id}")
            return False
        
        if not self.is_active or not self.connection:
            logger.warning("Stream not active, attempting reconnect")
            await self.reconnect()
            return False
        
        try:
            # Validate audio data
            if not audio_data or len(audio_data) == 0:
                return False
            
            # Send to Deepgram
            self.connection.send(audio_data)
            
            # Track metrics
            self.audio_chunks_processed += 1
            self.total_audio_bytes += len(audio_data)
            self.last_activity = datetime.utcnow()
            self.last_audio_time = datetime.utcnow()  # NEW: Track for supervisor
            
            if METRICS_ENABLED:
                stt_audio_chunks_processed.inc()
            
            # Check for silence
            if self._is_silence(audio_data):
                self.silence_chunks += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending audio: {str(e)}")
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= 3:
                await self.reconnect()
            
            return False
    
    def _is_silence(self, audio_data: bytes) -> bool:
        """Check if audio chunk is mostly silence."""
        if len(audio_data) < 100:
            return True
        
        # Simple energy-based detection
        samples = sum(abs(b - 128) for b in audio_data) / len(audio_data)
        return samples < 5
    
    async def _on_message(self, *args, **kwargs):
        """
        Handle transcript from Deepgram with filtering and classification.
        
        NEW: Applies confidence filtering, turn-boundary detection.
        """
        transcription_start = time.time()
        
        try:
            # Extract result
            result = args[1] if len(args) > 1 else args[0]
            
            if not result:
                return
            
            # Parse transcript
            transcript_data = result.channel.alternatives[0]
            transcript_text = transcript_data.transcript
            confidence = transcript_data.confidence
            is_final = result.is_final
            speech_final = result.speech_final if hasattr(result, 'speech_final') else False
            
            # Update timestamps
            self.last_transcript_time = datetime.utcnow()  # NEW
            self.last_activity = datetime.utcnow()
            self.consecutive_errors = 0
            
            # NEW: Apply confidence filter
            should_accept, reject_reason = self.filter.should_accept(
                transcript_text,
                confidence,
                is_final
            )
            
            if not should_accept:
                logger.debug(
                    f"Transcript filtered: reason={reject_reason} "
                    f"text='{transcript_text[:30]}...' confidence={confidence:.2f}"
                )
                
                if METRICS_ENABLED:
                    stt_transcripts_filtered.labels(reason=reject_reason).inc()
                
                return  # Reject transcript
            
            # NEW: Classify transcript and detect endpoints
            is_final_classified, endpoint_detected = self.filter.classify_transcript(
                transcript_text,
                confidence,
                is_final,
                speech_final
            )
            
            # NEW: Create structured output (NO raw strings)
            transcript = TranscriptMetadata(
                text=transcript_text,
                confidence=confidence,
                is_final=is_final_classified,
                endpoint_detected=endpoint_detected,
                timestamp=datetime.utcnow().isoformat(),
                language=self.language,
                duration=result.duration if hasattr(result, 'duration') else 0,
                word_count=len(transcript_text.split())
            )
            
            # Store transcript
            if is_final_classified:
                self.transcripts.append(transcript.to_dict())
                
                # Track endpoint detection
                if endpoint_detected:
                    logger.info(
                        f"ENDPOINT DETECTED: {self.session_id} - '{transcript_text}'"
                    )
                    
                    if METRICS_ENABLED:
                        stt_endpoint_detected.inc()
                
                # TRACK LATENCY
                if LATENCY_TRACKING_ENABLED:
                    transcription_latency_ms = (time.time() - transcription_start) * 1000
                    tracker = get_tracker(self.session_id)
                    tracker.track_latency(
                        LatencyType.STT,
                        transcription_latency_ms,
                        operation="deepgram_transcribe",
                        metadata={
                            "confidence": confidence,
                            "word_count": transcript.word_count,
                            "is_final": True,
                            "endpoint_detected": endpoint_detected
                        }
                    )
                
                # Track words
                self.total_words += transcript.word_count
                
                # Calculate WPM
                if self.start_time:
                    elapsed_minutes = (datetime.utcnow() - self.start_time).total_seconds() / 60
                    if elapsed_minutes > 0:
                        wpm = self.total_words / elapsed_minutes
                        if METRICS_ENABLED:
                            stt_words_per_minute.observe(wpm)
                
                # Track cost
                if self.start_time:
                    elapsed_minutes = (datetime.utcnow() - self.start_time).total_seconds() / 60
                    cost = elapsed_minutes * 0.0043
                    if METRICS_ENABLED:
                        stt_cost_dollars.inc(cost)
                
                logger.debug(
                    f"Final transcript: '{transcript_text[:50]}...' | "
                    f"Confidence: {confidence:.2f} | Words: {transcript.word_count} | "
                    f"Endpoint: {endpoint_detected}"
                )
            else:
                self.partial_transcripts.append(transcript.to_dict())
            
            # Track metrics
            if METRICS_ENABLED:
                stt_confidence_score.observe(confidence)
                if is_final_classified:
                    stt_requests_total.labels(
                        language=self.language,
                        status='success'
                    ).inc()
            
            # NEW: Emit transcript via callback (structured output)
            if self.on_transcript:
                await self.on_transcript(transcript)
            
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}", exc_info=True)
    
    async def _on_speech_started(self, *args, **kwargs):
        """Handle speech started event."""
        logger.debug(f"Speech started: {self.session_id}")
        self.last_audio_time = datetime.utcnow()
    
    async def _on_error(self, *args, **kwargs):
        """Handle Deepgram error."""
        error = args[1] if len(args) > 1 else args[0]
        logger.error(f"Deepgram error: {error}")
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= 3:
            await self.reconnect()
    
    async def _on_close(self, *args, **kwargs):
        """Handle connection close."""
        logger.info(f"Deepgram connection closed: {self.session_id}")
        self.is_active = False
    
    async def reconnect(self) -> bool:
        """Reconnect to Deepgram."""
        if self.reconnect_count >= MAX_RECONNECT_ATTEMPTS:
            logger.error(
                f"Max reconnection attempts reached: {self.session_id}"
            )
            if METRICS_ENABLED:
                stt_reconnections_total.labels(result='failed').inc()
            return False
        
        self.reconnect_count += 1
        logger.info(
            f"Attempting reconnection {self.reconnect_count}/{MAX_RECONNECT_ATTEMPTS}"
        )
        
        # Stop supervisor temporarily
        await self.supervisor.stop()
        
        # Close existing connection
        await self.stop()
        
        # Wait before reconnecting
        await asyncio.sleep(RECONNECT_DELAY)
        
        # Restart connection
        success = await self.start()
        
        if success:
            logger.info(f"Reconnection successful: {self.session_id}")
            if METRICS_ENABLED:
                stt_reconnections_total.labels(result='success').inc()
            self.consecutive_errors = 0
        else:
            logger.error(f"Reconnection failed: {self.session_id}")
            if METRICS_ENABLED:
                stt_reconnections_total.labels(result='failed').inc()
        
        return success
    
    async def stop(self):
        """Stop the STT stream and supervisor."""
        try:
            # NEW: Stop supervisor first
            await self.supervisor.stop()
            
            if self.connection:
                await self.connection.finish()
                self.connection = None
            
            self.is_active = False
            
            # Track stream duration
            if self.start_time and METRICS_ENABLED:
                duration = (datetime.utcnow() - self.start_time).total_seconds()
                stt_stream_duration_seconds.observe(duration)
                stt_active_streams.dec()
            
            logger.info(
                f"Deepgram stream stopped: {self.session_id} | "
                f"Transcripts: {len(self.transcripts)} | "
                f"Words: {self.total_words} | "
                f"Reconnects: {self.reconnect_count}"
            )
            
        except Exception as e:
            logger.error(f"Error stopping stream: {str(e)}")
    
    def get_transcripts(self, final_only: bool = True) -> List[Dict[str, Any]]:
        """Get transcripts from this session."""
        if final_only:
            return list(self.transcripts)
        else:
            return list(self.transcripts) + list(self.partial_transcripts)
    
    def get_latest_transcript(self, final_only: bool = True) -> Optional[Dict[str, Any]]:
        """Get the most recent transcript."""
        if final_only and len(self.transcripts) > 0:
            return self.transcripts[-1]
        elif not final_only and len(self.partial_transcripts) > 0:
            return self.partial_transcripts[-1]
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get stream health metrics."""
        now = datetime.utcnow()
        
        # Calculate uptime
        uptime = 0
        if self.start_time:
            uptime = (now - self.start_time).total_seconds()
        
        # Time since last activity
        last_activity_ago = 0
        if self.last_activity:
            last_activity_ago = (now - self.last_activity).total_seconds()
        
        # NEW: Time since last audio/transcript
        last_audio_ago = 0
        if self.last_audio_time:
            last_audio_ago = (now - self.last_audio_time).total_seconds()
        
        last_transcript_ago = 0
        if self.last_transcript_time:
            last_transcript_ago = (now - self.last_transcript_time).total_seconds()
        
        # Silence ratio
        silence_ratio = 0
        if self.audio_chunks_processed > 0:
            silence_ratio = self.silence_chunks / self.audio_chunks_processed
        
        return {
            "is_active": self.is_active,
            "session_id": self.session_id,
            "uptime_seconds": uptime,
            "last_activity_seconds_ago": last_activity_ago,
            "last_audio_seconds_ago": last_audio_ago,  # NEW
            "last_transcript_seconds_ago": last_transcript_ago,  # NEW
            "transcripts_count": len(self.transcripts),
            "partial_transcripts_count": len(self.partial_transcripts),
            "total_words": self.total_words,
            "audio_chunks_processed": self.audio_chunks_processed,
            "total_audio_mb": self.total_audio_bytes / 1024 / 1024,
            "silence_ratio": silence_ratio,
            "reconnect_count": self.reconnect_count,
            "consecutive_errors": self.consecutive_errors,
            "language": self.language,
            "model": STT_MODEL,
            "supervisor_running": self.supervisor._is_running,  # NEW
        }


# Global stream registry
_active_streams: Dict[str, DeepgramSTTStream] = {}


async def start_stt_stream(
    session_id: str,
    language: str = "en-US",
    sample_rate: int = 16000,
    cancel_token: Optional[CancelToken] = None,
    on_transcript: Optional[Callable] = None,
    on_timeout: Optional[Callable] = None
) -> Optional[DeepgramSTTStream]:
    """
    Start a new Deepgram STT stream with hardening.
    
    Args:
        session_id: Unique session identifier
        language: Language code
        sample_rate: Audio sample rate in Hz
        cancel_token: Optional cancellation token
        on_transcript: Optional callback for transcripts
        on_timeout: Optional callback for timeouts
        
    Returns:
        DeepgramSTTStream instance or None on error
    """
    try:
        # Check if stream already exists
        if session_id in _active_streams:
            logger.warning(f"Stream already exists: {session_id}")
            return _active_streams[session_id]
        
        # Create new stream with cancellation support
        stream = DeepgramSTTStream(
            language=language,
            sample_rate=sample_rate,
            session_id=session_id,
            cancel_token=cancel_token,  # NEW
            on_transcript=on_transcript,  # NEW
            on_timeout=on_timeout  # NEW
        )
        
        # Start stream
        if await stream.start():
            _active_streams[session_id] = stream
            logger.info(f"STT stream started: {session_id}")
            return stream
        else:
            logger.error(f"Failed to start STT stream: {session_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error starting STT stream: {str(e)}", exc_info=True)
        return None


async def stop_stt_stream(session_id: str) -> bool:
    """Stop an active STT stream."""
    try:
        if session_id not in _active_streams:
            logger.warning(f"Stream not found: {session_id}")
            return False
        
        stream = _active_streams[session_id]
        await stream.stop()
        
        del _active_streams[session_id]
        
        logger.info(f"STT stream stopped: {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error stopping STT stream: {str(e)}")
        return False


def get_stt_stream(session_id: str) -> Optional[DeepgramSTTStream]:
    """Get an active STT stream."""
    return _active_streams.get(session_id)


def get_all_streams() -> List[DeepgramSTTStream]:
    """Get all active STT streams."""
    return list(_active_streams.values())


def get_stream_health() -> Dict[str, Any]:
    """Get health status of all streams."""
    return {
        "active_streams": len(_active_streams),
        "streams": {
            session_id: stream.get_health_status()
            for session_id, stream in _active_streams.items()
        }
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_deepgram():
        print("Deepgram STT Module - Production Hardened v4.0")
        print("=" * 60)
        print(f"Model: {STT_MODEL} (Deepgram Nova-2)")
        print(f"Cost: $0.0043/min")
        print(f"Min Confidence: {MIN_CONFIDENCE_THRESHOLD}")
        print(f"Min Length: {MIN_TRANSCRIPT_LENGTH}")
        print(f"Silence Timeout: {SILENCE_TIMEOUT}s")
        print(f"Stall Timeout: {STALL_TIMEOUT}s")
        print("=" * 60)
        
        # Test stream creation with callbacks
        async def on_transcript(transcript: TranscriptMetadata):
            print(f"📝 Transcript: {transcript.text} (confidence={transcript.confidence:.2f}, "
                  f"final={transcript.is_final}, endpoint={transcript.endpoint_detected})")
        
        async def on_timeout(reason: str, duration: float):
            print(f"⏱️  Timeout: {reason} ({duration:.1f}s)")
        
        cancel_token = CancelToken()
        
        stream = await start_stt_stream(
            "test_session",
            cancel_token=cancel_token,
            on_transcript=on_transcript,
            on_timeout=on_timeout
        )
        
        if stream:
            print(f"✅ Stream created: {stream.session_id}")
            
            # Simulate audio
            await asyncio.sleep(2)
            
            # Get health
            health = stream.get_health_status()
            print(f"Stream health: {health}")
            
            # Stop stream
            await stop_stt_stream("test_session")
            print("✅ Stream stopped")
        else:
            print("❌ Failed to create stream")
    
    asyncio.run(test_deepgram())
