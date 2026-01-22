"""
WebSocket Audio Gateway (Production Hardened)
==============================================
Pure transport layer for real-time audio streaming.

HARDENING UPDATES (v3.0):
✅ Removed business state (pure transport layer)
✅ Controller delegation for cleanup
✅ Buffer overflow force disconnect
✅ Duplicate event detection
✅ Malformed frame handling
✅ Guaranteed cleanup with timeouts
✅ Comprehensive error metrics

Version: 3.0.0 (Hardened)
Last Updated: 2026-01-22
"""

import os
import asyncio
import logging
import signal
import uuid
import time
import json
from typing import Dict, Optional, Any, Set
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# Configuration
MAX_AUDIO_BUFFER_SIZE = 1024 * 1024  # 1MB hard limit
MAX_AUDIO_CHUNK_SIZE = 64 * 1024    # 64KB
HEARTBEAT_INTERVAL = 10              # seconds
HEARTBEAT_TIMEOUT = 30               # seconds
CONNECTION_TIMEOUT = 300             # 5 minutes max idle
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BASE_DELAY = 1.0          # seconds
MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONCURRENT_CALLS", "50"))
CLEANUP_TIMEOUT = 5.0               # seconds for cleanup operations


# Prometheus Metrics
if METRICS_ENABLED:
    ws_connections_total = Counter(
        'ws_connections_total',
        'Total WebSocket connections',
        ['status']
    )
    ws_active_connections = Gauge(
        'ws_active_connections',
        'Currently active WebSocket connections'
    )
    ws_audio_bytes_total = Counter(
        'ws_audio_bytes_total',
        'Total audio bytes transferred',
        ['direction']
    )
    ws_message_latency = Histogram(
        'ws_message_latency_seconds',
        'Message processing latency',
        ['message_type']
    )
    ws_reconnections_total = Counter(
        'ws_reconnections_total',
        'Total reconnection attempts',
        ['result']
    )
    ws_heartbeat_failures = Counter(
        'ws_heartbeat_failures_total',
        'Total heartbeat failures'
    )
    # NEW METRICS (Hardening)
    ws_buffer_overflows_total = Counter(
        'ws_buffer_overflows_total',
        'Buffer overflow events'
    )
    ws_malformed_frames_total = Counter(
        'ws_malformed_frames_total',
        'Malformed frame events',
        ['frame_type']
    )
    ws_cleanup_failures_total = Counter(
        'ws_cleanup_failures_total',
        'Cleanup failures',
        ['component']
    )
    ws_duplicate_events_total = Counter(
        'ws_duplicate_events_total',
        'Duplicate event detections'
    )


_shutdown_event = asyncio.Event()
_active_connections: Set[str] = set()
_connection_pool: Dict[str, 'CallConnection'] = {}


class ConnectionMetrics:
    """Track connection quality metrics."""
    
    def __init__(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.reconnect_count = 0
        self.heartbeat_failures = 0
        self.avg_latency_ms = 0.0
        self.packet_loss_rate = 0.0
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self._latency_samples = []
    
    def record_latency(self, latency_ms: float):
        """Record message latency."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 100:
            self._latency_samples.pop(0)
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        uptime = (datetime.utcnow() - self.created_at).total_seconds()
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "reconnect_count": self.reconnect_count,
            "heartbeat_failures": self.heartbeat_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "packet_loss_rate": round(self.packet_loss_rate, 4),
            "uptime_seconds": round(uptime, 2)
        }


class CancellationToken:
    """Token for cancelling operations."""
    
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


class CallConnection:
    """
    Pure transport connection for a single call.
    
    CHANGES (v3.0 Hardening):
    - Removed business state (is_speaking, is_listening, transfer_*)
    - Added controller_ref for delegation
    - Added duplicate event detection
    - Enhanced cleanup with guaranteed completion
    """
    
    def __init__(
        self,
        call_id: str,
        websocket: WebSocket,
        controller_ref: Optional[Any] = None,  # NEW: for delegation
        request_id: Optional[str] = None
    ):
        self.call_id = call_id
        self.request_id = request_id or str(uuid.uuid4())
        self.websocket = websocket
        self.controller_ref = controller_ref  # NEW
        
        # REMOVED: Business state (is_speaking, is_listening, transfer_*)
        # These belong in CallController, not transport layer
        
        # Transport state only
        self.is_active = True
        
        # Cancellation
        self.tts_token: Optional[CancellationToken] = None
        self.stt_active = False
        
        # Audio buffering
        self.audio_buffer = bytearray()
        self.audio_buffer_lock = asyncio.Lock()
        self.last_audio_time: Optional[datetime] = None
        
        # Heartbeat
        self.last_heartbeat = datetime.utcnow()
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.heartbeat_failures = 0
        
        # Metrics
        self.metrics = ConnectionMetrics()
        
        # Reconnection
        self.reconnect_attempts = 0
        self.last_reconnect_time: Optional[datetime] = None
        
        # Lifecycle
        self.start_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # NEW: Duplicate event detection
        self._processed_event_ids: Set[str] = set()
        
        # NEW: Cleanup tracking
        self._cleanup_started = False
        self._cleanup_completed = False
        
        # Track in pool
        _active_connections.add(call_id)
        _connection_pool[call_id] = self
        
        if METRICS_ENABLED:
            ws_active_connections.inc()
            ws_connections_total.labels(status='connected').inc()
        
        logger.info(f"CallConnection created: {call_id} (request_id: {self.request_id})")
    
    async def start_heartbeat(self):
        """Start heartbeat monitoring."""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self):
        """Monitor connection health with heartbeat."""
        try:
            while self.is_active:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                
                # Check if heartbeat timed out
                elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
                if elapsed > HEARTBEAT_TIMEOUT:
                    self.heartbeat_failures += 1
                    self.metrics.heartbeat_failures += 1
                    
                    if METRICS_ENABLED:
                        ws_heartbeat_failures.inc()
                    
                    logger.warning(
                        f"Heartbeat timeout for {self.call_id} "
                        f"(elapsed: {elapsed:.1f}s, failures: {self.heartbeat_failures})"
                    )
                    
                    # Disconnect after 3 failures
                    if self.heartbeat_failures >= 3:
                        logger.error(f"Too many heartbeat failures for {self.call_id}, disconnecting")
                        self.is_active = False
                        await self.cleanup(reason="heartbeat_timeout")
                        break
                
                # Send ping
                try:
                    await self.websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Failed to send ping: {str(e)}")
                    await self.cleanup(reason="ping_failure")
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat loop error: {str(e)}")
    
    async def _attempt_reconnection(self):
        """Attempt to reconnect the WebSocket."""
        if self.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            logger.error(f"Max reconnection attempts reached for {self.call_id}")
            await self.cleanup(reason="max_reconnect_attempts")
            return
        
        self.reconnect_attempts += 1
        self.metrics.reconnect_count += 1
        delay = min(RECONNECT_BASE_DELAY * (2 ** self.reconnect_attempts), 30)
        
        logger.info(
            f"Attempting reconnection #{self.reconnect_attempts} for {self.call_id} "
            f"(delay: {delay:.1f}s)"
        )
        
        await asyncio.sleep(delay)
        
        # Mark last reconnect time
        self.last_reconnect_time = datetime.utcnow()
        
        if METRICS_ENABLED:
            ws_reconnections_total.labels(result='attempted').inc()
        
        logger.warning(f"Reconnection attempted for {self.call_id} - awaiting client reconnect")
    
    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.heartbeat_failures = 0
    
    def record_activity(self):
        """Record connection activity."""
        self.last_activity = datetime.utcnow()
        self.metrics.last_activity = datetime.utcnow()
    
    async def send_audio(self, audio_data: bytes):
        """Send audio with metrics tracking."""
        start_time = time.time()
        try:
            await self.websocket.send_bytes(audio_data)
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(audio_data)
            
            if METRICS_ENABLED:
                ws_audio_bytes_total.labels(direction='sent').inc(len(audio_data))
                latency = (time.time() - start_time) * 1000
                ws_message_latency.labels(message_type='audio').observe(latency / 1000)
                self.metrics.record_latency(latency)
        
        except Exception as e:
            logger.error(f"Failed to send audio: {str(e)}")
            raise
    
    async def receive_audio(self) -> Optional[bytes]:
        """Receive audio with metrics tracking."""
        start_time = time.time()
        try:
            data = await self.websocket.receive_bytes()
            self.metrics.messages_received += 1
            self.metrics.bytes_received += len(data)
            self.record_activity()
            
            if METRICS_ENABLED:
                ws_audio_bytes_total.labels(direction='received').inc(len(data))
                latency = (time.time() - start_time) * 1000
                ws_message_latency.labels(message_type='audio').observe(latency / 1000)
                self.metrics.record_latency(latency)
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to receive audio: {str(e)}")
            return None
    
    def is_connection_stale(self) -> bool:
        """Check if connection is stale."""
        idle_time = (datetime.utcnow() - self.last_activity).total_seconds()
        return idle_time > CONNECTION_TIMEOUT
    
    def is_duplicate_event(self, event_id: str) -> bool:
        """
        NEW: Check if event has already been processed.
        
        Args:
            event_id: Unique event identifier
            
        Returns:
            True if duplicate
        """
        if event_id in self._processed_event_ids:
            logger.warning(f"Duplicate event detected: {self.call_id} event_id={event_id}")
            
            if METRICS_ENABLED:
                ws_duplicate_events_total.inc()
            
            return True
        
        self._processed_event_ids.add(event_id)
        
        # Limit set size (keep last 1000 events)
        if len(self._processed_event_ids) > 1000:
            self._processed_event_ids = set(list(self._processed_event_ids)[-1000:])
        
        return False
    
    def get_connection_quality(self) -> Dict[str, Any]:
        """Get connection quality metrics."""
        return {
            "call_id": self.call_id,
            "request_id": self.request_id,
            "is_active": self.is_active,
            "heartbeat_failures": self.heartbeat_failures,
            "reconnect_attempts": self.reconnect_attempts,
            "is_stale": self.is_connection_stale(),
            "metrics": self.metrics.get_stats()
        }
    
    async def cleanup(self, reason: str = "normal"):
        """
        ENHANCED: Guaranteed cleanup with controller delegation.
        
        Must complete even if components fail.
        """
        if self._cleanup_started:
            logger.debug(f"Cleanup already started for {self.call_id}")
            return
        
        self._cleanup_started = True
        
        logger.info(f"Cleaning up connection {self.call_id} (reason: {reason})")
        
        self.is_active = False
        cleanup_errors = []
        
        # 1. Cancel heartbeat (with timeout)
        try:
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await asyncio.wait_for(self.heartbeat_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        except Exception as e:
            cleanup_errors.append(f"heartbeat: {e}")
            if METRICS_ENABLED:
                ws_cleanup_failures_total.labels(component='heartbeat').inc()
        
        # 2. NEW: Stop STT via controller
        try:
            if self.controller_ref and hasattr(self.controller_ref, 'stop_stt'):
                await asyncio.wait_for(
                    self.controller_ref.stop_stt(),
                    timeout=2.0
                )
        except asyncio.TimeoutError:
            logger.warning(f"STT stop timeout for {self.call_id}")
            cleanup_errors.append("stt: timeout")
            if METRICS_ENABLED:
                ws_cleanup_failures_total.labels(component='stt').inc()
        except Exception as e:
            cleanup_errors.append(f"stt: {e}")
            if METRICS_ENABLED:
                ws_cleanup_failures_total.labels(component='stt').inc()
        
        # 3. NEW: Stop TTS via controller
        try:
            if self.controller_ref and hasattr(self.controller_ref, 'stop_tts'):
                await asyncio.wait_for(
                    self.controller_ref.stop_tts(),
                    timeout=2.0
                )
        except asyncio.TimeoutError:
            logger.warning(f"TTS stop timeout for {self.call_id}")
            cleanup_errors.append("tts: timeout")
            if METRICS_ENABLED:
                ws_cleanup_failures_total.labels(component='tts').inc()
        except Exception as e:
            cleanup_errors.append(f"tts: {e}")
            if METRICS_ENABLED:
                ws_cleanup_failures_total.labels(component='tts').inc()
        
        # 4. NEW: Cancel AI via controller
        try:
            if self.controller_ref and hasattr(self.controller_ref, 'cancel_ai'):
                await asyncio.wait_for(
                    self.controller_ref.cancel_ai(),
                    timeout=2.0
                )
        except asyncio.TimeoutError:
            logger.warning(f"AI cancel timeout for {self.call_id}")
            cleanup_errors.append("ai: timeout")
            if METRICS_ENABLED:
                ws_cleanup_failures_total.labels(component='ai').inc()
        except Exception as e:
            cleanup_errors.append(f"ai: {e}")
            if METRICS_ENABLED:
                ws_cleanup_failures_total.labels(component='ai').inc()
        
        # 5. Cancel TTS token (existing)
        try:
            if self.tts_token:
                await self.tts_token.cancel()
        except Exception as e:
            cleanup_errors.append(f"tts_token: {e}")
        
        # 6. NEW: Clear buffers (release memory)
        try:
            async with self.audio_buffer_lock:
                self.audio_buffer.clear()
        except Exception as e:
            cleanup_errors.append(f"buffers: {e}")
        
        # 7. NEW: Clear event tracking
        try:
            self._processed_event_ids.clear()
        except Exception as e:
            cleanup_errors.append(f"events: {e}")
        
        # 8. Remove from tracking
        _active_connections.discard(self.call_id)
        _connection_pool.pop(self.call_id, None)
        
        if METRICS_ENABLED:
            ws_active_connections.dec()
            ws_connections_total.labels(status='disconnected').inc()
        
        # 9. Close WebSocket (with timeout)
        try:
            await asyncio.wait_for(self.websocket.close(), timeout=2.0)
        except Exception:
            pass  # Expected if already closed
        
        self._cleanup_completed = True
        
        if cleanup_errors:
            logger.warning(
                f"Cleanup completed with errors for {self.call_id}: {cleanup_errors}"
            )
        else:
            logger.info(f"Connection cleaned up: {self.call_id}")


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("WebSocket Gateway starting up (Hardened v3.0)")
    
    # Startup
    yield
    
    # Shutdown
    logger.info("WebSocket Gateway shutting down")
    _shutdown_event.set()
    
    # Clean up all connections with timeout
    cleanup_tasks = []
    for call_id in list(_connection_pool.keys()):
        conn = _connection_pool.get(call_id)
        if conn:
            cleanup_tasks.append(
                asyncio.create_task(conn.cleanup(reason="shutdown"))
            )
    
    if cleanup_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*cleanup_tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("Shutdown cleanup timeout - some connections may not have cleaned up")


app = FastAPI(title="Voice AI WebSocket Gateway", version="3.0.0", lifespan=lifespan)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "active_connections": len(_active_connections),
        "max_connections": MAX_CONCURRENT_CONNECTIONS,
        "version": "3.0.0"
    })


# Detailed status endpoint
@app.get("/status")
async def status():
    """Get detailed status."""
    connections_status = []
    for call_id, conn in _connection_pool.items():
        connections_status.append(conn.get_connection_quality())
    
    return JSONResponse({
        "active_connections": len(_active_connections),
        "max_connections": MAX_CONCURRENT_CONNECTIONS,
        "utilization": len(_active_connections) / MAX_CONCURRENT_CONNECTIONS,
        "connections": connections_status
    })


# Metrics endpoint (Prometheus)
if METRICS_ENABLED:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )


# WebSocket endpoint
@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str, request: Request):
    """WebSocket connection handler with controller injection."""
    
    # Check connection limit
    if len(_active_connections) >= MAX_CONCURRENT_CONNECTIONS:
        logger.warning(f"Connection limit reached, rejecting {call_id}")
        await websocket.close(code=1008, reason="Connection limit reached")
        return
    
    # Extract request ID for tracing
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    logger.info(f"WebSocket connection attempt: {call_id} (request_id: {request_id})")
    
    await websocket.accept()
    
    # NEW: Get controller (would use DI in production)
    # For now, controller_ref=None (backwards compatible)
    # In production: controller = get_controller(call_id)
    controller_ref = None
    
    connection = CallConnection(
        call_id,
        websocket,
        controller_ref=controller_ref,  # NEW
        request_id=request_id
    )
    
    try:
        # Start heartbeat
        await connection.start_heartbeat()
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "call_id": call_id,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Handle messages
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while connection.is_active:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=1.0
                )
                
                # Reset error counter
                consecutive_errors = 0
                
                # Handle different message types
                if "text" in message:
                    data = message["text"]
                    await _handle_control_message(connection, data)
                
                elif "bytes" in message:
                    await _handle_audio_data(connection, message["bytes"])
                
                connection.update_heartbeat()
                
            except asyncio.TimeoutError:
                # Check for stale connection
                if connection.is_connection_stale():
                    logger.warning(f"Stale connection detected: {call_id}")
                    break
                continue
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {call_id}")
                break
            
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Message handling error for {call_id}: {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors for {call_id}")
                    break
    
    except Exception as e:
        logger.error(f"WebSocket error for {call_id}: {str(e)}")
    
    finally:
        # GUARANTEED cleanup
        try:
            await asyncio.wait_for(
                connection.cleanup(reason="connection_closed"),
                timeout=CLEANUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Cleanup timeout for {call_id}")
        except Exception as e:
            logger.error(f"Cleanup error for {call_id}: {str(e)}")


async def _handle_control_message(connection: CallConnection, data: str):
    """
    ENHANCED: Handle control messages with validation.
    """
    start_time = time.time()
    
    try:
        # NEW: Validate JSON
        try:
            msg = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Malformed JSON for {connection.call_id}: {str(e)}")
            if METRICS_ENABLED:
                ws_malformed_frames_total.labels(frame_type='invalid_json').inc()
            return  # Reject
        
        # NEW: Validate structure
        if not isinstance(msg, dict):
            logger.error(f"Invalid message structure for {connection.call_id}: {type(msg)}")
            if METRICS_ENABLED:
                ws_malformed_frames_total.labels(frame_type='invalid_structure').inc()
            return  # Reject
        
        msg_type = msg.get("type")
        
        if not msg_type:
            logger.warning(f"Missing message type for {connection.call_id}")
            return  # Reject
        
        # NEW: Check for duplicates
        event_id = msg.get("event_id")
        if event_id and connection.is_duplicate_event(event_id):
            logger.warning(f"Ignoring duplicate event: {connection.call_id} type={msg_type} id={event_id}")
            return  # Skip processing
        
        # Handle message types
        if msg_type == "pong":
            connection.update_heartbeat()
        
        elif msg_type == "start":
            # Just log - don't store business state here
            logger.info(f"Start event for {connection.call_id}")
            # Controller would handle actual start logic
        
        elif msg_type == "stop":
            connection.is_active = False
        
        connection.record_activity()
        
        # Track latency
        if METRICS_ENABLED:
            latency = (time.time() - start_time) * 1000
            ws_message_latency.labels(message_type='control').observe(latency / 1000)
    
    except Exception as e:
        logger.error(f"Failed to handle control message: {str(e)}")


async def _handle_audio_data(connection: CallConnection, audio_bytes: bytes):
    """
    ENHANCED: Handle incoming audio data with overflow protection.
    """
    try:
        # Validate audio size
        if len(audio_bytes) > MAX_AUDIO_CHUNK_SIZE:
            logger.warning(
                f"Audio chunk too large for {connection.call_id}: "
                f"{len(audio_bytes)} > {MAX_AUDIO_CHUNK_SIZE}"
            )
            
            if METRICS_ENABLED:
                ws_malformed_frames_total.labels(frame_type='oversized_audio').inc()
            
            # Truncate to max
            audio_bytes = audio_bytes[:MAX_AUDIO_CHUNK_SIZE]
        
        # Buffer audio with overflow protection
        async with connection.audio_buffer_lock:
            new_size = len(connection.audio_buffer) + len(audio_bytes)
            
            # NEW: Force disconnect on buffer overflow
            if new_size > MAX_AUDIO_BUFFER_SIZE:
                logger.error(
                    f"BUFFER OVERFLOW for {connection.call_id}: "
                    f"buffer={len(connection.audio_buffer)} "
                    f"incoming={len(audio_bytes)} "
                    f"max={MAX_AUDIO_BUFFER_SIZE}"
                )
                
                if METRICS_ENABLED:
                    ws_buffer_overflows_total.inc()
                
                # Force disconnect on runaway buffer
                connection.is_active = False
                asyncio.create_task(connection.cleanup(reason="buffer_overflow"))
                return
            
            connection.audio_buffer.extend(audio_bytes)
        
        connection.last_audio_time = datetime.utcnow()
        connection.record_activity()
    
    except Exception as e:
        logger.error(f"Failed to handle audio data: {str(e)}")


# Signal handlers
def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown")
    _shutdown_event.set()


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting WebSocket Gateway (Hardened v3.0)")
    
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )
