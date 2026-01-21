"""
WebSocket Audio Gateway (Production)
=====================================
Real-time audio streaming gateway for voice calls.
Handles audio I/O, barge-in, call lifecycle, and fault tolerance.

NO BUSINESS LOGIC - Pure audio orchestration only.
"""

import os
import asyncio
import logging
import signal
from typing import Dict, Optional, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


MAX_AUDIO_BUFFER_SIZE = 1024 * 1024  # 1MB
MAX_AUDIO_CHUNK_SIZE = 64 * 1024  # 64KB
HEARTBEAT_INTERVAL = 10  # seconds
HEARTBEAT_TIMEOUT = 30  # seconds
SILENCE_THRESHOLD = 500  # ms before processing audio


_shutdown_event = asyncio.Event()


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
    """Manages a single call connection with full lifecycle control."""
    
    def __init__(self, call_id: str, websocket: WebSocket):
        self.call_id = call_id
        self.websocket = websocket
        self.restaurant_id: Optional[str] = None
        self.customer_phone: Optional[str] = None
        self.detected_language: Optional[str] = None
        
        # State flags
        self.is_active = True
        self.is_speaking = False
        self.is_listening = False
        self.transfer_requested = False
        self.transfer_in_progress = False
        
        # Cancellation tokens
        self.tts_token: Optional[CancellationToken] = None
        self.stt_active = False
        
        # Audio buffering
        self.audio_buffer = bytearray()
        self.audio_buffer_lock = asyncio.Lock()
        self.last_audio_time: Optional[datetime] = None
        
        # Heartbeat
        self.last_heartbeat = datetime.utcnow()
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Cleanup tracking
        self.start_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        logger.info(f"CallConnection created: {call_id}")
    
    async def start_heartbeat(self):
        """Start heartbeat monitoring."""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self):
        """Monitor connection health."""
        try:
            while self.is_active:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                
                elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
                
                if elapsed > HEARTBEAT_TIMEOUT:
                    logger.warning(f"Heartbeat timeout for call {self.call_id}")
                    await self.close("heartbeat_timeout")
                    break
                
                try:
                    await self.websocket.send_json({"event": "ping"})
                except Exception as e:
                    logger.error(f"Heartbeat send failed for {self.call_id}: {str(e)}")
                    await self.close("heartbeat_failed")
                    break
                    
        except asyncio.CancelledError:
            pass
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    async def start_tts(self) -> Optional[CancellationToken]:
        """Start TTS with cancellation support."""
        if self.transfer_requested or self.transfer_in_progress:
            logger.warning(f"TTS blocked - transfer active: {self.call_id}")
            return None
        
        await self.stop_tts()
        
        self.is_speaking = True
        self.is_listening = False
        self.tts_token = CancellationToken()
        
        logger.debug(f"TTS started for {self.call_id}")
        return self.tts_token
    
    async def stop_tts(self):
        """Immediately stop TTS (barge-in)."""
        if self.tts_token and self.is_speaking:
            await self.tts_token.cancel()
            logger.info(f"Barge-in: TTS cancelled for {self.call_id}")
        
        self.is_speaking = False
        self.tts_token = None
    
    def start_listening(self):
        """Switch to listening mode."""
        if self.transfer_requested or self.transfer_in_progress:
            return
        
        self.is_speaking = False
        self.is_listening = True
        logger.debug(f"Listening mode for {self.call_id}")
    
    async def add_audio_chunk(self, chunk: bytes) -> bool:
        """
        Add audio chunk to buffer with backpressure protection.
        
        Returns:
            True if added, False if dropped (buffer full)
        """
        if not chunk:
            return True
        
        if len(chunk) > MAX_AUDIO_CHUNK_SIZE:
            logger.warning(f"Oversized audio chunk rejected: {len(chunk)} bytes")
            return False
        
        async with self.audio_buffer_lock:
            if len(self.audio_buffer) + len(chunk) > MAX_AUDIO_BUFFER_SIZE:
                logger.warning(
                    f"Audio buffer full for {self.call_id}, "
                    f"dropping {len(chunk)} bytes"
                )
                return False
            
            self.audio_buffer.extend(chunk)
            self.last_audio_time = datetime.utcnow()
            self.last_activity = datetime.utcnow()
            
            return True
    
    async def get_and_clear_audio(self) -> bytes:
        """Get buffered audio and clear buffer."""
        async with self.audio_buffer_lock:
            audio = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            return audio
    
    def mark_transfer_requested(self):
        """Mark that transfer has been requested."""
        self.transfer_requested = True
        self.is_listening = False
        logger.info(f"Transfer requested for {self.call_id}")
    
    async def close(self, reason: str = "normal"):
        """Close connection and cleanup resources."""
        if not self.is_active:
            return
        
        logger.info(f"Closing call {self.call_id}, reason: {reason}")
        
        self.is_active = False
        
        # Cancel TTS
        await self.stop_tts()
        
        # Cancel heartbeat
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Stop STT if active
        if self.stt_active:
            try:
                from stt_streaming import stop_stream
                await stop_stream(self.call_id)
                self.stt_active = False
            except Exception as e:
                logger.error(f"Error stopping STT: {str(e)}")
        
        # Notify main.py of call end
        try:
            from main import handle_call_end
            await handle_call_end(self.call_id)
        except Exception as e:
            logger.error(f"Error in call end handler: {str(e)}")
        
        logger.info(f"Call {self.call_id} closed")


_active_connections: Dict[str, CallConnection] = {}
_cleanup_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("WebSocket server starting")
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown")
        _shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start cleanup task
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    
    yield
    
    # Shutdown
    logger.info("WebSocket server shutting down")
    _shutdown_event.set()
    
    # Close all active connections
    tasks = []
    for conn in list(_active_connections.values()):
        tasks.append(conn.close("server_shutdown"))
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Cancel cleanup task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    
    logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_calls": len(_active_connections),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/inbound_call")
async def handle_inbound_call(request: Request):
    """Handle inbound call from Twilio."""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        
        if not call_sid:
            return Response(
                content="<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Hangup/></Response>",
                media_type="application/xml"
            )
        
        from config import get_config
        config = get_config()
        base_url = config.server.base_url
        
        ws_url = f"{base_url}/ws/{call_sid}".replace("http://", "ws://").replace("https://", "wss://")
        
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}"/>
    </Connect>
</Response>"""
        
        logger.info(f"Inbound call: {call_sid}, WebSocket: {ws_url}")
        
        return Response(content=twiml, media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error handling inbound call: {str(e)}", exc_info=True)
        return Response(
            content="<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Hangup/></Response>",
            media_type="application/xml"
        )


@app.post("/call_status")
async def handle_call_status(request: Request):
    """Handle call status updates from Twilio."""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        
        logger.info(f"Call status: {call_sid} -> {call_status}")
        
        if call_status in ["completed", "failed", "busy", "no-answer"]:
            conn = _active_connections.get(call_sid)
            if conn:
                await conn.close(f"call_status_{call_status}")
                _active_connections.pop(call_sid, None)
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error handling call status: {str(e)}")
        return {"status": "error", "error": str(e)}


@app.websocket("/ws/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    """WebSocket endpoint for audio streaming."""
    await websocket.accept()
    
    conn = CallConnection(call_sid, websocket)
    _active_connections[call_sid] = conn
    
    try:
        await conn.start_heartbeat()
        
        # Start call
        await _handle_call_start(conn)
        
        # Main message loop
        async for message in websocket.iter_json():
            if _shutdown_event.is_set():
                break
            
            if not conn.is_active:
                break
            
            await _process_message(conn, message)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_sid}")
    except Exception as e:
        logger.error(f"WebSocket error for {call_sid}: {str(e)}", exc_info=True)
    finally:
        await conn.close("websocket_closed")
        _active_connections.pop(call_sid, None)


async def _handle_call_start(conn: CallConnection):
    """Handle call start event."""
    try:
        # Extract restaurant ID (from Twilio custom params if available)
        conn.restaurant_id = os.getenv("DEFAULT_RESTAURANT_ID", "rest_001")
        
        # Start STT stream
        from stt_streaming import start_stream
        await start_stream(conn.call_id, None)
        conn.stt_active = True
        
        # Notify main.py
        from main import handle_call_start
        response = await handle_call_start(
            call_id=conn.call_id,
            restaurant_id=conn.restaurant_id
        )
        
        # Play greeting
        greeting = response.get("greeting", "")
        greeting_lang = response.get("language", "en")
        
        if greeting:
            token = await conn.start_tts()
            if token:
                from tts_streaming import stream_tts
                await stream_tts(
                    text=greeting,
                    language_code=greeting_lang,
                    websocket=conn.websocket,
                    call_id=conn.call_id,
                    cancel_event=token
                )
            
            conn.start_listening()
        
    except Exception as e:
        logger.error(f"Error in call start: {str(e)}", exc_info=True)


async def _process_message(conn: CallConnection, message: Dict[str, Any]):
    """Process WebSocket message."""
    try:
        event = message.get("event")
        
        if not event:
            return
        
        conn.update_heartbeat()
        
        if event == "start":
            await _handle_start_event(conn, message)
        
        elif event == "media":
            await _handle_media_event(conn, message)
        
        elif event == "stop":
            await conn.close("stream_stopped")
        
        elif event == "pong":
            pass
        
        else:
            logger.debug(f"Unknown event: {event}")
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)


async def _handle_start_event(conn: CallConnection, message: Dict[str, Any]):
    """Handle stream start event."""
    try:
        start_data = message.get("start", {})
        custom_params = start_data.get("customParameters", {})
        
        if "From" in start_data:
            conn.customer_phone = start_data["From"]
        
        logger.info(f"Stream started for {conn.call_id}")
        
    except Exception as e:
        logger.error(f"Error in start event: {str(e)}")


async def _handle_media_event(conn: CallConnection, message: Dict[str, Any]):
    """Handle incoming audio media."""
    try:
        if conn.transfer_requested or conn.transfer_in_progress:
            return
        
        media = message.get("media", {})
        payload = media.get("payload")
        
        if not payload:
            return
        
        # Decode audio
        import base64
        try:
            audio_chunk = base64.b64decode(payload)
        except Exception as e:
            logger.warning(f"Failed to decode audio: {str(e)}")
            return
        
        # Barge-in detection
        if conn.is_speaking:
            await conn.stop_tts()
        
        # Buffer audio
        added = await conn.add_audio_chunk(audio_chunk)
        
        if not added:
            return
        
        # Check for silence
        if conn.last_audio_time:
            silence_ms = (datetime.utcnow() - conn.last_audio_time).total_seconds() * 1000
            
            if silence_ms >= SILENCE_THRESHOLD and conn.is_listening:
                await _process_buffered_audio(conn)
        
        # Feed to STT stream
        if conn.stt_active:
            from stt_streaming import feed_audio, get_transcript
            await feed_audio(conn.call_id, audio_chunk)
            
            # Check for transcript
            transcript = await get_transcript(conn.call_id)
            
            if transcript and transcript.get("is_final"):
                await _handle_transcript(conn, transcript)
        
    except Exception as e:
        logger.error(f"Error in media event: {str(e)}", exc_info=True)


async def _process_buffered_audio(conn: CallConnection):
    """Process buffered audio through STT."""
    try:
        audio_bytes = await conn.get_and_clear_audio()
        
        if len(audio_bytes) < 1000:
            return
        
        # Legacy batch processing (fallback)
        from stt_streaming import transcribe_audio
        result = await transcribe_audio(audio_bytes, conn.detected_language)
        
        text = result.get("text", "").strip()
        
        if text:
            await _handle_user_input(conn, text, result.get("language_code"))
        
    except Exception as e:
        logger.error(f"Error processing audio buffer: {str(e)}")


async def _handle_transcript(conn: CallConnection, transcript: Dict[str, Any]):
    """Handle final transcript from streaming STT."""
    try:
        text = transcript.get("text", "").strip()
        language = transcript.get("language_code")
        
        if not text:
            return
        
        logger.info(f"Transcript for {conn.call_id}: {text}")
        
        if not conn.detected_language and language:
            conn.detected_language = language
        
        await _handle_user_input(conn, text, language)
        
    except Exception as e:
        logger.error(f"Error handling transcript: {str(e)}")


async def _handle_user_input(conn: CallConnection, text: str, language: Optional[str]):
    """Handle user text input."""
    try:
        from main import handle_user_text
        
        response = await handle_user_text(
            text=text,
            call_id=conn.call_id,
            detected_language=language
        )
        
        # Check for transfer
        if response.get("actions", {}).get("transfer_requested"):
            await _execute_transfer(conn, response["actions"].get("transfer_details", {}))
            return
        
        # Get AI response
        response_text = response.get("response_text", "")
        response_lang = response.get("language", "en")
        
        if not response_text:
            return
        
        # Speak response
        token = await conn.start_tts()
        
        if not token:
            return
        
        from tts_streaming import stream_tts
        success = await stream_tts(
            text=response_text,
            language_code=response_lang,
            websocket=conn.websocket,
            call_id=conn.call_id,
            cancel_event=token
        )
        
        if success:
            conn.start_listening()
        
    except Exception as e:
        logger.error(f"Error handling user input: {str(e)}", exc_info=True)


async def _execute_transfer(conn: CallConnection, transfer_details: Dict[str, Any]):
    """Execute call transfer."""
    try:
        transfer_number = transfer_details.get("transfer_number")
        reason = transfer_details.get("reason", "Customer request")
        
        if not transfer_number:
            logger.error("Transfer requested but no number provided")
            return
        
        logger.info(f"Executing transfer for {conn.call_id} to {transfer_number}")
        
        conn.transfer_in_progress = True
        conn.mark_transfer_requested()
        
        # Stop TTS
        await conn.stop_tts()
        
        # Play handoff message
        handoff = "One moment please, I'm transferring you to a team member."
        
        if conn.detected_language and conn.detected_language != "en":
            try:
                from translate import from_english
                handoff = await from_english(handoff, conn.detected_language)
            except:
                pass
        
        token = CancellationToken()
        from tts_streaming import stream_tts
        await stream_tts(
            text=handoff,
            language_code=conn.detected_language or "en",
            websocket=conn.websocket,
            call_id=conn.call_id,
            cancel_event=token
        )
        
        await asyncio.sleep(0.5)
        
        # Execute Twilio transfer
        from twilio.rest import Client
        from config import get_config
        
        config = get_config()
        client = Client(config.twilio.account_sid, config.twilio.auth_token)
        
        call = client.calls(conn.call_id).update(
            twiml=f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Number>{transfer_number}</Number>
    </Dial>
</Response>'''
        )
        
        logger.info(f"Transfer initiated: {conn.call_id}")
        
        # Log to database
        from db import db
        db.store_call_log({
            "restaurant_id": conn.restaurant_id,
            "caller_phone": conn.customer_phone or "unknown",
            "call_sid": conn.call_id,
            "direction": "inbound",
            "status": "transferred",
            "transcript": f"Transferred to {transfer_number}. Reason: {reason}"
        })
        
        await conn.close("transferred")
        
    except Exception as e:
        logger.error(f"Transfer failed: {str(e)}", exc_info=True)
        
        # Inform user
        error_msg = "I apologize, but I'm unable to transfer your call right now."
        
        if conn.detected_language and conn.detected_language != "en":
            try:
                from translate import from_english
                error_msg = await from_english(error_msg, conn.detected_language)
            except:
                pass
        
        token = CancellationToken()
        from tts_streaming import stream_tts
        await stream_tts(
            text=error_msg,
            language_code=conn.detected_language or "en",
            websocket=conn.websocket,
            call_id=conn.call_id,
            cancel_event=token
        )


async def _cleanup_loop():
    """Background cleanup of stale connections."""
    while not _shutdown_event.is_set():
        try:
            await asyncio.sleep(60)
            
            now = datetime.utcnow()
            stale = []
            
            for call_id, conn in _active_connections.items():
                idle_time = (now - conn.last_activity).total_seconds()
                
                if idle_time > 300:
                    stale.append(call_id)
            
            for call_id in stale:
                logger.warning(f"Cleaning up stale connection: {call_id}")
                conn = _active_connections.get(call_id)
                if conn:
                    await conn.close("stale_timeout")
                _active_connections.pop(call_id, None)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from config import get_config
    config = get_config()
    
    logger.info("Starting WebSocket Audio Gateway")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.port}")
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info"
    )
