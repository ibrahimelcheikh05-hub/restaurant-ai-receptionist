"""
WebSocket Audio Gateway
=======================
Real-time audio streaming server using Twilio and Vocode.
Handles audio I/O, STT/TTS coordination, and barge-in detection.

NO BUSINESS LOGIC - Pure audio orchestration only.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import uvicorn

from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.server.base import TelephonyServer
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationConfig
)

# Import our modules (business logic lives here)
from stt import transcribe_audio
from tts import stream_tts, CancellationToken
from detect import detect_language
from main import handle_call_start, handle_user_text, handle_call_end


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
BASE_URL = os.getenv("BASE_URL", f"http://{HOST}:{PORT}")

# Restaurant Configuration (default)
DEFAULT_RESTAURANT_ID = os.getenv("DEFAULT_RESTAURANT_ID", "rest_001")

# Audio Configuration
AUDIO_ENCODING = "linear16"
SAMPLE_RATE = 16000  # Hz
CHUNK_SIZE = 4096  # Bytes


# ============================================================================
# CALL STATE MANAGEMENT
# ============================================================================

class CallSession:
    """
    Manages state for a single active call.
    Tracks audio streaming, barge-in, and TTS state.
    """
    
    def __init__(self, call_id: str, restaurant_id: str):
        self.call_id = call_id
        self.restaurant_id = restaurant_id
        self.customer_phone: Optional[str] = None
        self.detected_language: Optional[str] = None
        
        # Audio streaming state
        self.is_speaking = False  # TTS is playing
        self.is_listening = False  # Listening for user input
        self.tts_token: Optional[CancellationToken] = None
        
        # Transfer state
        self.transfer_requested = False
        self.transfer_in_progress = False
        
        # Timing
        self.start_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Barge-in detection
        self.barge_in_detected = False
        self.audio_buffer = bytearray()
        
        logger.info(f"CallSession created: {call_id}")
    
    async def start_tts(self) -> CancellationToken:
        """Start TTS playback with cancellation support."""
        # Don't start TTS if transfer is requested
        if self.transfer_requested or self.transfer_in_progress:
            logger.warning(f"TTS blocked - transfer active: {self.call_id}")
            return None
        
        self.is_speaking = True
        self.is_listening = False
        self.tts_token = CancellationToken()
        self.barge_in_detected = False
        return self.tts_token
    
    async def stop_tts(self):
        """Stop TTS playback (barge-in)."""
        if self.tts_token and self.is_speaking:
            await self.tts_token.cancel()
            self.barge_in_detected = True
            logger.info(f"Barge-in detected for call {self.call_id}")
        
        self.is_speaking = False
        self.is_listening = True
    
    def mark_transfer_requested(self):
        """Mark that transfer has been requested."""
        self.transfer_requested = True
        self.is_listening = False
        logger.info(f"Transfer requested for call {self.call_id}")
    
    def start_listening(self):
        """Switch to listening mode."""
        # Don't listen if transfer is active
        if self.transfer_requested or self.transfer_in_progress:
            return
        
        self.is_speaking = False
        self.is_listening = True
        self.audio_buffer.clear()
    
    def add_audio_chunk(self, chunk: bytes):
        """Buffer audio chunk for STT."""
        self.audio_buffer.extend(chunk)
        self.last_activity = datetime.utcnow()
    
    def get_and_clear_audio(self) -> bytes:
        """Get buffered audio and clear buffer."""
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


# Active call sessions
active_calls: Dict[str, CallSession] = {}


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Voice Agent WebSocket Server")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_calls": len(active_calls),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# TWILIO WEBHOOK HANDLERS
# ============================================================================

@app.post("/inbound_call")
async def inbound_call(
    CallSid: str,
    From: str,
    To: str
):
    """
    Handle inbound Twilio call.
    Returns TwiML to connect call to WebSocket.
    """
    logger.info(f"Inbound call: {CallSid} from {From}")
    
    # Return TwiML to connect to our WebSocket
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{BASE_URL}/ws/{CallSid}" />
    </Connect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.post("/call_status")
async def call_status(
    CallSid: str,
    CallStatus: str
):
    """Handle call status updates from Twilio."""
    logger.info(f"Call status update: {CallSid} - {CallStatus}")
    
    # Handle call completion
    if CallStatus in ["completed", "failed", "busy", "no-answer"]:
        if CallSid in active_calls:
            await _cleanup_call(CallSid)
    
    return {"status": "ok"}


# ============================================================================
# WEBSOCKET AUDIO STREAM HANDLER
# ============================================================================

@app.websocket("/ws/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    """
    Main WebSocket handler for audio streaming.
    Coordinates STT, TTS, and barge-in detection.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {call_sid}")
    
    session: Optional[CallSession] = None
    
    try:
        # Initialize call session
        session = await _initialize_call_session(call_sid, websocket)
        
        # Main audio streaming loop
        await _audio_stream_loop(websocket, session)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_sid}")
    
    except Exception as e:
        logger.error(f"WebSocket error for {call_sid}: {str(e)}", exc_info=True)
    
    finally:
        # Cleanup
        await _cleanup_call(call_sid)


async def _initialize_call_session(
    call_sid: str,
    websocket: WebSocket
) -> CallSession:
    """
    Initialize call session and start conversation.
    
    Args:
        call_sid: Twilio call SID
        websocket: WebSocket connection
        
    Returns:
        Initialized CallSession
    """
    # Create session
    session = CallSession(
        call_id=call_sid,
        restaurant_id=DEFAULT_RESTAURANT_ID
    )
    active_calls[call_sid] = session
    
    # Initialize call in main.py (business logic)
    try:
        await handle_call_start(
            call_id=call_sid,
            restaurant_id=session.restaurant_id,
            customer_phone=session.customer_phone
        )
        logger.info(f"Call initialized: {call_sid}")
        
        # Send initial greeting
        await _send_greeting(websocket, session)
        
    except Exception as e:
        logger.error(f"Failed to initialize call {call_sid}: {str(e)}")
        raise
    
    return session


async def _send_greeting(websocket: WebSocket, session: CallSession):
    """Send initial greeting to caller."""
    greeting = "Hello! Thank you for calling. How can I help you today?"
    
    # Detect language (default to English for greeting)
    language = session.detected_language or "en"
    
    try:
        # Start TTS
        token = await session.start_tts()
        
        # Stream greeting
        success = await stream_tts(
            text=greeting,
            language_code=language,
            websocket=websocket,
            cancellation_token=token
        )
        
        if success:
            session.start_listening()
        else:
            logger.warning(f"Greeting interrupted for {session.call_id}")
        
    except Exception as e:
        logger.error(f"Failed to send greeting: {str(e)}")
        session.start_listening()


async def _audio_stream_loop(websocket: WebSocket, session: CallSession):
    """
    Main audio streaming loop.
    Handles incoming audio, STT, and coordinates responses.
    """
    silence_threshold = 0.5  # Seconds of silence to trigger STT
    last_audio_time = datetime.utcnow()
    
    session.start_listening()
    
    while True:
        try:
            # Receive message from Twilio
            message = await websocket.receive_json()
            
            # Handle different message types
            event = message.get("event")
            
            if event == "connected":
                logger.info(f"Stream connected: {session.call_id}")
                continue
            
            elif event == "start":
                logger.info(f"Stream started: {session.call_id}")
                # Extract customer phone if available
                start_data = message.get("start", {})
                custom_params = start_data.get("customParameters", {})
                session.customer_phone = custom_params.get("from")
                continue
            
            elif event == "media":
                # Audio data received
                media = message.get("media", {})
                payload = media.get("payload")
                
                if not payload:
                    continue
                
                # Decode audio
                audio_chunk = base64.b64decode(payload)
                
                # Check for barge-in
                if session.is_speaking:
                    # User is speaking while TTS is playing - barge-in!
                    await session.stop_tts()
                
                # Buffer audio if listening
                if session.is_listening:
                    session.add_audio_chunk(audio_chunk)
                    last_audio_time = datetime.utcnow()
                
                # Check for silence (end of utterance)
                silence_duration = (datetime.utcnow() - last_audio_time).total_seconds()
                
                if silence_duration >= silence_threshold and len(session.audio_buffer) > 0:
                    # Process buffered audio
                    await _process_user_audio(websocket, session)
            
            elif event == "stop":
                logger.info(f"Stream stopped: {session.call_id}")
                break
            
            else:
                logger.debug(f"Unknown event: {event}")
        
        except WebSocketDisconnect:
            break
        
        except Exception as e:
            logger.error(f"Error in audio loop: {str(e)}", exc_info=True)
            # Continue processing


async def _process_user_audio(websocket: WebSocket, session: CallSession):
    """
    Process buffered user audio through STT and AI pipeline.
    
    Args:
        websocket: WebSocket connection
        session: Call session
    """
    # Don't process if transfer is requested
    if session.transfer_requested or session.transfer_in_progress:
        logger.info(f"Audio processing blocked - transfer active: {session.call_id}")
        return
    
    try:
        # Get buffered audio
        audio_bytes = session.get_and_clear_audio()
        
        if len(audio_bytes) < 1000:  # Too short, ignore
            return
        
        logger.info(f"Processing audio chunk: {len(audio_bytes)} bytes")
        
        # Step 1: Speech-to-Text
        stt_result = await transcribe_audio(
            audio_bytes=audio_bytes,
            language_hint=session.detected_language
        )
        
        text = stt_result.get("text", "").strip()
        
        if not text:
            logger.debug("No text transcribed")
            return
        
        logger.info(f"Transcribed: {text}")
        
        # Update detected language
        detected_lang = stt_result.get("language_code")
        if detected_lang and not session.detected_language:
            session.detected_language = detected_lang
        
        # Step 2: Process through main AI pipeline
        response = await handle_user_text(
            text=text,
            call_id=session.call_id,
            detected_language=detected_lang
        )
        
        # Step 2.5: Check for transfer request
        if response.get("actions", {}).get("transfer_requested"):
            logger.info(f"Transfer requested via AI pipeline: {session.call_id}")
            
            transfer_details = response["actions"].get("transfer_details", {})
            
            # Execute transfer
            await execute_call_transfer(
                call_id=session.call_id,
                websocket=websocket,
                transfer_number=transfer_details.get("transfer_number"),
                reason=transfer_details.get("reason", "Customer request")
            )
            return
        
        response_text = response.get("response_text", "")
        response_language = response.get("language", "en")
        
        if not response_text:
            logger.warning("No response from AI")
            return
        
        logger.info(f"AI Response: {response_text}")
        
        # Step 3: Text-to-Speech and stream back
        token = await session.start_tts()
        
        if not token:
            # TTS blocked (transfer active)
            return
        
        success = await stream_tts(
            text=response_text,
            language_code=response_language,
            websocket=websocket,
            cancellation_token=token
        )
        
        if success:
            # TTS completed without interruption
            session.start_listening()
        else:
            # TTS was interrupted (barge-in)
            logger.info(f"TTS interrupted by barge-in: {session.call_id}")
            session.start_listening()
        
    except Exception as e:
        logger.error(f"Error processing user audio: {str(e)}", exc_info=True)
        # Continue listening
        session.start_listening()


async def execute_call_transfer(
    call_id: str,
    websocket: WebSocket,
    transfer_number: str,
    reason: str = "Customer requested transfer"
):
    """
    Execute call transfer to human agent.
    
    This function:
    1. Stops any ongoing TTS
    2. Plays handoff message
    3. Initiates Twilio call transfer
    4. Logs transfer event
    5. Closes WebSocket connection
    
    Args:
        call_id: Twilio call SID
        websocket: Active WebSocket connection
        transfer_number: Phone number to transfer to (E.164 format)
        reason: Reason for transfer
        
    Raises:
        RuntimeError: If transfer fails
    """
    try:
        session = active_calls.get(call_id)
        
        if not session:
            raise RuntimeError(f"No active session for call: {call_id}")
        
        logger.info(f"Executing call transfer: {call_id} -> {transfer_number}")
        
        # Mark transfer in progress
        session.transfer_in_progress = True
        session.mark_transfer_requested()
        
        # Stop any ongoing TTS
        if session.is_speaking:
            await session.stop_tts()
        
        # Play handoff message
        handoff_message = "One moment please, I'm transferring you to a team member."
        
        # Translate handoff message if needed
        if session.detected_language and session.detected_language != "en":
            try:
                from translate import from_english
                handoff_message = await from_english(
                    handoff_message,
                    session.detected_language
                )
            except Exception as e:
                logger.warning(f"Failed to translate handoff message: {str(e)}")
        
        # Stream handoff message
        try:
            handoff_token = CancellationToken()
            await stream_tts(
                text=handoff_message,
                language_code=session.detected_language or "en",
                websocket=websocket,
                cancellation_token=handoff_token
            )
        except Exception as e:
            logger.warning(f"Failed to play handoff message: {str(e)}")
        
        # Small delay to ensure message is heard
        await asyncio.sleep(0.5)
        
        # Initiate Twilio transfer using API
        from twilio.rest import Client
        from config import get_config
        
        config = get_config()
        twilio_client = Client(
            config.twilio.account_sid,
            config.twilio.auth_token
        )
        
        # Update the call to transfer
        try:
            call = twilio_client.calls(call_id).update(
                twiml=f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Number>{transfer_number}</Number>
    </Dial>
</Response>'''
            )
            
            logger.info(f"Twilio transfer initiated: {call_id}")
            
        except Exception as e:
            logger.error(f"Twilio transfer failed: {str(e)}")
            raise RuntimeError(f"Failed to transfer call: {str(e)}")
        
        # Log transfer to database
        try:
            # Update call log with transfer info
            from db import db
            db.store_call_log({
                "restaurant_id": session.restaurant_id,
                "caller_phone": session.customer_phone or "unknown",
                "call_sid": call_id,
                "direction": "inbound",
                "status": "transferred",
                "transcript": f"Call transferred to {transfer_number}. Reason: {reason}"
            })
            
            logger.info(f"Transfer logged to database: {call_id}")
            
        except Exception as e:
            logger.error(f"Failed to log transfer to database: {str(e)}")
        
        # Clean up session (don't call handle_call_end to avoid circular import)
        # Cleanup will happen naturally through _cleanup_call
        if call_id in active_calls:
            del active_calls[call_id]
        
        logger.info(f"Call transfer complete: {call_id}")
        
    except Exception as e:
        logger.error(f"Call transfer execution failed: {str(e)}", exc_info=True)
        
        # Try to inform caller of error
        try:
            error_message = "I apologize, but I'm unable to transfer your call at this time. Please try calling back."
            
            if session and session.detected_language and session.detected_language != "en":
                from translate import from_english
                error_message = await from_english(error_message, session.detected_language)
            
            error_token = CancellationToken()
            await stream_tts(
                text=error_message,
                language_code=session.detected_language if session else "en",
                websocket=websocket,
                cancellation_token=error_token
            )
        except:
            pass
        
        raise


async def _cleanup_call(call_sid: str):
    """
    Clean up call session and finalize in database.
    
    Args:
        call_sid: Call SID to clean up
    """
    try:
        session = active_calls.get(call_sid)
        
        if not session:
            return
        
        logger.info(f"Cleaning up call: {call_sid}")
        
        # Stop any ongoing TTS
        if session.is_speaking:
            await session.stop_tts()
        
        # Finalize call in business logic
        try:
            await handle_call_end(call_sid)
        except Exception as e:
            logger.error(f"Error finalizing call {call_sid}: {str(e)}")
        
        # Remove from active calls
        if call_sid in active_calls:
            del active_calls[call_sid]
        
        logger.info(f"Call cleanup complete: {call_sid}")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def cleanup_stale_calls():
    """
    Background task to clean up stale call sessions.
    Runs periodically to remove inactive calls.
    """
    stale_threshold = 300  # 5 minutes
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            now = datetime.utcnow()
            stale_calls = []
            
            for call_sid, session in active_calls.items():
                idle_time = (now - session.last_activity).total_seconds()
                
                if idle_time > stale_threshold:
                    stale_calls.append(call_sid)
            
            # Clean up stale calls
            for call_sid in stale_calls:
                logger.warning(f"Cleaning up stale call: {call_sid}")
                await _cleanup_call(call_sid)
        
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    asyncio.create_task(cleanup_stale_calls())
    logger.info("WebSocket server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all active calls on shutdown."""
    logger.info("Shutting down server...")
    
    # Clean up all active calls
    call_sids = list(active_calls.keys())
    for call_sid in call_sids:
        await _cleanup_call(call_sid)
    
    logger.info("Server shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run the WebSocket server."""
    # Validate configuration
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.error("Missing Twilio credentials in environment")
        raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN required")
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Base URL: {BASE_URL}")
    
    # Run server
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
