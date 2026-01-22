"""
Main Entry Point (Production)
==============================
Minimal bootstrap for AI voice call system.

Responsibilities:
- Boot the server
- Validate configuration
- Create CallController instances
- Route incoming events to controllers

Does NOT contain:
- Business logic
- AI prompts
- Database access
- Orchestration logic
"""

import os
import sys
import asyncio
import logging
import structlog
from typing import Dict, Optional, Any
from datetime import datetime

# Configuration validation
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Core components
from call_controller import CallController
from safety_rails import SafetyLimits
from call_state import CallState

# External handlers (business logic layer)
from handlers import (
    handle_greeting,
    handle_ai_request,
    handle_ai_fallback,
    handle_call_end,
    handle_transfer_request
)

# Metrics
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# Metrics
if METRICS_ENABLED:
    active_controllers = Gauge(
        'active_call_controllers',
        'Currently active call controllers'
    )
    controller_lifecycle = Counter(
        'controller_lifecycle_total',
        'Controller lifecycle events',
        ['event']
    )
    api_requests = Counter(
        'api_requests_total',
        'API requests',
        ['endpoint', 'status']
    )


# Active controller registry
_controllers: Dict[str, CallController] = {}


class ConfigValidationError(Exception):
    """Configuration validation failed."""
    pass


def validate_config() -> Dict[str, Any]:
    """
    Validate required configuration.
    
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigValidationError: If configuration is invalid
    """
    required_env_vars = [
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
    ]
    
    missing = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        raise ConfigValidationError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    
    # Build config
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "deepgram_api_key": os.getenv("DEEPGRAM_API_KEY"),
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_key": os.getenv("SUPABASE_KEY"),
        "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
        "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
        "twilio_phone_number": os.getenv("TWILIO_PHONE_NUMBER"),
        "max_call_duration": int(os.getenv("MAX_CALL_DURATION", "900")),
        "max_turns": int(os.getenv("MAX_TURNS_PER_CALL", "100")),
        "max_silence_duration": int(os.getenv("MAX_SILENCE_DURATION", "30")),
        "max_ai_timeout": int(os.getenv("MAX_AI_TIMEOUT", "15")),
    }
    
    logger.info(
        "configuration_validated",
        max_call_duration=config["max_call_duration"],
        max_turns=config["max_turns"]
    )
    
    return config


def create_safety_limits(config: Dict[str, Any]) -> SafetyLimits:
    """
    Create safety limits from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SafetyLimits instance
    """
    return SafetyLimits(
        max_call_duration_seconds=config["max_call_duration"],
        max_turns=config["max_turns"],
        max_silence_duration_seconds=config["max_silence_duration"],
        max_ai_timeout_seconds=config["max_ai_timeout"],
    )


def create_event_handlers() -> Dict[str, Any]:
    """
    Create event handler mapping.
    
    Returns:
        Handler dictionary
    """
    return {
        "on_greeting": handle_greeting,
        "on_ai_request": handle_ai_request,
        "on_ai_fallback": handle_ai_fallback,
        "on_call_end": handle_call_end,
        "on_transfer_request": handle_transfer_request,
    }


async def handle_call_start(
    call_id: str,
    restaurant_id: str,
    customer_phone: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle incoming call start event.
    
    Args:
        call_id: Unique call identifier
        restaurant_id: Restaurant ID
        customer_phone: Customer phone number
        
    Returns:
        Initial response
    """
    if call_id in _controllers:
        logger.warning("call_already_exists", call_id=call_id)
        return {"error": "Call already exists"}
    
    try:
        # Validate config
        config = validate_config()
        
        # Create safety limits
        safety_limits = create_safety_limits(config)
        
        # Create event handlers
        event_handlers = create_event_handlers()
        
        # Create controller
        controller = CallController(
            call_id=call_id,
            restaurant_id=restaurant_id,
            customer_phone=customer_phone,
            safety_limits=safety_limits,
            event_handlers=event_handlers
        )
        
        # Store controller
        _controllers[call_id] = controller
        
        if METRICS_ENABLED:
            active_controllers.inc()
            controller_lifecycle.labels(event='created').inc()
        
        # Start call
        response = await controller.start()
        
        logger.info(
            "call_started",
            call_id=call_id,
            restaurant_id=restaurant_id,
            request_id=controller.request_id
        )
        
        if METRICS_ENABLED:
            controller_lifecycle.labels(event='started').inc()
            api_requests.labels(endpoint='call_start', status='success').inc()
        
        return response
    
    except Exception as e:
        logger.error(
            "call_start_failed",
            call_id=call_id,
            error=str(e),
            exc_info=True
        )
        
        # Cleanup on failure
        if call_id in _controllers:
            try:
                await _controllers[call_id].close(reason="start_failure")
            except Exception:
                pass
            del _controllers[call_id]
        
        if METRICS_ENABLED:
            api_requests.labels(endpoint='call_start', status='error').inc()
        
        raise


async def handle_user_text(
    text: str,
    call_id: str,
    detected_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle user text input.
    
    Args:
        text: User input text
        call_id: Call identifier
        detected_language: Detected language code
        
    Returns:
        Response dictionary
    """
    controller = _controllers.get(call_id)
    
    if not controller:
        logger.error("controller_not_found", call_id=call_id)
        if METRICS_ENABLED:
            api_requests.labels(endpoint='user_text', status='not_found').inc()
        return {
            "error": "Call not found",
            "response_text": "I apologize, but I've lost track of our conversation."
        }
    
    try:
        response = await controller.handle_user_input(text, detected_language)
        
        if METRICS_ENABLED:
            api_requests.labels(endpoint='user_text', status='success').inc()
        
        return response
    
    except Exception as e:
        logger.error(
            "user_text_failed",
            call_id=call_id,
            error=str(e),
            exc_info=True
        )
        
        if METRICS_ENABLED:
            api_requests.labels(endpoint='user_text', status='error').inc()
        
        # Return fallback
        return {
            "response_text": "I apologize, could you please repeat that?",
            "language": detected_language or "en"
        }


async def handle_speech_complete(call_id: str) -> Dict[str, Any]:
    """
    Handle TTS speech completion.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Status dictionary
    """
    controller = _controllers.get(call_id)
    
    if not controller:
        logger.warning("controller_not_found_speech_complete", call_id=call_id)
        return {"status": "not_found"}
    
    try:
        await controller.handle_speech_complete()
        return {"status": "ok"}
    
    except Exception as e:
        logger.error(
            "speech_complete_failed",
            call_id=call_id,
            error=str(e)
        )
        return {"status": "error"}


async def handle_call_end(call_id: str) -> Dict[str, Any]:
    """
    Handle call end event.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Closure summary
    """
    controller = _controllers.get(call_id)
    
    if not controller:
        logger.warning("controller_not_found_call_end", call_id=call_id)
        if METRICS_ENABLED:
            api_requests.labels(endpoint='call_end', status='not_found').inc()
        return {"status": "not_found"}
    
    try:
        # Close controller
        response = await controller.close(reason="call_ended")
        
        # Remove from registry
        del _controllers[call_id]
        
        if METRICS_ENABLED:
            active_controllers.dec()
            controller_lifecycle.labels(event='closed').inc()
            api_requests.labels(endpoint='call_end', status='success').inc()
        
        logger.info(
            "call_ended",
            call_id=call_id,
            duration=response.get("duration_seconds")
        )
        
        return response
    
    except Exception as e:
        logger.error(
            "call_end_failed",
            call_id=call_id,
            error=str(e),
            exc_info=True
        )
        
        # Force cleanup
        if call_id in _controllers:
            del _controllers[call_id]
            if METRICS_ENABLED:
                active_controllers.dec()
        
        if METRICS_ENABLED:
            api_requests.labels(endpoint='call_end', status='error').inc()
        
        return {"status": "error", "error": str(e)}


def get_session_state(call_id: str) -> Optional[str]:
    """
    Get current state of call session.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Current state or None
    """
    controller = _controllers.get(call_id)
    return controller.state_machine.current_state.value if controller else None


def get_active_sessions() -> list[str]:
    """
    Get list of active session IDs.
    
    Returns:
        List of call IDs
    """
    return list(_controllers.keys())


def get_controller_stats(call_id: str) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a controller.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Stats dictionary or None
    """
    controller = _controllers.get(call_id)
    return controller.get_stats() if controller else None


async def shutdown():
    """Gracefully shutdown all active calls."""
    logger.info(
        "shutting_down",
        active_calls=len(_controllers)
    )
    
    # Close all active controllers
    close_tasks = []
    for call_id, controller in list(_controllers.items()):
        close_tasks.append(controller.close(reason="shutdown"))
    
    if close_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*close_tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("shutdown_timeout")
    
    _controllers.clear()
    
    if METRICS_ENABLED:
        active_controllers.set(0)
    
    logger.info("shutdown_complete")


# Public API exports
__all__ = [
    "handle_call_start",
    "handle_user_text",
    "handle_speech_complete",
    "handle_call_end",
    "get_session_state",
    "get_active_sessions",
    "get_controller_stats",
    "shutdown",
]


if __name__ == "__main__":
    # Basic validation on import
    try:
        config = validate_config()
        print("✅ Configuration validated successfully")
        print(f"   Max call duration: {config['max_call_duration']}s")
        print(f"   Max turns: {config['max_turns']}")
        print(f"   Max silence: {config['max_silence_duration']}s")
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
