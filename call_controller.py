"""
Call Controller (Production)
=============================
Central orchestration for voice call lifecycle.

Responsibilities:
- Coordinate state transitions
- Manage cancellation tokens
- Enforce safety rails
- Route events to appropriate handlers
- NO business logic, AI prompts, or database access
"""

import asyncio
import logging
import structlog
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import uuid

from call_state import CallState, CallStateMachine, StateTransitionError
from cancel_token import (
    CancelToken,
    OperationType,
    create_cancel_token,
    cancel_all_operations,
    cleanup_tokens
)
from safety_rails import SafetyRails, SafetyLimits

# Structured logging
logger = structlog.get_logger(__name__)


class CallController:
    """
    Call controller - orchestrates single call lifecycle.
    
    This class:
    - Manages state machine
    - Coordinates cancellation
    - Enforces safety rails
    - Routes events to external handlers
    
    This class does NOT:
    - Make business decisions
    - Call databases directly
    - Embed AI prompts
    - Implement voice pipeline logic
    """
    
    def __init__(
        self,
        call_id: str,
        restaurant_id: str,
        customer_phone: Optional[str] = None,
        safety_limits: Optional[SafetyLimits] = None,
        event_handlers: Optional[Dict[str, Callable]] = None
    ):
        self.call_id = call_id
        self.restaurant_id = restaurant_id
        self.customer_phone = customer_phone
        self.request_id = str(uuid.uuid4())
        
        # Core components
        self.state_machine = CallStateMachine(call_id)
        self.safety_rails = SafetyRails(
            call_id,
            limits=safety_limits,
            on_violation=self._handle_safety_violation
        )
        
        # Event handlers (injected dependencies)
        self.event_handlers = event_handlers or {}
        
        # Cancellation tokens (created on-demand)
        self._tokens: Dict[OperationType, CancelToken] = {}
        
        # Metadata
        self.start_time = datetime.utcnow()
        self.detected_language: Optional[str] = None
        
        # Lifecycle state
        self._started = False
        self._closed = False
        
        logger.info(
            "call_controller_created",
            call_id=call_id,
            restaurant_id=restaurant_id,
            request_id=self.request_id
        )
    
    async def start(self) -> Dict[str, Any]:
        """
        Start the call session.
        
        Returns:
            Initial response with greeting
        """
        if self._started:
            logger.warning("call_already_started", call_id=self.call_id)
            return {"error": "Call already started"}
        
        self._started = True
        
        logger.info(
            "call_starting",
            call_id=self.call_id,
            request_id=self.request_id,
            state=self.state_machine.current_state.value
        )
        
        try:
            # Transition to GREETING
            self.state_machine.transition(CallState.GREETING, reason="call_start")
            
            # Start safety watchdogs
            await self.safety_rails.start_watchdogs()
            
            # Call external greeting handler
            greeting_response = await self._call_handler(
                "on_greeting",
                call_id=self.call_id,
                restaurant_id=self.restaurant_id,
                customer_phone=self.customer_phone
            )
            
            # Transition to LISTENING
            self.state_machine.transition(CallState.LISTENING, reason="greeting_complete")
            
            logger.info(
                "call_started",
                call_id=self.call_id,
                state=self.state_machine.current_state.value
            )
            
            return {
                "status": "started",
                "call_id": self.call_id,
                "request_id": self.request_id,
                **greeting_response
            }
        
        except Exception as e:
            logger.error(
                "call_start_failed",
                call_id=self.call_id,
                error=str(e),
                exc_info=True
            )
            
            # Force close on startup failure
            await self._force_close("start_failure")
            
            raise
    
    async def handle_user_input(
        self,
        text: str,
        detected_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle user input and generate response.
        
        Args:
            text: User input text
            detected_language: Detected language code
            
        Returns:
            Response dictionary
        """
        if self._closed:
            logger.warning("input_on_closed_call", call_id=self.call_id)
            return {"error": "Call is closed"}
        
        start_time = datetime.utcnow()
        
        # Update language
        if detected_language:
            self.detected_language = detected_language
        
        # Update activity
        self.safety_rails.update_activity()
        
        # Check turn limit
        if not self.safety_rails.increment_turn():
            logger.warning(
                "max_turns_exceeded",
                call_id=self.call_id,
                turns=self.safety_rails.turn_count
            )
            await self.close(reason="max_turns")
            return {
                "response_text": "Thank you for calling. Goodbye.",
                "actions": {"end_call": True}
            }
        
        logger.info(
            "user_input_received",
            call_id=self.call_id,
            turn=self.safety_rails.turn_count,
            state=self.state_machine.current_state.value,
            text_length=len(text)
        )
        
        try:
            # Transition to THINKING
            self.state_machine.transition(CallState.THINKING, reason="user_input")
            
            # Create LLM cancel token
            llm_token = await create_cancel_token(self.call_id, OperationType.LLM)
            self._tokens[OperationType.LLM] = llm_token
            
            # Call external AI handler with timeout and cancellation
            try:
                ai_response = await asyncio.wait_for(
                    self._call_handler(
                        "on_ai_request",
                        call_id=self.call_id,
                        user_text=text,
                        language=self.detected_language,
                        cancel_token=llm_token
                    ),
                    timeout=self.safety_rails.limits.max_ai_timeout_seconds
                )
                
                # Reset AI error count on success
                self.safety_rails.reset_ai_errors()
            
            except asyncio.TimeoutError:
                logger.error(
                    "ai_timeout",
                    call_id=self.call_id,
                    timeout=self.safety_rails.limits.max_ai_timeout_seconds
                )
                
                if not self.safety_rails.increment_ai_error():
                    await self.close(reason="max_ai_errors")
                    return {
                        "response_text": "I'm having technical difficulties. Please call back.",
                        "actions": {"end_call": True}
                    }
                
                # Fallback response
                ai_response = await self._call_handler(
                    "on_ai_fallback",
                    call_id=self.call_id,
                    error_type="timeout"
                )
            
            # AI NEVER directly changes state - we interpret and transition
            next_action = ai_response.get("suggested_action")
            
            if next_action == "end_call":
                await self.close(reason="ai_suggested_end")
                return {
                    **ai_response,
                    "actions": {"end_call": True}
                }
            
            # Transition to SPEAKING
            self.state_machine.transition(CallState.SPEAKING, reason="ai_response_ready")
            
            # Create TTS cancel token
            tts_token = await create_cancel_token(self.call_id, OperationType.TTS)
            self._tokens[OperationType.TTS] = tts_token
            
            # Return response (TTS will be handled externally)
            response = {
                **ai_response,
                "call_id": self.call_id,
                "state": self.state_machine.current_state.value,
                "turn": self.safety_rails.turn_count
            }
            
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(
                "user_input_processed",
                call_id=self.call_id,
                state=self.state_machine.current_state.value,
                latency_ms=latency_ms
            )
            
            return response
        
        except StateTransitionError as e:
            logger.error(
                "state_transition_error",
                call_id=self.call_id,
                error=str(e)
            )
            
            # Return fallback without changing state
            return await self._call_handler(
                "on_ai_fallback",
                call_id=self.call_id,
                error_type="state_error"
            )
        
        except Exception as e:
            logger.error(
                "input_processing_error",
                call_id=self.call_id,
                error=str(e),
                exc_info=True
            )
            
            if not self.safety_rails.increment_ai_error():
                await self.close(reason="max_errors")
                return {
                    "response_text": "I'm experiencing technical difficulties. Goodbye.",
                    "actions": {"end_call": True}
                }
            
            return await self._call_handler(
                "on_ai_fallback",
                call_id=self.call_id,
                error_type="exception"
            )
    
    async def handle_speech_complete(self):
        """Handle completion of TTS playback."""
        if self._closed:
            return
        
        logger.debug(
            "speech_complete",
            call_id=self.call_id,
            state=self.state_machine.current_state.value
        )
        
        # Transition back to LISTENING
        try:
            self.state_machine.transition(CallState.LISTENING, reason="speech_complete")
        except StateTransitionError:
            logger.warning(
                "speech_complete_invalid_state",
                call_id=self.call_id,
                state=self.state_machine.current_state.value
            )
    
    async def handle_transfer_request(self, reason: str) -> Dict[str, Any]:
        """
        Handle request to transfer call to human.
        
        Args:
            reason: Reason for transfer
            
        Returns:
            Transfer details
        """
        logger.info(
            "transfer_requested",
            call_id=self.call_id,
            reason=reason
        )
        
        try:
            self.state_machine.transition(CallState.CLOSING, reason="transfer")
            
            transfer_response = await self._call_handler(
                "on_transfer_request",
                call_id=self.call_id,
                reason=reason
            )
            
            return transfer_response
        
        except Exception as e:
            logger.error(
                "transfer_request_failed",
                call_id=self.call_id,
                error=str(e)
            )
            raise
    
    async def close(self, reason: str = "normal") -> Dict[str, Any]:
        """
        Gracefully close the call.
        
        Args:
            reason: Reason for closing
            
        Returns:
            Closure summary
        """
        if self._closed:
            logger.debug("call_already_closed", call_id=self.call_id)
            return {"status": "already_closed"}
        
        logger.info(
            "call_closing",
            call_id=self.call_id,
            reason=reason,
            turns=self.safety_rails.turn_count
        )
        
        try:
            # Transition to CLOSING
            try:
                self.state_machine.transition(CallState.CLOSING, reason=reason)
            except StateTransitionError:
                # Force transition if invalid
                self.state_machine.force_transition(CallState.CLOSING, reason=reason)
            
            # Cancel all ongoing operations
            await cancel_all_operations(self.call_id, reason="call_closing")
            
            # Stop watchdogs
            await self.safety_rails.stop_watchdogs()
            
            # Call external cleanup handler
            cleanup_result = await self._call_handler(
                "on_call_end",
                call_id=self.call_id,
                restaurant_id=self.restaurant_id,
                customer_phone=self.customer_phone,
                stats=self.get_stats()
            )
            
            # Transition to CLOSED
            self.state_machine.transition(CallState.CLOSED, reason="cleanup_complete")
            
            # Cleanup tokens
            await cleanup_tokens(self.call_id)
            
            self._closed = True
            
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            
            logger.info(
                "call_closed",
                call_id=self.call_id,
                reason=reason,
                duration_seconds=duration,
                turns=self.safety_rails.turn_count
            )
            
            return {
                "status": "closed",
                "call_id": self.call_id,
                "duration_seconds": duration,
                **cleanup_result
            }
        
        except Exception as e:
            logger.error(
                "call_close_error",
                call_id=self.call_id,
                error=str(e),
                exc_info=True
            )
            
            # Force close even on error
            await self._force_close("close_error")
            
            raise
    
    async def _force_close(self, reason: str):
        """
        Force close call (emergency cleanup).
        
        Args:
            reason: Reason for forced close
        """
        logger.warning(
            "force_closing_call",
            call_id=self.call_id,
            reason=reason
        )
        
        self._closed = True
        
        # Force state transition
        self.state_machine.force_transition(CallState.CLOSED, reason=reason)
        
        # Cancel everything
        try:
            await cancel_all_operations(self.call_id, reason=f"force_close_{reason}")
        except Exception as e:
            logger.error("cancel_operations_failed", error=str(e))
        
        # Stop watchdogs
        try:
            await self.safety_rails.stop_watchdogs()
        except Exception as e:
            logger.error("stop_watchdogs_failed", error=str(e))
        
        # Cleanup tokens
        try:
            await cleanup_tokens(self.call_id)
        except Exception as e:
            logger.error("cleanup_tokens_failed", error=str(e))
        
        logger.warning(
            "call_force_closed",
            call_id=self.call_id,
            reason=reason
        )
    
    async def _handle_safety_violation(self, violation_type: str):
        """Handle safety rail violation."""
        logger.error(
            "safety_violation",
            call_id=self.call_id,
            violation_type=violation_type
        )
        
        # Force close on safety violation
        await self.close(reason=f"safety_violation_{violation_type}")
    
    async def _call_handler(self, handler_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call external event handler.
        
        Args:
            handler_name: Name of handler
            **kwargs: Handler arguments
            
        Returns:
            Handler response
        """
        handler = self.event_handlers.get(handler_name)
        
        if not handler:
            logger.warning(
                "handler_not_found",
                call_id=self.call_id,
                handler=handler_name
            )
            return {}
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**kwargs)
            else:
                result = handler(**kwargs)
            
            return result or {}
        
        except Exception as e:
            logger.error(
                "handler_error",
                call_id=self.call_id,
                handler=handler_name,
                error=str(e),
                exc_info=True
            )
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get call statistics."""
        return {
            "call_id": self.call_id,
            "request_id": self.request_id,
            "restaurant_id": self.restaurant_id,
            "customer_phone": self.customer_phone,
            "detected_language": self.detected_language,
            "state": self.state_machine.current_state.value,
            "duration_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "safety_stats": self.safety_rails.get_stats(),
            "state_history": self.state_machine.get_history(),
            "is_closed": self._closed
        }
    
    def __repr__(self):
        return (
            f"<CallController call_id={self.call_id} "
            f"state={self.state_machine.current_state.value}>"
        )
